# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""llm-scaler v27 (KNOWN_ISSUES #11): a dedicated oneCCL communicator for
the speculative drafter's collectives.

Background. With TP=2 + MTP speculative decoding, two independent host-side
sites enqueue oneCCL collectives onto the SAME ProcessGroupXCCL communicator
during steady decode:

* the target's verify/decode step (collectives issued by the piecewise
  wrapper between captured XPU-graph pieces, or captured inside them), and
* the eager MTP drafter (per-forward head all-reduces plus the
  vocab-parallel argmax gathers in ``get_top_tokens``; ~6 colls/step per
  the CCL_LOG_LEVEL=debug trace, 2026-08-30).

Under ``--async-scheduling`` the drafter of step N+1 overlaps the target of
step N, so the per-rank ISSUE ORDER into the shared communicator can differ
between the two ranks (and replayed graphs bake capture-time communicator
state). oneCCL matching then inverts and both ranks spin in collective
kernels that never retire: the >=32k serve wedge (py-spy: one rank stuck
submitting the align-mode D2H, the peer a step ahead inside the eager MTP
head; both engines 100% at ~22% EU; no xe reset). Every single-site
configuration (no-spec graphs 0/10, enforce-eager spec 0/13) is clean, and
``VLLM_XPU_ALLOW_COMM_IN_GRAPH=0`` (all collectives eager on one comm) only
moves the wedge to >=133k - the shared communicator is the common factor.

Fix. Give the drafter its own communicator. Every drafter collective then
matches against a communicator whose op sequence is issued in identical
order on both ranks (the drafter is host-lockstep), independent of how the
target's collectives interleave; the target keeps the stock TP group
untouched. Implemented by swapping the TP GroupCoordinator's
``device_communicator``/``device_group`` for the duration of
``LLMBaseProposer.propose``/``dummy_run`` (single-threaded host, restored
in ``finally``; never applied while an XPU graph is capturing).

``VLLM_XPU_DRAFTER_PG=0`` restores the stock shared-communicator behavior.

llm-scaler v48 (DFlash2 real-live perf): drafter TP=1 replication.

The DFlash2 drafter is a 5-layer Qwen3-style model that rides the engine's
TP=2, so every propose step pays ~11 eager oneCCL collectives (2 all-reduces
per layer x 5 + the vocab-parallel head gather) at ~2 ms each on this
fabric: dforward d=25.4 ms / h=60.6 ms vs the 1-layer MTP drafter's
d=1.2 / h=5.5, while both drafters' yields are nearly equal (2.4-2.5
tok/step). The collectives also back-pressure every eager kernel submission
between them, inflating the TQ draft-attention wrapper wall 7.6 ms/call vs
0.45 ms/call on the MTP lane (same file; solo cProfile 1 ms).

Fix: run the drafter under a world=1 TP view - construction, weight load,
and every forward. Each rank then materializes FULL drafter weights
(+1.8 GB/rank bf16, 3.6 GB checkpoint) and all TP collectives short-circuit
(parallel_state: ``if self.world_size == 1: return input_``). The drafter is
tiny next to the target; replication is strictly cheaper than 11 eager
collectives/step. Implemented as ``dflash_tp1()`` - a context manager that
swaps the global TP GroupCoordinator's rank/world/group attributes for a
per-rank self-group (gloo; no device communicator is created at world 1),
entered from DFlash2Proposer._get_model / propose / dummy_run only
(method-scoped; MTP/EAGLE lanes untouched). ``VLLM_XPU_DFLASH_TP1=0``
restores the stock TP-sharded drafter.
"""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

from vllm.logger import init_logger

logger = init_logger(__name__)

_DRAFTER_COORD = None
_INIT_FAILED = False

# v48: TP=1 drafter replication state.
_TP1_COORD = None
_TP1_FAILED = False
_TP1_ACTIVE = False

F = TypeVar("F", bound=Callable[..., Any])


def drafter_pg_enabled() -> bool:
    value = os.environ.get("VLLM_XPU_DRAFTER_PG", "1")
    return value.strip().lower() not in ("0", "false", "no", "off")


def dflash_tp1_enabled() -> bool:
    """v48: whether DFlash2 runs its drafter replicated at TP=1.

    v50: default flipped to OFF — the lane is convicted corrupt (v48
    acceptance 2-10% vs ~59% stock) and co-factor of the fatal
    block_hashes assert (R2/R3 2026-09-05). See dflash2.py v50 note.
    """
    value = os.environ.get("VLLM_XPU_DFLASH_TP1", "0")
    return value.strip().lower() not in ("0", "false", "no", "off")


def get_drafter_coordinator():
    """Lazily build a dedicated TP-sized GroupCoordinator (``drafter_tp``).

    Called symmetrically on every TP rank at the first drafter step; the
    underlying ``torch.distributed.new_group(backend="xccl")`` handshake
    gives it a fresh oneCCL communicator, separate from the stock TP
    group's. Returns ``None`` (and permanently falls back to the shared TP
    communicator) when disabled, TP=1, non-XPU, or on any init failure.
    """
    global _DRAFTER_COORD, _INIT_FAILED
    if not drafter_pg_enabled() or _INIT_FAILED:
        return None
    if _DRAFTER_COORD is not None:
        return _DRAFTER_COORD
    try:
        import torch

        from vllm.platforms import current_platform

        if not current_platform.is_xpu():
            _INIT_FAILED = True
            return None

        from vllm.distributed.parallel_state import (
            get_tp_group,
            get_world_group,
            init_model_parallel_group,
        )

        tp = get_tp_group()
        if tp.world_size <= 1:
            _INIT_FAILED = True
            return None

        coord = init_model_parallel_group(
            group_ranks=[tp.ranks],
            local_rank=get_world_group().local_rank,
            backend="xccl",
            use_message_queue_broadcaster=False,
            group_name="drafter_tp",
        )
        # Warm the fresh communicator eagerly (outside any graph capture) so
        # the first real drafter step never pays - or captures - lazy
        # communicator/kernel setup.
        warm_ar = torch.zeros(8, dtype=torch.float32, device=tp.device)
        coord.all_reduce(warm_ar)
        warm_ag = torch.ones(1, dtype=torch.float32, device=tp.device)
        coord.all_gather(warm_ag, dim=-1)
        torch.xpu.synchronize()
        _DRAFTER_COORD = coord
        logger.info(
            "v27 drafter PG: dedicated xccl communicator ready (ranks=%s)",
            tp.ranks,
        )
    except Exception:
        _INIT_FAILED = True
        logger.exception(
            "v27 drafter PG init failed; the drafter keeps the shared TP "
            "communicator (set VLLM_XPU_DRAFTER_PG=0 to silence this)"
        )
        return None
    return _DRAFTER_COORD


def _get_tp1_coordinator():
    """v48: build (once) a per-rank self GroupCoordinator (``drafter_tp1``).

    ``group_ranks=[[r] for r in tp.ranks]`` is symmetric across ranks (each
    rank calls ``new_group`` for every row and keeps its own); world=1 rows
    skip device-communicator creation entirely, so this never touches
    oneCCL. Returns ``None`` when TP is already 1, non-XPU, or on failure.
    """
    global _TP1_COORD, _TP1_FAILED
    if _TP1_FAILED:
        return None
    if _TP1_COORD is not None:
        return _TP1_COORD
    try:
        from vllm.platforms import current_platform

        if not current_platform.is_xpu():
            _TP1_FAILED = True
            return None

        from vllm.distributed.parallel_state import (
            get_tp_group,
            get_world_group,
            init_model_parallel_group,
        )

        tp = get_tp_group()
        if tp.world_size <= 1:
            _TP1_FAILED = True
            return None

        coord = init_model_parallel_group(
            group_ranks=[[r] for r in tp.ranks],
            local_rank=get_world_group().local_rank,
            backend="gloo",
            use_message_queue_broadcaster=False,
            group_name="drafter_tp1",
        )
        _TP1_COORD = coord
        logger.info(
            "v48 dflash TP1: self-group ready (global rank %s -> world=1 "
            "TP view for the drafter)",
            coord.rank,
        )
    except Exception:
        _TP1_FAILED = True
        logger.exception(
            "v48 dflash TP1 group init failed; the drafter stays TP-sharded "
            "(set VLLM_XPU_DFLASH_TP1=0 to silence this)"
        )
        return None
    return _TP1_COORD


@contextmanager
def dflash_tp1():
    """v48: run the block with the global TP coordinator seen as world=1.

    Swaps ``ranks``/``world_size``/``rank_in_group``/``cpu_group``/
    ``device_group``/``device_communicator`` on the TP GroupCoordinator
    singleton for the duration (single-threaded host; restored in
    ``finally``). Layers constructed and weights loaded inside see full
    (unsharded) shapes; every collective inside short-circuits at
    world_size == 1 without touching oneCCL. Never applied while an XPU
    graph is capturing. No-op when disabled or inapplicable.
    """
    global _TP1_ACTIVE
    if not dflash_tp1_enabled():
        yield
        return
    # Never swap while a graph is capturing - the target's graph capture
    # must bake its own (stock) group state, and the swap must not leak
    # into any captured region.
    try:
        import torch

        if torch.xpu.is_current_stream_capturing():
            yield
            return
    except Exception:
        pass
    coord = _get_tp1_coordinator()
    if coord is None:
        yield
        return
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    if tp.world_size <= 1:
        yield
        return
    saved = (
        tp.ranks,
        tp.world_size,
        tp.rank_in_group,
        tp.cpu_group,
        tp.device_group,
        tp.device_communicator,
    )
    tp.ranks = coord.ranks
    tp.world_size = coord.world_size
    tp.rank_in_group = coord.rank_in_group
    tp.cpu_group = coord.cpu_group
    tp.device_group = coord.device_group
    tp.device_communicator = coord.device_communicator
    _TP1_ACTIVE = True
    try:
        yield
    finally:
        _TP1_ACTIVE = False
        (
            tp.ranks,
            tp.world_size,
            tp.rank_in_group,
            tp.cpu_group,
            tp.device_group,
            tp.device_communicator,
        ) = saved


@contextmanager
def drafter_communicator():
    """Run the block with the TP group's collectives on the drafter's
    dedicated oneCCL communicator (XPU only; no-op otherwise)."""
    # v48: under the active TP1 drafter view there are no cross-rank
    # collectives at all (world=1 short-circuits them before any
    # communicator is consulted), so the dedicated world-2 communicator
    # swap is unnecessary - and its capture guard already ran above.
    if _TP1_ACTIVE:
        yield
        return
    # Check capture state BEFORE anything else: never create (and warm) the
    # communicator while a graph is capturing - its setup collectives would
    # be captured, and the swap must not be baked into any graph. The MTP
    # drafter is eager on XPU; this guards other drafter modes (e.g. dflash
    # piecewise capture).
    try:
        import torch

        if torch.xpu.is_current_stream_capturing():
            yield
            return
    except Exception:
        pass

    coord = get_drafter_coordinator()
    if coord is None:
        yield
        return

    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    saved_comm = tp.device_communicator
    saved_group = tp.device_group
    tp.device_communicator = coord.device_communicator
    tp.device_group = coord.device_group
    try:
        yield
    finally:
        tp.device_communicator = saved_comm
        tp.device_group = saved_group


def with_drafter_communicator(fn: F) -> F:
    """Decorator: route every collective the call issues through the
    drafter's dedicated oneCCL communicator. Zero overhead when disabled."""

    if not drafter_pg_enabled():
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        with drafter_communicator():
            return fn(*args, **kwargs)

    return wrapper
