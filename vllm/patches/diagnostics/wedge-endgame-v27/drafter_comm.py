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

F = TypeVar("F", bound=Callable[..., Any])


def drafter_pg_enabled() -> bool:
    value = os.environ.get("VLLM_XPU_DRAFTER_PG", "1")
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


@contextmanager
def drafter_communicator():
    """Run the block with the TP group's collectives on the drafter's
    dedicated oneCCL communicator (XPU only; no-op otherwise)."""
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
    drafter's dedicated communicator. Zero overhead when disabled."""

    if not drafter_pg_enabled():
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        with drafter_communicator():
            return fn(*args, **kwargs)

    return wrapper
