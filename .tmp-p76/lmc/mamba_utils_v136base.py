# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import itertools
import os
from collections.abc import Callable
from typing import Any

import torch

from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch

# llm-scaler v52f (crash-4 R1 onset probe): state-copy event logging on
# every mamba align-mode boundary carry (PRE-COPY / POST-COPY / DEF-PP).
# Detect-only; copies are rare (~1 per block_size tokens per request).
logger = init_logger(__name__)

# llm-scaler v52k (P77 MTP rewrite): the original batch_memcpy_kernel gave
# ONE triton program per copy entry and walked the whole state serially in
# 1 KB iterations — a single-CU multi-megabyte chew on the critical path of
# every step (decode boundary carries AND every chunked-prefill chunk),
# which is the device-side stall mass behind the v52f fingerprint. The
# rewrite is a 2-D chunked grid: axis0 = copy entry, axis1 = BLOCK-sized
# chunk, so the copy saturates the device instead of one CU. Semantics are
# byte-identical (pure memcpy). VLLM_V52K_TIMER=1 arms per-event device
# timing (p50/p99/max rolled every 64 events) for before/after gates.
_MEMCPY_BLOCK = 8192
_V52K_TIMER = os.environ.get("VLLM_V52K_TIMER", "0") == "1"

# llm-scaler v138c (NaN-state spiral root-kill): telem_c3 (2026-10-09)
# proved dest pages are zeroed at alloc (v138b: B-side hashes n0/a0.0 at
# every boundary) yet a request's LIVE conv state can still go 100% NaN
# mid-block (fp16 accumulation overflow -> inf -> fp8_e4m3fn cast = NaN,
# bytes 0x7F/0xFF; the 448 store-clamp only catches finite overflow) and
# then propagate boundary-to-boundary forever via the PRE-COPY carry
# (A-side md5 1f0e8545 n393216/anan at prev=5->6 AND 6->7 — the zombie
# spiral behind the '!!!!' outputs). Sanitizing every carried page to
# finite (NaN/Inf -> 0, the proven-benign zeroed-page value) heals the
# request at its next 512-token boundary instead of spiraling.
# VLLM_MAMBA_SANITIZE_CARRY=0 disables.
_MAMBA_SANITIZE_CARRY = os.environ.get("VLLM_MAMBA_SANITIZE_CARRY", "1") != "0"


def _lmc_sanitize_state_page(page: torch.Tensor) -> None:
    """Zero NaN/Inf elements of one mamba state page, in place (v138c).

    fp8_e4m3fn has no Inf and NaN is exactly bytes 0x7F/0xFF, so the conv
    page is scrubbed through a uint8 view (comparison ops are unreliable
    on fp8). All other dtypes (ssm float16/32/bfloat16) take an isfinite
    masked fill. Stream-ordered: callers enqueue it after the carry copy.
    """
    if page.dtype == torch.float8_e4m3fn:
        b = page.view(torch.uint8)
        b[(b == 0x7F) | (b == 0xFF)] = 0
    else:
        page[~torch.isfinite(page)] = 0
_v52k_window: list = []

# llm-scaler v135 (MambaDType float16 support, 2026-10-09): fp8 SSM pools
# survive the adversarial seeded-full-page battery because their quantize
# casts saturate (e4m3 has no inf; e5m2 measured clean, 0 non-finite pages
# across the full battery), but float16 pages can hold inf: a seeded pass
# whose diverged trajectory exceeds 65504 materializes inf, and the NEXT
# seeded pass consuming that seed NaNs the whole lineage (fp16 legs:
# '!!!!'x24 + all-NaN conv tensor from verified-clean inputs). Repair
# seed pages in place at PRE-COPY staging so no forward ever consumes a
# non-finite seed (nan->0, +-inf->+-65504). No-op for finite pages;
# skipped entirely for non-float16 pools, so the float32 production
# default pays nothing. VLLM_MAMBA_FP16_SEED_REPAIR=0 disables.
_FP16_SEED_REPAIR = (
    os.environ.get("VLLM_MAMBA_FP16_SEED_REPAIR", "1") != "0"
)

# TEMP (LMCDbgFwd): at PRE-COPY time (the moment the forward's initial
# mamba state is staged), log the worker-side block tables and md5 the
# source/dest state pages per group. The LMCache load verified byte-perfect
# at load time, so any divergence must appear here: wrong worker-side page
# ids or zeroed/garbage page content at forward time.
_LMC_DBG_FWD = os.environ.get("LMC_DBG_FWD", "0") == "1"

# llm-scaler telemetry: one-time pool truth (dtype/shape per mamba layer
# tensor — settles the CLI-vs-engine dtype dump discrepancy) + nan-count
# parser for hash strings of the form "md5/n<n>/a<a>/c<c>".
_LMC_POOL_LOGGED = False


def _lmc_nf_of_hash(s: str) -> int:
    try:
        return int(s.split("/n")[1].split("/")[0])
    except Exception:
        return 0


def _lmc_fwd_hash_page(state, block_id: int) -> str:
    import hashlib

    # TEMP (probe10): FULL flattened page (no 262144 cap — the cap made
    # hashes partly structural/prompt-independent), raw-bytes md5, plus
    # non-finite count and abs-max to catch fp16 state overflow (the
    # '!!!!'x24 / EOS@1 outputs are degenerate-distribution signatures).
    pg = state[block_id].reshape(-1)
    torch.xpu.synchronize()
    cpu = pg.detach().cpu()
    if cpu.numel() == 0:
        return "empty"
    f = cpu.float()
    nf = int((~torch.isfinite(f)).sum().item())
    am = float(f.abs().max().item())
    h = hashlib.md5(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()[:8]
    return f"{h}/n{nf}/a{am:.1e}/c{cpu.numel()}"


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    entry = tl.program_id(0)
    chunk = tl.program_id(1)

    src_ptr = tl.load(src_ptrs + entry)
    dst_ptr = tl.load(dst_ptrs + entry)
    size = tl.load(sizes + entry)

    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    curr_src_ptr = (src_ptr + offsets).to(tl.pointer_type(tl.uint8))
    curr_dst_ptr = (dst_ptr + offsets).to(tl.pointer_type(tl.uint8))

    data = tl.load(curr_src_ptr, mask=mask)
    tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs, dst_ptrs, sizes, max_size=None):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    if batch == 0:
        return
    if max_size is None:
        # external callers without a host-side view; costs one tiny D2H
        max_size = int(sizes.max().item())

    grid = (batch, cdiv(max_size, _MEMCPY_BLOCK))
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=_MEMCPY_BLOCK)


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


@dataclasses.dataclass
class MambaCopyBuffers:
    src_ptrs: CpuGpuBuffer
    dst_ptrs: CpuGpuBuffer
    sizes: CpuGpuBuffer
    mamba_group_ids: list[int]
    mamba_spec: MambaSpec
    offset: int = 0
    # llm-scaler v138c: dest page views of the queued carry entries,
    # sanitized (NaN/Inf -> 0) after the batch copy in
    # do_mamba_copy_block. Stream-ordered with the copy and the step's
    # forward kernels, so the GDN kernel reads finite state.
    sanitize_pages: list[torch.Tensor] = dataclasses.field(default_factory=list)

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaCopyBuffers":
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
        entries_per_req = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        ) * len(copy_funcs)
        n = max_num_reqs * entries_per_req
        return cls(
            src_ptrs=make_buffer(n, dtype=torch.uint64),
            dst_ptrs=make_buffer(n, dtype=torch.uint64),
            sizes=make_buffer(n, dtype=torch.int32),
            mamba_group_ids=mamba_group_ids,
            mamba_spec=mamba_spec,
        )


def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        # llm-scaler v52k: structural containment for the accepted-token
        # bias (v52e NaN-poison family). The temporal copy reads
        # block_ids[src + bias]; an acc_cpu beyond what the GDN kernel
        # actually wrote selects an unwritten or foreign block. Clamping
        # the read to the request's own block range means a dishonest
        # counter can only degrade, never poison across requests. The
        # v52f rolling-history detector stays armed as telemetry.
        group_bias = min(accept_token_bias, len(block_ids) - 1 - src_block_idx)
        if group_bias < 0:
            group_bias = 0
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, group_bias + 1
                )

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                if _MAMBA_SANITIZE_CARRY:
                    copy_bufs.sanitize_pages.append(state[dest_block_id])
                offset += 1

    copy_bufs.offset = offset


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        return
    sizes_np = copy_bufs.sizes.np
    max_size = int(sizes_np[:n].max())
    timer_ok = False
    if _V52K_TIMER:
        try:
            ev_start = torch.xpu.Event(enable_timing=True)
            ev_end = torch.xpu.Event(enable_timing=True)
            ev_start.record()
            timer_ok = True
        except Exception:
            timer_ok = False
    batch_memcpy(
        copy_bufs.src_ptrs.copy_to_gpu(n),
        copy_bufs.dst_ptrs.copy_to_gpu(n),
        copy_bufs.sizes.copy_to_gpu(n),
        max_size,
    )
    # llm-scaler v138c: NaN/Inf -> 0 on every carried page, enqueued
    # after the copy on the same stream — the request heals at this
    # boundary instead of propagating a poisoned state page forever.
    if _MAMBA_SANITIZE_CARRY and copy_bufs.sanitize_pages:
        for _pg in copy_bufs.sanitize_pages:
            _lmc_sanitize_state_page(_pg)
        copy_bufs.sanitize_pages.clear()
    if _V52K_TIMER and timer_ok:
        try:
            ev_end.record()
            _v52k_window.append((ev_start, ev_end, int(sizes_np[:n].sum()), n))
            if len(_v52k_window) >= 64:
                torch.xpu.synchronize()
                times = sorted(
                    ev_start.elapsed_time(ev_end)
                    for ev_start, ev_end, _, _ in _v52k_window
                )
                total_mb = sum(b for _, _, b, _ in _v52k_window) / 1e6
                p50 = times[len(times) // 2]
                p99 = times[min(len(times) - 1, int(len(times) * 0.99))]
                logger.info(
                    "llm-scaler v52k COPY-TIMER: events=%d p50=%.3fms p99=%.3fms "
                    "max=%.3fms bytes=%.1fMB eff=%.1fGB/s entries_last=%d",
                    len(times), p50, p99, times[-1], total_mb,
                    total_mb / max(sum(times), 1e-9) * 1e3, n)
                _v52k_window.clear()
        except Exception:
            _v52k_window.clear()


# TEMP (LMCDbgM0): p0-liveness bracket around the batched mamba state
# copies. Req-2's attn page 0 (local-prefix KV) reads zero at load staging
# but was real at the scheduler zero call ~16ms earlier; the PRE-COPY carry
# is the only pool write in that window. Hash attn layer-7 page 0 before /
# after the copies and log ONLY on change. Remove after diagnosis.
_orig_do_mamba_copy_block = do_mamba_copy_block


def _lmc_m0_p0hash() -> str:
    import hashlib
    import sys

    m = sys.modules.get(
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
        ".vllm_v1_adapter"
    )
    ad = getattr(m, "_LMC_DBG_ADAPTER_REF", None) if m is not None else None
    ts = getattr(ad, "_attn_kvcaches", None) if ad is not None else None
    if not ts:
        return "n/a"
    t = ts[1] if len(ts) > 1 else ts[0]
    r = t[:, 0].reshape(-1)
    torch.xpu.synchronize()
    return hashlib.md5(r.cpu().numpy().tobytes()).hexdigest()[:8]


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):
    import os

    if os.environ.get("LMC_DBG_WATCH") != "1":
        return _orig_do_mamba_copy_block(copy_bufs)
    try:
        pre = _lmc_m0_p0hash()
    except Exception:
        pre = "err"
    n_before = copy_bufs.offset
    _orig_do_mamba_copy_block(copy_bufs)
    try:
        torch.xpu.synchronize()
        post = _lmc_m0_p0hash()
        if pre != post:
            logger.info(
                "LMCDbgM0 MAMBA COPY CHANGED p0 n=%d %s -> %s",
                n_before,
                pre,
                post,
            )
    except Exception:
        pass


# llm-scaler v136 (LMCache mamba store): worker-side boundary registry.
# Align mode nulls every non-running boundary block, so a historical
# boundary state only exists between the copy that materializes it and the
# next nulling pass -- the adapter's store can never read it back from the
# pool (the null-table law). Snapshot the just-materialized boundary bytes
# (one page per mamba layer, every group) right after the copy that
# materializes them; wait_for_save consumes the registry to bank REAL
# boundary states into the mamba LMCache engine. Layout: group-major, then
# layer order within the group, then conv/ssm per layer -- the same order
# MambaPageConnector packs and unpacks MemoryObjs.
_LMC_BOUNDARY_REGISTRY: dict[str, list[tuple[int, torch.Tensor]]] = {}
_LMC_BOUNDARY_PER_REQ = int(os.environ.get("LMC_BOUNDARY_KEEP", "8"))
# Leak bound for registry entries the finishing store never consumed
# (aborted requests, engine shutdown): retire oldest request first.
_LMC_BOUNDARY_GLOBAL_CAP = int(os.environ.get("LMC_BOUNDARY_CAP", "48"))
# Retrieved boundary states parked by the adapter's retrieve (CPU bytes);
# preprocess_mamba consumes-and-pops the entry for its request. Value:
# (boundary token count B, flat uint8 CPU tensor of all layer pages).
_LMC_SEED_OVERRIDE: dict[str, tuple[int, torch.Tensor]] = {}


def _lmc_snap_boundary(
    req_id: str,
    aligned_tokens: int,
    dest_block_idx: int,
    req_state: "CachedRequestState",
    kv_cache_config: "KVCacheConfig",
    mamba_group_ids: list[int],
    forward_context: dict[str, Any],
) -> None:
    """D2H-copy the boundary state pages the caller's do_mamba_copy_block
    just materialized (the copies above are stream-ordered before this).
    One flat uint8 CPU tensor per boundary; failures degrade to a missing
    boundary (later lookups recompute), never to a wrong one."""
    # llm-scaler LMC dbg: bisect sentinel — skip boundary snapshots (the
    # entry-time GPU gather + D2H sync) entirely via touch file.
    if os.path.exists("/tmp/lmc_snap_off"):
        return
    try:
        pages: list[torch.Tensor] = []
        pinfo: list[tuple[int, int, torch.dtype]] = []
        total = 0
        for gid in mamba_group_ids:
            bid = req_state.block_ids[gid][dest_block_idx]
            for layer_name in kv_cache_config.kv_cache_groups[gid].layer_names:
                for st in forward_context[layer_name].kv_cache:
                    pg = st[bid].reshape(-1).view(torch.uint8).reshape(-1)
                    pages.append(pg)
                    pinfo.append((total, pg.numel(), st.dtype))
                    total += pg.numel()
        if total == 0:
            return
        flat = torch.empty(total, dtype=torch.uint8, device=pages[0].device)
        off = 0
        for pg in pages:
            flat[off:off + pg.numel()].copy_(pg)
            off += pg.numel()
        cpu = flat.cpu()  # one D2H, synced after the materializing copies
        # llm-scaler telemetry: content stats of the just-materialized
        # boundary — the WRITE-side truth pair for PRE-COPY's read-side
        # hashes (post-copy clean + next-read NaN = poisoned in between).
        # LMC_BOUND_STATS=0 skips the whole block (race-window bisect:
        # this CPU pause perturbs async-overlap timing), LMC_BOUND_MD5=0
        # keeps the nan/absmax scan but drops the md5 cost.
        if os.environ.get("LMC_BOUND_STATS", "1") != "0":
            try:
                import hashlib as _hl

                nf = 0
                am = 0.0
                for off2, nb2, dt2 in pinfo:
                    if dt2 == torch.uint8:
                        continue
                    f2 = cpu[off2:off2 + nb2].view(dt2).float()
                    nf += int((~torch.isfinite(f2)).sum().item())
                    if f2.numel():
                        am = max(am, float(f2.abs().max().item()))
                hh = "-"
                if os.environ.get("LMC_BOUND_MD5", "1") != "0":
                    hh = _hl.md5(cpu.numpy().tobytes()).hexdigest()[:8]
                ids_dbg = [
                    req_state.block_ids[g][dest_block_idx]
                    for g in mamba_group_ids
                ]
                logger.info(
                    "LMCBOUND req=%s B=%d dest=%d ids=%s %s/n%d/a%.2e/c%d",
                    req_id, aligned_tokens, dest_block_idx, ids_dbg,
                    hh, nf, am, total,
                )
            except Exception as e:
                logger.info("LMCBOUND ERR req=%s %s", req_id, repr(e)[:120])
        lst = _LMC_BOUNDARY_REGISTRY.setdefault(req_id, [])
        if lst and lst[-1][0] == aligned_tokens:
            lst[-1] = (aligned_tokens, cpu)
        else:
            lst.append((aligned_tokens, cpu))
        while len(lst) > _LMC_BOUNDARY_PER_REQ:
            lst.pop(0)
        # Global FIFO bound: the finishing request's store pops its own
        # entry at wait_for_save; anything the store never consumed
        # (aborted requests, shutdown) is retired oldest-first here. A
        # dropped entry only turns a later hit into a clean recompute.
        while len(_LMC_BOUNDARY_REGISTRY) > _LMC_BOUNDARY_GLOBAL_CAP:
            oldest = next(iter(_LMC_BOUNDARY_REGISTRY))
            if oldest == req_id:
                break
            _LMC_BOUNDARY_REGISTRY.pop(oldest)
    except Exception as e:
        logger.warning(
            "LMC boundary snapshot failed req=%s: %s", req_id, repr(e)[:120]
        )


def _lmc_seed_write(
    req_id: str,
    seed_tokens: int,
    prev_state_idx: int,
    block_size: int,
    req_state: "CachedRequestState",
    kv_cache_config: "KVCacheConfig",
    mamba_group_ids: list[int],
    forward_context: dict[str, Any],
    seed_bytes: torch.Tensor,
) -> None:
    """Write a retrieved boundary state (seed) into the pages the resume
    will carry FROM. The joint lookup booked exactly B = block_size*j
    tokens as computed, so prev_state_idx = B/block_size - 1 IS the
    boundary's dest block. The normal PRE-COPY then carries the seed into
    the running-state block and the v135 fp16 repair covers this write."""
    want_idx = seed_tokens // block_size - 1
    if want_idx != prev_state_idx or prev_state_idx < 0:
        logger.warning(
            "LMC seed mismatch req=%s B=%d want_idx=%d prev=%d; dropping",
            req_id, seed_tokens, want_idx, prev_state_idx,
        )
        return
    off = 0
    written = 0
    for gid in mamba_group_ids:
        bid = req_state.block_ids[gid][prev_state_idx]
        for layer_name in kv_cache_config.kv_cache_groups[gid].layer_names:
            for st in forward_context[layer_name].kv_cache:
                pg = st[bid].reshape(-1).view(torch.uint8).reshape(-1)
                pg.copy_(
                    seed_bytes[off:off + pg.numel()].to(pg.device, non_blocking=False)
                )
                off += pg.numel()
                written += 1
    logger.info(
        "LMC seed write req=%s B=%d page_idx=%d pages=%d bytes=%d",
        req_id, seed_tokens, prev_state_idx, written, off,
    )


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    # We need to clear mamba_state_idx for resumed requests. When requests are
    # force-preempted (e.g., during reset_prefix_cache / KV cache flush),
    # they appear in resumed_req_ids without a corresponding entry in
    # preempted_req_ids, leaving stale mamba_state_idx entries that can
    # point to block indices beyond the new (smaller) block allocation.
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)
        # llm-scaler v136: a preempted/resumed request re-plans from
        # scratch (fresh joint lookup). The seed override is one-shot;
        # boundary registry entries are ONLY popped by the finishing
        # request's wait_for_save store (which fires later in the SAME
        # step as this cleanup lists the request as finished — popping
        # here would race it away), with a global FIFO cap as the leak
        # bound.
        _LMC_SEED_OVERRIDE.pop(req_id, None)

    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        # llm-scaler v136 (LMCache mamba load): consume the retrieved
        # boundary seed the adapter's retrieve parked for this request.
        # Written into the carry-from page BEFORE any PRE-COPY below, so
        # the normal machinery carries it forward.
        _lmc_seed = _LMC_SEED_OVERRIDE.pop(req_id, None)
        if _lmc_seed is not None:
            _lmc_seed_write(
                req_id,
                _lmc_seed[0],
                prev_state_idx,
                block_size,
                req_state,
                kv_cache_config,
                mamba_group_ids,
                forward_context,
                _lmc_seed[1],
            )

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            # TEMP (LMCDbgFwd): worker-side truth for the forward's initial
            # state. Tables + source/dest page hashes per mamba group.
            if _LMC_DBG_FWD:
                global _LMC_POOL_LOGGED
                if not _LMC_POOL_LOGGED:
                    _LMC_POOL_LOGGED = True
                    try:
                        for _gid in mamba_group_ids:
                            _ln = kv_cache_config.kv_cache_groups[
                                _gid
                            ].layer_names[0]
                            for _st in forward_context[_ln].kv_cache:
                                logger.info(
                                    "LMCPOOL g=%d shape=%s dtype=%s",
                                    _gid, tuple(_st.shape), _st.dtype,
                                )
                    except Exception as e:
                        logger.info("LMCPOOL ERR %s", repr(e)[:120])
                try:
                    parts = []
                    nf_pieces: list[str] = []
                    for gid in mamba_group_ids:
                        bids = req_state.block_ids[gid]
                        lname = kv_cache_config.kv_cache_groups[
                            gid
                        ].layer_names[0]
                        att = forward_context[lname]
                        hashes = []
                        for st in att.kv_cache:
                            hh2 = (
                                _lmc_fwd_hash_page(st, bids[prev_state_idx])
                                + ">"
                                + _lmc_fwd_hash_page(st, bids[curr_state_idx])
                            )
                            hashes.append(hh2)
                            nf_pieces.append(hh2)
                        parts.append(
                            f"g{gid}:{bids[:12]}[{hashes}]"
                        )
                    for gi in range(len(kv_cache_config.kv_cache_groups)):
                        spec = kv_cache_config.kv_cache_groups[gi].kv_cache_spec
                        if type(spec).__name__ != "MambaSpec":
                            aids = req_state.block_ids[gi][:12]
                            ahs = []
                            try:
                                import hashlib as _hl
                                per_log = max(1, 1024 // spec.block_size)
                                gnames = kv_cache_config.kv_cache_groups[gi].layer_names
                                for ln in (gnames[0], gnames[len(gnames) // 2]):
                                    att2 = forward_context[ln]
                                    for st2 in att2.kv_cache:
                                        for b_i in range(min(3, len(aids))):
                                            # probe10 fix: index by the
                                            # request's ACTUAL logical page
                                            # id (was b_i — pooled pages).
                                            lo = aids[b_i] * per_log
                                            pg = st2[lo:lo + per_log].reshape(-1)[:262144]
                                            torch.xpu.synchronize()
                                            cpu2 = pg.detach().cpu()
                                            f2 = cpu2.float()
                                            ahs.append(
                                                _hl.md5(
                                                    cpu2.view(torch.uint8).numpy().tobytes()
                                                ).hexdigest()[:8]
                                                + "/n"
                                                + str(int((~torch.isfinite(f2)).sum().item()))
                                                + "/a"
                                                + f"{float(f2.abs().max().item()):.1e}"
                                            )
                                            nf_pieces.append(ahs[-1])
                            except Exception as e:
                                ahs.append("ERR" + repr(e)[:200])
                            parts.append(
                                f"a{gi}:{aids}[{','.join(ahs)}]"
                            )
                    logger.info(
                        "LMCDbgFwd %s prev=%d curr=%d %s",
                        req_id, prev_state_idx, curr_state_idx,
                        " ".join(parts),
                    )
                    # llm-scaler telemetry: any non-finite state page at
                    # read time — dump the full block tables for post-hoc
                    # correlation with the EngineCore LMCPJ journal.
                    nf_total = 0
                    for _hs in nf_pieces:
                        for _seg in _hs.split(">"):
                            nf_total += _lmc_nf_of_hash(_seg)
                    if nf_total > 0:
                        logger.info(
                            "LMCNANDETECT req=%s prev=%d curr=%d nf=%d"
                            " tables=%s attn=%s",
                            req_id, prev_state_idx, curr_state_idx, nf_total,
                            [
                                list(req_state.block_ids[g])
                                for g in mamba_group_ids
                            ],
                            [
                                list(req_state.block_ids[gi])[:8]
                                for gi in range(
                                    len(kv_cache_config.kv_cache_groups)
                                )
                                if type(
                                    kv_cache_config.kv_cache_groups[gi]
                                    .kv_cache_spec
                                ).__name__
                                != "MambaSpec"
                            ],
                        )
                except Exception as e:
                    logger.info("LMCDbgFwd %s ERR %s", req_id, repr(e))
            # llm-scaler v52f (crash-4 R1 onset probe): log every 2048-
            # boundary running-state carry (rare: ~1 per block_size tokens
            # per request). The bias (= accepted - 1) selects which state
            # slot the copy reads; a bias beyond the state actually written
            # by the GDN kernel reads garbage -> NaN poison (v52e verdict:
            # zombie = all-NaN target logits starting just past a 2048
            # multiple).
            try:
                logger.info(
                    "llm-scaler v52f PRE-COPY: %s slot=%d prev=%d curr=%d "
                    "bias=%d (acc_cpu=%d) computed=%d sched=%d blocksz=%d",
                    req_id, i, prev_state_idx, curr_state_idx,
                    int(input_batch.num_accepted_tokens_cpu[i]) - 1,
                    int(input_batch.num_accepted_tokens_cpu[i]),
                    req_state.num_computed_tokens,
                    int(num_scheduled_tokens), block_size)
            except Exception:
                pass
            # llm-scaler v135 (MambaDType float16 support): repair the
            # seed pages in place BEFORE staging — see the module-header
            # note. Runs only when a copy actually stages a seed (resume
            # / boundary carry); same-slot steps skip (their state was
            # written by this request's own finite passes).
            if _FP16_SEED_REPAIR:
                try:
                    for gid in mamba_group_ids:
                        _lname = kv_cache_config.kv_cache_groups[
                            gid
                        ].layer_names[0]
                        _bids = req_state.block_ids[gid]
                        for _st in forward_context[_lname].kv_cache:
                            if _st.dtype == torch.float16:
                                _pg = _st[_bids[prev_state_idx]]
                                _pg.copy_(
                                    torch.nan_to_num(
                                        _pg,
                                        nan=0.0,
                                        posinf=65504.0,
                                        neginf=-65504.0,
                                    )
                                )
                except Exception as e:
                    logger.warning(
                        "llm-scaler v135 fp16 seed repair failed: %s",
                        repr(e)[:120],
                    )
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    If a blocks is converted from partial block to full block in this step, copy the
    state from the block for running state to the new full block.
    """
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    copy_bufs.offset = 0
    boundary_rows: list[tuple[str, int, int, CachedRequestState]] = []
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        # TODO: how to ensure all blocks that cache_blocks called are cached here?
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            # llm-scaler v52f: partial->full block conversion copy (sync
            # path; async uses run_deferred_postprocess below).
            try:
                logger.info(
                    "llm-scaler v52f POST-COPY: %s slot=%d src=%d dest=%d "
                    "bias=%d running=%d newc=%d aligned=%d acc=%d",
                    req_id, i, src_block_idx, dest_block_idx,
                    accept_token_bias, num_tokens_running_state,
                    new_num_computed_tokens, aligned_new_computed_tokens,
                    num_accepted_tokens)
            except Exception:
                pass
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
            # llm-scaler v136: this boundary just materialized (or was
            # already in place); snapshot it after the batched copies.
            if dest_block_idx >= 0:
                boundary_rows.append(
                    (req_id, aligned_new_computed_tokens, dest_block_idx, req_state)
                )
    do_mamba_copy_block(copy_bufs)
    for rid, aligned, dest_idx, rstate in boundary_rows:
        _lmc_snap_boundary(
            rid,
            aligned,
            dest_idx,
            rstate,
            kv_cache_config,
            mamba_group_ids,
            forward_context,
        )


def run_deferred_postprocess(model_runner, rows):
    """llm-scaler v32: align-mode postprocess deferred from the previous
    step's _update_states_after_model_execute.

    Counts are on host (async pinned copy, event already synced by the
    caller, still in PREVIOUS-step order -- the _prepare_inputs remap has
    not run yet at the execute_model entry consumption point). Per-request
    arithmetic uses the N-step snapshots stashed at deferral. Rows whose
    request finished/dropped, or whose mamba state pointer moved since
    (preempt/resume re-migration), are skipped: their copies are either
    useless (dead state) or obsolete (handled by the migration path).
    """
    copy_bufs = model_runner._get_mamba_copy_bufs()
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    block_size = mamba_spec.block_size
    cpu = model_runner.input_batch.num_accepted_tokens_cpu
    copy_bufs.offset = 0
    boundary_rows: list[tuple[str, int, int, CachedRequestState]] = []
    for i, (rid, computed, sched, draft, src) in enumerate(rows):
        if rid not in model_runner.requests:
            continue
        if src < 0 or model_runner.mamba_state_idx.get(rid, -1) != src:
            continue
        running = computed + sched - draft
        newc = running + int(cpu[i]) - 1
        aligned = newc // block_size * block_size
        if aligned >= running:
            bias = aligned - running
            dest = aligned // block_size - 1
            # llm-scaler v52f: async deferred partial->full copy — the
            # variant that runs on this lane (async scheduling). cpu[i] is
            # the accepted count of the step whose state was last written.
            try:
                logger.info(
                    "llm-scaler v52f DEF-PP: %s row=%d src=%d dest=%d "
                    "bias=%d running=%d newc=%d aligned=%d acc=%d "
                    "(snapshot computed=%d sched=%d draft=%d)",
                    rid, i, src, dest, bias, running, newc, aligned,
                    int(cpu[i]), computed, sched, draft)
            except Exception:
                pass
            collect_mamba_copy_meta(
                copy_bufs,
                model_runner.kv_cache_config,
                model_runner.model.get_mamba_state_copy_func(),
                mamba_group_ids,
                src,
                dest,
                bias,
                model_runner.requests[rid],
                model_runner.compilation_config.static_forward_context,
            )
            if src == dest:
                cpu[i] = 1
            # llm-scaler v136: deferred boundary materialization — snapshot
            # after the batched copies (runs at this step's entry, i.e.
            # BEFORE this step's wait_for_save can consume it).
            if dest >= 0:
                boundary_rows.append(
                    (rid, aligned, dest, model_runner.requests[rid])
                )
    do_mamba_copy_block(copy_bufs)
    for rid, aligned, dest_idx, rstate in boundary_rows:
        _lmc_snap_boundary(
            rid,
            aligned,
            dest_idx,
            rstate,
            model_runner.kv_cache_config,
            mamba_group_ids,
            model_runner.compilation_config.static_forward_context,
        )
