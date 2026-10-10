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
_v52k_window: list = []


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

    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

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
    do_mamba_copy_block(copy_bufs)


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
    do_mamba_copy_block(copy_bufs)
