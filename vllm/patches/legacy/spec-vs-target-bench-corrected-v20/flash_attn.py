# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import copy
import importlib
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

# llm-scaler v19c (KNOWN_ISSUES #07): float(layer._k_scale) below is a D2H
# sync; inside XPU graph capture it raises "wait method cannot be used for an
# event associated with a command graph" and kills boot for
# fp8 KV x nospec x FULL_DECODE_ONLY graphs (4/4 deterministic). Fix caches
# the static load-time scales as python floats; VLLM_ESIMD_F8_SCALE_FIX=0
# restores the original (capture-crashing) read for A/B.
_ESIMD_F8_SCALE_FIX = os.getenv("VLLM_ESIMD_F8_SCALE_FIX", "1") != "0"

from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    flash_attn_supports_quant_query_input,
    get_flash_attn_version,
    is_fa_version_supported,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.worker.workspace import current_workspace_manager

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
import vllm.envs as envs
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)
_DISABLE_XPU_DRAFT_METADATA_REUSE = (
    os.environ.get("DISABLE_XPU_DRAFT_METADATA_REUSE", "0") == "1"
)


class FlashAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    forward_includes_kv_cache_update: bool = False

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if current_platform.is_xpu():
            return max(default_block_size, 64)
        return super().get_preferred_block_size(default_block_size)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        if head_size % 8 != 0:
            return False
        if head_size <= 256:
            return True
        # XPU: triton unified_attention (called via flash_attn_varlen_func)
        # supports head_size up to 512 — needed for gemma4 full attention layers
        if current_platform.is_xpu() and head_size <= 512:
            return True
        if is_fa_version_supported(4):
            return head_size <= 512
        return False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if is_quantized_kv_cache(kv_cache_dtype):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True


def _xpu_split_mixed_causal_varlen(
    flash_attn_varlen_func,
    *,
    query,
    key_cache,
    value_cache,
    output,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal_per_req,
    alibi_slopes,
    window_size,
    block_table,
    softcap,
    fa_version,
    q_descale,
    k_descale,
    v_descale,
    num_splits,
    s_aux,
):
    """XPU fallback for per-request mixed causal/bidirectional attention.

    The XPU FlashAttention (vllm_xpu_kernels FA2) ``is_causal`` arg is a scalar
    bool and has no per-request causal input (unlike upstream FA4/Triton, which
    expose a ``per_seq_causal`` tensor). DiffusionGemma needs causal masking for
    encoder-phase requests and bidirectional attention for denoise-phase
    requests within the same packed batch.

    We split the packed batch into a causal group and a bidirectional group,
    run each through a scalar-causal ``varlen_fwd``, and scatter the per-token
    outputs back into ``output``. Correctness-first (no perf tuning); gated by
    the ``DISABLE_XPU_MIXED_CAUSAL_SPLIT`` env var at the call site.
    """
    device = query.device
    num_reqs = causal_per_req.shape[0]
    num_tokens = query.shape[0]

    # Fast path: a single request (the diffusion canvas case, max_num_seqs=1) is
    # always homogeneous in causal flag, so the split/index/scatter machinery
    # below is pure overhead (~0.3ms/layer of repeat_interleave + nonzero +
    # index_select x5 + .any() syncs + index_copy_). Call varlen_fwd once
    # directly into ``output``. One scalar read replaces two .any() syncs.
    if num_reqs == 1:
        grp = bool(causal_per_req[0])
        group_window_size = window_size
        if (not grp and group_window_size is not None
                and len(group_window_size) == 2 and group_window_size[1] == 0):
            # Non-causal (denoise): symmetric sliding window (see below).
            group_window_size = [group_window_size[0], group_window_size[0]]
        flash_attn_varlen_func(
            q=query, k=key_cache, v=value_cache, out=output,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale, causal=grp, alibi_slopes=alibi_slopes,
            window_size=group_window_size, block_table=block_table,
            softcap=softcap, fa_version=fa_version,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            num_splits=num_splits, s_aux=s_aux,
        )
        return

    # Map each query token to its request, then to that request's causal flag.
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(num_reqs, device=device), q_lens.long()
    )
    causal_per_token = causal_per_req[req_ids]

    # index_copy_ writes back through this view into the real output storage.
    out_flat = output.view(num_tokens, -1)

    for grp in (True, False):
        req_mask = causal_per_req == grp
        if not bool(req_mask.any()):
            continue
        req_idx = req_mask.nonzero(as_tuple=True)[0]
        tok_idx = (causal_per_token == grp).nonzero(as_tuple=True)[0]

        q_g = query.index_select(0, tok_idx)
        q_lens_g = q_lens.index_select(0, req_idx)
        cu_g = torch.zeros(
            req_idx.shape[0] + 1, dtype=cu_seqlens_q.dtype, device=device
        )
        torch.cumsum(q_lens_g, 0, out=cu_g[1:])
        seqused_g = seqused_k.index_select(0, req_idx)
        block_table_g = block_table.index_select(0, req_idx)
        out_g = q_g.new_empty(q_g.shape[:-1] + (value_cache.shape[-1],))

        # q/k/v_descale are per-tensor fp8 scales broadcast to
        # (num_reqs, num_kv_heads) via .expand() (zero-stride). The XPU kernel
        # requires that zero-stride single-scalar view, so re-slice the first
        # n_g rows (slicing preserves zero stride) instead of index_select,
        # which would materialize a strided copy and break the assertion.
        n_g = req_idx.shape[0]
        qd = q_descale[:n_g] if q_descale is not None else None
        kd = k_descale[:n_g] if k_descale is not None else None
        vd = v_descale[:n_g] if v_descale is not None else None

        # For denoise-phase DiffusionGemma requests, non-causal attention must
        # use a symmetric sliding window. Decoder layers store the normal
        # causal sliding-window shape as (left, 0); if we pass that unchanged
        # with causal=False, each canvas token still cannot see tokens on its
        # right, which breaks bidirectional block denoising.
        group_window_size = window_size
        if (
            not grp
            and group_window_size is not None
            and len(group_window_size) == 2
            and group_window_size[1] == 0
        ):
            group_window_size = [group_window_size[0], group_window_size[0]]

        flash_attn_varlen_func(
            q=q_g,
            k=key_cache,
            v=value_cache,
            out=out_g,
            cu_seqlens_q=cu_g,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_g,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=grp,
            alibi_slopes=alibi_slopes,
            window_size=group_window_size,
            block_table=block_table_g,
            softcap=softcap,
            fa_version=fa_version,
            q_descale=qd,
            k_descale=kd,
            v_descale=vd,
            num_splits=num_splits,
            s_aux=s_aux,
        )
        out_flat.index_copy_(0, tok_idx, out_g.reshape(tok_idx.shape[0], -1))


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model.

    Only inspects FlashAttentionImpl layers. Other backends (e.g.
    TurboQuant, MLA) use their own metadata builders and are skipped.
    """
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        if not isinstance(layer.impl, FlashAttentionImpl):
            continue
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        if self.dcp_world_size > 1:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self._dcp_context_kv_lens = torch.zeros(
                max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None
        # Draft MTP steps only change the scalar/query/cache tensor fields.
        # Keep the fast-build metadata object and update those fields in place
        # instead of reconstructing it for every sequential draft token.
        self._draft_metadata: FlashAttentionMetadata | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # Disable AOT schedule for spec-decode proposer (not worth the overhead)
        # and for batch invariance (schedule varies with max_seqlen_q/k).
        aot_schedule = (
            self.aot_schedule and not fast_build and not envs.VLLM_BATCH_INVARIANT
        )

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if is_quantized_kv_cache(cache_dtype):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            context_kv_lens = seq_lens - query_lens
            local_context_kv_lens = get_dcp_local_seq_lens(
                context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            self._dcp_context_kv_lens[:num_reqs] = local_context_kv_lens
            self._dcp_context_kv_lens[num_reqs:] = 0
            dcp_context_kv_lens = self._dcp_context_kv_lens[:num_reqs]

            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> FlashAttentionMetadata:
        """Reuse fast-build metadata across sequential draft steps."""
        if (
            _DISABLE_XPU_DRAFT_METADATA_REUSE
            or draft_index == 0
            or self._draft_metadata is None
        ):
            metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=True,
            )
            reusable = (
                not metadata.use_cascade
                and metadata.dcp_context_kv_lens is None
                and metadata.scheduler_metadata is None
                and metadata.prefix_scheduler_metadata is None
                and metadata.cu_prefix_query_lens is None
                and metadata.prefix_kv_lens is None
                and metadata.suffix_kv_lens is None
            )
            self._draft_metadata = metadata if reusable else None
            return metadata

        metadata = self._draft_metadata
        metadata.num_actual_tokens = common_attn_metadata.num_actual_tokens
        metadata.max_query_len = common_attn_metadata.max_query_len
        metadata.query_start_loc = common_attn_metadata.query_start_loc
        metadata.max_seq_len = common_attn_metadata.max_seq_len
        metadata.seq_lens = common_attn_metadata.seq_lens
        metadata.block_table = common_attn_metadata.block_table_tensor
        metadata.slot_mapping = common_attn_metadata.slot_mapping
        metadata.causal = common_attn_metadata.causal
        return metadata

    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        # llm-scaler v19c (#07): per-layer fp8 KV descale factors, cached as
        # python floats at the first eager call so the ESIMD decode fast path
        # never D2H-syncs inside XPU graph capture. Keyed by id(layer); the
        # scale tensors are static after model load.
        self._esimd_kv_scales: dict[int, tuple[float, float]] = {}

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        logger.info_once(
            "Using FlashAttention version %s",
            self.vllm_flash_attn_version,
        )
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT

        if is_quantized_kv_cache(self.kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        self.supports_quant_query_input = flash_attn_supports_quant_query_input()

        vllm_config = get_current_vllm_config_or_none()
        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else cp_lse_ag_out_rs

        self._dcp_dtype: torch.dtype | None = None
        if vllm_config is not None and self.dcp_world_size > 1:
            self._dcp_dtype = vllm_config.model_config.dtype

        # Opt-in (DGEMMA_FUSED_CAUSAL=1): single-launch CUTLASS-SYCL flash with
        # a per-sequence causal tensor, replacing the 2-call split helper for
        # DiffusionGemma's mixed causal/bidirectional batch. Resolved ONCE here
        # (worker reads os.environ at model-load; spawn inherits it). Output is
        # validated bit-identical to the split path.
        self._use_fused_causal = (
            os.environ.get("DGEMMA_FUSED_CAUSAL", "0") == "1"
        )
        if self._use_fused_causal and self.head_size == 256:
            logger.info_once(
                "DiffusionGemma: using fused per-seq-causal CUTLASS flash "
                "(single launch) in place of the split-FA2 path."
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if os.environ.get("PROFILE_ATTN", "0") == "1" and attn_metadata is not None and attn_metadata.max_query_len == 1:
            import time as _time
            import sys as _sys
            _nq = query.shape[1] if query.dim() >= 2 else 0
            _nkv = kv_cache.shape[3] if kv_cache.dim() >= 4 else 0
            _gqa = _nq // _nkv if _nkv else 0
            _sw = self.sliding_window if hasattr(self, "sliding_window") else None
            _is_sliding = (_sw is not None and _sw[0] != -1)
            torch.xpu.synchronize()
            _t0 = _time.perf_counter()
            _ret = self._inner_forward(layer, query, key, value, kv_cache,
                                        attn_metadata, output,
                                        output_scale, output_block_scale)
            torch.xpu.synchronize()
            _dt = (_time.perf_counter() - _t0) * 1000.0
            global _ATTN_STATS_GLOBAL
            try:
                _ATTN_STATS_GLOBAL
            except NameError:
                _ATTN_STATS_GLOBAL = {}
            _stats = _ATTN_STATS_GLOBAL
            _key = (_gqa, _is_sliding)
            _e = _stats.setdefault(_key, [0, 0.0])
            _e[0] += 1; _e[1] += _dt
            _stats["_call_n"] = _stats.get("_call_n", 0) + 1
            if _stats["_call_n"] % 600 == 0:
                _to_print = dict((k, v) for k, v in _stats.items() if k != "_call_n")
                print(f"[ATTN STATS @{_stats['_call_n']}] {_to_print}", flush=True, file=_sys.stderr)
            return _ret
        return self._inner_forward(layer, query, key, value, kv_cache,
                                    attn_metadata, output,
                                    output_scale, output_block_scale)

    def _inner_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)
        # Fix degenerate strides on size-1 dims (e.g. num_kv_heads=1 with TP).
        # FA3/4 on H100+ uses TMA, which requires ≥16-byte stride alignment.
        # See vllm.utils.torch_utils.canonicalize_singleton_dim_strides.
        fixed_k = canonicalize_singleton_dim_strides(key_cache)
        fixed_v = canonicalize_singleton_dim_strides(value_cache)
        if fixed_k is not key_cache or fixed_v is not value_cache:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashAttention): "
                "shape=%s, key strides before=%s after=%s, "
                "value strides before=%s after=%s",
                key_cache.shape,
                key_cache.stride(),
                fixed_k.stride(),
                value_cache.stride(),
                fixed_v.stride(),
            )
        key_cache, value_cache = fixed_k, fixed_v

        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            q_descale = (
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                return output
            else:
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                )

                # ---- PAGED_ATTN_ESIMD_INSERTED_v1 ----
                # XPU paged attention decode shortcut (esimd custom kernel).
                # Gate: head_dim==256 AND GQA>=2. The kernel now has a NATIVE
                # GQA=2 path (no q-head padding), so gemma4 sliding (GQA=2,
                # hd=256) computes exactly 2 q-heads instead of padding to 4.
                # GQA in {2, 4, 8, ...} go straight to the kernel; other
                # non-multiples-of-4 (e.g. GQA=6) still pad up to 4 below.
                _q_check = query[:num_actual_tokens]
                _num_q_check = _q_check.shape[1] if _q_check.dim() >= 2 else 0
                _num_kv_check = kv_cache.shape[3] if kv_cache.dim() >= 4 else 0
                _gqa_check = (_num_q_check // _num_kv_check) if _num_kv_check > 0 else 0
                if (
                    os.environ.get("DISABLE_ESIMD_PAGE_ATTN") != "1"
                    and current_platform.is_xpu()
                    and attn_type == AttentionType.DECODER
                    and attn_metadata.max_query_len == 1
                    and self.head_size == 256
                    and _gqa_check >= 2
                    # esimd page_attn_decode has NO sliding-window arg; it
                    # attends the FULL cached KV. Sliding-window layers (e.g.
                    # gemma4 sliding, window=1024, hd=256, GQA=2) MUST NOT take
                    # this path or decode attends beyond the window once
                    # seqlen>window -> progressive garbage on long context.
                    # Route them to flash_attn_varlen_func (passes window_size).
                    and self.sliding_window[0] == -1
                    # page_attn_decode is fp16-only and graphs-only:
                    #   - bf16 query: the kernel TORCH_CHECKs query==kHalf,
                    #     so a bf16 serve (qwen3.8 --dtype bfloat16) crashes
                    #     at eagle.sycl entry on the first decode step.
                    #   - fp16 query in EAGER mode (VLLM_XPU_ENABLE_XPU_GRAPH
                    #     unset/0): the kernel returns garbage on this build
                    #     (verified: qwen3.8-27b eager fp16 decode produces
                    #     incoherent text while graph-replayed steps with the
                    #     identical weights/config are correct).
                    # Restrict to the only validated configuration (fp16
                    # query under XPU graphs); everything else falls through
                    # to flash_attn_varlen_func, which handles bf16/eager
                    # correctly. DISABLE_ESIMD_PAGE_ATTN=1 remains a manual
                    # opt-out.
                    and query.dtype == torch.float16
                    and os.environ.get(
                        "VLLM_XPU_ENABLE_XPU_GRAPH", "0") not in ("", "0")
                ):
                    try:
                        eagle_ops = importlib.import_module(
                            "custom_esimd_kernels_vllm.eagle_ops")
                    except ImportError:
                        eagle_ops = None

                    if eagle_ops is not None:
                        _q = query[:num_actual_tokens]
                        # Kernel has built-in softmax_scale = 1/sqrt(head_dim).
                        # Gemma4 uses scale=1.0 (Q/K already normalized).
                        # Compensate: multiply Q by (self.scale / kernel_scale)
                        # = self.scale * sqrt(head_size)
                        _pa_kernel_scale = 1.0 / (self.head_size ** 0.5)
                        if abs(self.scale - _pa_kernel_scale) > 1e-6:
                            _q = (_q * (self.scale / _pa_kernel_scale)).half()
                        _o = output[:num_actual_tokens]
                        _num_q = _q.shape[1]
                        _num_kv = kv_cache.shape[3]
                        _gqa = _num_q // _num_kv if _num_kv > 0 else 0
                        # GQA=2 has a native kernel path; only odd / non-4k
                        # ratios (e.g. 6) still need q-head padding to 4.
                        _need_pad = (_gqa > 2 and _gqa % 4 != 0)

                        # KV cache view: must reinterpret the [2,num_blocks,page,kv_h,hd]
                        # tensor as the actual fp8 dtype the writer used.
                        if self.kv_cache_dtype.startswith("fp8"):
                            _kv_dtype = (
                                torch.float8_e5m2
                                if self.kv_cache_dtype == "fp8_e5m2"
                                else torch.float8_e4m3fn)
                            _kv_for_esimd = kv_cache.view(_kv_dtype)
                            # llm-scaler v19c (#07): float(layer._k/_v_scale)
                            # D2H-syncs and kills XPU graph capture (see the
                            # module-top comment). Cache the static python
                            # floats at the first eager call; if capture ever
                            # precedes every eager call, skip the ESIMD fast
                            # path (falls through to flash_attn_varlen_func)
                            # rather than bake wrong scales into the graph.
                            _sc = self._esimd_kv_scales.get(id(layer))
                            if _sc is None and _ESIMD_F8_SCALE_FIX:
                                if torch.xpu.is_current_stream_capturing():
                                    eagle_ops = None  # uncached at capture
                                else:
                                    _sc = self._esimd_kv_scales[id(layer)] = (
                                        float(layer._k_scale),
                                        float(layer._v_scale),
                                    )
                            if _sc is not None or not _ESIMD_F8_SCALE_FIX:
                                if _sc is not None:
                                    _k_scale, _v_scale = _sc
                                else:
                                    _k_scale = float(layer._k_scale)
                                    _v_scale = float(layer._v_scale)
                            else:
                                _k_scale = 1.0  # unused (fast path skipped)
                                _v_scale = 1.0
                        else:
                            _kv_for_esimd = kv_cache
                            _k_scale = 1.0
                            _v_scale = 1.0

                        if _need_pad and eagle_ops is not None:
                            _padded_gqa = ((_gqa + 3) // 4) * 4
                            _pad_per_kv = _padded_gqa - _gqa
                            _bs, _, _hd = _q.shape
                            _q_grouped = _q.reshape(_bs, _num_kv, _gqa, _hd)
                            _q_pad = torch.nn.functional.pad(
                                _q_grouped, (0, 0, 0, _pad_per_kv))
                            _q_pad = _q_pad.reshape(
                                _bs, _num_kv * _padded_gqa, _hd).contiguous()
                            _o_pad = torch.zeros_like(_q_pad)
                            eagle_ops.page_attn_decode(
                                _q_pad, _kv_for_esimd, block_table, seqused_k,
                                _o_pad, 1, attn_metadata.max_seq_len,
                                _k_scale, _v_scale)
                            _o_grouped = _o_pad.reshape(
                                _bs, _num_kv, _padded_gqa, _hd)
                            _o.copy_(_o_grouped[:, :, :_gqa, :].reshape(
                                _bs, _num_q, _hd))
                        elif eagle_ops is not None and (
                                _gqa == 2 or _gqa >= 4):
                            eagle_ops.page_attn_decode(
                                _q, _kv_for_esimd, block_table, seqused_k,
                                _o, 1, attn_metadata.max_seq_len,
                                _k_scale, _v_scale)
                        else:
                            eagle_ops = None  # unsupported gqa, fall through

                        if eagle_ops is not None:
                            return output
                # ---- end PAGED_ATTN_ESIMD_INSERTED_v1 ----

                # XPU: DiffusionGemma passes a per-request causal bool
                # tensor (encoder=causal, denoise=bidirectional) within one
                # packed batch. The XPU varlen_fwd is_causal arg is scalar, so
                # split the batch by causal flag and run each group once.
                if (
                    isinstance(attn_metadata.causal, torch.Tensor)
                    and os.environ.get("DISABLE_XPU_MIXED_CAUSAL_SPLIT", "0")
                    != "1"
                ):
                    # Opt-in (DGEMMA_FUSED_CAUSAL=1): single-launch CUTLASS-SYCL
                    # flash with a per-sequence causal tensor — handles the
                    # mixed causal(encoder)+bidirectional(denoise) batch in ONE
                    # call, replacing the split-into-two-FA2-calls helper. The
                    # kernel reads per_seq_causal[seq]; bidir seqs use the full
                    # kv range + symmetric window. Output validated bit-identical
                    # to the split path (vs_split=0.0000, 6/6 cases).
                    if self._use_fused_causal:
                        flash_attn_varlen_func(
                            q=query[:num_actual_tokens],
                            k=key_cache,
                            v=value_cache,
                            out=output[:num_actual_tokens],
                            cu_seqlens_q=cu_seqlens_q,
                            max_seqlen_q=max_seqlen_q,
                            seqused_k=seqused_k,
                            max_seqlen_k=max_seqlen_k,
                            softmax_scale=self.scale,
                            causal=True,  # compile-time path; per-seq decides
                            window_size=sliding_window_size,
                            block_table=block_table,
                            softcap=self.logits_soft_cap,
                            fa_version=self.vllm_flash_attn_version,
                            q_descale=q_descale,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            num_splits=attn_metadata.max_num_splits,
                            s_aux=self.sinks,
                            per_seq_causal=attn_metadata.causal,
                        )
                    else:
                        _xpu_split_mixed_causal_varlen(
                            flash_attn_varlen_func,
                            query=query[:num_actual_tokens],
                            key_cache=key_cache,
                            value_cache=value_cache,
                            output=output[:num_actual_tokens],
                            cu_seqlens_q=cu_seqlens_q,
                            seqused_k=seqused_k,
                            max_seqlen_q=max_seqlen_q,
                            max_seqlen_k=max_seqlen_k,
                            softmax_scale=self.scale,
                            causal_per_req=attn_metadata.causal,
                            alibi_slopes=self.alibi_slopes,
                            window_size=sliding_window_size,
                            block_table=block_table,
                            softcap=self.logits_soft_cap,
                            fa_version=self.vllm_flash_attn_version,
                            q_descale=q_descale,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            num_splits=attn_metadata.max_num_splits,
                            s_aux=self.sinks,
                        )
                    return output

                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                    # XPU MTP+graph: route paged multi-query to branch1
                    # (chunk_prefill) so capture avoids branch2's .item().
                    is_mix_batch=False,
                )
                return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        # Scatter write into the KV cache using slot_mapping indices.
        # No TMA kernel is invoked here, so stride canonicalization is not needed.
        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        n = query_across_dcp.shape[0]
        (dcp_context_out,) = current_workspace_manager().get_simultaneous(
            (
                (n, self.num_heads * self.dcp_world_size, self.head_size),
                self._dcp_dtype,
            ),
        )
        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=dcp_context_out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        # FA returns LSE in shape [ H, B ] but DCP combine wants [ B, H ]
        context_attn_out_cor, context_lse_cor = self.dcp_combine(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        (dcp_query_out,) = current_workspace_manager().get_simultaneous(
            ((query.shape[0], self.num_heads, self.head_size), self._dcp_dtype),
        )
        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=dcp_query_out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # For encoder attention, process FP8 quantization if needed
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        # Call flash attention directly on Q, K, V tensors
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape)
            if self.supports_quant_query_input
            else None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        # s_aux is incorporated into prefix_lse inside the GPU kernel,
        # enabling its effect during the final attention merge.
        s_aux=s_aux,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
