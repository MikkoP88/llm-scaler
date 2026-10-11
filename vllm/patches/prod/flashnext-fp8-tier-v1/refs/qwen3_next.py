# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.esimd_utils import (
    esimd_fused_add_rms_norm,
    esimd_fused_add_rms_norm_batched,
    esimd_gemv_fp8_blockscale_fp16_fused2,
    esimd_gemv_fp8_pert,
    esimd_gemv_fp8_pert_fused2,
    esimd_gemv_fp16,
    esimd_gemv_int4,
    esimd_qkv_split_norm_rope,
    esimd_resadd_norm_gemv_fp8_pert,
    esimd_resadd_norm_gemv_int4_pert,
    moe_forward_full,
    moe_forward_full_v2,
)
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig

from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


def _require_fp16_block_moe_activations(hidden_states: torch.Tensor) -> None:
    if hidden_states.dtype != torch.float16:
        raise RuntimeError(
            "Qwen3 XPU FP8 block MoE currently requires FP16 "
            f"activations, but received {hidden_states.dtype}. Relaunch "
            "vLLM with `--dtype float16`; native BF16 block-MoE support "
            "is not implemented yet."
        )


def _dequantize_fp8_gate_weight(
    gate, weight: torch.Tensor
) -> torch.Tensor:
    """Reconstruct an fp16 router gate weight from an FP8 quantized gate.

    Handles both a per-channel/per-tensor ``weight_scale`` and a DeepSeek-style
    128x128 block ``weight_scale_inv`` ([N/128, K/128]). Returns fp16 [N, K].
    """
    w = weight.to(torch.float32)
    block_scale = getattr(gate, "weight_scale_inv", None)
    if block_scale is not None:
        scale = block_scale.data.to(torch.float32)
        n, k = w.shape
        # The block size must come from the layer, not be inferred from the
        # weight/scale shape ratio: with a non-multiple dimension (e.g. N=129
        # -> 2 scale rows) a ratio-derived size gives 65/65 instead of the
        # correct 128 + 1, silently mis-scaling every row.
        block_size = getattr(gate, "weight_block_size", None)
        if block_size is None or len(block_size) != 2:
            raise RuntimeError(
                "Block-quantized router gate has weight_scale_inv but no "
                "usable weight_block_size; refusing to guess the block shape "
                "for the fused router GEMV. Set DISABLE_ESIMD_MOE_NORM=1 to "
                "fall back to the unfused router."
            )
        bn, bk = int(block_size[0]), int(block_size[1])
        expected = ((n + bn - 1) // bn, (k + bk - 1) // bk)
        if tuple(scale.shape) != expected:
            raise RuntimeError(
                f"Router gate weight_scale_inv shape {tuple(scale.shape)} does "
                f"not match weight shape {(n, k)} with block size {(bn, bk)} "
                f"(expected {expected})."
            )
        # Expand each block scale over its full bn x bk tile, then trim the
        # tail padding when N/K are not multiples of the block size.
        expanded = scale.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
        return (w * expanded[:n, :k]).to(torch.float16)
    scale = getattr(gate, "weight_scale", None)
    if scale is None:
        raise RuntimeError(
            "FP8 router gate has neither weight_scale nor weight_scale_inv; "
            "cannot build the fused router GEMV weight."
        )
    scale = scale.data.to(torch.float32)
    if scale.numel() > 1:
        scale = scale.reshape(-1, 1)
    return (w * scale).to(torch.float16)


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
            or config.shared_expert_intermediate_size <= 0
        ):
            self.shared_expert = None
        else:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_expert",
            )

        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=1 if self.shared_expert is None else None,
            shared_expert_gate=self.shared_expert_gate
            if self.shared_expert is None
            else None,
        )

        # Eligibility for the fused ESIMD MoE decode kernel (moe_forward_full).
        # Decided once. Requires: ESIMD available + not disabled, FP8 quant, a
        # separate (non-None) shared expert MLP whose weights the kernel reads
        # directly (the shared_expert-fused-into-FusedMoE layout used when
        # shared_expert is None is not supported), and n_experts % 16 == 0 (the
        # topk kernel loads logits in 16-wide chunks). Per-call gating to
        # num_tokens <= 128 lives in forward(). The weight dtype check (e4m3 vs
        # e5m2) is handled inside the kernel's host dispatch.
        from vllm.model_executor.layers.esimd_utils import (
            ESIMD_AVAILABLE,
            disable_esimd_moe,
            quant_is_block_fp8,
            quant_is_fp8,
        )

        self._moe_esimd_ok = (
            ESIMD_AVAILABLE
            and not disable_esimd_moe()
            and not parallel_config.enable_expert_parallel
            and quant_is_fp8(quant_config)
            and not quant_is_block_fp8(quant_config)
            and self.shared_expert is not None
            and self.n_routed_experts % 16 == 0
        )
        self._moe_block_esimd_ok = (
            ESIMD_AVAILABLE
            and not disable_esimd_moe()
            and not parallel_config.enable_expert_parallel
            and quant_is_block_fp8(quant_config)
            and self.shared_expert is not None
            and self.n_routed_experts % 16 == 0
            and hasattr(torch.ops.moe_ops, "moe_forward_full_fp8_block")
        )
        self._moe_block_output_buffers: dict[int, torch.Tensor] = {}

        from vllm.model_executor.layers.esimd_utils import (
            disable_esimd_int4,
            quant_is_sym_int4,
        )

        self._moe_int4_esimd_ok = (
            ESIMD_AVAILABLE
            and not disable_esimd_int4()
            and not parallel_config.enable_expert_parallel
            and quant_is_sym_int4(quant_config)
            and self.shared_expert is not None
            and self.n_routed_experts % 16 == 0
        )
        self._int4_moe_weights_cached = False

        # Router logits the DecoderLayer may precompute when it folds the
        # post_attention_layernorm and this gate's GEMV into one fused kernel
        # (see fused_post_attn_router). Consumed (and cleared) in forward().
        self._precomputed_router_logits = None
        # Lazily-built FP8 shadow of the (fp16) gate weight, used only by the
        # fused router GEMV kernel. The fp16 self.gate stays the source of truth
        # for the upstream/prefill router path.
        self._gate_fp8_weight = None
        self._gate_fp8_scale = None
        self._router_buf = None
        self._normed_buf = None

    def fused_post_attn_router(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm_weight_fp16: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse residual-add + post_attn RMSNorm + router GEMV (decode, M==1).

        Replaces the DecoderLayer's ``post_attention_layernorm`` (a fused
        add+norm) plus this block's separate gate GEMV with a single ESIMD
        kernel. Returns ``(router_logits, normed_hidden)`` — both backed by
        reused per-layer buffers. ``residual`` is updated in place.

        ``norm_weight_fp16`` is the Gemma-adjusted ``(w + 1)`` fp16 weight,
        owned/cached by the DecoderLayer. The gate here is fp16 (unlike the
        reference, which quantizes it), so we keep a lazily-quantized FP8
        shadow of the gate weight for the per-tensor GEMV kernel.
        """
        from vllm import _custom_ops as ops
        if self._gate_fp8_weight is None:
            # The fused router GEMV (esimd_resadd_norm_gemv_fp8_pert) needs the
            # gate weight as per-tensor FP8 [N, K] = [n_experts, hidden] with a
            # scalar fp32 scale. When the gate is quantized (quant_config passed
            # through, as upstream now does for qwen3_next), it already holds
            # exactly that -- consume it directly, matching v0.14.0's
            # moe_router_forward(x, gate.weight, gate.weight_scale). Only an
            # unquantized (fp16) gate needs a lazily-quantized FP8 shadow.
            # Re-quantizing an already-FP8 gate.weight crashes: scaled_fp8_quant
            # dispatches on the input dtype and has no FP8 case
            # ("not implemented for Float8_e4m3fn").
            # When LoRA is enabled the gate is wrapped in a *WithLoRA layer;
            # its .weight property proxies through to the base layer, but
            # weight_scale is not proxied. Unwrap to the base layer so an
            # already-FP8 quantized gate is detected instead of being re-
            # quantized (which crashes on FP8 input).
            gate = getattr(self.gate, "base_layer", self.gate)
            gw = gate.weight
            gs = getattr(gate, "weight_scale", None)
            if (
                gw.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                and gs is not None
                and gs.numel() == 1
            ):
                self._gate_fp8_weight = gw.data
                self._gate_fp8_scale = gs.data
            else:
                src = gw.data
                if src.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    # A block-quantized (or otherwise non-per-tensor) FP8 gate:
                    # scaled_fp8_quant dispatches on the input dtype and has no
                    # FP8 case, so dequantize to fp16 first. Block-FP8
                    # checkpoints for this model leave mlp.gate unquantized, so
                    # this is a defensive path for other checkpoints; it runs
                    # once per layer at first decode.
                    src = _dequantize_fp8_gate_weight(gate, src)
                qw, scale = ops.scaled_fp8_quant(
                    src.contiguous(), scale=None
                )
                self._gate_fp8_weight = qw
                self._gate_fp8_scale = scale
            dev = hidden_states.device
            self._router_buf = torch.empty(
                1, self.n_routed_experts, dtype=torch.float16, device=dev
            )
            self._normed_buf = torch.empty(
                1, hidden_states.shape[-1], dtype=torch.float16, device=dev
            )

        esimd_resadd_norm_gemv_fp8_pert(
            hidden_states,
            residual,
            norm_weight_fp16,
            self._gate_fp8_weight,
            self._gate_fp8_scale,
            self._router_buf,
            self._normed_buf,
            eps,
        )
        return self._router_buf, self._normed_buf

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Decode fast path: fuse topk + routed up/down + shared expert + finalize
        # into one ESIMD kernel, replacing the FusedMoE dispatch + separate
        # shared-expert MLP. FP8 is gated to num_tokens <= 128; the INT4 path is
        # gated to num_tokens <= 64 because its grouped GEMM kernel only computes
        # the first 64 M-rows (M_TILES=8 x M_TILE=8) and leaves rows 64.. as
        # uninitialized garbage -> NaN/!!!! cascade for larger batches.
        # prefill/larger batches keep the upstream path. Not under sequence
        # parallelism (the kernel produces the full per-token output directly).
        # A precomputed router-logits buffer (set by the DecoderLayer's fused
        # post_attn_norm + router path) is consumed once here, then cleared so
        # a later non-fused forward recomputes its own logits.
        precomputed_logits = self._precomputed_router_logits
        self._precomputed_router_logits = None

        if (
            self._moe_block_esimd_ok
            and num_tokens <= 12
            and not self.is_sequence_parallel
        ):
            out = self._forward_esimd_moe_block(hidden_states, precomputed_logits)
            if out is not None:
                return out.view(orig_shape)

        if (
            self._moe_esimd_ok
            and num_tokens <= 128
            and not self.is_sequence_parallel
        ):
            out = self._forward_esimd_moe(
                hidden_states, num_tokens, precomputed_logits
            )
            if out is not None:
                return out.view(orig_shape)

        if (
            self._moe_int4_esimd_ok
            and num_tokens <= 64
            and not self.is_sequence_parallel
        ):
            out = self._forward_int4_esimd_moe(
                hidden_states, num_tokens, precomputed_logits
            )
            if out is not None:
                return out.view(orig_shape)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(orig_shape)

    def _forward_esimd_moe(self, hidden_states, num_tokens, precomputed_logits=None):
        """Fused MoE decode via moe_forward_full (BSZ=1) / _v2 (BSZ>1).

        Returns the [num_tokens, hidden] output, or None to fall through to the
        upstream path (e.g. warmup before weights are FP8-quantized). The kernel
        does topk internally, so we pass the raw gate logits. All weights are
        consumed in their native v0.21.0 layout (no transpose; see progress doc).

        ``precomputed_logits`` (when not None) are router logits the DecoderLayer
        already produced inside its fused post_attn_norm + router GEMV kernel,
        letting us skip this block's gate GEMV entirely.
        """
        experts = self.experts
        w13 = getattr(experts, "w13_weight", None)
        # Before process_weights_after_loading the weights are not yet FP8;
        # bail to the upstream path (covers the profile/warmup run too).
        if w13 is None or w13.dtype not in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            return None

        # This fused kernel is per-tensor only (reads scalar per-expert
        # w13_weight_scale). Block-quant experts carry w13_weight_scale_inv
        # instead -> fall through to the modular XPUExpertsFp8Block path.
        if getattr(experts, "w13_weight_scale", None) is None:
            return None

        if precomputed_logits is not None:
            router_logits = precomputed_logits
        elif (
            2 <= hidden_states.shape[0] <= 8
            and esimd_gemv_fp16 is not None
            and hidden_states.dtype == torch.float16
            and hidden_states.is_contiguous()
            and self.gate.weight.dtype == torch.float16
            and self.gate.weight.is_contiguous()
            and self.gate.bias is None
        ):
            # Offline block checkpoints keep the accuracy-sensitive router in
            # fp16. Preserve that weight exactly while avoiding oneDNN's
            # small-M setup path; no FP8 transcode or persistent weight copy.
            router_logits = torch.empty(
                hidden_states.shape[0],
                self.gate.weight.shape[0],
                dtype=torch.float16,
                device=hidden_states.device,
            )
            esimd_gemv_fp16(hidden_states, self.gate.weight, router_logits)
        else:
            router_logits, _ = self.gate(hidden_states)
        # Route through torch.ops.moe_ops.* (dispatcher op) instead of the
        # raw pybind binding so the MoE call appears as an fx node and the
        # platform's splitting_ops can split it out of the captured XPU
        # graph (the ESIMD MoE kernel does host-side alloc/sync and cannot
        # be replayed inside a graph). Same args as the pybind signature.
        moe_fn = (
            torch.ops.moe_ops.moe_forward_full
            if num_tokens == 1
            else torch.ops.moe_ops.moe_forward_full_v2
        )
        out = moe_fn(
            hidden_states,
            router_logits,
            experts.w13_weight,
            experts.w13_weight_scale,
            self.shared_expert.gate_up_proj.weight,
            self.shared_expert.gate_up_proj.weight_scale,
            experts.w2_weight,
            experts.w2_weight_scale,
            self.shared_expert.down_proj.weight,
            self.shared_expert.down_proj.weight_scale,
            self.shared_expert_gate.weight,
            experts.top_k,
            1,  # num_shared_experts
            self.n_routed_experts,
        )
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _forward_esimd_moe_block(self, hidden_states, precomputed_logits=None):
        """Small-batch fused MoE decode for offline 128x128 FP8 block weights."""
        _require_fp16_block_moe_activations(hidden_states)

        experts = self.experts
        w13 = getattr(experts, "w13_weight", None)
        s13 = getattr(experts, "w13_weight_scale_inv", None)
        w2 = getattr(experts, "w2_weight", None)
        s2 = getattr(experts, "w2_weight_scale_inv", None)
        shared_w13 = getattr(self.shared_expert.gate_up_proj, "weight", None)
        shared_s13 = getattr(
            self.shared_expert.gate_up_proj, "weight_scale_inv", None
        )
        shared_w2 = getattr(self.shared_expert.down_proj, "weight", None)
        shared_s2 = getattr(
            self.shared_expert.down_proj, "weight_scale_inv", None
        )
        required = (w13, s13, w2, s2, shared_w13, shared_s13, shared_w2, shared_s2)
        if any(t is None for t in required):
            return None
        if w13.dtype != torch.float8_e4m3fn:
            return None

        if precomputed_logits is not None:
            router_logits = precomputed_logits
        elif (
            2 <= hidden_states.shape[0] <= 8
            and esimd_gemv_fp16 is not None
            and hidden_states.dtype == torch.float16
            and hidden_states.is_contiguous()
            and self.gate.weight.dtype == torch.float16
            and self.gate.weight.is_contiguous()
            and self.gate.bias is None
        ):
            # Offline block checkpoints keep the router in fp16. Preserve the
            # exact checkpoint weight while avoiding oneDNN setup on every
            # small-M decode layer; expert and shared-expert weights remain
            # native block FP8 with their original 128x128 scales.
            router_logits = torch.empty(
                hidden_states.shape[0],
                self.gate.weight.shape[0],
                dtype=torch.float16,
                device=hidden_states.device,
            )
            esimd_gemv_fp16(hidden_states, self.gate.weight, router_logits)
        else:
            router_logits, _ = self.gate(hidden_states)
        logger.info_once(
            "Using fused ESIMD FP8-block MoE decode path for up to 12 tokens."
        )
        num_tokens = hidden_states.shape[0]
        output = self._moe_block_output_buffers.get(num_tokens)
        if (
            output is None
            or output.shape != hidden_states.shape
            or output.dtype != hidden_states.dtype
            or output.device != hidden_states.device
        ):
            output = torch.empty_like(hidden_states)
            self._moe_block_output_buffers[num_tokens] = output
        out = torch.ops.moe_ops.moe_forward_full_fp8_block(
            hidden_states,
            router_logits,
            output,
            w13,
            s13,
            shared_w13,
            shared_s13,
            w2,
            s2,
            shared_w2,
            shared_s2,
            self.shared_expert_gate.weight,
            experts.top_k,
            1,
            self.n_routed_experts,
        )
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _forward_int4_esimd_moe(self, hidden_states, num_tokens, precomputed_logits=None):
        """Fused INT4 MoE decode via moe_forward_full_cutlass_nmajor_int4.

        Returns [num_tokens, hidden] output, or None to fall through (e.g.
        before weights are ready on the first forward / profile run).
        """
        if not self._int4_moe_weights_cached:
            qm = getattr(self.experts, "quant_method", None)
            impl = getattr(qm, "_fused_moe_impl", None)
            if impl is None:
                return None
            self._int4_w13 = impl.w13
            self._int4_w13_scales = impl.gemm1_scales
            self._int4_w2 = impl.w2
            self._int4_w2_scales = impl.gemm2_scales
            self._int4_shared_gu_w = self.shared_expert.gate_up_proj.weight
            self._int4_shared_d_w = self.shared_expert.down_proj.weight
            self._int4_shared_gate_w = self.shared_expert_gate.weight
            from custom_esimd_kernels_vllm.ops import (
                moe_forward_full_cutlass_nmajor_int4 as _moe_int4_bsz1_fn,
                moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits as _moe_int4_tiny_fn,
            )
            self._moe_int4_bsz1_fn = _moe_int4_bsz1_fn
            self._moe_int4_tiny_fn = _moe_int4_tiny_fn
            self._int4_moe_weights_cached = True

        if precomputed_logits is not None:
            router_logits = precomputed_logits
        else:
            router_logits, _ = self.gate(hidden_states)

        _moe_fn = self._moe_int4_bsz1_fn if num_tokens == 1 else self._moe_int4_tiny_fn
        out = _moe_fn(
            hidden_states,
            router_logits,
            self._int4_w13,
            self._int4_w13_scales,
            self._int4_w2,
            self._int4_w2_scales,
            self._int4_shared_gu_w,
            self._int4_shared_d_w,
            self._int4_shared_gate_w,
            self.experts.top_k,
            1,  # num_shared_experts
            self.n_routed_experts,
        )
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Eligibility for the fused ESIMD QKV split+norm+RoPE(+gate) fast path,
        # decided once here since model/rope/quant shape is fixed. The kernel
        # hardcodes headDim==256, q/k GemmaRMSNorm (weight+1, eps=1e-6) and
        # NeoX-style partial RoPE reading rotary_emb.cos_sin_cache directly. The
        # per-call positions.ndim==1 (text-only) check stays in forward(): a
        # multimodal request passes [3, n] T/H/W positions that need the mRoPE
        # section remap this kernel cannot express. Calibrated vs the upstream
        # path on XPU: q/k rel_err ~2e-4, v/gate bit-exact.
        from vllm.model_executor.layers.esimd_utils import (
            ESIMD_AVAILABLE,
            disable_esimd_qkv,
        )

        # ESIMD_AVAILABLE is only True on XPU with the kernel package installed,
        # so it doubles as the platform guard (no current_platform import here).
        self._qkv_esimd_ok = (
            ESIMD_AVAILABLE
            and not disable_esimd_qkv()
            and self.head_dim == 256
            and getattr(self.rotary_emb, "is_neox_style", False)
            and config.rms_norm_eps == 1e-6
        )
        self._qkv_rotary_dim = int(
            self.head_dim * getattr(config, "partial_rotary_factor", 1.0)
        )

        from vllm.model_executor.layers.esimd_utils import (
            disable_esimd_int4,
            quant_is_sym_int4,
        )
        from vllm.model_executor.layers.quantization.sym_int4 import (
            QK4_GROUP_SIZE,
        )

        self._use_esimd_int4_gemv = (
            ESIMD_AVAILABLE
            and quant_is_sym_int4(quant_config)
            and not disable_esimd_int4()
            and QK4_GROUP_SIZE == 128
        )
        self._decode_qkv_int4_buf = None
        self._decode_o_int4_buf = None

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        esimd_positions = self._get_esimd_positions(positions)
        # ntoks <= 128 cap: the kernel dispatches a 2D (totalHeads, ntoks) grid
        # with a 1x1 work-group, which stalls the worker at prefill-scale token
        # counts (observed to hang at ntoks~800). This is a decode / small-batch
        # fast path; prefill falls through to the upstream split+norm+RoPE.
        if (
            self._qkv_esimd_ok
            and not self._use_esimd_int4_gemv
            and esimd_positions is not None
            and hidden_states.shape[0] <= 128
        ):
            return self._forward_esimd_qkv(
                esimd_positions, output, hidden_states
            )

        _ntoks = hidden_states.shape[0]
        if _ntoks == 1 and self._use_esimd_int4_gemv:
            if self._decode_qkv_int4_buf is None:
                N = self.qkv_proj.weight_esimd.shape[0]
                self._decode_qkv_int4_buf = torch.empty(
                    1, N, dtype=torch.float16, device=hidden_states.device
                )
            esimd_gemv_int4(
                hidden_states,
                self.qkv_proj.weight_esimd,
                self.qkv_proj.scale_esimd,
                self._decode_qkv_int4_buf,
            )
            qkv = self._decode_qkv_int4_buf
        elif _ntoks <= 64 and self._use_esimd_int4_gemv:
            from vllm.model_executor.layers.esimd_utils import esimd_gemm_int4_pgrp

            N = self.qkv_proj.weight_esimd.shape[0]
            qkv = torch.empty(
                _ntoks, N, dtype=torch.float16, device=hidden_states.device
            )
            esimd_gemm_int4_pgrp(
                hidden_states,
                self.qkv_proj.weight_esimd,
                self.qkv_proj.scale_esimd,
                qkv,
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)

        # The INT4 projection produces the same fp16 QKV layout consumed by
        # the fused split + q/k norm + RoPE kernel. Keep the INT4 GEMV while
        # avoiding the generic PyTorch post-processing path below.
        if (
            self._qkv_esimd_ok
            and self._use_esimd_int4_gemv
            and esimd_positions is not None
            and _ntoks <= 64
        ):
            self._esimd_qkv_core(qkv, esimd_positions, output)
            return

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        self._project_attention_output(attn_output, output)

    def _project_attention_output(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Project attention output, preserving the INT4 decode fast path."""
        ntoks = attn_output.shape[0]
        if ntoks == 1 and self._use_esimd_int4_gemv:
            if self._decode_o_int4_buf is None:
                N = self.o_proj.weight_esimd.shape[0]
                self._decode_o_int4_buf = torch.empty(
                    1, N, dtype=torch.float16, device=attn_output.device
                )
            esimd_gemv_int4(
                attn_output,
                self.o_proj.weight_esimd,
                self.o_proj.scale_esimd,
                self._decode_o_int4_buf,
            )
            output[:1] = tensor_model_parallel_all_reduce(self._decode_o_int4_buf)
        elif ntoks <= 64 and self._use_esimd_int4_gemv:
            from vllm.model_executor.layers.esimd_utils import esimd_gemm_int4_pgrp

            N = self.o_proj.weight_esimd.shape[0]
            o_buf = torch.empty(
                ntoks, N, dtype=torch.float16, device=attn_output.device
            )
            esimd_gemm_int4_pgrp(
                attn_output,
                self.o_proj.weight_esimd,
                self.o_proj.scale_esimd,
                o_buf,
            )
            output[:ntoks] = tensor_model_parallel_all_reduce(o_buf)
        else:
            output[:], _ = self.o_proj(attn_output)

    @staticmethod
    def _get_esimd_positions(positions: torch.Tensor) -> torch.Tensor | None:
        """Return plain-RoPE positions for the fused QKV decode path."""
        if positions.ndim == 1:
            return positions
        if positions.ndim != 2:
            return None

        # M-RoPE models pass [3, num_tokens] positions even for text-only
        # generation. During a uniform decode all three coordinates advance
        # together, so the temporal row is equivalent to plain RoPE. Prefill
        # and mixed batches must keep the full M-RoPE section remapping.
        batch_descriptor = get_forward_context().batch_descriptor
        if batch_descriptor is not None and batch_descriptor.uniform:
            return positions[0]
        return None

    def _forward_esimd_qkv(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        """Text-only fast path: one ESIMD kernel does QKV split + q/k RMSNorm +
        NeoX partial RoPE (+ sigmoid gate), replacing split/chunk/reshape, two
        GemmaRMSNorm dispatches, the RoPE call and the gate sigmoid. Gated by
        _qkv_esimd_ok and plain/uniform-decode positions in forward()."""
        qkv, _ = self.qkv_proj(hidden_states)
        self._esimd_qkv_core(qkv, positions, output)

    def forward_precomputed_qkv(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ):
        """Attention forward when the qkv projection was already computed by
        the DecoderLayer's fused input_norm + qkv GEMV kernel. Runs only the
        ESIMD split + q/k RMSNorm + RoPE core, attention and o_proj. Requires
        _qkv_esimd_ok (same kernel constraints as _forward_esimd_qkv); decode
        M==1 path."""
        self._esimd_qkv_core(qkv, positions, output)

    def _esimd_qkv_core(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ):
        ntoks = qkv.shape[0]
        dev = qkv.device

        # Preallocate q/k/v/gate decode buffers once (sliced to ntoks, a leading
        # slice so contiguous); the kernel overwrites them fully. ntoks is gated
        # <=128 by the caller. Avoids 3-4 torch.empty per full-attn layer/step.
        if getattr(self, "_qkv_q_buf", None) is None:
            mb = 128
            self._qkv_q_buf = torch.empty(
                mb, self.q_size, dtype=torch.float16, device=dev
            )
            self._qkv_k_buf = torch.empty(
                mb, self.kv_size, dtype=torch.float16, device=dev
            )
            self._qkv_v_buf = torch.empty(
                mb, self.kv_size, dtype=torch.float16, device=dev
            )
            self._qkv_gate_buf = (
                torch.empty(mb, self.q_size, dtype=torch.float16, device=dev)
                if self.attn_output_gate
                else torch.empty(1, dtype=torch.float16, device=dev)
            )
        q = self._qkv_q_buf[:ntoks]
        k = self._qkv_k_buf[:ntoks]
        v = self._qkv_v_buf[:ntoks]
        gate = (
            self._qkv_gate_buf[:ntoks] if self.attn_output_gate else self._qkv_gate_buf
        )

        # cos_sin_cache is fp16 (model dtype) -- kernel reads it directly.
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        esimd_qkv_split_norm_rope(
            qkv,
            q,
            gate,
            k,
            v,
            self.q_norm.weight,
            self.k_norm.weight,
            positions.to(torch.int32),
            self.num_heads,
            self.num_kv_heads,
            self.attn_output_gate,
            self._qkv_rotary_dim,
            cos_sin_cache,
        )

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            # gate already has sigmoid applied inside the kernel.
            attn_output = attn_output * gate

        self._project_attention_output(attn_output, output)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

        self._init_esimd_flags(quant_config, vllm_config.lora_config is not None)

    def _init_esimd_flags(self, quant_config, lora_enabled: bool) -> None:
        """Decide the ESIMD decode fast-path eligibility flags + buffer slots.

        Reads only attributes every DecoderLayer __init__ sets before calling
        this (``layer_type``, ``layer_scale``, ``mlp``, ``linear_attn`` /
        ``self_attn``) plus the ``quant_config`` (passed in, used to gate fp8 vs
        sym_int4 paths), so subclasses that intentionally bypass
        ``Qwen3NextDecoderLayer.__init__`` (e.g. ``Qwen3_5DecoderLayer``, which
        needs hf_text_config / a different GDN layout / RMSNorm class) can build
        their own submodules and then call this once to get the same ESIMD
        wiring -- instead of silently missing every flag (which disables the
        decode fast paths and AttributeErrors on the lazy-ready checks).
        """
        # Eligibility for folding post_attention_layernorm + the MoE router
        # GEMV into one ESIMD kernel at decode (M==1). Requires an FP8 MoE block
        # that itself runs the fused-MoE decode kernel, no layer_scale (the
        # scale multiply happens after the norm and the kernel can't express
        # it), and the kernel available + not disabled. Decided once.
        from vllm.model_executor.layers.esimd_utils import (
            ESIMD_AVAILABLE,
            disable_esimd_attn_input_norm,
            disable_esimd_gdn_input_norm,
            disable_esimd_moe_norm,
            quant_is_fp8,
        )

        from vllm.model_executor.layers.esimd_utils import (
            disable_esimd_int4,
            quant_is_sym_int4,
        )

        _is_fp8 = quant_is_fp8(quant_config)
        _is_int4 = quant_is_sym_int4(quant_config)

        # Both the per-tensor and the 128x128 block FP8 MoE decode fast paths
        # accept router logits precomputed by the fused resadd + RMSNorm +
        # router GEMV kernel, so both may fold this layer's
        # post_attention_layernorm and the block's gate GEMV into one launch.
        # The fusion kernel itself is per-tensor FP8 and is independent of the
        # expert weight layout: it only consumes the (separately quantized)
        # router gate weight, which block-FP8 checkpoints leave unquantized.
        # The fused router GEMV consumes a cached snapshot of the *base* gate
        # weight, so an active LoRA adapter on mlp.gate would silently not
        # affect routing (the unfused path calls self.gate(...) and does apply
        # it). Disable the fusion whenever LoRA is enabled at all -- the LoRA
        # wrapper is substituted after model init, so we cannot inspect
        # self.mlp.gate here to see whether this particular adapter targets it.
        self._moe_norm_esimd_ok = (
            ESIMD_AVAILABLE
            and _is_fp8
            and not disable_esimd_moe_norm()
            and not lora_enabled
            and isinstance(self.mlp, Qwen3NextSparseMoeBlock)
            and (
                getattr(self.mlp, "_moe_esimd_ok", False)
                or getattr(self.mlp, "_moe_block_esimd_ok", False)
            )
            and not self.layer_scale
        )
        if self._moe_norm_esimd_ok:
            logger.info_once(
                "Using fused ESIMD post-attn RMSNorm + MoE router GEMV "
                "decode path."
            )

        self._moe_norm_int4_esimd_ok = (
            ESIMD_AVAILABLE
            and _is_int4
            and not disable_esimd_moe_norm()
            and not disable_esimd_int4()
            and isinstance(self.mlp, Qwen3NextSparseMoeBlock)
            and not self.layer_scale
        )
        self._int4_router_buf = None
        self._int4_normed_buf = None
        # Lazily-cached fp16 (w + 1) post-attn norm weight for the fused kernel.
        # Private clone+contiguous: the XPU cache allocator may otherwise hand
        # this storage to another tensor mid-inference and corrupt the cached
        # weight (same caveat as GemmaRMSNorm._gemma_w_fp16 / the spec).
        self._post_norm_w_fp16 = None

        # Eligibility for folding input_layernorm + the GDN in_proj dual-GEMV
        # into one ESIMD kernel at decode (M==1), for linear_attention layers.
        # Requires the GDN's own dual-GEMV fast path to be enabled (so the proj
        # weights are FP8 per-tensor) and no layer_scale. Decided once; the
        # weight-readiness (FP8 + scalar scale) is re-checked lazily in forward.
        self._gdn_input_norm_esimd_ok = (
            ESIMD_AVAILABLE
            and _is_fp8
            and not disable_esimd_gdn_input_norm()
            and self.layer_type == "linear_attention"
            and getattr(self.linear_attn, "_gdn_proj_esimd_ok", False)
            and not self.layer_scale
        )
        # Lazily-cached fp16 (w + 1) input norm weight + proj output buffers.
        self._input_norm_w_fp16 = None
        self._gdn_fused_qkvz_buf = None
        self._gdn_fused_ba_buf = None
        self._gdn_input_norm_is_block = False
        self._gdn_input_norm_ready = None  # None=undecided; True/False once checked

        # Eligibility for folding input_layernorm + the attention qkv_proj GEMV
        # into one ESIMD kernel at decode (M==1), for full_attention layers.
        # Requires the attention's own ESIMD qkv split+norm+rope fast path (so
        # the rope/head shape is kernel-compatible) and no layer_scale. Decided
        # once; FP8 weight readiness is re-checked lazily in forward.
        self._attn_input_norm_esimd_ok = (
            ESIMD_AVAILABLE
            and _is_fp8
            and not disable_esimd_attn_input_norm()
            and self.layer_type == "full_attention"
            and getattr(self.self_attn, "_qkv_esimd_ok", False)
            and not self.layer_scale
        )
        self._attn_input_norm_w_fp16 = None
        self._attn_fused_qkv_buf = None
        self._attn_input_norm_is_block = False
        self._attn_input_norm_ready = None  # None=undecided; True/False once checked
        # Preallocated [1, hidden] attention-output buffer for the M==1 fused
        # decode path, reused across steps instead of a per-step empty_like.
        self._decode_attn_out_buf = None

    def _fused_gdn_input_proj(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> bool:
        """Fuse input_layernorm + GDN in_proj dual-GEMV (decode M==1).

        Two-kernel path matching v0.14.0: esimd_fused_add_rms_norm_batched does
        residual += hidden and writes normed = rmsnorm(residual)*(w+1) back into
        hidden_states in place, then esimd_gemv_fp8_pert_fused2 does the dual
        in_proj GEMV (qkvz + ba) off the normed hidden. This replaces the single
        fused esimd_resadd_norm_gemv2_fp8_pert kernel: at the large in_proj
        output dim ({3088} per rank) the folded-norm GEMV profiles slower
        (~22.7us/call) than the split norm + plain dual-GEMV (~3.8 + 16.7us),
        same as the attn qkv path.
        Returns True if it ran (buffers now hold the projections), else False to
        fall through to the standard input_layernorm + GDN forward.

        Readiness (both projections FP8 per-tensor, no bias) is decided lazily on
        the first decode -- weight_scale only exists after weight loading.
        """
        gdn = self.linear_attn
        if self._gdn_input_norm_ready is None:
            wq = gdn.in_proj_qkvz.weight
            wb = gdn.in_proj_ba.weight
            sq = getattr(gdn.in_proj_qkvz, "weight_scale", None)
            sb = getattr(gdn.in_proj_ba, "weight_scale", None)
            sq_block = getattr(gdn.in_proj_qkvz, "weight_scale_inv", None)
            q_block_size = getattr(gdn.in_proj_qkvz, "weight_block_size", None)
            per_tensor_ready = (
                wq.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                and wb.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                and sq is not None
                and sb is not None
                and sq.numel() == 1
                and sb.numel() == 1
            )
            block_ready = (
                esimd_gemv_fp8_blockscale_fp16_fused2 is not None
                and wq.dtype == torch.float8_e4m3fn
                and wb.dtype == torch.float16
                and sq_block is not None
                and tuple(q_block_size or ()) == (128, 128)
            )
            self._gdn_input_norm_ready = (
                (per_tensor_ready or block_ready)
                and gdn.in_proj_qkvz.bias is None
                and gdn.in_proj_ba.bias is None
            )
            if self._gdn_input_norm_ready:
                self._gdn_input_norm_is_block = block_ready
                dev = hidden_states.device
                self._input_norm_w_fp16 = (
                    (self.input_layernorm.weight.data + 1.0)
                    .half()
                    .clone()
                    .contiguous()
                )
                self._gdn_fused_qkvz_buf = torch.empty(
                    1, wq.shape[0], dtype=torch.float16, device=dev
                )
                self._gdn_fused_ba_buf = torch.empty(
                    1, wb.shape[0], dtype=torch.float16, device=dev
                )
        if not self._gdn_input_norm_ready:
            return False

        # residual += hidden; hidden = rmsnorm(residual) * (input_norm.weight+1),
        # written back into hidden_states in place (matches input_layernorm).
        esimd_fused_add_rms_norm_batched(
            hidden_states,
            residual,
            self._input_norm_w_fp16,
            self.input_layernorm.variance_epsilon,
        )
        if self._gdn_input_norm_is_block:
            esimd_gemv_fp8_blockscale_fp16_fused2(
                hidden_states,
                gdn.in_proj_qkvz.weight,
                gdn.in_proj_qkvz.weight_scale_inv,
                self._gdn_fused_qkvz_buf,
                gdn.in_proj_ba.weight,
                self._gdn_fused_ba_buf,
            )
        else:
            # Original per-tensor path remains independently dispatched.
            esimd_gemv_fp8_pert_fused2(
                hidden_states,
                gdn.in_proj_qkvz.weight,
                gdn.in_proj_qkvz.weight_scale,
                self._gdn_fused_qkvz_buf,
                gdn.in_proj_ba.weight,
                gdn.in_proj_ba.weight_scale,
                self._gdn_fused_ba_buf,
            )
        return True

    def _fused_attn_input_proj(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ):
        """Fuse input_layernorm + attention qkv_proj GEMV (decode M==1).

        Two-kernel path matching v0.14.0: esimd_fused_add_rms_norm updates
        residual += hidden and writes normed = rmsnorm(residual)*(w+1) back into
        hidden_states in place, then esimd_gemv_fp8_pert does the plain qkv GEMV
        off the normed hidden. This replaces the single fused
        esimd_resadd_norm_gemv_fp8_pert kernel, which folds the RMSNorm into the
        GEMV: that fusion regresses badly at the large qkv output dim (~2560 per
        rank) -- profiled at ~42us/call vs ~14.5us for the plain GEMV plus a
        cheap batched norm. The router GEMV (small {512} output) keeps the fused
        kernel; only this large-output attn qkv path is split.
        Returns the [1, qkv_out] projection buffer, or None to fall through.

        Readiness is decided lazily on the first decode after weight loading.
        The qkv projection must have no bias and use either per-tensor FP8
        (scalar weight_scale) or 128x128 block-scaled FP8
        (weight_scale_inv with weight_block_size=(128, 128)).
        """
        attn = self.self_attn
        if self._attn_input_norm_ready is None:
            w = attn.qkv_proj.weight
            ws = getattr(attn.qkv_proj, "weight_scale", None)
            ws_block = getattr(attn.qkv_proj, "weight_scale_inv", None)
            weight_block_size = getattr(attn.qkv_proj, "weight_block_size", None)
            block_shape = (
                tuple(weight_block_size) if weight_block_size is not None else ()
            )
            per_tensor_ready = ws is not None and ws.numel() == 1
            block_ready = (
                w.dtype == torch.float8_e4m3fn
                and ws_block is not None
                and block_shape == (128, 128)
            )
            self._attn_input_norm_ready = (
                w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                and (per_tensor_ready or block_ready)
                and attn.qkv_proj.bias is None
            )
            if self._attn_input_norm_ready:
                self._attn_input_norm_is_block = block_ready
                dev = hidden_states.device
                self._attn_input_norm_w_fp16 = (
                    (self.input_layernorm.weight.data + 1.0)
                    .half()
                    .clone()
                    .contiguous()
                )
                if not block_ready:
                    self._attn_fused_qkv_buf = torch.empty(
                        1, w.shape[0], dtype=torch.float16, device=dev
                    )
        if not self._attn_input_norm_ready:
            return None

        # residual += hidden; hidden = rmsnorm(residual) * (input_norm.weight+1),
        # written back into hidden_states in place (matches the non-fused
        # input_layernorm(hidden, residual) semantics the slow path uses).
        esimd_fused_add_rms_norm_batched(
            hidden_states,
            residual,
            self._attn_input_norm_w_fp16,
            self.input_layernorm.variance_epsilon,
        )
        if self._attn_input_norm_is_block:
            # The block-scaled qkv projection remains a W8A16 GEMV, then feeds
            # the same fused split + q/k norm + RoPE core as per-tensor FP8.
            # This keeps the original 128x128 scales intact while removing the
            # dozens of eager tensor kernels in Qwen3NextAttention.forward.
            qkv, _ = attn.qkv_proj(hidden_states)
            return qkv
        # Plain per-tensor FP8 qkv GEMV off the normed hidden.
        esimd_gemv_fp8_pert(
            hidden_states,
            attn.qkv_proj.weight,
            attn.qkv_proj.weight_scale,
            self._attn_fused_qkv_buf,
        )
        return self._attn_fused_qkv_buf

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        # Decode fast path (M==1, FP8): fold input_layernorm (residual-add +
        # RMSNorm) and the attention input projection GEMV into one ESIMD
        # kernel, then run attention off the precomputed projection. Saves a
        # kernel launch + a normed-buffer round-trip per layer.
        #   - linear_attention: dual-GEMV (in_proj_qkvz + in_proj_ba) ->
        #     forward_xpu_precomputed_proj
        #   - full_attention: single qkv_proj GEMV -> forward_precomputed_qkv
        decode_m1 = (
            residual is not None
            and hidden_states.shape[0] == 1
            and hidden_states.dtype == torch.float16
        )
        fused_gdn = (
            decode_m1
            and self._gdn_input_norm_esimd_ok
            and self._fused_gdn_input_proj(hidden_states, residual)
        )
        fused_attn = False
        if not fused_gdn and decode_m1 and self._attn_input_norm_esimd_ok:
            qkv = self._fused_attn_input_proj(hidden_states, residual)
            fused_attn = qkv is not None
        if fused_gdn or fused_attn:
            # M==1 fused decode: reuse one preallocated [1, hidden] output
            # buffer instead of a per-step torch.empty_like. A layer is either
            # linear_attention or full_attention, so the two branches never
            # share a step and one buffer suffices. Safe under in-order stream
            # execution (this step's writes/reads are enqueued before the next
            # step reuses it), matching the reference's _m_attn_output reuse.
            if self._decode_attn_out_buf is None:
                self._decode_attn_out_buf = torch.empty_like(hidden_states)
            self_attention_output = self._decode_attn_out_buf
            if fused_gdn:
                self.linear_attn.forward_xpu_precomputed_proj(
                    self._gdn_fused_qkvz_buf,
                    self._gdn_fused_ba_buf,
                    self_attention_output,
                )
            else:
                self.self_attn.forward_precomputed_qkv(
                    qkv, positions, self_attention_output
                )
            hidden_states = self_attention_output
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            elif hidden_states.shape[0] == 1 and hidden_states.dtype == torch.float16:

                if getattr(self, "_input_norm_w_fp16_cache", None) is None:
                    self._input_norm_w_fp16_cache = (
                        (self.input_layernorm.weight.data + 1.0)
                        .half()
                        .clone()
                        .contiguous()
                    )
                esimd_fused_add_rms_norm(
                    hidden_states,
                    residual,
                    self._input_norm_w_fp16_cache,
                    self.input_layernorm.variance_epsilon,
                )
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            if hidden_states.shape[0] == 1:
                if self._decode_attn_out_buf is None:
                    self._decode_attn_out_buf = torch.empty_like(hidden_states)
                self_attention_output = self._decode_attn_out_buf
            else:
                self_attention_output = torch.empty_like(hidden_states)
            if self.layer_type == "linear_attention":
                if hidden_states.shape[0] == 1:
                    self.linear_attn.forward_xpu(hidden_states, self_attention_output)
                else:
                    self.linear_attn(
                        hidden_states=hidden_states,
                        output=self_attention_output,
                    )
            elif self.layer_type == "full_attention":
                if hidden_states.shape[0] == 1:
                    self.self_attn.forward(
                        positions=positions,
                        output=self_attention_output,
                        hidden_states=hidden_states,
                    )
                else:
                    self.self_attn(
                        hidden_states=hidden_states,
                        output=self_attention_output,
                        positions=positions,
                    )
            else:
                raise ValueError("Invalid layer_type")
            hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        # Fully Connected
        # Decode fast path (M==1, FP8 MoE): one ESIMD kernel does the
        # post_attention_layernorm residual-add + RMSNorm AND the router gate
        # GEMV, handing the logits to the SparseMoeBlock so it skips its own
        # gate GEMV. residual is updated in place; normed hidden is returned.
        if (
            self._moe_norm_esimd_ok
            and residual is not None
            and hidden_states.shape[0] == 1
            and hidden_states.dtype == torch.float16
        ):
            if self._post_norm_w_fp16 is None:
                self._post_norm_w_fp16 = (
                    (self.post_attention_layernorm.weight.data + 1.0)
                    .half()
                    .clone()
                    .contiguous()
                )
            router_logits, normed = self.mlp.fused_post_attn_router(
                hidden_states,
                residual,
                self._post_norm_w_fp16,
                self.post_attention_layernorm.variance_epsilon,
            )
            self.mlp._precomputed_router_logits = router_logits
            hidden_states = self.mlp.forward(normed)
        elif (
            self._moe_norm_int4_esimd_ok
            and residual is not None
            and hidden_states.shape[0] == 1
            and hidden_states.dtype == torch.float16
        ):
            if self._post_norm_w_fp16 is None:
                self._post_norm_w_fp16 = (
                    (self.post_attention_layernorm.weight.data + 1.0)
                    .half()
                    .clone()
                    .contiguous()
                )
            if self._int4_router_buf is None:
                gate = self.mlp.gate
                n_exp = gate.weight_esimd.view(torch.int32).shape[0]
                dev = hidden_states.device
                self._int4_router_buf = torch.empty(
                    1, n_exp, dtype=torch.float16, device=dev
                )
                self._int4_normed_buf = torch.empty(
                    1, hidden_states.shape[-1], dtype=torch.float16, device=dev
                )
            esimd_resadd_norm_gemv_int4_pert(
                hidden_states,
                residual,
                self._post_norm_w_fp16,
                self.mlp.gate.weight_esimd.view(torch.int32),
                self.mlp.gate.scale_esimd,
                self._int4_router_buf,
                self._int4_normed_buf,
                self.post_attention_layernorm.variance_epsilon,
            )
            self.mlp._precomputed_router_logits = self._int4_router_buf
            hidden_states = self.mlp.forward(self._int4_normed_buf)
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        num_experts = getattr(self.config, "num_experts", 0)
        if rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
            num_experts += 1
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        is_fse = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        num_routed = getattr(self.config, "num_experts", 0)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # FSE: remap shared_expert weights to the fused expert slot
            if is_fse and "mlp.shared_expert." in name:
                name = name.replace(
                    "mlp.shared_expert.",
                    f"mlp.experts.{num_routed}.",
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class QwenNextMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
