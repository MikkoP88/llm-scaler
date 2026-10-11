# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.lora_experts_mixin import LoRAExpertsMixin
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kMxfp4Static,
)
from vllm.platforms import current_platform

if current_platform.is_xpu():
    from vllm_xpu_kernels.fused_moe_interface import (
        XpuFusedMoe,
        fused_moe_activation,
    )


def prepare_fp8_moe_layer_for_xpu(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return w13.transpose(-1, -2).contiguous(), w2.transpose(-1, -2).contiguous()


def _xpu_block_moe_forward(
    experts: "XPUExperts",
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
) -> None:
    """CUTLASS routed-expert forward for 128x128 block FP8 (w8a16).

    Reuses the persistent grouped-GEMM scheduler from the per-tensor path,
    while selecting scales by the weight's 128x128 block. Weights stay in
    [E, N, K] layout with scales [E, N/128, K/128].
    """
    if hidden_states.dtype != torch.float16:
        raise RuntimeError(
            "XPU FP8 block MoE currently requires FP16 activations, but "
            f"received {hidden_states.dtype}. Relaunch vLLM with "
            "`--dtype float16`; native BF16 block-MoE support is not "
            "implemented yet."
        )

    num_experts = experts.moe_config.num_local_experts
    ep_rank = experts.moe_config.ep_rank
    ep_size = experts.moe_config.ep_size
    device = hidden_states.device
    dtype = hidden_states.dtype

    num_rows, hidden = hidden_states.shape
    topk = topk_ids.size(-1)
    total = num_rows * topk

    N1 = w1.shape[1]  # 2 * intermediate (gate_up width)
    inter = N1 // 2
    K2 = w2.shape[2]  # gemm2 contraction dim (== intermediate for gated act)

    # Expert map for expert parallelism (mirrors XpuFusedMoe.__init__).
    expert_map = None
    if ep_size > 1:
        expert_map = torch.empty(
            num_experts * ep_size, dtype=torch.int32, device=device
        )
        torch.ops._moe_C.init_expert_map(expert_map, num_experts, ep_rank, ep_size)
    total_experts_num = (
        expert_map.shape[0] if expert_map is not None else num_experts * ep_size
    )

    # Scatter tokens into expert-contiguous order.
    remapped = torch.empty((total, hidden), dtype=dtype, device=device)
    rows_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=device)
    unpermuted_row_to_permuted_row = torch.empty(
        (num_rows, topk), dtype=torch.int32, device=device
    )
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=hidden_states,
        hidden_states_scales=None,
        remapped_hidden_states=remapped,
        remapped_hidden_states_scales=None,
        expert_map=expert_map,
        rows_per_expert=rows_per_expert,
        unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
        topk_ids=topk_ids,
        total_experts_num=total_experts_num,
        local_experts_num=num_experts,
    )

    # gemm1: gate_up = scattered_x @ dequant(w13)^T
    gemm1_output = torch.empty((total, N1), dtype=dtype, device=device)
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=remapped,
        ptr_B=w1,
        ptr_scales=w1_scale,
        ptr_bias=None,
        ptr_D=gemm1_output,
        rows_per_expert=rows_per_expert,
        N=N1,
        K=hidden,
        num_experts=num_experts,
        is_B_int4=False,
        is_B_mxfp4=False,
        is_B_fp8_block=True,
    )

    # activation (silu/gelu/... and mul)
    inter_size_scale = 2 if activation.value == "relu2_no_mul" else 1
    act_output = torch.empty(
        (total, inter * inter_size_scale), dtype=dtype, device=device
    )
    fused_moe_activation(act_output, gemm1_output, activation.value)

    # gemm2: out = act @ dequant(w2)^T
    gemm2_output = torch.empty((total, hidden), dtype=dtype, device=device)
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=act_output,
        ptr_B=w2,
        ptr_scales=w2_scale,
        ptr_bias=None,
        ptr_D=gemm2_output,
        rows_per_expert=rows_per_expert,
        N=hidden,
        K=K2,
        num_experts=num_experts,
        is_B_int4=False,
        is_B_mxfp4=False,
        is_B_fp8_block=True,
    )

    # weighted gather back to per-token output
    torch.ops._moe_C.moe_gather(
        output, gemm2_output, topk_weights,
        unpermuted_row_to_permuted_row, num_experts,
    )


class XPUExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.is_fp8 = False
        self.is_mxfp4 = False
        self.is_block_fp8 = False
        self.fused_moe_impl = None

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if self.is_block_fp8:
            # Route 128x128 block FP8 experts through the persistent CUTLASS
            # grouped GEMM while preserving the checkpoint weight and scale
            # layouts. Reuse the packaged scatter/activation/gather glue.
            _xpu_block_moe_forward(
                self,
                output=output,
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                w1_scale=self.w1_scale,
                w2_scale=self.w2_scale,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
            )
            return

        if self.fused_moe_impl is None:
            topk = topk_ids.size(-1)
            self.fused_moe_impl = XpuFusedMoe(
                w13=w1,
                w13_scales=self.w1_scale,
                w13_bias=self.w1_bias,
                w2=w2,
                w2_scales=self.w2_scale,
                w2_bias=self.w2_bias,
                n_experts_per_token=topk,
                activation=activation.value,
                num_experts=self.moe_config.num_local_experts,
                ep_rank=self.moe_config.ep_rank,
                ep_size=self.moe_config.ep_size,
                is_fp8=self.is_fp8,
                is_int4=getattr(self, "is_int4", False),
                is_mxfp4=self.is_mxfp4,
                is_block_fp8=self.is_block_fp8,
            )
        self.fused_moe_impl.apply(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )


class XPUExpertsFp8(LoRAExpertsMixin, XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.is_fp8 = True

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Keep the existing monolithic XPU fast path when this experts object
        # is used without a LoRA context.
        ctx = self._lora_context
        if ctx is None or ctx.punica_wrapper.no_lora:
            return super().apply(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )

        del a1q_scale, a2_scale, workspace13, workspace2
        del expert_tokens_meta, apply_router_weight_on_input

        num_tokens, hidden_size = hidden_states.shape
        top_k = topk_ids.shape[-1]
        num_moe_inputs = num_tokens * top_k
        num_experts = self.moe_config.num_local_experts
        inter_size = w1.shape[-1] // 2

        remapped_hidden_states = torch.empty(
            (num_moe_inputs, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        rows_per_expert = torch.zeros(
            num_experts, dtype=torch.int32, device=hidden_states.device
        )
        route_to_permuted_row = torch.empty(
            (num_tokens, top_k), dtype=torch.int32, device=hidden_states.device
        )
        total_num_experts = (
            expert_map.shape[0] if expert_map is not None else global_num_experts
        )
        if total_num_experts == -1:
            total_num_experts = num_experts

        torch.ops._moe_C.remap_hidden_states(
            hidden_states=hidden_states,
            hidden_states_scales=None,
            remapped_hidden_states=remapped_hidden_states,
            remapped_hidden_states_scales=None,
            expert_map=expert_map,
            rows_per_expert=rows_per_expert,
            unpermuted_row_to_permuted_row=route_to_permuted_row,
            topk_ids=topk_ids,
            total_experts_num=total_num_experts,
            local_experts_num=num_experts,
        )

        gemm1_output = torch.empty(
            (num_moe_inputs, 2 * inter_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=remapped_hidden_states,
            ptr_B=w1,
            ptr_scales=self.w1_scale,
            ptr_bias=self.w1_bias,
            ptr_D=gemm1_output,
            rows_per_expert=rows_per_expert,
            N=2 * inter_size,
            K=hidden_size,
            num_experts=num_experts,
            is_B_int4=False,
            is_B_mxfp4=False,
        )

        routing = route_to_permuted_row.reshape(-1)
        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = self.apply_w13_lora(
            ctx,
            y=gemm1_output,
            x=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=expert_map,
            w1=w1,
            w2=w2,
            num_tokens=num_tokens,
            top_k_num=top_k,
            c_row_mapping=routing,
        )

        act_output = torch.empty(
            (num_moe_inputs, inter_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        fused_moe_activation(act_output, gemm1_output, activation.value)

        gemm2_output = torch.empty(
            (num_moe_inputs, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=act_output,
            ptr_B=w2,
            ptr_scales=self.w2_scale,
            ptr_bias=self.w2_bias,
            ptr_D=gemm2_output,
            rows_per_expert=rows_per_expert,
            N=hidden_size,
            K=inter_size,
            num_experts=num_experts,
            is_B_int4=False,
            is_B_mxfp4=False,
        )

        self.apply_w2_lora(
            ctx,
            y=gemm2_output.view(num_tokens, top_k, hidden_size),
            x=act_output,
            topk_weights=topk_weights,
            sorted_token_ids_lora=sorted_token_ids_lora,
            expert_ids_lora=expert_ids_lora,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            token_lora_mapping=token_lora_mapping,
            num_tokens=num_tokens,
            w1=w1,
            w2=w2,
            top_k_num=top_k,
            row_mapping=routing,
        )

        torch.ops._moe_C.moe_gather(
            output,
            gemm2_output,
            topk_weights,
            route_to_permuted_row,
            num_experts,
        )


class XPUExpertsFp8Block(XPUExperts):
    """Routed-expert path for DeepSeek-style 128x128 block FP8 (weight
    128x128 static block scale + activation 1x128 dynamic per-token-group).

    Mirrors the dense XPUFp8BlockScaledMMKernel (F1) at the MoE level. The
    routed experts use the persistent Xe2 CUTLASS grouped block-scaled GEMM,
    while reusing the packaged scatter, activation, and weighted-gather
    operations.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        # Preserve the FP8/block-FP8 weight preparation contract and select
        # _xpu_block_moe_forward in XPUExperts.apply.
        self.is_fp8 = True
        self.is_block_fp8 = True

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_xpu()

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsMXFp4(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.is_mxfp4 = True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsInt4(LoRAExpertsMixin, XPUExperts):
    """LoRA-aware XPU grouped-GEMM experts for W4A16 MoE.

    Unlike :class:`XPUExperts`, this keeps the two grouped GEMMs visible so
    routed-expert LoRA can be added before the activation and before the final
    expert reduction.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config, quant_config, max_num_tokens, num_dispatchers
        )
        self.is_int4 = True

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if (
            self._lora_context is None
            or self._lora_context.punica_wrapper.no_lora
        ):
            return super().apply(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )

        # Routed LoRA bypasses XPUExperts.apply, but it still needs the exact
        # one-time INT4 weight conversion performed by XpuFusedMoe.  Retain the
        # initialized object so subsequent base-model requests use the original
        # fast path and so the converted allocation is owned for the lifetime
        # of this expert layer.
        if self.fused_moe_impl is None:
            self.fused_moe_impl = XpuFusedMoe(
                w13=w1,
                w13_scales=self.w1_scale,
                w13_bias=self.w1_bias,
                w2=w2,
                w2_scales=self.w2_scale,
                w2_bias=self.w2_bias,
                n_experts_per_token=topk_ids.size(-1),
                activation=activation.value,
                num_experts=self.moe_config.num_local_experts,
                ep_rank=self.moe_config.ep_rank,
                ep_size=self.moe_config.ep_size,
                is_int4=True,
            )

        del global_num_experts, a1q_scale, a2_scale, workspace13, workspace2
        del expert_tokens_meta, apply_router_weight_on_input

        num_tokens, hidden_size = hidden_states.shape
        top_k = topk_ids.size(-1)
        num_moe_inputs = num_tokens * top_k
        num_experts = self.moe_config.num_local_experts
        inter_size = w1.shape[-2] // 2

        if not hasattr(w1, "xpu_fused_moe"):
            raise RuntimeError(
                "XPU INT4 MoE weights were not converted during weight loading"
            )

        remapped_hidden_states = torch.empty(
            (num_moe_inputs, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        rows_per_expert = torch.zeros(
            num_experts, dtype=torch.int32, device=hidden_states.device
        )
        unpermuted_row_to_permuted_row = torch.empty(
            (num_tokens, top_k), dtype=torch.int32, device=hidden_states.device
        )
        torch.ops._moe_C.remap_hidden_states(
            hidden_states=hidden_states,
            hidden_states_scales=None,
            remapped_hidden_states=remapped_hidden_states,
            remapped_hidden_states_scales=None,
            expert_map=expert_map,
            rows_per_expert=rows_per_expert,
            unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
            topk_ids=topk_ids,
            total_experts_num=(
                expert_map.shape[0] if expert_map is not None else num_experts
            ),
            local_experts_num=num_experts,
        )
        gemm1_output = torch.empty(
            (num_moe_inputs, 2 * inter_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=remapped_hidden_states,
            ptr_B=w1,
            ptr_scales=self.w1_scale,
            ptr_bias=self.w1_bias,
            ptr_D=gemm1_output,
            rows_per_expert=rows_per_expert,
            N=2 * inter_size,
            K=hidden_size,
            num_experts=num_experts,
            is_B_int4=True,
            is_B_mxfp4=False,
        )
        ctx = self._lora_context
        assert ctx is not None
        routing = unpermuted_row_to_permuted_row.reshape(-1)
        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = self.apply_w13_lora(
            ctx,
            y=gemm1_output,
            x=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=expert_map,
            w1=w1,
            w2=w2,
            num_tokens=num_tokens,
            top_k_num=top_k,
            c_row_mapping=routing,
        )

        act_output = torch.empty(
            (num_moe_inputs, inter_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        apply_moe_activation(activation, act_output, gemm1_output)

        gemm2_output = torch.empty(
            (num_moe_inputs, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=act_output,
            ptr_B=w2,
            ptr_scales=self.w2_scale,
            ptr_bias=self.w2_bias,
            ptr_D=gemm2_output,
            rows_per_expert=rows_per_expert,
            N=hidden_size,
            K=inter_size,
            num_experts=num_experts,
            is_B_int4=True,
            is_B_mxfp4=False,
        )

        self.apply_w2_lora(
            ctx,
            y=gemm2_output.view(num_tokens, top_k, hidden_size),
            x=act_output,
            topk_weights=topk_weights,
            sorted_token_ids_lora=sorted_token_ids_lora,
            expert_ids_lora=expert_ids_lora,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            token_lora_mapping=token_lora_mapping,
            num_tokens=num_tokens,
            w1=w1,
            w2=w2,
            top_k_num=top_k,
            row_mapping=routing,
        )

        torch.ops._moe_C.moe_gather(
            output,
            gemm2_output,
            topk_weights,
            unpermuted_row_to_permuted_row,
            num_experts,
        )
