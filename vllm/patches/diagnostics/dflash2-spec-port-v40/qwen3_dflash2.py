# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash 2 draft model (llm-scaler era-3 port of upstream PR #52816).

Grouped dynamic depthwise convolutions inside the draft decoder layers plus
a candidate selector (top-K per slot, edge-scored beam walk). Adapted to
the fork's V1 DFlashProposer machinery: the fork layer signature (layer_type,
no layer_idx) and the fork LogitsProcessor's XPU-padded all-gather for the
vocab-parallel top-k reduction.

Numerics (llm-scaler era-3): the DFlash2 checkpoint is bf16-trained and its
residual stream legitimately reaches ~7e4 by draft layer 3 (embed 0.5 ->
2.8e4 -> 4.3e4 -> residual add 6.6e4 > fp16 max 65504 -> inf -> NaN, i.e.
0% acceptance under fp16). The TARGET must stay fp16 (its fp8 dequant path
is fp16-only on this fork), but the draft carries no quantized weights, so
every draft-owned parameter is cast to bf16 post-init; embed_tokens/lm_head
are shared with the fp16 target (has_own_* = False) and stay fp16, with
explicit boundary casts at each crossing (embed_input_ids, precompute
context states, lm_head logits, combine output).
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Mapping

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.platforms import current_platform

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import maybe_prefix


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    # fp32 internals: the checkpoint is BF16-trained and its finish-side
    # conv coefficients exceed the fp16 range with +inf and -inf terms that
    # CANCEL; in fp16 the GEMM saturates to +-inf first and the
    # cancellation yields NaN. The conv OUTPUT magnitudes are O(10), so
    # only the internal math needs fp32; the result casts back to the
    # model dtype losslessly.
    hs = hidden_states.float()
    blocks = hs.unflatten(-1, (num_groups, group_size))
    coefficients = base.float().view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hs.shape[0], device=hs.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2).to(hidden_states.dtype)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # fp32 GEMM (see _grouped_conv): the finish-side coefficient columns
        # overflow fp16 inside the projection itself. The weight is
        # [2*taps*groups, hidden]; upcast both operands (bias-free).
        w = self.kernel_projection.weight
        proj = F.linear(hidden_states.float(), w.float())
        coefficients = proj.reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        cache_config=None,
        quant_config=None,
        layer_type: str = "full_attention",
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            layer_type=layer_type,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Query tokens per request: the bonus token plus the mask tokens.
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # Without its own tag the selector shares the draft head's compile cache.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor, *a, **kw) -> torch.Tensor:
        # Boundary cast: the shared fp16 target embed feeds the bf16 draft
        # stream.
        return (
            super().embed_input_ids(input_ids, *a, **kw).to(torch.bfloat16)
            * self.input_embedding_scale
        )

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        # Boundary cast: fp16 target hidden states -> bf16 draft internals
        # (the fused-KV F.linear and rms_norm below are draft-owned, bf16).
        return super().precompute_and_store_context_kv(
            context_states.to(torch.bfloat16),
            context_positions,
            context_slot_mapping,
        )


class _DFlash2TopKProcessor(LogitsProcessor):
    """LogitsProcessor with a vocab-parallel top-k reduction.

    Mirrors upstream PR #52816's get_top_k_tokens, with the fork's XPU
    workaround from get_top_tokens: the tiny per-row (value, id) pairs are
    padded to a 64-wide payload so the XPU custom all-gather does not hit the
    broken eager-message path. fp32 carries vocab ids exactly
    (248320 < 2^24).
    """

    def get_top_k_tokens(
        self,
        lm_head,
        hidden_states: torch.Tensor,
        k: int,
        embedding_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vocab-parallel top-k without all-gathering full logits.

        Returns (ids int64 [N, k], values fp32 [N, k]). Scale and soft cap
        are applied to the k selected values; both are monotonic so the
        selection is unchanged.
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local top-k reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = get_tensor_model_parallel_world_size()

        # Boundary cast: draft hidden is bf16, the shared lm_head is fp16.
        w_dtype = getattr(getattr(lm_head, "weight", None), "dtype", None)
        if w_dtype is not None and hidden_states.dtype != w_dtype:
            hidden_states = hidden_states.to(w_dtype)

        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        values, ids = torch.topk(logits, k, dim=-1)
        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index

        if tp_size > 1:
            if current_platform.is_xpu():
                # Interleave (value, id) pairs: even slots = values,
                # odd slots = ids; pad to gather_width so the collective
                # avoids the XPU tiny-payload corruption path.
                local = torch.stack(
                    [values.float(), ids.float()], dim=-1
                ).flatten(-2)  # [N, 2k]
                gather_width = 64
                group = get_tp_group()
                gather_input = torch.zeros(
                    (*local.shape[:-1], gather_width),
                    dtype=local.dtype,
                    device=local.device,
                )
                gather_input[..., : local.shape[-1]].copy_(local)
                gathered_flat = torch.empty(
                    (tp_size * gather_input.shape[0], gather_width),
                    dtype=gather_input.dtype,
                    device=gather_input.device,
                )
                torch.distributed.all_gather_into_tensor(
                    gathered_flat, gather_input, group=group.device_group
                )
                gathered = gathered_flat.reshape(
                    (tp_size, *gather_input.shape)
                ).movedim(0, -1)  # [N, width, tp]
                gathered = gathered[..., : local.shape[-1], :]  # [N, 2k, tp]
                values_all = gathered[:, 0::2, :].reshape(
                    -1, tp_size * k
                )  # [N, k*tp]
                ids_all = gathered[:, 1::2, :].reshape(-1, tp_size * k)
            else:
                values_all = tensor_model_parallel_all_gather(values, dim=-1)
                ids_all = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = torch.topk(values_all, k, dim=-1)
            ids = ids_all.long().gather(-1, selected)

        values = values.float()
        if self.scale != 1.0:
            values = values * self.scale
        if self.soft_cap is not None:
            values = torch.tanh(values / self.soft_cap) * self.soft_cap
        return ids, values


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.candidate_logits_processor = _DFlash2TopKProcessor(
            vllm_config.model_config.get_vocab_size(),
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )
        # The DFlash2 checkpoint ships neither embed_tokens nor lm_head:
        # the base proposer shares the target's weights when these flags
        # are False (process_eagle_weight never sees the keys to flip them).
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False
        # See the module docstring: cast every draft-owned parameter to
        # bf16 (fp16 drafts NaN in the residual stream at ~6.6e4).
        for name, p in self.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                continue
            p.data = p.data.to(torch.bfloat16)
        for name, b in self.named_buffers():
            if b.is_floating_point():
                b.data = b.data.to(torch.bfloat16)
        # Linear layers keep a `params_dtype` attribute from construction
        # and fork code casts inputs against it (e.g. combine_hidden_states
        # -> self.model.fc). Sync it wherever the weight data is now bf16,
        # or those guards pass fp16 inputs straight into bf16 GEMMs
        # ("expected mat1 and mat2 to have the same dtype").
        for module in self.modules():
            pd = getattr(module, "params_dtype", None)
            w = getattr(module, "weight", None)
            if (
                pd is not None
                and pd != torch.bfloat16
                and isinstance(w, torch.Tensor)
                and w.dtype == torch.bfloat16
            ):
                module.params_dtype = torch.bfloat16

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # Boundary cast: draft hidden is bf16, the shared lm_head is fp16.
        w_dtype = getattr(getattr(self.lm_head, "weight", None), "dtype", None)
        if w_dtype is not None and hidden_states.dtype != w_dtype:
            hidden_states = hidden_states.to(w_dtype)
        return super().compute_logits(hidden_states)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = super().combine_hidden_states(hidden_states)
        return out if out.dtype == torch.bfloat16 else out.to(torch.bfloat16)


EntryClass = DFlash2Qwen3ForCausalLM
