# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

import torch

from vllm.distributed import (
    get_tp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform


# --8<-- [start:logits_processor]
@PluggableLayer.register("logits_processor")
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    # --8<-- [end:logits_processor]

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)

        # llm-scaler v78 engine variant: follow the lm_head's own captured
        # sharding instead of the engine-wide TP size (same law as
        # get_top_tokens below). A replicated (TP1) lm_head inside a TP2
        # engine already holds the full vocab logits on every rank; the
        # group gather would concatenate duplicate full-vocab tensors.
        # For sharded lm_heads (tp_size == engine TP) this is identical
        # to the previous unconditional gather.
        tp_size = getattr(lm_head, "tp_size", None)
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # Gather logits for TP
            logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size).
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        # llm-scaler v78 R1b: follow the lm_head's own captured sharding
        # instead of the engine-wide TP size. A replicated (TP1) MTP head
        # lm_head inside a TP2 engine must take the local fast path - the
        # engine-level query would route it through the (value, index)
        # all-gather against the TP2 group. For shared/sharded lm_heads
        # (tp_size == engine TP) behavior is identical to before.
        tp_size = getattr(lm_head, "tp_size", None)
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()

        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        local_max_vals, local_max_indices = logits.max(dim=-1)

        # Convert shard-local indices to global vocab indices.
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        if tp_size == 1:
            return global_indices

        # All-gather (value, index) pairs, then reduce to global argmax.
        # Use float32 to avoid bf16 precision loss on large vocab indices.
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        # [batch, 2] -> [batch, 2 * tp_size]
        if current_platform.is_xpu():
            # The XPU custom all-gather can corrupt the peer half for this
            # tiny [batch, 2] payload. Pad the native collective to avoid the
            # broken eager-message path, then keep only the pair fields.
            group = get_tp_group()
            gather_width = 64
            gather_input = torch.zeros(
                (*local_pair.shape[:-1], gather_width),
                dtype=local_pair.dtype,
                device=local_pair.device,
            )
            gather_input[..., :2].copy_(local_pair)
            gathered_flat = torch.empty(
                (tp_size * gather_input.shape[0], *gather_input.shape[1:]),
                dtype=gather_input.dtype,
                device=gather_input.device,
            )
            torch.distributed.all_gather_into_tensor(
                gathered_flat, gather_input, group=group.device_group
            )
            gathered = gathered_flat.reshape(
                (tp_size, *gather_input.shape)
            ).movedim(0, gather_input.dim() - 1)
            gathered = gathered[..., :2]
            gathered = gathered.reshape(
                (*local_pair.shape[:-1], tp_size * local_pair.shape[-1])
            )
        else:
            gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        # [batch, tp_size, 2] where [:, :, 0]=values, [:, :, 1]=indices
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
