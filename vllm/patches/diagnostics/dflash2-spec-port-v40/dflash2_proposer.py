# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash 2 proposer (llm-scaler era-3 port of upstream PR #52816).

Rides the fork's V1 DFlashProposer; differences from DFlash/DSpark v1:

1. Query layout widens to 1 + num_speculative_tokens rows per request
   (bonus anchor + N masks). The grouped conv in the DFlash2 layers cycles
   positions over flattened rows with block_size = 1 + N, so the row count
   must match or the conv misaligns every request after the first.
2. Only the N mask rows are sampled (the anchor row's hidden state is not
   used); the input-expansion kernel gets SKIP_BONUS=1.
3. Draft tokens come from a beam walk over candidate-selector edge scores
   instead of a logits argmax chain. Greedy walk only in this port
   (upstream's probabilistic gumbel path is not ported; the fork's spec
   lanes are greedy for the correctness gate anyway).
"""

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.spec_timing import spec_seg

logger = init_logger(__name__)


class DFlash2Proposer(DFlashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        # 1 + N query rows: bonus anchor + N masks (matches conv block_size
        # and the checkpoint's trained layout). set_inputs_first_pass reads
        # this attribute (era-3 patch); buffers are already sized for
        # max_batch_size * (1 + num_speculative_tokens).
        self.num_query_per_req = 1 + self.num_speculative_tokens
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", {}
        )
        ckpt_block = int(dflash_config.get("block_size", self.num_query_per_req))
        if ckpt_block != self.num_query_per_req:
            raise ValueError(
                f"DFlash2 checkpoint conv block_size={ckpt_block} does not "
                f"match 1 + num_speculative_tokens="
                f"{self.num_query_per_req}. Adjust num_speculative_tokens."
            )
        self._anchor_indices = torch.arange(
            self.max_batch_size, dtype=torch.int64, device=device
        ) * self.num_query_per_req
        # Checkpoint top-level is_causal=false, but upstream PR #52816's
        # causality matrix treats an ALL-sliding draft as causal regardless
        # (tests/v1/spec_decode/test_dflash_causality.py: all-sliding +
        # is_causal=False -> True). Keep the fork default
        # sliding_attention_causal=True; setting it False made the draft's
        # context attention non-causal and acceptance decayed with context
        # length (72% @2k -> 21% @16k -> 9% @65k; True: 30% @16k).
        logger.info(
            "DFlash2Proposer: num_query_per_req=%d (1 anchor + %d masks), "
            "greedy selector walk, causal sliding draft attention",
            self.num_query_per_req,
            self.num_speculative_tokens,
        )

    def propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=None,
        slot_mappings=None,
    ):
        # The per-request bonus (anchor) token keys the selector's layer-0
        # edge scores; equivalently input_ids[row * num_query_per_req].
        self._dflash2_anchor = next_token_ids.long()
        return super().propose(
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            token_indices_to_sample,
            common_attn_metadata,
            sampling_metadata,
            mm_embed_inputs=mm_embed_inputs,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            slot_mappings=slot_mappings,
        )

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Selector beam walk over the N sampled mask-row hidden states.

        hidden_states: [batch * N, hidden] request-major (the gathered rows
        selected by token_indices_to_sample). Returns flat [batch * N]; the
        base proposer reshapes to [batch, N].
        """
        with spec_seg("greedy"):
            anchor = self._dflash2_anchor
            num_reqs = anchor.shape[0]
            num_spec = self.num_speculative_tokens
            selector = self.model.model.candidate_selector
            top_k = selector.top_k

            candidate_ids, unary_logits = self.model.compute_candidates(
                hidden_states
            )  # [B*N, K] int64 / fp32
            cand = candidate_ids.view(num_reqs, num_spec, top_k)
            unary = unary_logits.view(num_reqs, num_spec, top_k)
            hidden = hidden_states.view(num_reqs, num_spec, -1)
            scores = selector(cand, unary, hidden, anchor)  # [B, N, K, K]

            prev = torch.zeros(
                num_reqs, dtype=torch.int64, device=scores.device
            )
            tokens = torch.empty(
                num_reqs, num_spec, dtype=torch.int64, device=scores.device
            )
            rows = torch.arange(num_reqs, device=scores.device)
            for step in range(num_spec):
                idx = scores[rows, step, prev].argmax(dim=-1)
                tokens[:, step] = (
                    cand[rows, step].gather(1, idx.unsqueeze(1)).squeeze(1)
                )
                prev = idx
            return tokens.reshape(-1)
