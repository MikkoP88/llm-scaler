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

llm-scaler v42 (Path A): emission-width knob DFLASH2_EMIT_K (1..N). The
draft ALWAYS runs its native 1 + N row layout — the grouped conv cycles
row phases mod block_size = 1 + N, which the checkpoint was trained under,
so a narrower internal layout would cross-request-contaminate phases.
The knob only truncates what propose() EMITS: the greedy walk is
prefix-consistent (step i depends only on the prev chain from steps < i),
so row[:k] is exactly what a k-wide drafter would emit. The truncated
list-of-lists return reuses the fork's variable-width draft contract
(scheduler reads per-request lengths from scheduled_spec_decode_tokens;
verify runs draft_len + 1 target rows) — the same contract the base
_truncate_by_confidence uses. Verify shrinks 8 -> k + 1 rows; draft
compute stays at the native width (accepted cost of exactness).

llm-scaler v48 (DFlash2 real-live perf, VLLM_XPU_DFLASH_TP1): the drafter
runs REPLICATED at TP=1 — construction/load/propose/dummy_run enter the
drafter_comm.dflash_tp1() world=1 TP view, so every rank holds full
drafter weights and the ~11 per-step eager oneCCL collectives of the
TP=2-sharded 5-layer drafter disappear (parallel_state short-circuits all
collectives at world 1). =0 restores the stock sharded drafter.
"""

import os
from dataclasses import replace

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.drafter_comm import dflash_tp1  # llm-scaler v48
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
        # llm-scaler v42 (Path A): emission-width knob, 1..N. Default off
        # (k >= N passes the tensor through untouched -> byte-identical to
        # v41 behavior). Invalid values fail loudly at engine init.
        self._emit_k: int | None = None
        env_k = os.environ.get("DFLASH2_EMIT_K", "")
        if env_k:
            k = int(env_k)
            if not 1 <= k <= self.num_speculative_tokens:
                raise ValueError(
                    f"DFLASH2_EMIT_K={k} out of range 1.."
                    f"{self.num_speculative_tokens} (num_speculative_tokens)"
                )
            self._emit_k = k
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
        if self._emit_k is not None:
            logger.info(
                "DFlash2 emission truncation (v42): emitting %d of %d "
                "drafted tokens per step (draft layout unchanged)",
                self._emit_k,
                self.num_speculative_tokens,
            )
        # llm-scaler v48: drafter TP=1 replication state (log-only here;
        # the view itself is entered per-span via dflash_tp1()).
        logger.info(
            "DFlash2Proposer v48: drafter TP=1 replication %s "
            "(VLLM_XPU_DFLASH_TP1)",
            "ON" if os.environ.get("VLLM_XPU_DFLASH_TP1", "1")
            not in ("0", "false", "no", "off") else "OFF",
        )

    def _get_model(self):
        # llm-scaler v48 (VLLM_XPU_DFLASH_TP1): construct AND weight-load
        # the drafter under the world=1 TP view — layer constructors and
        # weight loaders query the TP coordinator dynamically, so shapes
        # come out full (unsharded) on every rank. VLLM_XPU_DFLASH_TP1=0
        # restores the stock TP-sharded drafter.
        with dflash_tp1():
            return super()._get_model()

    def dummy_run(self, *args, **kwargs):
        # llm-scaler v48: warmup/capture dummies must see the same world=1
        # view as real proposes (head gathers etc. query the TP group at
        # forward time).
        with dflash_tp1():
            return super().dummy_run(*args, **kwargs)

    def _create_draft_vllm_config(self):
        # llm-scaler v48 companion (see load_model): keep the draft config
        # clone's block_size at 1024 too, so anything the drafter's backend
        # builds from the clone (metadata defaults) matches the 1024-token
        # draft pool installed by load_model below.
        base = super()._create_draft_vllm_config()
        if os.environ.get("VLLM_XPU_DFLASH_TP1", "1") in (
            "0",
            "false",
            "no",
            "off",
        ):
            return base
        draft_block = int(os.environ.get("DFLASH2_DRAFT_BLOCK", "1024"))
        if draft_block <= 0:
            return base
        base = replace(
            base,
            cache_config=replace(base.cache_config, block_size=draft_block),
        )
        return base

    def load_model(self, target_model):
        # llm-scaler v48 (VLLM_XPU_DFLASH_TP1): under TP1 replication the
        # draft pool carries FULL 8 KV heads per rank; at the hybrid block
        # size (2048, set by the mamba interface AFTER model load) its page
        # (~3.67 MB at k8v4) exceeds every target page and forces the
        # engine's unified KV page size from 2 to 4 MiB — doubling the
        # per-token cost of ALL 67 groups (57.2 vs 28.8 KB/tok) and
        # halving KV capacity (262144-token boots OOM: "14.26 GiB needed,
        # 6.4 available"). Attention.get_kv_cache_spec refreshes
        # block_size from the vllm_config it is HANDED at spec-build time
        # (the engine config, 2048), so the clone in
        # _create_draft_vllm_config is not enough: wrap every drafter
        # module's get_kv_cache_spec to substitute block_size=1024, giving
        # a ~1.83 MB draft page under the stock 2 MiB unification. The TQ
        # kernels read block size from the pool tensor shape
        # (turboquant_attn: block_size = kv_cache.shape[1]), so pool,
        # metadata, and kernels stay consistent.
        super().load_model(target_model)
        if os.environ.get("VLLM_XPU_DFLASH_TP1", "1") in (
            "0",
            "false",
            "no",
            "off",
        ):
            return
        draft_block = int(os.environ.get("DFLASH2_DRAFT_BLOCK", "1024"))
        if draft_block <= 0:
            # Bisect mode (e.g. DFLASH2_DRAFT_BLOCK=2048 with a 4bit draft
            # pool at MAXLEN<=131072): leave the spec path stock.
            return
        patched = 0
        for mod in self.model.modules():
            orig = getattr(mod, "get_kv_cache_spec", None)
            if orig is None:
                continue

            def wrapped(vllm_config, _orig=orig, _b=draft_block):
                cfg = replace(
                    vllm_config,
                    cache_config=replace(
                        vllm_config.cache_config, block_size=_b
                    ),
                )
                return _orig(cfg)

            mod.get_kv_cache_spec = wrapped
            patched += 1
        logger.info(
            "DFlash2Proposer v48: draft KV block=%d on %d drafter spec "
            "sites (full-head draft page ~1.83 MB; keeps unified page at "
            "2 MiB)",
            draft_block,
            patched,
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
        # llm-scaler v48 (VLLM_XPU_DFLASH_TP1): the entire propose span —
        # anchor bookkeeping, the base propose (input expansion, draft-KV
        # precompute, drafter forward, selector walk) and emission
        # truncation — runs under the world=1 TP view so the replicated
        # drafter issues zero cross-rank collectives.
        with dflash_tp1():
            # The per-request bonus (anchor) token keys the selector's layer-0
            # edge scores; equivalently input_ids[row * num_query_per_req].
            self._dflash2_anchor = next_token_ids.long()
            out = super().propose(
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
            # llm-scaler v42 (Path A): emission truncation. Return a k-wide
            # variable-width draft LIST per request — the scheduler and verify
            # paths read per-request lengths from the list form (same contract
            # as the base _truncate_by_confidence), so verify runs k + 1 target
            # rows while the draft keeps its native 1 + N layout. Greedy walk
            # is prefix-consistent -> row[:k] is the exact k-wide draft.
            k = self._emit_k
            if k is None or k >= self.num_speculative_tokens:
                return out
            if isinstance(out, list):
                return [row[:k] for row in out]
            assert torch.is_tensor(out) and out.dim() == 2, (
                f"DFlash2 v42 emission truncation: unexpected propose() return "
                f"type {type(out)}"
            )
            return out[:, :k].tolist()

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
