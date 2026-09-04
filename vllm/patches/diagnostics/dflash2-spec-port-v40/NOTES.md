# DFlash 2 spec-decode port study (v40)

Port of upstream vLLM **PR #52816** (DFlash 2 drafter: grouped dynamic
depthwise convolutions + candidate-selector beam walk) onto this fork
(`vllm-0.21.1.dev0+gad7125a43.xpu`, B70 XPU, TP=2), benchmarked against the
fork's MTP lane. Engagement Sep 4 on host `10.20.3.65`.

**Verdict: WORKING lane, NOT promoted to prod** (prod stays 4bit TQ nospec).
Acceptance 0.46–0.51 @k=7 (4.43 tok/step), correctness gate passed, perf par
with MTP at short context and conc8, but loses badly at long context due to an
open acceptance-decay defect (§6). Do not bake until §6 is root-caused.

- Upstream material: `mindstudio.ai/blog/run-dflash2-vllm-sglang-locally`
  (how-to) and PR #52816 diff (`pr52816.diff`, copy on server at
  `/root/build/pr52816.diff`).
- Checkpoint: `/models/qwen3.8-27b-dflash2` — 5 draft layers (fork target
  layers 64–68, `target_layer_ids [5,19,33,47,61]`), all-sliding
  (`sliding_window 2048`, top-level `is_causal: false`), `dflash_config`:
  block_size 8 (= 1 + num_speculative_tokens 7), conv_group_size 16,
  conv_kernel_size 2, selector_rank 256, selector_top_k 16,
  mask_token_id 248070. Ships neither embed_tokens nor lm_head (shares the
  target's).

## 1. Architecture deltas vs DFlash/DSpark v1 (`qwen3_dflash.py`)

1. Query layout widens to `1 + num_speculative_tokens` rows per request
   (bonus anchor + N masks); the grouped conv cycles positions over flattened
   rows with `block_size = 1 + N`, so the row count must match or the conv
   misaligns every request after the first.
2. Only the N mask rows are sampled (anchor row's hidden state unused); the
   input-expansion kernel gets `SKIP_BONUS=1`.
3. Draft tokens come from a beam walk over candidate-selector edge scores
   (top-K per slot via vocab-parallel top-k, `unary + <pred·hidden, succ>`
   edge scores), not a logits argmax chain. Greedy walk only — upstream's
   probabilistic gumbel path not ported (fork spec lanes are greedy for the
   correctness gate anyway).

## 2. Port mechanics (era-3 style)

`dflash2_edit.py` — fail-loud anchor-replace patcher, idempotent, applies 10
hooks to the installed tree and injects the two new files:

| fork file | hook |
|---|---|
| `spec_decode/utils.py` | `SKIP_BONUS` constexpr + skip-bonus sample indices |
| `spec_decode/dflash.py` | `num_query_per_req` hook + `SKIP_BONUS` kernel arg |
| `models/qwen3_dflash.py` | decoder_layer_cls / model_cls hooks + construction via hook |
| `models/registry.py` | `DFlash2DraftModel` entry |
| `v1/worker/gpu_model_runner.py` | `DFlash2Proposer` selection |

New files: `qwen3_dflash2.py` (model), `dflash2_proposer.py` (proposer).
Boot: `dt_dflash2_boot.sh` — 4 phases (patch-gen with
`DFLASH2_VERBATIM_LIDS=1` → await-patch container → `docker cp` 7 files →
restart + health poll, ~230 s). Lane: `dt_dflash2_serve.sh` — image
`llm-scaler-prod:v1`, identical flags to the MTP lane (block 512,
turboquant_4bit_nc, prefix caching, async-sched, FULL_DECODE_ONLY graphs),
spec config `{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}`.
MTP comparison lane: `dt_mtp_boot.sh` / `dt_mtp_serve.sh` (k=1 MTP drafter).

## 3. Landmines found (all fixed in the port files)

1. **The draft MUST run bf16; the target stays fp16.** The DFlash2 checkpoint
   is bf16-trained and its residual stream legitimately reaches ~7e4 by draft
   layer 3 (telemetry: embed 0.5 → 2.8e4 → 4.3e4 → residual add 6.6e4 >
   fp16 max 65504 → inf → NaN → **0% acceptance**; final hidden 100864).
   Fix: post-init cast of every draft-owned param/buffer to bf16
   (`embed_tokens`/`lm_head` names skipped — shared with the fp16 target via
   `has_own_embed_tokens/lm_head = False`), with explicit boundary casts at
   every crossing (embed_input_ids, precompute context states, lm_head
   logits, combine output). The target's fp8 dequant path is fp16-only on
   this fork, so the target cannot simply go bf16.
2. **Stale `params_dtype` attributes crash bf16 drafts.** Fork guards cast
   inputs against the construction-time `params_dtype` (fp16); after the
   weight-data cast those guards pass fp16 inputs into bf16 GEMMs
   (`Half != BFloat16` in `combine_hidden_states` → `model.fc`). Fix: sync
   `module.params_dtype` to bf16 wherever weight data is now bf16.
3. **Conv internals need fp32.** The finish-side conv coefficients exceed
   fp16 range with +inf/−inf terms that CANCEL; in fp16 the projection
   GEMM saturates first and the cancellation yields NaN. fp32 GEMM + fp32
   conv math only; outputs are O(10) and cast back losslessly.
4. **Causality semantics.** Checkpoint top-level `is_causal: false`, but
   upstream's test matrix (`tests/v1/spec_decode/test_dflash_causality.py`
   in the PR) treats an ALL-sliding draft as causal regardless. Keep the
   fork default `sliding_attention_causal=True`. Setting it False made the
   draft's context attention non-causal and acceptance decayed with context
   (72% @2k → 21% @16k → 9% @65k; True: 30% @16k).

## 4. Correctness gate — PASSED

Probe battery `dt_probe2.py` (6 prompts, temp 0, sha256 of full output):

- p1/p2/p5/p6 bit-stable across runs AND identical to the pre-port
  references: `3cecc747d7885a11` / `c87e27c45fcafcb6` / `194e1de8d589b61b`
  / `1f9c461b0f2bd1cb`.
- p3 (and p4 across larger config deltas) are run-to-run BISTABLE on this
  lane — divergent greedy trajectories (e.g. p4 378 vs 400 completion
  tokens, both hashes observed on back-to-back runs of the SAME build).
  The MTP control lane shows the same p3 behavior → lane-level target
  nondeterminism (fp16 knife-edge argmax class, cf. KNOWN_ISSUES #18), not
  a port defect.
- FULL_DECODE_ONLY graphs: bit-stable vs eager on the stable probes.
- Stripped final build re-verified after instrumentation removal ( Sept 4).

## 5. Benchmark vs MTP (`dt_bench3.py`, single-stream decode tok/s)

| context | DFlash2 (k=7) | MTP (k=1) | ratio |
|---|---|---|---|
| 2k  | 329–396 | 380 | par |
| 16k | 174–212 | 268 | 0.7x |
| 65k | 27–46   | 181 | 0.2x |
| conc8 aggregate | 119–125 | 126 | par |

TTFT @65k: DFlash2 60–72 s vs MTP 41 s (5-layer context-KV precompute over
74k tokens dominates). Acceptance: MTP 0.68–0.81, DFlash2 0.46–0.51
(blended over battery).

Profiling (fork `VLLM_SPEC_TIMING`, stripped from the final lane config):
`propose` ≈ 1.3 ms/step host total (dforward 0.40 ms device, precompute
incremental ≈ 0.006 ms, greedy walk 0.02 ms). Both lanes are dominated by
~20–45 ms/step fork step overhead single-request, so single-stream probes
are overhead-bound; emission pays at concurrency (conc8 par despite
k=7 vs k=1 draft cost).

## 6. OPEN DEFECT — acceptance decays with context

72.3% @2.3k → 48.1% @6.9k → 28.6% @11.4k → 30.4% @18.2k → 9.3% @74k
(measured via `dt_b65.py` per-round `/metrics` deltas). Gradual — not a
cliff at 2048 (window) or 8192. `dforward` device time is FLAT with context
(≈0.3 ms/step @65k) → the 2048 window IS enforced in cost, but the decay
pattern suggests the draft attends the WRONG 2048 window range over the
precomputed context KV (window rooted at context start vs trailing).
Not root-caused; next step is dumping the draft attention metadata
(block table / start offsets) at 16k. **This is the gate for any prod
promotion.**

## 7. Repro

```
# DFlash2 lane (patch + boot + probe, ~4 min):
bash /root/build/dt_dflash2_boot.sh
python3 /root/build/dt_probe2.py       # hashes + acceptance
python3 /root/build/dt_bench3.py       # perf battery
# MTP comparison lane:
bash /root/build/qwen38-dflash2/dt_mtp_boot.sh
```

## 8. Files

| file | what |
|---|---|
| `dflash2_edit.py` | era-3 patcher: 10 fork hooks + file injection |
| `qwen3_dflash2.py` | draft model (convs, selector, bf16 scheme, top-k processor) |
| `dflash2_proposer.py` | proposer (1+N query rows, anchor capture, beam walk) |
| `dt_dflash2_serve.sh` / `dt_dflash2_boot.sh` | lane boot (final, instrumentation-free) |
| `dt_mtp_serve.sh` / `dt_mtp_boot.sh` | MTP comparison lane |
| `dt_probe2.py` | correctness probe (hashes + acceptance) |
| `dt_bench3.py` | TTFT/decode/ctx/conc8 benchmark |
| `dt_b65.py` | acceptance-vs-context probe |
