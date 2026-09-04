# DFlash2 port vs upstream vLLM PR #52816 — feature-by-feature comparison

Deep-research comparison of our implemented DFlash2 (v40 port + v41 window
fix + v42 k-adjustable emission) against the public upstream implementation
(PR #52816, "[Spec Decode] DFlash2: local convolution + candidate selector",
merged 2026-08-21, head `3406ec1dae9916f920b90f0dbf90dcf54923d042`, 16
commits, 14 files +866/−44). Engagement Sep 4; all our references are the
committed patch dirs `../dflash2-spec-port-v40/`, `../dflash2-window-fix-v41/`
and this dir.

**Verdict: MATCH on the drafter math (formula-identical), SUPERIOR on
operational capability for this platform (k-adjustable emission with async
scheduling AND graphs — none of which exist upstream), NOTHING upstream has
that we need.** Post-merge upstream activity contains exactly one substantive
fix and it is in a code path we deliberately did not port.

## 1. Drafter architecture — MATCH (formula-level)

| component | upstream PR #52816 | ours | verdict |
|---|---|---|---|
| grouped dynamic depthwise conv | `out[i,c] = Σ_t (base[t,c] + δ[i,t,g(c)])·x[i−t,c]`, taps zeroed across the block boundary, `block_size = 1 + num_speculative_tokens`, position modulo `& (bs−1)` | `qwen3_dflash2.py::_grouped_conv` — identical: `coefficients = base.view(1,taps,groups,gsize) + delta.unsqueeze(-1)`, per-tap `(position >= tap)` gate, `position & (bs−1)` fast modulo, prepare/finish two-sided kernel projection | **match** |
| candidate selector edge score | `unary[l,c] + einsum("blpr,blcr->blpc", pred[p]⊙project(h), succ[c])`, anchor token = slot-0 predecessor | `qwen3_dflash2.py::_score_edges` — identical formula, identical anchor expansion, same predecessor/successor codebooks + rank-d hidden projection | **match** |
| candidate generation (top-k) | FlashInfer radix top-k where available, `torch.topk` fallback; `get_top_k_tokens` = shard-local top-k + all-gather of 2k pairs per rank | `_DFlash2TopKProcessor.get_top_k_tokens` mirrors the fallback design + adds the XPU-required 64-wide padded all-gather payload (fork XPU tiny-payload corruption workaround); FlashInfer does not exist on XPU | **match** (XPU-adapted) |
| beam walk | Triton beam walk, `gumbel_noised_argmax` (probabilistic) / argmax (greedy) | greedy sequential `argmax` walk over `scores[rows, step, prev]` (dflash2_proposer.py) — matches the greedy path exactly; probabilistic path not ported (fork spec lanes are temp-0 greedy for the correctness gate) | **match** (greedy path) |
| quant handling | conv `kernel_projection` + selector `hidden_projection` hardcoded unquantized (`quant_config=None`) | same — both `ReplicatedLinear(..., quant_config=None)` | **match** |
| checkpoint keys | `dflash_config`: block_size, conv_kernel_size, conv_group_size, selector_rank, selector_top_k, mask_token_id, input_embedding_scale, output_multiplier, final_logit_softcapping | all read, same defaults; shares target embed/lm_head (`has_own_*=False`) | **match** |
| block_size validation | derives `1 + num_spec_tokens` | proposer asserts checkpoint block_size == 1 + num_speculative_tokens at boot | **match** (fail-louder) |

## 2. Runtime machinery — SUPERIOR on this platform (by necessity or by feature)

| capability | upstream | ours | verdict |
|---|---|---|---|
| model runner | **V2 only** — `use_v2_model_runner` forced; on V1 the DFlashProposer silently degrades to DFlash1 behavior | full DFlash2 on the fork's V1 machinery (`VLLM_USE_V2_MODEL_RUNNER=0`; V2 does not exist on this XPU fork) | **superior-for-platform** (upstream's constraint is unusable here; ours preserves the fork's certified serving stack) |
| graph mode | drafter default-eager (`_generate_draft(cudagraph_runtime_mode=NONE)`); piecewise CUDA graphs on NVIDIA | drafter INSIDE the fork's whole-step XPU graphs (FULL_DECODE_ONLY, KNOWN_ISSUES #11 posture) | **different lineage, equal-or-better** (v41/v42 prove bit-stable under graphs) |
| k-adjustable emission | **does not exist** — k is fixed to `num_speculative_tokens` (checkpoint block_size−1) | `DFLASH2_EMIT_K=1..7` at serve time: drafter cap + runner local-pad + async placeholder cap (v42c) + 3-site width-aware graph capture (v42d: runner udql, config lattice bs·(1+k), dispatcher width) | **superior** (unique capability; verified k∈{3,5,7}, k=4 convicted fork graph defect, guarded) |
| async scheduling | not addressed | works under the fork AsyncScheduler (v42c placeholder-width cap) | **superior** (upstream has no async + DFlash2 combination at all) |
| TP | yes (CUDA NCCL) | yes, verified TP=2 B70 oneCCL | match |
| KV cache | standard | `turboquant_4bit_nc` + **v41 sliding-window read fix** (fork TurboQuantAttentionImpl silently dropped `sliding_window`; fixed to match trained trailing-2048 semantics: acceptance 9.3%→62.2% @74k) | **superior-for-platform** (restores trained semantics the fork lost) |
| draft numerics | homogeneous bf16 | bf16 draft / fp16 target scheme + fp32 conv internals + params_dtype sync (all REQUIRED: bf16-trained checkpoint residual reaches ~6.6e4 > fp16 max; fork target dequant is fp16-only) | **necessary adaptations**, absent upstream because unnecessary there |

## 3. Post-merge upstream activity — nothing we need

- **PR #54282** (2026-08-29, commit `fe755c8`) — the ONLY substantive
  dflash2-speculator commit since merge: Philox noise-stream decoupling
  (draft and verify shared an offset, biasing rejection re-sampling).
  Affects **only `draft_sample_method="probabilistic"`** — a path we
  deliberately did not port (greedy lanes). N/A for us.
- **PR #52559** (OPEN draft, unmerged, needs rebase) — adaptive batch-level K
  for DFlash **v1** constrained to a CUDA-graph lattice {0,1,3,7,15}. Our
  static `DFLASH2_EMIT_K` covers the operational need; note their lattice
  skips k=4 (consistent with our k=4 graph corruption conviction, though
  theirs is a graph-shape constraint, not a numerics defect).
- Nothing else touched conv/selector/proposer semantics.

**Improvements needed: NONE.** The drafter math is identical, the only
post-merge fix is in an unported path, FlashInfer is CUDA-only, and the V2
runner is unavailable on this fork. Adopting any upstream delta would add
risk with zero expected gain — failing the "improvement only if needed"
test.

## 4. Performance standing (different hardware — compare ratios, not absolute)

Upstream H200, Qwen3.8-27B @k=7: acceptance 5.34 tok/step (≈0.67/position),
conc1 224.6 tok/s = **3.51× AR**, conc8 2.91×, conc32 2.20×.

Ours, B70 XPU TP=2 @k=7 (v41 lane, AR baseline measured Sep 4 on the
restored prod nospec lane, same bench harness):

| row | AR (nospec) | DFlash2 k=7 | ratio | MTP k=4 | ratio |
|---|---|---|---|---|---|
| 2k decode | 136.6 | 411.6 | **3.01×** | 354.1 | 2.59× |
| 16k decode | 160.8 | 262.0 | 1.63× | 431.1 | 2.68× |
| 65k decode | 130.9 | 189.1 | 1.44× | — | — |
| conc8 aggregate | 219.3 | 123.6 | 0.56× | 142.4 | 0.65× |

- Single-stream @2k: 3.01× AR — same magnitude as upstream's 3.51× conc1
  despite the fork's whole-step overhead and B70 silicon; acceptance
  6.4 tok/step @2k / 5.3 @74k (v41 ladder 77.7/70.6/62.2%) is at or above
  upstream's 5.34.
- conc8 aggregate: spec (both methods) loses to AR on this fork — the
  fork's step-overhead-dominated regime at batch 8 plus k=7 draft cost;
  DFlash2 tracks MTP (0.56× vs 0.65×). Lane-generic, not a port defect;
  upstream's 2.91× conc8 reflects H200 compute-bound decode that this
  platform/fork does not replicate. Known posture: k is a compute-saving /
  tail-acceptance-tuning knob (conc8 @k=5 125.7 ≈ k=7's 123.6).

## 5. Conclusion for this engagement

1. Our implementation **matches** the upstream drafter exactly where it
   matters (conv, selector, top-k, greedy walk, TP, quant posture).
2. It is **superior operationally** on this platform: k-adjustable emission
   with async scheduling and width-aware whole-step graphs is a capability
   upstream does not have; the v41 window fix restores trained semantics on
   the fork's TQ KV path.
3. No upstream deltas are worth adopting (sole post-merge fix is in the
   unported probabilistic path).
4. Action taken instead of code changes: full re-validation of the chain +
   this image bake (`llm-scaler-exp:dflash2-v42`, see NOTES.md).
