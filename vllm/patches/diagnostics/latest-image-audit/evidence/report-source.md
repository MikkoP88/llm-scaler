# Latest-image audit and new improvement plan

Date: 2026-09-07. Audience: maintainers and operators of the TP=2 Intel XPU deployment at `10.20.3.65`. Repository HEAD inspected: `cdda818d5c3662c37b4e2b7e1bfa8d96e5ae638d`; the audit and v125-fixes directories are existing untracked work, so HEAD alone does not identify these overlays.

## 1. Decision and evidence boundaries

**Keep v1.2.5 as the comparison baseline. The newest locally available image, v1.2.6t2, is an experimental observability/guarded-feature build, not a demonstrated performance upgrade.** Prioritize (1) trustworthy measurement and loop diagnosis, (2) scheduler-owned speculative widths, (3) tiled compressed-KV attention, and (4) memory/admission accounting. Do not enable proposer-only dynamic narrowing: its validation recorded a worker wedge.

There is no configuration that minimizes latency and maximizes throughput, memory capacity, and quality for every workload. Historical results favor MTP4/TQ4 for short single-client work, nonspec/FP8 for deep decode and short concurrent traffic, and nonspec/TQ4 when KV capacity matters. These are starting points, not universal optima or latest-image certifications.

Evidence labels throughout:

- **Fresh**: host state or raw results retrieved during this review; collection did not launch the already-running benchmark job.
- **Historical**: existing archived measurements, whose configuration and harness restrictions still apply.
- **Code**: an inspected execution path; occurrence and performance impact may remain unmeasured.
- **Proposed**: implementation or experiment to perform, with no promised speedup.

The host changed from t2 to v1.2.5 during the review. At 16:54 UTC, `master127.sh`/`run_lane127.sh` were actively running a four-lane benchmark with one active request. No competing inference, server replacement, driver reset, or kernel patch was performed by this audit. The collected `bench127` snapshot is partial. Exhaustive all-dtype, all-client, semantic 128k/262k validation remains to be run under the plan below. Neither a green health endpoint nor fixed-output throughput establishes semantic correctness.

Primary local evidence: [live inventory](evidence/live-inventory.txt), [latest image overlay hashes and dtype enum](evidence/latest-image-packages.txt), [package list](evidence/packages.txt), [fresh benchmark snapshot](evidence/bench127-partial.log), [loop results](evidence/loop8-partial.out), [prior audit](../v125-audit/REPORT.md), and [v53 implementation/post-mortem](../../v125-fixes/PLAN.md). Source reads, runtime evidence and upstream references were reconciled before writing this plan. A planning tool was unavailable; discovery, follow-up, synthesis and verification were tracked in the conversation.

## 2. What the latest image actually contains

| Identity | Verified value | Interpretation |
|---|---|---|
| Latest local tag | `llm-scaler-exp:v1.2.6t2` | Created 2026-09-07 14:31:24 UTC; local host inventory, not a global registry claim |
| Latest image ID | `sha256:006791e5f95c2d262f962198df5929953b441a2be5fe0309eb12fde7cea72c5c` | Pin this ID for same-host A/B; record a registry digest separately when publishing |
| Baseline image ID | `sha256:a522bf15be2b666e7dcaab1d59557c8cc21e3bbcdf6be0a9c6bbf686b31f0ff8` | v1.2.5 and v1.2.5t5 |
| vLLM | `0.21.1.dev0+gad7125a43.d20260826.xpu` | Wheel version does not describe subsequent overlays |
| PyTorch / Triton | `2.11.0+xpu` / `triton-xpu 3.7.0` | XPU stack; CUDA tuning claims do not transfer automatically |
| Transformers | `5.8.0` | Actual installed version, distinct from model config's recorded development version |
| XPU kernels | `0.1.8.3.dev0+g3cab97a.d20260830` | Also `custom-esimd-kernels-vllm 0.1.0` |
| Target | Served alias `qwen3.8-27b-fp8`; config architecture `Qwen3_5ForConditionalGeneration` | Alias is not a model identity; hash config, tokenizer, template and weight manifest |
| Hybrid architecture | 64 layers: 16 full attention, 48 linear attention; 24 Q heads, 4 KV heads, full-attention head dimension 256 | Full-attention KV compression does not compress all GDN/SSM state |
| Drafter | `/models/dflash2`, `DFlash2DraftModel`, five sliding-attention layers, block size 8 | User's `method=dflash` must resolve to this model implementation; record selected class in startup manifest |

Model details are from the archived [installed source/config inventory](../v125-audit/evidence/installed-source-inventory.txt); their weight contents were not rehashed in this review. Latest packages and three overlays were read from an isolated CPU-only, network-disabled container, without GPU devices or model mounts.

The [t1 Dockerfile](../../v125-fixes/bake_v126t1_Dockerfile) adds three overlays to v1.2.5; [t2](../../v125-fixes/bake_v126t2_Dockerfile) replaces the proposer again. Verified t2 MD5 values:

| In-image file | MD5 | Behavior |
|---|---|---|
| `v1/spec_decode/llm_base_proposer.py` | `7215ed994b24f842dc9b7bc0b1a6247e` | Context-based width calculation; narrowing default-disabled by `VLLM_V53_L2_UNSTABLE` |
| `v1/attention/backends/flash_attn.py` | `4500df595690ff51a4edf7c214efbe70` | Ragged front-padding route, default on; optional routing counters |
| `v1/attention/backends/turboquant_attn.py` | `192f701929d8614e898e0b750e65d11a` | Optional split-tier logging |

Deep warmup `dt_warmup_v53.py` is host-side, **not baked into t2**. Baseline inheritance includes v52 scheduler boundary clamp, zero-emission handling, GDN/spec input guards, TQ graph fixes, FP8 MQ kernels and disconnect cleanup. The prior [wheel divergence inventory](../v125-audit/evidence/wheel-divergence.txt) lists 27 modified and seven added files: auditing upstream version strings alone misses the deployed behavior. Record imported module paths too: duplicate old/new OpenAI serving layouts exist in this wheel.

## 3. Baselines and corrections to earlier conclusions

### 3.1 Fresh baseline from the active host job

All rows below are **v1.2.5**, DFlash2 K=7, FP8 e4m3, GMU 0.9, maxlen 262144, TP2, MNBT8192, max sequences64, block512, FP16 model/SSM and the user's parser/graph/async settings. They are one-run capacity measurements from the [raw snapshot](evidence/bench127-partial.log), not new t2 performance claims.

| Actual prompt tokens | Completion tokens | TTFT seconds | Reported decode tokens/s |
|---:|---:|---:|---:|
| 2,015 | 256 | 1.2 | 23.3 |
| 16,665 | 256 | 19.3 | 19.3 |
| 33,231 | 256 | 21.4 | 12.9 |
| 67,485 | 256 | 37.1 | 8.0 |
| 136,191 | 256 | 125.7 | 5.4 |
| 261,757 | 256 | 209.5 | 5.2 |

The last row reaches 262,013 total tokens, **131 below the limit**, not exactly 262,144. Reported prefill rates 1084/1249 tokens/s are prompt-tokens/TTFT proxies, including non-kernel work. Aggregate acceptance 1099/3269=0.336 was from the short battery, not those deep requests.

Final collection captured this lane's completed concurrency tests: C8 at about2k achieved **100.3 aggregate tokens/s**,8/8 successful; C4 mixed2k/16k achieved19.8 aggregate tokens/s,4/4 successful; C2 about65k achieved3.1 aggregate tokens/s including prefill,2/2 successful. These denominators differ from post-TTFT per-client decode rates. Lane health counters reported STALLS20, STRIKES0, F8=1, JIT13, ERRS0 and DEVLOST0. Small sample counts do not establish tail percentiles or reliability. At the final check the external job had advanced to DFlash/TQ4 warmup; the full matrix remained incomplete. [Final host snapshot](evidence/final-host-check.txt), [completed FP8 lane report](evidence/df7-fp8-completed-report.txt).

Fresh natural-stop study: four selected thinking prompts, 8192 output budget; three exhausted the budget without final content, one stopped at 6840 tokens. One of the three had a detected periodic tail (`!!!!`, period 4 characters, 128 repetitions). The other two had no detected periodic tail. **Budget exhaustion and periodic repetition must be different outcome classes.** This contradicts a blanket “no loops” reading of the older audit. It does not prove FP8 is causal or that t2 reproduces it; full raw reasoning and paired target-only trials are required. [Loop snapshot](evidence/loop8-partial.out).

### 3.2 Historical comparison, not latest-image certification

| Lane | ~2k decode t/s | 136,191 prompt decode t/s | 242,761 prompt decode t/s | 8-client ~2k aggregate t/s |
|---|---:|---:|---:|---:|
| nonspec FP8 | 23.5 | 25.6 | 21.2 | 200.6 |
| nonspec TQ4nc | 33.7 | 17.7 | 9.6 | 160.5 |
| MTP4 TQ4nc | 61.8 | 11.7 | 6.1 | 117.5 |
| DFlash7 TQ4nc | 46.9 | 4.2 | 7.3 | 93.4 |
| MTP4 FP8 | 31.6 | 6.7 | 4.4 | 61.4 |
| DFlash7 FP8 | 23.2 | 9.7 | 4.0 | 90.1 |

Source: [v125 audit tables and archived lane files](../v125-audit/REPORT.md). Prompt generation, warm-cache state and sample count constrain comparisons. The fresh DFlash/FP8 deep result differs materially from the historical 9.7; retain both, do not average unlike runs. Exact-token historical checks exist only for MTP4/TQ4: 131072 prompt tokens and 261888+256=262144, at 13.06 and 7.55 reported decode t/s. They establish capacity completion, not long-context reasoning quality. [Exact probe JSONL](../v125-audit/evidence/v125_audit_fresh_deep.jsonl).

### 3.3 Findings that supersede older recommendations

1. **Adaptive width is an end-to-end protocol change.** The v53 trial narrowed only proposer output; recorded failure was a silent 300-second RPC timeout. Async placeholder width was fixed at boot. Accounting/graph mismatch is a strong code-backed explanation, but the precise blocked device/collective instruction was not isolated. Keep the unstable gate closed. [Post-mortem](../../v125-fixes/PLAN.md).
2. **Ragged routing is a latent path, not an established cause of the GMU cliff.** P-1 observed 2000 fixed-width MTP/FP8 verify calls, all uniform. Investigate other workloads separately; do not project a ragged-padding speedup onto these measured lanes.
3. **Deep warmup did not solve deep decode.** Different-text deep results reproduced prior slow values after extended warmup. Same-prefix TTFT gains demonstrate cache reuse. They do not prove transferable JIT removal; Triton specialization normally depends on shapes/constexprs, not text semantics. Do not promise cold 262k TTFT below 10 seconds from warmup alone.
4. **The TQ host tier ladder is excluded, not the DFlash depth anomaly itself.** In FULL_DECODE_ONLY, `_tq_adaptive_splits` is false and the tier function early-returns. Device-side partitioning, acceptance, windows, addresses and request history remain candidates. Calling the whole anomaly “closed” is too strong.
5. **A retained dequantized prefix per layer is not a cheap workspace fix.** Layers overwrite shared scratch; cross-chunk reuse needs per-layer data or a redesign. Use tiled in-attention dequantization instead.
6. **The memory and health posture is not universally clean.** Later v125 restore logs recorded DEVICE_LOST during warmup. This differs from the silent L2 wedge; neither zero errors in an earlier 3.5-hour run nor a successful retry excludes intermittent faults.
7. **Draft dtype inheritance is real.** MTP inherits cache configuration; DFlash defaults to target dtype and has an override. Do not assume always-FP16 draft KV. Nor is “K+1 full scans” a measured bandwidth equation: fused multi-query verification can reuse KV tiles, and draft attention differs from target attention.
8. **Historical workload boundaries need correction.** MTP4/TQ4 at 16k (46.6) exceeds nonspec/FP8 (33.5), so nonspec is not the winner throughout 16k–65k. TQ's 1.3M vs 0.705M pool is about 1.84× capacity, not 4.9×; 4.9 is approximately the number of full-length requests before overhead, not an improvement ratio.

## 4. KV dtype coverage and memory model

The latest image enum contains 15 values. **Accepted argument spelling, tensor representation, backend routing, successful startup, and correct generation are separate support gates.** Even backend declarations disagree with exercised FP8 routes: the inspected Flash backend's advertised list contains only auto/FP16/BF16, while patched FP8 code and live runs exist. Add an executable capability check rather than treating either enum or class list as sufficient.

| Dtype | Status for this deployment | New validation / recommendation |
|---|---|---|
| `auto` | Resolves to model execution dtype, here FP16 | Required uncompressed-KV quality reference in nonspec/MTP/DFlash; still FP8 weights |
| `float16` | Enum and dtype mapping exist; older ledger recorded rejection | Explicit boot smoke on t2, compare to auto; report version-specific result |
| `bfloat16` | Mapping exists; no current full-matrix result | Test cache/backend compatibility separately from changing model dtype to BF16 |
| `fp8` | Generic byte storage / alias-like route | Confirm actual format/scales, compare output to e4m3; do not count as independent precision by name |
| `fp8_e4m3` | Exercised baseline | Primary decode/capacity candidate; audit scales and long-context quality |
| `fp8_e5m2` | Historical checkpoint rejection plus downstream failure when bypassed | Expected rejection for this FP8 checkpoint unless complete backend support is implemented; never bypass only the guard |
| `fp8_inc` | Maps to native float8 tensor, unlike e4m3 byte storage | Backend-specific experimental gate; not an interchangeable alias |
| `fp8_ds_mla` | Enum/mapping only here; target is GQA/hybrid, not MLA | Mark inapplicable to this target unless backend proves otherwise |
| `turboquant_k8v4` | Implemented FP8 keys / 4-bit values; prior quality study exists | Useful intermediate candidate; all three decode modes still require latest validation |
| `turboquant_4bit_nc` | Exercised 4-bit keys/values with norm correction | Primary capacity and short-MTP candidate |
| `turboquant_k3v4_nc` | Implemented 3-bit keys / 4-bit values | Aggressive experimental precision; mandatory retrieval/reasoning differential gates |
| `turboquant_3bit_nc` | Implemented 3-bit keys/values | Same gate; no assumed proportional throughput improvement |
| `int8_per_token_head` | Enum/mapping, no deployed-kernel proof | Validate scale layout, paged storage, prefill, verify and graphs before serving |
| `fp8_per_token_head` | Enum/mapping, no deployed-kernel proof | Same; do not confuse with per-tensor FP8 scaling |
| `nvfp4` | Packed uint8 mapping, no demonstrated B70 implementation | Do not advertise NVIDIA-specific kernels as XPU support; expected unsupported/experimental classification |

Evidence: [latest enum](evidence/latest-image-packages.txt), [actual dtype mapping](evidence/torch_utils.py), [TQ layout](../v125-audit/evidence/tq-layout-and-markers.txt), [KNOWN_ISSUES #19](../../../KNOWN_ISSUES.md), [Flash backend](../../v125-fixes/flash_attn_v53.py).

For full-attention target KV only, assuming KV heads shard evenly over TP2, bytes/token/rank = 16 layers × 2 local KV heads × bytes per K+V head vector. Derived packed storage:

| KV mode | Bytes/K+V head (D=256) | KiB/token/rank | At 262144 tokens/rank |
|---|---:|---:|---:|
| FP16/BF16 | 1024 | 32 | 8 GiB |
| FP8 e4m3 | 512 | 16 | 4 GiB |
| TQ K8V4 | 388 | 12.125 | 3.03125 GiB |
| TQ4nc | 262 | 8.1875 | 2.046875 GiB |
| TQ K3V4nc | 230 | 7.1875 | 1.796875 GiB |
| TQ3nc | 198 | 6.1875 | 1.546875 GiB |

These are **derived lower components**, not allocator reservations. Add scales outside slots, page alignment, grouping/padding, MTP/DFlash pools, GDN state/checkpoints, graphs, activations, scratch, communication and safety headroom. Historical TQ config docstring PPL percentages are not quality measurements for this target.

Continuation-prefill at 262k with local Hkv=2, D=256, FP16 has roughly 512 MiB cached K+V, 512 MiB concatenated K+V and 256 MiB rotated K, before possible reshape copies and attention scratch. Actual peak/lifetimes need profiling. Across 32 chunks of 8192, full-prefix re-dequant visits 8192×32×31/2 = 4,063,232 cached token positions per layer, about 15.5× final length. This is avoidable extra conversion work; causal attention itself still requires its normal QK work.

`--max-num-seqs 64` is a scheduling ceiling, not 64×262k capacity. Historical DFlash/FP8 pool 373507 tokens cannot hold two unrelated full-length 262k requests simultaneously without reclamation/preemption/queueing. Admission should reserve prompt+output budget and backend lookahead, include hybrid grouping, and report queueing rather than promising latency at impossible concurrency.

## 5. New fixes and improvements backlog

Each item is a proposed change unless explicitly stated otherwise. No numerical gain below is claimed as measured. Preserve rollback switches until correctness, performance and soak gates pass.

| ID / priority | Problem and evidence | Concrete fix / investigation | Required validation |
|---|---|---|---|
| N01 / P0 | Measurements can cross container replacement; bench scripts sometimes assume 256 output tokens | Record container ID+image ID before and after each batch; abort attribution if changed. Consume final usage, HTTP errors and DONE; distinguish characters, events and tokens | Unit fixtures with grouped spec chunks, empty content, missing usage, disconnect; result cannot pass without identity and counts |
| N02 / P0 | Fresh DFlash/FP8 periodic tail and no final answer | Archive full reasoning/content/token IDs and sampler settings; paired nonspec/MTP/DFlash at identical KV/seed/prompt, then auto KV reference. Check EOS masking, rejected-token cleanup, NaNs, state rollback and parser separation | 4k/8k/16k budgets, temp0/0.7/1.0, repetitions; zero structural token errors; semantic score and loop rate reported separately |
| N03 / P0 | Proposer-local narrowing disagrees with async placeholders/graphs | Keep unstable gate disabled. Scheduler chooses width at a step boundary; serialize width with request/step identity to both ranks, runner, proposer, verifier and output accounting. Capture compatible width buckets; drain pending old-width steps | Width transitions both directions, K=0/1/2/4/7 where valid, prefill tails, cancellation, preemption, mixed batches, 512/2048 boundaries, 262k; no orphan placeholders or divergent collective order |
| N04 / P0 | DEVICE_LOST and silent RPC wedge share downstream timeout symptoms | Preserve first-error timestamps and per-rank phase IDs, correlate kernel/driver logs; bounded diagnostics and graceful lifecycle. Fail a lane on device loss; diagnose before retry | Repeated clean boots, deep chunked prefill and teardown; no lost devices, reset events or hidden failed attempts |
| N05 / P1 | FP8/spec slowdown at GMU0.9 remains unexplained | Factorial GMU0.80/0.85/0.90 × spec mode, fixed model/maxlen; log actual pool/group/block sizes, allocator peaks, graph buckets, kernel routes/times, draft/target/collective time. Compare equal cache reservations if supported | Randomized paired runs, at least three fresh processes and distinct prompts; isolate memory-size effects from warm-cache/order effects |
| N06 / P1 | TQ continuation re-dequantizes and materializes the full prefix per chunk | Add paged tiled attention that dequantizes K/V inside tile loading and performs stable online softmax; preserve rotation and norm correction. Reuse bounded scratch per execution stream; keep FA fallback | All four TQ layouts, target D256 and draft D128, q1/q2–8/prefill, page tails, GQA, causal/noncausal/window cases; logits/output tolerance and memory peak A/B |
| N07 / P1 | TQ deep decode slower than FP8 despite fewer bytes | Profile unpack, centroid lookup, rotations, occupancy/spills, bandwidth and split reduction; specialize q1 and verify independently. Tune KV tiling/warps/splits against measured bottleneck | Matched-shape microbench plus real 128k/262k streams; verify numerical tolerance and no low-context/concurrency regressions |
| N08 / P1 | Hybrid state precision/boundaries can degrade long outputs | Compare FP16 vs FP32 SSM cache with same weights/KV; validate accepted-token state copy/rollback and prefix-cache alignment. Make dtype-dependent effective block requirements explicit | Exact 512/1024/2048±1 boundaries, reject-all/accept-all patterns, resumed prefixes, long natural generation; no cross-request state contamination |
| N09 / P1 | Capacity and lifetime insufficiently characterized | Worker-side allocated/reserved/peak memory, free blocks per group, graph/scratch attribution, request lifecycle counters. Add admission budgets and reclaim checks after abort/EOS/preemption | Long→short cycles, 100+ cancellations, 30-cycle plateau, 24h soak; distinguish reusable reserved memory from live leaks |
| N10 / P1 | Spec host/IPC/collective costs erase concurrency gains | Profile actual exposed critical path, not stack-sample percentage. Move only proven bottlenecks into a bounded worker-resident loop; fuse metadata copies; preserve scheduler fairness/cancel opportunities | C1/2/4/8/16/32/64, open-loop arrivals and aborts; compare aggregate goodput and p95/p99, not one headline t/s |
| N11 / P2 | Ragged FP8 padding has allocations and broad exception fallback | First measure hit rate. Use preallocated gather/scatter buffers or direct ragged kernel with cu_seqlens; catch only recoverable shape/unsupported cases, propagate device/OOM failures | q_i=0…8, waste threshold, empty partitions, per-layer scales, graph replay. Cross-check front-padding causal mapping; no swallowed device faults |
| N12 / P2 | `_DEQUANT_BUFS` fallback keyed only by device; checks length/head count but not D/dtype/ownership | Key reusable workspace by device, head dimension, dtype and stream/ownership; ensure event-protected reuse and capture lifetime. This is a latent generalization risk, not a reproduced target crash | Alternate models/head dimensions and draft/target paths; concurrent-stream stress; bounded memory after model unload |
| N13 / P2 | Warmup increases boot time without proven transferable deep benefit | Separate compile warmup, prefix-cache priming and cold unique requests. Size buckets by tokenizer IDs; retain only buckets that improve expected production workload | Fresh-process A/B, distinct equal-length prompts, prefix hits measured; report readiness time plus first unique TTFT |
| N14 / P2 | Mixed prefill/decode may cause tails; old N=2 percentile claim is insufficient | Timestamp admission/schedule/prefill/decode per request. Test MNBT and admission caps before changing scheduler fairness; add aging only if traces show starvation | Mixed 2k/16k/128k/262k, staggered arrivals, at least 100 requests for tails; report per-client distribution |
| N15 / P1 | Enum/backend/kernel support mismatch | Startup capability validation using model architecture, target+draft dtype, dimensions, scale layout, graphs and kernel availability. Error before device work for unsupported combinations | All 15 enum values × three modes; unsupported outcomes count as explicit rejection tests, not generation passes |
| N16 / P2 | Image reproducibility depends on many overlays and shared libraries | Bake source SHA256 manifest, dependency lock, imported-path check and loaded-library identities; reject stale patch bases | Rebuild by digest, compare manifest, launch semantic smoke; no mutable hot-swap results presented as clean image results |

Code anchors: [proposer](../../v125-fixes/llm_base_proposer_v53.py) `_v53_k_eff` around 537–605 and call sites 699/711; [TQ backend](../../v125-fixes/turboquant_attn_v53.py) adaptive gate 523, early return 594, continuation 1512 onward, fallback 1560–1570; [FP8 backend](../../v125-fixes/flash_attn_v53.py) ragged path 1381–1472; [benchmark script](evidence/master127.sh) post-phase character accumulation and constant-256 numerator. The original `live_probe.py` validates usage but its post-first-content rate still includes the first event's tokens in its numerator: label it a proxy, not exact ITL.

N03 should assess upstream Dynamic SD before inventing another incompatible API. Current official docs explicitly limit MRv1 dynamic SD to piecewise graphs; full graphs require MRv2. That does **not** authorize switching this heavily patched XPU/GDN/DFlash deployment to MRv2 without a port and full validation. [vLLM Dynamic SD, updated July 7 2026](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/).

## 6. Environment and parameter strategy

Use the user's four commands unchanged for primary comparisons, and add two nonspec commands by omitting `--speculative-config`. Keep the same image ID, model/tokenizer/template, parser kwargs, output budget and host conditions. A separate BF16 or FP32-SSM accuracy arm must not silently replace the FP16 baseline.

| Workload | Initial candidate | What remains to optimize |
|---|---|---|
| Single client, short prompt | MTP4 + TQ4nc, GMU0.9 | Compare MTP K1/2/3/4, natural stop and code/tool accuracy |
| 16k transition region | MTP4/TQ4 vs nonspec/FP8 | Measure prompt-specific crossover; old 16k result favors MTP |
| 32k–262k decode-heavy | nonspec/FP8, GMU0.9 | Validate exact boundaries and quality; cannot extrapolate 242k measurement to 262k |
| Many long prompts/capacity | nonspec/TQ4nc | Use actual hybrid pool/admission accounting; profile prefill scratch |
| High concurrency | nonspec FP8 or TQ4 | Optimize goodput under TTFT/ITL SLO; prioritize capacity if queueing dominates |
| FP8 plus speculation required | Include GMU0.85 as historical mitigation candidate | Repeat matched A/B on latest image; ensure 262k fits before accepting it |
| DFlash2 | Keep K7 as baseline; experimental K alternatives must satisfy draft block and scheduler contract | No default recommendation until periodic-tail regression and deep speed are explained |
| Quality-critical long reasoning | Compare auto KV and FP32 SSM against compressed/FP16 SSM baseline | Output budget, reasoning settings and dtype must be separate experiment axes |

Retain observed host environment as a reproducible starting point: `VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_USE_AOT_COMPILE=0`, `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `VLLM_XPU_FP8_MQ=1`, `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`, `ZE_AFFINITY_MASK=0,1`. Keep the captured oneCCL DRM IPC/P2P/SYCL settings for the initial runs; they are host-specific, not a claim of global optimality. [Actual environment](evidence/live-inventory.txt).

Leave `VLLM_SPEC_CTX_K` unset and `VLLM_V53_L2_UNSTABLE` unset/0. Route counters `VLLM_V125_P1=1` are temporary diagnostic arms on t2; measure overhead and disable for clean throughput runs. Host tier logging cannot observe the inactive graph-tier branch. Retain existing split defaults; prior trials rejected MQ splits64 and FP8 q1 rerouting. Do not treat the graph-disable knob alone as an eager correctness reference: the local ledger records an all-reduce compiler aliasing failure; use a separately validated eager configuration.

Parameter sweep order:

1. Choose quality-passing dtype+spec mode at baseline8192/64/0.9.
2. Sweep MNBT `{2048,4096,8192}` for mixed workloads; 16384 only as a controlled experiment because prior local trials rejected it. Sweep max sequences `{1,4,8,16,32,64}`. Record effective hybrid block size: requested512 is not necessarily the final grouping unit.
3. Sweep GMU `{0.80,0.85,0.90}` only where capacity permits; leave physical runtime headroom. Compare actual cache bytes/group layouts, not GMU alone.
4. Tune fixed K and graph buckets in configurations supported by the implementation. DFlash's block size8/K7 relationship makes arbitrary width changes nontrivial; K0 requires the nonspec lifecycle, not an empty proposer list.
5. Compare prefix caching on/off and unique/reused prefixes; then NUMA/CPU affinity and verified P2P topology. One bounded metrics sampler; no stacked GPU monitoring loops.
6. Only after stage profiles justify it, tune kernel tiles/warps/CCL thresholds/allocator policy one factor at a time. Offline model-loading variables affect startup reproducibility, not a proven decode speedup.

For chat, `reasoning_effort=xhigh` and preserved thinking can consume output/context and delay final answers. Keep them for baseline fidelity; create separate explicitly labeled thinking-off or bounded-reasoning product arms. Verify the rendered chat template actually responds to these kwargs. A repetition penalty or stop rule changes sampling semantics and is not a kernel correctness fix.

Official vLLM describes speculation as workload-dependent and chunk-budget tuning as a TTFT/ITL tradeoff. Its generic recommendation for larger budgets is not evidence against local XPU regressions. [Speculation overview](https://docs.vllm.ai/en/latest/features/speculative_decoding/), [optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization/). FP8 calibration deserves a separate quality arm, but supported scale layouts must be verified in these patched kernels. [Quantized KV guide, updated August 3 2026](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/).

## 7. Comprehensive real-scenario validation plan

### 7.1 Coverage and staging

**Stage A — support and lifecycle:** all 15 enum entries × nonspec/MTP4/DFlash7 on the latest immutable image; auto/e4m3/TQ4 also on baseline. Record supported, expected-reject, unexpected-boot-fail, generation-fail or pass. For unsupported target/backend combinations stop at a clean informative rejection. Probe each servable lane with 32/2048-token inputs, natural EOS, one tool call and one cancellation before long runs.

**Stage B — primary performance matrix:** six primary lanes (two baseline dtypes × three modes); prompt tokens `{2048,16384,32768,65536,131072,261888}`, output256, concurrency `{1,2,4,8}`. Extend C16/32/64 at short/mixed lengths and feasible deep capacity. At long lengths, distinguish resident concurrency from queued clients; insufficient capacity is a finding. Repeat at least three independent processes and three unique equal-length prompts per cell. Report sample counts and uncertainty. Then screen other servable dtypes and promote only quality-passing candidates into the same matrix.

**Stage C — natural application workloads:** each candidate executes the scenarios below, with output budgets `{256,1024,4096,8192}` and a targeted16384 reasoning arm. Temperature0 plus sampled0.7/1.0, saved seeds/top-p/top-k/penalties; don't combine across settings. For distributional checks, use a small controlled sampling task with many independent samples; fixed-seed byte identity is not generally expected across different numerical precision/batching.

| Scenario | Inputs / clients | Outcome to score |
|---|---|---|
| Interactive chat | 2k/16k, C1/4/16, 10–30 turns | Correct final answers, context accumulation, first reasoning vs first final content, natural stop |
| Coding | Multi-file bug repair, test generation, structured patch output; 16k/128k | Compile/tests on output, no duplicated blocks or invalid tool JSON |
| RAG single needle | Unique facts at 1/10/25/50/75/90/99% depth in 128k/near262k | Exact answer plus cited record; distractor and no-answer controls |
| RAG multi-hop | Join 3–5 records far apart, similar identifiers, conflicting dated entries | Correct joins and temporal selection; not merely literal needle retrieval |
| Long document | Varied prose/code/tables, summary and exact quotations | Rubric coverage, factual errors, source position sensitivity |
| Tool agent | Zero/one/multiple calls, streamed args, tool-result continuation, mixed thinking | Valid call IDs/schema, parser equivalence streamed vs nonstreamed; no spurious duplicate calls |
| Thinking regression | The four fresh loop prompts plus arithmetic/proof/DB tasks | Periodicity, coherent budget exhaustion, final answer correctness, reasoning leakage separately |
| Prefix reuse/isolation | Identical system prefixes, partial overlap, per-client unique facts; C2/8 | Cache hits vs cold TTFT, no answer contamination, state validity after cancel/evict |
| Mixed arrivals | 80% short, 15%16k, 5%128k/near262k; staggered and burst arrivals | Goodput under SLO, queue delay, starvation, preemptions |
| Streaming failure | Cancel before first token, mid-prefill, mid-decode, client socket reset, slow reader | Prompt abort, release of blocks/placeholders, healthy next request |
| Long generation | Short and deep prompts, 4k–16k output where context permits | EOS/length distinction, repetition onset, accepted/rejected token integrity |
| Memory pressure | Unique long requests at capacity±one, then short requests | Bounded queue/preemption, no OOM/device loss, return to steady allocated memory |

### 7.2 Exact 128k and 262k protocol

- Define 128k as **131072** and 262k as **262144** tokens; preserve actual counts everywhere. Also log decimal lengths if users mean128000/262000.
- Capacity test: 131072 prompt+256 output; maximum-boundary test: **261888 prompt+256 output=262144**. Tokenize through the actual server/tokenizer and submit token IDs for exact completion tests. For chat, count the fully rendered template, role delimiters, tools and reasoning history; trim filler until exact budget fits.
- Negative boundary tests: prompt beyond limit; prompt+requested output beyond limit; boundary−1/exact/+1 total, with documented API behavior. Distinguish server clipping from rejection and from client truncation.
- Long-generation alternative: 258048 prompt+4096 output=262144; for8192 output use253952 input. A full262144-token prompt leaves no positive output budget under this configuration.
- Both lengths require C1 and C2 unique-prefix runs, C4 when capacity allows, plus mixed long+short C8. Simultaneous exact262k C2 DFlash/FP8 may queue because the measured pool is too small: test graceful admission, not an impossible “both resident” throughput target.
- Run cold-process, warmed-unique-prefix, exact-repeat-prefix, eviction-then-repeat and reversed length order. Use multiple seeds and equal token lengths. Compare 120k/128k/136k/144k and 240k/252k/near262k to localize the DFlash anomaly without changing unrelated factors.
- For semantic tasks insert known answers before tokenization and verify every needle survives trimming; store token offsets. Use at least20 documents ×7 positions per target length for retrieval; include cross-document reasoning and multilingual/code records. Capacity filler alone cannot pass this gate.

### 7.3 Metrics and acceptance gates

Record request ID, client ID, image/container IDs, source manifest, UTC+monotonic times, tokenized prompt hash/count, full request settings, final usage, finish reason and exact outcome. Keep raw content/reasoning separately with token IDs where supported. Never count SSE chunks or characters as tokens.

Measure TTFT to first nonempty reasoning/text, time to first final content, end-to-end latency, output tokens/wall time, and event gaps. Label client event-gap percentiles as event gaps; speculative chunks may contain multiple tokens. Obtain true token ITL from server/token timestamps, or state that it is unavailable. Aggregate throughput = sum successfully completed output tokens / common run interval; also report failed/unfinished work and quality-passing goodput. Percentiles require adequate sample counts: minimum100 completed requests for indicative p95, preferably1000 for p99; deep small-N runs report individual values/median/range instead.

Collect per-rank allocated/reserved/peak memory, KV occupancy by group, prefix hits, preemptions, running/waiting, draft proposed/accepted by position, emitted tokens/step, graph/eager routes, H2D/D2H/collective/kernel timing and errors. Use request-window metric deltas, not process-lifetime acceptance for a single deep case. Profile only representative runs and confirm low-overhead unprofiled repetitions.

Release gates (proposed, tune SLO values with deployment requirements):

1. Zero silent corruption, cross-request leakage, out-of-range IDs, invalid accepted-token accounting, crashes, device resets, or missing terminal protocol events in the deterministic structural suite.
2. No statistically supported regression beyond predeclared task tolerance against same-weight auto-KV reference; exact synthetic retrieval should be perfect on the controlled fixture, and natural tasks need scored confidence intervals. Quantized KV is not assumed lossless. Compare FP32 SSM separately.
3. No new periodic loop pattern on the regression corpus; report natural budget exhaustion separately. Target-only periodic failure means model/numerical investigation, while spec-only failure prioritizes rollback/sampling paths.
4. Performance promotion requires ≥10% improvement in the target workload median with uncertainty supporting a gain, ≤5% regression in protected workloads, and no worse SLO goodput or memory failures. These are acceptance thresholds, not forecasts.
5. Memory gate: no monotonic live allocation/block retention after repeated identical workload/cancel cycles once warm pools stabilize. Set a measured tolerance and explain reserved allocator caching. Queue length and post-abort running requests must recover.
6. Run at least2 hours mixed-client stress for screening and24 hours for release, including long contexts, natural stops, cancellations and graceful restarts. Preserve failed attempts in the denominator; a retry cannot erase a failed run.

### 7.4 Execution sequence and rollback

Finish/capture the existing host job first. Use an exclusive host benchmark lock and pin the container ID per phase; do not kill another run. Archive its completed raw files before launching a new lane. For each lane: manifest→boot→health+semantic smoke→cold tests→warm unique/cache tests→performance→semantic/loop→cancellation/memory→graceful teardown→error-log capture. Stop the lane immediately on device loss or structural corruption; preserve the first error and mark remaining cells not-run.

Implement N01/N02/N15/N16 first, then N05/N08/N09 profiling. Develop N03 and N06/N07 as separate patches and images. Validate N11 only on actual ragged fixtures. Combine successful patches only after individual A/B gates pass, then rerun the full matrix. Keep the original v125 digest available; no retagging of baseline and no production hot-swaps during comparative runs.

## 8. Issue inventory and remaining gaps

The [local KNOWN_ISSUES ledger](../../../KNOWN_ISSUES.md) contains historical issues01–20 (16 is listed after19). Their presence does not establish that all remain active in t2:

| Family | Ledger coverage | Disposition in this plan |
|---|---|---|
| Platform/transport | 01 installation firmware,02 PCIe topology,03 shutdown wedges,05 device-loss/memory/boot family | Retain environment baseline and lifecycle checks; intermittent failure remains relevant |
| Compiler/graph/state correctness | 04 custom-op aliasing,06 context-blind TQ graphs,07 FP8 graph synchronization,09 MTP head/collectives,11 long-prompt TP wedges | Preserve inherited fixes; graph/runner upgrades require regression ports |
| Measurement and reproducibility | 08 SSE events counted as tokens,12 large chunked prompt instability,18 historical FP8 race/knife-edge results | N01 measurement gates; distinguish race from benign numeric divergence |
| Prefix/deep/precision | 10 prefix granularity,13 repetitive prompt degeneration,14 TQ register pressure,15 FP8/spec depth,19 unsupported e5m2 | N05–N09, dtype support table and real semantic corpus |
| Spec host/concurrency | 16 acceptance dependency,17 eager proposer concurrency | N03/N10 plus workload routing |
| Thinking behavior | 20 prior budget exhaustion without periodic loops | Reopened narrowly by fresh periodic-tail detector; N02 differential diagnosis |

Upstream issue discovery checked the Intel repository and relevant official vLLM pages. [Intel issue420](https://github.com/intel/llm-scaler/issues/420) concerns an older0.14.0-b8.2.1 Qwen image-processor startup failure; it is a compatibility regression scenario, **not evidence that this0.21.1 overlay image has that bug**. Upstream generic KV docs describe different supported hardware paths, so installed code and runtime evidence take precedence for B70.

This is a scoped deep audit of the serving path, overlays, issue ledger and available live baselines, not a proof that every repository line or public issue was exhaustively verified. Outstanding evidence: completed bench127 matrix, clean t2 generation matrix for all servable dtypes, fresh nonspec controls, semantic128k/262k scores, exact deep multi-client results, raw reproduction of the periodic tail, allocator timelines and causal profiles for GMU/depth anomalies. Source retrieval stopped after the major design claims had code/runtime support or explicit gaps; more broad searches would not replace these measurements.

The deliverable is a new implementation and test plan. No inference-kernel fixes are represented as implemented or validated by this document.
