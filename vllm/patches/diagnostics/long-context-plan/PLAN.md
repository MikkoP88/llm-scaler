# Fixing long-context degradation: nonspec, MTP and DFlash2

Date: 2026-09-07. Status: researched implementation plan; fixes and new performance tests below are **not yet implemented or executed**.

Target: TP2 Intel Arc Pro B70, Qwen hybrid target at `/models/target`, 262144 context limit, FP16 execution/SSM, async scheduling, prefix caching, FULL_DECODE_ONLY graphs. Compare the v1.2.5 baseline with individually patched images; use v1.2.6t2 only with its unsafe proposer-width feature disabled. This plan narrows the [latest-image audit](../latest-image-audit/REPORT.md) to long-context latency, throughput, quality and memory.

**User scope refinement: token-generation degradation is the primary objective.** Optimize steady-state decode milliseconds per emitted token and concurrent output goodput at128k/262k. Prefill TTFT is a separate secondary metric; quality/state tests are mandatory regression gates, not substitutes for measuring decode speed.

## 1. Recommended direction

**For immediate long-context operation, nonspec FP8 is the strongest historical decode baseline; nonspec TQ4 is the capacity-oriented alternative. For a real code fix, optimize long-context attention for both modes and make speculation conditional on its measured cost per emitted token.** Improving acceptance alone or lowering KV precision alone does not ensure faster generation.

Treat four distinct problems separately:

| Degradation | Observable symptom | Required remedy |
|---|---|---|
| Expected dense-attention growth | Decode time increases smoothly with attended length | Improve bandwidth/parallelism and tile reuse; exact full attention still reads historical KV |
| Excess implementation overhead | Cliffs, low occupancy, repeated conversion, allocator stalls, exposed host waits | Kernel, workspace, scheduler and communication fixes below |
| Speculation becomes unprofitable | Spec has higher time per emitted token than matched nonspec | Lower scheduler-owned verify width or select nonspec; preserve draft/state contracts |
| Quality degrades | Wrong retrieval, state contamination, periodic output, missing final answer | Numerical/state/sampling differential tests; model limitations require model-level or explicit retrieval changes |

Do not promise constant latency to262k with exact dense attention, universal speculative speedups, or recovery of model reasoning quality through a kernel optimization alone.

## 2. Evidence and what is still unknown

Historical v125 results, capacity prompts with256 generated tokens:

| Mode / KV | ~67k decode t/s | 136191 prompt decode t/s | 242761 prompt decode t/s |
|---|---:|---:|---:|
| nonspec FP8 | 29.6 | 25.6 | 21.2 |
| nonspec TQ4nc | 23.2 | 17.7 | 9.6 |
| MTP4 TQ4nc | 21.0 | 11.7 | 6.1 |
| DFlash7 TQ4nc | 14.3 | 4.2 | 7.3 |

These are [archived measurements](../v125-audit/REPORT.md), not equal-length semantic benchmarks. A later DFlash7/FP8 run measured5.4 t/s at136191 and5.2 at261757 prompt tokens; the latter plus256 output is262013 total, not the exact limit. Its short thinking study detected one periodic tail among four selected prompts, while two other prompts exhausted their budgets without periodicity. [Later baseline evidence](../latest-image-audit/evidence/df7-fp8-completed-report.txt), [loop evidence](../latest-image-audit/evidence/loop8-partial.out).

At17:12 UTC this review found the external `bench127` job still running on v1.2.5, in the DFlash/TQ4 deep phase. Its preceding ~67k result was14.3 t/s. No competing GPU benchmark was started. The in-progress [v127 draft report](../v127-audit/REPORT.md) has unfilled result sections and causal assertions that need the corrections below; it is not a completed experiment record.

### New source findings that change the fix design

1. **DFlash2 is not seven autoregressive model forwards per step.** It constructs a native eight-row block and selects seven candidates; its selector walk has a short sequential dependency. Reducing emitted candidates can save target verification work while leaving native draft work largely intact. The deployed [DFlash2 source](evidence/dflash2.py), lines61–77 and234–330, enforces the checkpoint's block size and shows the selector path. This agrees with the parallel-drafting architecture in the [DFlash paper](https://arxiv.org/abs/2602.06036), Chen et al., revised2026-05-28; its published speedups are not predictions for this XPU port.
2. **`DSPARK_ADAPTIVE_BLOCK=1` is not an established DFlash2 solution.** The base [DFlash source](evidence/dflash.py), lines139 and173–178, resets confidence then truncates only if confidence is populated. Confidence is produced in the base Markov-head sampling path. DFlash2 overrides `_greedy_sample` with a candidate-selector walk and does not populate `_draft_confidence` in that inspected path. This knob alone therefore does not activate the proposed confidence policy for the ordinary DFlash2 selector path. Add real selector confidence/calibration or use measured acceptance; test the actual hook firing.
3. **DFlash2 internal width must remain checkpoint-compatible.** The native layout is one anchor plus seven masks; narrower emission uses a list conversion after drafting. The source's variable-width comments do not prove compatibility with all async placeholder and full-graph paths. Historical emission settings and v53 proposer narrowing have failed. Scheduler/runner verification is mandatory.
4. **DFlash2 TP1 replication is not a safe communications shortcut.** The installed source, lines110–135, records a default-off corruption history involving draft remapping/hash granularity. Leave `VLLM_XPU_DFLASH_TP1=0` until that separate defect is fixed and tested.
5. **TQ conversion is repeated, but attention rereads are not all removable.** [TQ continuation prefill](../../v125-fixes/turboquant_attn_v53.py), around1512–1675, dequantizes old KV, rotates keys and concatenates each layer/chunk. Fusing removes global scratch traffic and separate conversion passes. Future query chunks still need their KV tiles for exact attention; do not claim the fused kernel removes dense attention's quadratic prefill work.
6. **An allocator cause for the FP8/GMU cliff remains a hypothesis.** Overriding the KV block count down at GMU0.9 also frees physical memory. A speedup cannot distinguish pool layout from free-memory margin. A larger TQ pool does not exclude FP8-specific address, page, format or cache effects. Use the controlled experiments in LC-5.
7. **Counters and comments are insufficient attribution.** Lane-total DFLASH_STALL counts do not establish stalls occurred during slow decode. Its timer measures host elapsed time, not necessarily synchronized device execution. The inactive TQ host tier ladder is ruled out under current graphs, but the depth anomaly itself remains open. Small-N per-client rates do not prove a particular fairness mechanism.
8. **Executable defaults take precedence over comments.** The archived [installed FP8 MQ source](../v125-audit/evidence/installed-triton_fp8_mq.py) comments describe tile16/one warp, while assignments around61–67 default to tile32/four warps. Log effective constants; do not tune from the header description.

## 3. Cost model and success definition

Measure request-window totals, not process-lifetime acceptance:

`nonspec_cost_per_token = total_nonspec_step_time / emitted_target_tokens`

`spec_cost_per_token(K) = total_spec_step_time(K) / emitted_target_tokens(K)`

For the conceptual split, spec time contains draft, verification, sampling/state commit and exposed scheduling/communication. These overlap in places, so use a device/host timeline rather than blindly adding profiled durations. Emitted tokens include accepted drafts and any valid bonus token. Use sum(time)/sum(tokens), not the unweighted mean of individual step ratios.

Choose speculation only when the measured cost is lower than the matched nonspec alternative by a margin larger than noise. For an initial controller gate use10% with at least64 comparable completed steps and hysteresis; these are tuning starting points, not validated constants. At high concurrency optimize quality-passing throughput under latency SLOs as well as single-request cost.

Do not infer K+1 independent full-cache reads: a multi-query verification kernel can reuse loaded KV across query rows. MTP drafting, DFlash drafting and target verification have different layer counts and window policies. Profile each separately.

## 4. Ordered implementation plan

### Priority specifically for token-generation speed

Start with LC-0 measurement and the LC-6 correctness screen. Then prioritize **LC-1 dedicated long-context decode kernels**, **LC-3 MTP cost-based width/nonspec fallback**, and **LC-4 DFlash draft/verify cost reduction**. Run LC-5 alongside profiling when memory pressure appears on the decode critical path. LC-2 is primarily a prefill/memory improvement: do not claim a steady-state decode gain from it unless a measured reduction in allocator pressure or interference produces one.

Historical timings expressed as milliseconds per emitted token make the deficit concrete (1000 / reported decode tokens/s; approximate because the original rate is a client-side proxy):

| Lane | At136191 prompt tokens | At242761 prompt tokens | Immediate comparison target |
|---|---:|---:|---|
| nonspec FP8 | 39.1 ms | 47.2 ms | Optimize its measured attention critical path; retain as reference |
| nonspec TQ4 | 56.5 ms | 104.2 ms | Close TQ-specific overhead without losing capacity advantage |
| MTP4 TQ4 | 85.5 ms | 163.9 ms | First recover same-dtype nonspec cost via safe fallback, then demonstrate profitable speculation |
| DFlash7 TQ4 | 238.1 ms | 137.0 ms | Explain anomaly and separate native draft cost from target verify cost before width tuning |

These are historical comparison points, not promised patch results or exact262k measurements. Compare each candidate to a fresh matched baseline; changing dtype, concurrency, prompt content or cache state cannot be credited as a kernel-only improvement.

For every proposed optimization, record the fraction `f` of baseline decode time it can actually affect. Even an infinitely fast replacement has an end-to-end speedup ceiling of `1/(1-f)` if other work is unchanged. Use measured critical-path fractions, including overlap, to prioritize work: a faster prefill or a rare ragged route cannot explain a large steady-state decode improvement. Request-level routing to a nonspec deployment is the lowest-complexity first mitigation; switching an already-running request requires LC-3's state protocol.

### LC-0 — Make the degradation reproducible before altering kernels

Implement a long-context measurement harness and manifest recorder. Save image/container ID, actual imported source hashes, effective block sizes, model/tokenizer/template hashes, target/draft cache groups, settings and current device health. Detect container replacement before/after each run. Use exact token-ID capacity prompts and separately rendered natural chat prompts.

Record per-request accepted counts by position, emitted tokens, target verification rows, draft rows, prefill chunks, graph/eager routes, KV occupancy, peak allocated/reserved bytes, CPU wait and device timing. Correlate stall logs with request/step IDs. Capture the first error from each rank.

Use final API usage; count neither characters nor SSE events as tokens. Speculative chunks can emit several tokens: client event gaps are not true token ITL. Use server timestamps for token ITL or explicitly label the proxy. Preserve natural EOS and reasoning/content separately.

**Done:** repeat the historical depth curves with three fresh-process repetitions and distinct equal-length prompts, or document the changed result. Every subsequent patch depends on this baseline.

### LC-1 — Nonspec: optimize the q=1 long-context attention path

Targets: FP8 and TQ attention backends plus their decode kernels. Keep current FP8 q1→FA2 as the reference; prior forcing of q1 into the MQ split kernel regressed. TQ already has split decoding; adding “split KV” again is not a new fix.

Implementation sequence:

1. Profile128k/262k, C1/C2/C8: effective bandwidth, occupancy, register spills, dequant/rotation instructions, reduction time and launches. Compare matched shapes across FP8, auto KV and TQ4.
2. Implement or tune a dedicated TQ q1 specialization: unpack/load tiles once per KV group, reuse across query heads where profitable, retain FP32 stable softmax accumulation, bound register use. Evaluate vectorized unpack/centroid loads and fused norm correction.
3. Tune active splits and tile/warp combinations offline per dtype/head dimension/concurrency. Graph replay must use runtime device sequence lengths; fixed capture grids can mask inactive work. If dispatch chooses different grids, capture each legal variant and select consistently across ranks.
4. Reuse bounded reduction scratch with explicit shape/dtype/stream ownership. Test empty splits and tails; all-empty padded rows must not inject NaNs into real outputs.
5. Add a direct ragged/multi-query specialization only where LC-0 shows actual use; avoid global routing changes that penalize q1.

The [Flash-Decoding design](https://crfm.stanford.edu/2023/10/12/flashdecoding.html), Dao et al.,2023-10-12, supports sequence-parallel partial attention with stable reduction. Its NVIDIA measurements do not establish the best B70 split count. Promotion requires end-to-end gains, not just one fast attention microbenchmark.

**Done:** ≥10% median improvement at the target deep workload, confidence supporting a gain, ≤5% regression in protected short/mixed workloads, unchanged quality gates. If FP8 remains best, retain it; no requirement to force a TQ win.

### LC-2 — Both modes: eliminate full-prefix dequant materialization during prefill

Implement paged TQ continuation attention with compressed KV loaded/dequantized in on-chip tiles and stable online softmax. Handle cached quantized prefix and current raw chunk deliberately: either preserve the baseline's raw-current-chunk numerical behavior, or classify quantizing that chunk first as a separate numerical change.

Preserve key rotations, norm correction, GQA mapping, scale/zero metadata, causal limits, page offsets and draft window semantics. Evaluate rotating Q into the stored-key space rather than inverse-rotating every cached K, but only after proving the exact transform convention, scaling and finite-precision tolerance against current kernels. This is an optimization candidate, not an algebraic shortcut to assume safe.

Use bounded reusable scratch, no `[full_context, heads, dimension]` temporary for every chunk. Keep the reference FA continuation path behind a rollback switch. Validate all four TQ layouts and targetD256/draftD128 separately.

Memory sanity check atTP2 with two local target KV heads andD256: full target FP16 KV across16 full-attention layers is8 GiB/rank at262144 tokens. Retaining an additional dequantized copy costs roughly another8 GiB/rank, before draft/SSM/graphs; it is not a small cache. One layer's cached K+V is512 MiB, concatenated K+V another512 MiB, rotated K256 MiB before other copies. Measure peak rather than reusing earlier unqualified2-GiB estimates.

At32 chunks of8192, the existing prefix conversion visits4,063,232 cached positions/layer (15.5× final prompt length). The fused path removes separate full-prefix conversion/materialization traffic; it does not eliminate repeated attention to those positions by different queries.

**Done:** lower cold unique-prefix TTFT and measured peak live memory at128k/262k, correctness across chunk/page boundaries, no growth after repeated runs. Keep a change only if end-to-end measurements justify it.

### LC-3 — MTP: make speculation conditional without breaking async execution

Immediate experiment: fresh-server fixed MTP K1/K2/K4 plus true nonspec at equal dtype, context and resident load. Use boot-consistent widths; keep `VLLM_V53_L2_UNSTABLE=0`. K1 still pays drafter/verification overhead and is not nonspec.

Then implement a scheduler-owned width contract:

1. First support one width per batch/step to limit complexity. Scheduler emits a step ID, selected width and pending-placeholder count; TP ranks agree before execution.
2. Propagate width through scheduler output, runner input preparation, draft generation, verification, rejection sampler, GDN state commit, output placeholders and KV lookahead accounting.
3. Drain old-width in-flight work before switching; graph dispatch uses explicitly validated `(batch bucket, width)` variants. Never mutate proposer width behind scheduler accounting.
4. Implement K0 as a real target-only path. Define how draft KV/hidden states catch up before re-enabling speculation. If catch-up is not implemented, make the transition to nonspec one-way for that request; re-enable only for new requests.
5. Seed a static policy from measured context/concurrency cells, then add acceptance-and-time estimates with hysteresis. Use scheduler CPU metadata; avoid D2H synchronization to choose width. Don't explore risky widths on live requests.

Current [upstream Dynamic SD documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/), updated2026-07-07, restricts MRv1 dynamic SD to piecewise graphs and full graphs to MRv2. Evaluate its protocol before porting; it is not a drop-in solution for the current MRv1 FULL_DECODE_ONLY XPU overlays.

**Done:** all width transitions and rejection patterns preserve token/state accounting; deep policy approaches matched nonspec cost when speculation loses and retains measured short-context gains. Include switching/catch-up overhead in the result.

### LC-4 — DFlash2: repair acceptance/cost at depth using its actual architecture

First distinguish low acceptance from slow verification or slow drafting. Record per-depth/per-position acceptance and calibrated candidate-score margins, window/position metadata, draft dtype, native row count and selector cost. Never compare short-battery acceptance to deep-request timing as causal evidence.

Correctness checks before policy changes:

- Native anchor+mask rows and grouped convolution phase stay consistent across batches, prefill continuation, accepted-tail rollback and reused requests.
- Sliding attention uses the intended causal rule and absolute/window positions; KV rollover retains required entries. Source comments record depth-sensitive acceptance when causality was changed. Reproduce against the checkpoint/reference rather than changing `is_causal` by intuition.
- Selected target hidden layers, bonus anchor, candidate IDs and selector edges align with the actual target tokenizer/checkpoint. Hash both model configurations.
- Sampled decoding uses a correct target rejection/verification algorithm. A greedy drafter can be compatible with an exact verifier; absence of the upstream probabilistic draft path alone does not prove biased target sampling. Test the actual rejection contract.

Optimization sequence: fuse/preallocate selector scratch; profile CPU launch and TP costs; then scheduler-owned **emitted** width using LC-3's protocol while native draft width stays8. Measure whether saved verification pays for unchanged draft cost and list/D2H overhead. If draft remains expensive, choose true nonspec rather than endlessly narrowing output. Do not offer the DSpark knob as a working controller without wiring and validating confidence.

For longer-term acceptance improvement, evaluate depth-diverse draft training/calibration with the actual quantized target and tokenizer, preserving held-out long-context tasks. More draft precision or a wider draft window may improve acceptance but consumes bandwidth/memory; optimize total cost per emitted token. Draft-only approximation can preserve target distribution only if the verifier is correct; target attention truncation changes the answer distribution.

**Done:** no new periodic tails or state errors, profitable deep speculation where enabled, and safe fallback where it is unprofitable. Keep TP1 replication and arbitrary internal-width changes disabled.

### LC-5 — Both modes: identify memory pressure, capacity and address effects

Use actual bytes/groups rather than GMU labels. All following are isolated lab experiments, never ballast in production:

| Experiment | Hold fixed | Change / interpretation |
|---|---|---|
| GMU comparison with equal actual pool bytes | Image, graphs, group layout, prompts, pool reservation | If behavior differs, inspect remaining startup decisions; verify physical free memory rather than assuming it differs |
| Pool-size sweep | Model/settings and representative workload | Maps effect but pool size and free margin covary; not causal separation |
| Same pool plus inert device allocation after setup | Pool layout, graphs, prompt | Tests sensitivity to reduced free margin; control allocation timing and verify no allocator side effects invalidate comparison |
| Different pool with matched measured free margin | Use controlled reserved ballast in the smaller-pool arm | Helps separate pool/address effects; compare allocator stats/timelines, not just final free bytes |

At each run collect allocated/reserved/peak, allocation retries, free blocks per KV group, graph buffers and per-rank timing. Do not infer allocator stalls solely from Python stack samples. Keep failed attempts in the record.

Admission must reserve prompt+output+lookahead and hybrid cache/state constraints. The historical DFlash/FP8 pool373507 is below two unrelated262k requests: they may queue/preempt but cannot both fully reside under that capacity. Test queueing fairness, not an impossible64×262k promise. Prefix blocks are potentially reclaimable; cache retention is not automatically a leak.

For mixed workloads try MNBT2048/4096/8192 and concurrent-prefill/admission limits before a scheduler rewrite. Splitting prefill among clients trades first-client TTFT against others and can change numerical batching: it is not risk-free. Respect effective512/1024/2048 hybrid boundaries rather than hardcoding a1024-token floor.

**Done:** identify the causal slowdown or leave it explicitly open; choose a measured memory/latency operating point, with bounded post-cancel retention and no device loss.

### LC-6 — Quality: distinguish cache quantization, SSM drift, verifier bugs and model limits

Reference ladder at each length, same target weights/tokenizer/template:

1. nonspec + auto KV + FP32 SSM;
2. nonspec + auto KV + baselineFP16 SSM;
3. nonspec + FP8/TQ KV + each selected SSM precision;
4. MTP/DFlash + the same KV/SSM combinations that pass startup/capacity gates.

The first reference still has FP8 target weights; it does not isolate weight-quantization error. If all these fail similarly, a higher-precision weight/reference-model study is separate work. FP32 SSM changes state memory and possibly effective block sizing: record this instead of pretending it is cost-free.

For quality divergence, replay a common teacher-forced token prefix and compare layer outputs/logits at chosen checkpoints. Locate the first divergence before free-running generations branch. Check NaNs/infinities, scales and saturation, target-state commit after rejected candidates, cache block reuse and position IDs. Greedy byte differences alone do not distinguish benign rounding from corruption; use tolerances and task outcomes.

Fix based on the isolated cause: correct stale state/offsets for spec-only errors; correct scale/layout or protect sensitive full-attention layers for quantization-specific errors; retain FP32 SSM if recurrence drift is demonstrated and capacity permits. Layer skipping must use actual full-attention layer IDs and supported mixed-cache grouping, not dense-model first/last-layer heuristics.

If the uncompressed-KV reference also loses distant facts, evaluate model-level limitations and explicit RAG/context selection. That is an application change with potential recall loss, not an exact262k attention fix. Never silently truncate context or change RoPE to improve a speed score. The recorded model limit is already262144; a permissive max-length flag does not improve learned positional generalization.

**Done:** semantic regressions bounded against a declared reference and a clear failure taxonomy: wrong answer, periodic loop, coherent budget exhaustion, parser-only failure, or engine/state error.

## 5. Required long-context tests

### Exact sizes and workloads

| Test | Prompt / output | Clients | Purpose |
|---|---|---|---|
| Mid-depth curve | 32768/65536/98304/131072 +256 | 1,2,4 | Find crossover and excess cost slope |
| Deep curve | 163840/196608/229376/261888 +256 | 1,2;4 if capacity allows | Quantify monotonicity and resident vs queued performance |
| Exact max boundary | **261888+256=262144** | 1,2 | Correct budget and termination, with unique prefixes |
| Long output at max total | **258048+4096=262144**; **253952+8192=262144** | 1,2 | Stateful decode, EOS and repetition |
| Hybrid boundaries | 512/1024/2048 multiples±1 around64k/128k/near262k | 1,2,4 | Chunk/state/block transitions |
| Mixed arrivals | Short2k/16k plus131072/261888 | 4,8,16 | Goodput, prefill interference, fairness and admission |

Run six primary lanes: nonspec/MTP4/DFlash7 × FP8/TQ4; include auto KV quality controls and TQ K8V4/K3V4/3bit as secondary precision candidates. Other enum values must pass the [capability gates](../latest-image-audit/REPORT.md) before inclusion. No unsupported dtype becomes “passed” through argument parsing alone.

Capacity runs may use ignore_eos with exact token counts. Natural application tests must not: cover code repair with executable tests, structured tool calls, long document synthesis, single/multiple needles, multi-hop record joins, no-answer distractors, multilingual records and long reasoning. Place facts at1/10/25/50/75/90/99% depth and verify trimming preserves them. Use at least20 documents per position/length for initial retrieval scoring; expand when uncertainty prevents a decision. [RULER](https://arxiv.org/abs/2404.06654), Hsieh et al.,2024, motivates testing aggregation and tracing beyond one-needle retrieval; it is not evidence of this target's quality.

At128k/262k test cold process, warmed unique prompt, exact prefix reuse, eviction/reuse and reversed length order. Use equal lengths with distinct seeds to distinguish content/acceptance from shape/state effects. Long-context warmup that merely primes the same prefix cannot pass the unique-prompt gate.

### Patch-specific invariants

- Accept0…K candidates, natural EOS inside proposed block, output-budget clipping, preemption, cancelled old-width step, K0 re-entry/catch-up, TP agreement and no stale placeholders.
- Ragged query rows, zero-length padded partitions, per-group block sizes, scales, page tails, sliding windows and graph replay with changed device seq_lens.
- Two clients with identical system prefix but distinct secret synthetic facts; no contamination after cancellation/eviction/reuse.
- Cancel before first token, during deep prefill, during verification and mid-generation; next short request succeeds and live allocations/blocks recover.
- At least100 short/mixed requests for indicativep95; deep small-N runs publish individual timings/range. Use1000+ for crediblep99 estimation. Three fresh-process repeats for performance comparisons; profile representative runs separately from clean timing runs.

## 6. Delivery milestones and rollback

| Milestone | Deliverable | Exit gate |
|---|---|---|
| M0 | LC-0 harness, manifests, six-lane depth curves, raw outputs | Reproducible attribution and valid counts |
| M1 | LC-6 quality diagnosis and LC-5 memory experiments | Identify correctness blockers; no tuning around corrupted output |
| M2 | Dedicated q1 improvements from LC-1 | Deep nonspec gain with quality/short-context protection |
| M3 | Tiled continuation kernel from LC-2 | Lower cold TTFT/peak memory; all layouts and boundary tests pass |
| M4 | Scheduler width protocol, fixed policy and safe K0 from LC-3 | State/async/graph transitions pass; MTP deep fallback works |
| M5 | DFlash selector/window fixes and emitted-width policy from LC-4 | Measured net gain or safe nonspec fallback; no loop regression |
| M6 | Combined candidate, mixed traffic and lifecycle soak | 2h screening then24h release soak; zero structural/device failures |

Use one patch per image with immutable digest and explicit rollback. Kernel changes retain old routing until promotion. Width switching defaults off until its entire state-machine suite passes. Do not change the active external benchmark job; take the benchmark lock after it completes and preserve its logs.

Performance gate: target ≥10% median improvement with evidence beyond noise, ≤5% protected-workload regression, and no quality/SLO goodput regression. These are proposed acceptance thresholds, not predicted gains. Reliability gate: zero device resets, silent corruption, cross-request leakage or missing terminal events; bounded memory after repeated workload/abort cycles. Numerical changes require an explicit tolerance and task score, not universal bit identity.

## 7. Research scope and open questions

Research directly checked deployed DFlash/DFlash2 source, local TQ/FP8 kernels, v53 failure records, prior raw baselines and current host-job state; upstream sources above supply algorithm and compatibility context. Source snapshots under `evidence/` were read-only captures from the running v125 container. No new kernels or load tests were deployed in this turn.

Still open: exact cause of GMU-sensitive FP8 latency; DFlash128k anomaly after per-step acceptance/stall correlation; true262k semantic quality; actual FP32-SSM accuracy benefit; profitable width boundaries; and latest-image results for secondary dtypes. These must be measured, not replaced with a confident env-variable recommendation.
