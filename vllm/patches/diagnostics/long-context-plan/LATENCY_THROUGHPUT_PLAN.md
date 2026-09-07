# Long-context latency and maximum token throughput plan

Date: 2026-09-07. Status: researched proposal, not deployed. Scope: the user's TP2 B70/Qwen hybrid deployment, nonspec, MTP and DFlash2, with 128k and262k contexts. This is a separate operating-point and optimization plan; [PLAN.md](PLAN.md) supplies detailed kernel/state-machine repair designs.

## 1. What to optimize

**Start from nonspec FP8 for long-context decode speed and nonspec TQ4 for capacity-sensitive traffic. Retain speculation only where measured time per emitted token is lower.** These are historical starting points, not proven optima for every prompt or concurrency level.

Minimum latency and maximum aggregate throughput are competing objectives. Produce three validated profiles instead of calling one setting optimal:

| Profile | Primary objective | Constraints |
|---|---|---|
| First-token latency | Minimize queue delay plus cold unique-prompt TTFT | Preserve output quality and acceptable decode latency |
| Interactive decode | Minimize milliseconds per emitted token and final-answer latency | Bound prefill interference and queueing |
| Concurrent throughput | Maximize quality-passing output tokens/s under declared latency SLOs | No memory failures, starvation or semantic regression |

Also record time to first final-answer content. The baseline's preserved thinking and xhigh reasoning can produce early reasoning tokens while delaying the answer substantially. Changing thinking settings is a separate user-visible behavior experiment, not a kernel speedup.

Optimize a measured latency decomposition:

`request latency = queue + tokenization/preprocessing + prefill + decode + finalization`

`decode cost/token = sum(decode wall time) / sum(emitted tokens)`

`aggregate output rate = completed output tokens / common measurement interval`

For SLO goodput count only requests passing quality, protocol and declared latency gates; include failed and unfinished work in load/error accounting. Do not reward periodic repetitions or padded output as useful generation. A prefix hit improves prefill work; it does not remove full-context attention during subsequent decode.

## 2. Research evidence and fresh update

The [prior audit](../latest-image-audit/REPORT.md) identifies the latest locally available experimental image as v1.2.6t2, with vLLM0.21.1.dev0, Torch2.11 XPU and three v53 overlays. Its adaptive proposer narrowing is unsafe when enabled; defaults do not establish a performance upgrade. The active host benchmark remains v1.2.5. Pin exact image/source identity in all new trials.

Historical single-client decode at136191/242761 prompt tokens:

| Lane | Decode tokens/s | Interpretation |
|---|---|---|
| nonspec FP8 | 25.6 /21.2 | Strongest historical deep-decode baseline |
| nonspec TQ4 | 17.7 /9.6 | More compressed cache, slower measured decode |
| MTP4 TQ4 | 11.7 /6.1 | Extra speculation cost exceeds benefit on those prompts |
| DFlash7 TQ4 | 4.2 /7.3 | Non-monotonic curve; mechanism remains unproven |

At17:21 UTC this review observed the external bench127 job progressing through DFlash/TQ4. Its136191/261757-token prompts produced4.2/6.7 reported decode tokens/s, with TTFT88.7/128.7 seconds. Its four selected8192-budget thinking tests detected three periodic tails and four budget exhaustions. This extends the earlier FP8 observation: periodic tails are **not exclusive to FP8** in these samples, but the cause is still unknown. Raw content and matched nonspec controls are needed. [Captured host evidence](evidence/latency-throughput-host-snapshot.txt).

The261757+256 run totals262013 tokens, not exactly262144. All comparisons above are capacity/performance evidence, not semantic long-context certification. No competing inference was added and the running job was not interrupted.

Research conclusions carried into this plan:

- TQ continuation prefill materializes dequantized historical KV per chunk; the avoidable memory traffic affects TTFT and possibly runtime memory margin.
- TQ and FP8 decode routes differ; lower packed bytes do not guarantee lower decode time. The TQ path already splits KV, so a new generic split flag is not a complete optimization.
- DFlash2 drafts a native eight-row block in parallel; lowering emitted width primarily saves target verify work. Its selector path does not populate the base DSpark confidence hook in the inspected implementation.
- MTP/DFlash adaptive width must agree with scheduler placeholders, verification, graphs and hybrid-state rollback. Enabling a proposer-local knob is not a safe latency fix.
- Warmup did not establish transferable deep-decode improvement. GMU-related FP8 slowdown remains causally unresolved.

Source/code anchors and qualifications: [source-backed findings and cost model](PLAN.md), [DFlash2 snapshot](evidence/dflash2.py), [TQ backend](../../v125-fixes/turboquant_attn_v53.py), [v53 post-mortem](../../v125-fixes/PLAN.md).

## 3. Concrete initial configurations

Use this baseline command template, inside the existing image with existing model mounts and host environment. Variables are illustrative shell inputs; no launch was performed by this document.

```bash
KV=fp8_e4m3
GMU=0.9
MNBT=8192
SEQS=64
SPEC=()  # nonspec; replace with one option below for a spec comparison

vllm serve --model /models/target --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 --gpu-memory-utilization "$GMU" \
  --max-model-len 262144 --max-num-batched-tokens "$MNBT" \
  --max-num-seqs "$SEQS" --block-size 512 --dtype float16 \
  --kv-cache-dtype "$KV" --mamba-ssm-cache-dtype float16 \
  --async-scheduling --enable-prefix-caching --trust-remote-code \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  "${SPEC[@]}" --port 8000
```

For MTP comparison set `SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":4}')` before the command. For DFlash2 set `SPEC=(--speculative-config '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}')`. For the capacity comparison set `KV=turboquant_4bit_nc`. Omit speculation entirely for the true nonspec reference.

| Candidate | Initial settings | Controlled sweep |
|---|---|---|
| Fast long-context decode | nonspec FP8, GMU0.9, MNBT8192, SEQS64 baseline | MNBT2048/4096/8192 under mixed load; SEQS1/4/8/16/32/64 subject to capacity |
| Capacity/long concurrency | nonspec TQ4, same baseline | Compare FP8 at equal feasible resident load, then equal offered load |
| MTP required | MTP4, FP8 and TQ4 | Fresh-process fixed K1/2/4; GMU0.85 vs0.9 for FP8 if262k fits |
| DFlash required | DFlash7, FP8 and TQ4 | Fix/measure draft, verify and selector cost first; no arbitrary native-width reduction |
| TTFT-focused | Best quality-passing prefill lane, MNBT8192 baseline | Smaller chunks for interference control; prefix reuse as a separately measured workload |

Keep baseline environment recorded in [host inventory](../latest-image-audit/evidence/live-inventory.txt): MRv1, spawn, XPU graphs, existing oneCCL/P2P settings and expandable segments. Preserve it during first comparisons. Leave `VLLM_SPEC_CTX_K` unset, `VLLM_V53_L2_UNSTABLE=0`, `VLLM_XPU_DFLASH_TP1=0`; do not introduce unvalidated emission widths, q1 MQ rerouting or split64/128 changes. Prior local results rejected those classes; change them only in a trace-backed experiment.

MNBT16384 is not a recommended setting for this host: earlier local trials rejected it. Generic upstream advice is not a reason to override that evidence. Increasing SEQS or GMU cannot create bandwidth and may increase queueing, interference or runtime allocation pressure. Requested block512 can become a larger effective hybrid grouping unit; record the actual value.

Secondary dtype study: auto KV reference, TQ K8V4, K3V4nc and3bitnc only after quality/support checks. FP8 aliases and other accepted enum names are not proof of kernel support. Use the [full dtype gate table](../latest-image-audit/REPORT.md). Neither lower precision nor FP32 SSM is automatically optimal; the latter is an accuracy-control arm with a memory cost.

## 4. Optimization workstreams

| Order | Change | Latency component improved | Implementation / verification |
|---|---|---|---|
| 1 | Correct measurement and reference profiles | Enables all decisions | Record final token usage, identity, prefix state, per-rank timings, accepted/emitted counts and error denominators |
| 2 | Admission and interference control | Queue delay, p95 TTFT/ITL | Budget actual prompt+output+lookahead by KV group; cap resident long work; sweep chunk budget; trace starvation before changing fairness |
| 3 | Dedicated long-context q1 TQ kernel work | Nonspec decode; target-only fallback | Profile bandwidth/unpack/rotation/spills/reduction; tune tile reuse across GQA heads and split count with stable FP32 accumulation |
| 4 | Fused compressed-KV continuation attention | Cold TTFT and peak memory | In-tile dequant, online softmax, no full-prefix scratch per chunk; retain reference route and all layout/causality tests |
| 5 | Safe speculation policy | MTP/DFlash decode | Scheduler-owned widths, matching graphs/placeholders/GDN commits; true K0 and explicit draft catch-up or one-way fallback |
| 6 | DFlash selector and metadata overhead | Draft/verify step latency | Preallocate/fuse selector work; validate windows/anchors/hidden layers; retain native eight-row drafting |
| 7 | Exposed CPU/communication chain | Spec concurrency and tail latency | Timeline exposed waits, batch metadata transfers, remove measured synchronizations; bounded worker loop only after state/cancel protocol tests |
| 8 | Graph/workspace/allocator tuning | Runtime stalls and stability | Log actual pools/graphs; isolate free-memory margin from pool layout; no per-step empty_cache or global synchronize added as a speed fix |

The low-risk mitigation before adaptive code exists is selecting a fixed nonspec configuration for a long-context deployment. Routing between separately provisioned engines requires enough hardware; do not assume two additional TP2 services fit the same pair of cards. Restarting or migrating an in-flight request to another engine adds prefill/state-transfer costs and must not be hidden in speed claims.

### Speculation decision rule

For each measured `(mode, KV, context bucket, resident batch, output workload)` compare total step time divided by emitted tokens against nonspec. Enable spec only with a margin exceeding noise; initially require10%. Measure verification and draft costs separately, but account for timeline overlap. High acceptance can still be slow, and DFlash emission truncation does not remove the native draft pass.

A future controller chooses scheduler-owned width at safe boundaries, with hysteresis and minimum evidence. Start with one width per batch/step. K0 means real target-only execution; resuming spec requires draft state catch-up. Keep deterministic/static policy fallback. Never reinterpret the current unsafe proposer-only mechanism as the controller.

### Memory and serving limits

Historical DFlash/FP8 pool373507 tokens cannot fully hold two unrelated262k requests. C2 may queue or preempt; report that separately from simultaneous resident decode. For FP8/TQ comparison test both equal resident concurrency and equal offered load: TQ may win useful throughput by avoiding queue/preemption even when its single-client decode is slower.

Use byte-level group reservations, not `floor(total_pool/maxlen)` as a guaranteed admission count. Include SSM checkpoints, draft pools, graph scratch and output lookahead. Reclaimable prefix cache is not necessarily a leak. Test cancellation and eviction return live allocations/blocks to their stable state while allowing allocator-reserved memory to remain cached.

A smaller KV pool also frees memory; a GMU/block-count comparison alone cannot isolate allocator pressure. Use the controlled memory/layout experiments in [LC-5](PLAN.md), and inspect device/host timelines rather than assigning causality from a single counter.

## 5. Benchmark design that selects the optimal operating points

### Stage A: single-client surface

Run six lanes: nonspec/MTP4/DFlash7 × FP8/TQ4, plus auto KV controls where capacity permits. Use prompt counts32768,65536,131072,196608 and261888;256 output for capacity comparisons. At every deep point use three distinct equal-length prompts and at least three fresh-process repetitions. Retain identical sampling/template settings.

Measure cold unique-prompt TTFT, warm unique-prompt TTFT, exact-prefix reuse TTFT, decode tokens/s, end-to-end latency and first final-answer latency separately. Reverse length order to expose history effects. Obtain true ITL from server token timestamps; otherwise report SSE event gaps as event gaps.

### Stage B: concurrency and saturation

For each candidate test C1/2/4/8 at128k and near262k; explicitly classify queued/preempted versus resident clients. Test C16/32/64 with mixed short+long traffic when feasible. Use both synchronized bursts and open-loop arrivals, increasing offered load until queues or latency SLOs fail. Choose the highest sustainable quality-passing throughput below that point, with operational headroom; maximum transient throughput is not sustainable capacity.

Suggested reproducible synthetic mix:80%2k/16k,15%131072 and5%near262k prompts. Also test an all-long workload. Declare output-length distribution, prefix reuse ratio, temperature and thinking settings. Start with fixed256 output, then replay realistic256/1024/4096/8192 budgets with natural stopping. Do not extrapolate short output throughput to long reasoning.

For indicative p95 collect at least100 completed requests; use1000+ for meaningful p99 estimation and show uncertainty. Publish individual/range statistics for small-N deep cells. Keep failed, cancelled and unfinished requests in separate counts, not silently removed.

### Stage C: exact boundaries and natural usefulness

| Purpose | Exact prompt+output budget |
|---|---|
| 128k prompt capacity | 131072+256 |
| Maximum total boundary | **261888+256=262144** |
| Long output near maximum | **258048+4096=262144** |
| Long reasoning budget | **253952+8192=262144** |

Tokenize using the actual server. For chat count the fully rendered tools/roles/history. Test total limit−1/exact/+1 with documented rejection/clipping behavior. A262144-token prompt leaves no positive output budget.

Natural scenarios: long code repair with executable tests; distant-record retrieval and multi-hop joins; no-answer controls; long document synthesis; streamed tool calls; and the periodic-tail regression prompts. Place required facts across the context and verify trimming retains them. Separate periodic looping, coherent budget exhaustion, parser failures and correct completion. Faster incorrect output fails promotion.

### Stage D: choose profiles, not a single winner

Build a table of feasible candidates with columns: cold/warm TTFT, decode ms/token, end-to-end latency, aggregate output rate, quality/SLO goodput, peak memory, queue/preemption rate, loop/error rate and sample count. Remove candidates that are slower and use more memory without a compensating benefit. From the remaining candidates select the three profiles in§1.

Use the small parameter grid first: MNBT2048/4096/8192, GMU0.85/0.9 where capacity permits, and workload-relevant SEQS limits. Sweep fixed MTP K separately. Avoid a huge all-knob Cartesian product; follow stage profiling to refine only sensitive dimensions.

## 6. Implementation milestones and promotion gates

1. Capture the external benchmark when complete; preserve raw outputs and identity. Add benchmark locking and manifest checks. Do not overlap lane changes on this host.
2. Establish the six-lane exact-depth and realistic-output baseline. Publish measurement uncertainty and open quality failures.
3. Tune serving budgets/admission; select initial low-TTFT, low-ITL and throughput profiles with no kernel changes.
4. Build one isolated image each for q1 decode, fused continuation, scheduler-owned speculation and DFlash selector work. Use the detailed source/invariant designs in[PLAN.md](PLAN.md).
5. Compare each patch to matched baseline. Promote only a supported target-workload improvement, initially≥10% median with uncertainty, ≤5% protected-workload regression and no quality/SLO regressions. These thresholds are proposed gates, not speedup forecasts.
6. Combine successful patches and retest. Run2h mixed-load screening and24h release soak including exact boundaries, long outputs, cancellations, preemption, prefix reuse and graceful restart. Any device reset, silent state corruption or cross-request leakage fails the lane; retries do not erase failures.

Rollback: immutable baseline image, original routes behind feature switches, static spec width unless the new protocol passes, baseline env preserved. Do not deploy experimental DFlash TP1 replication, arbitrary native-width changes or unvalidated target KV pruning. Sparse attention/context compression can reduce work but changes full-context semantics and belongs in a separate opt-in product experiment.

Expected deliverables from implementation: machine-readable benchmark results, per-profile launch recipes, stage timing/memory traces, a tested patch set and a measured crossover policy. This document does not claim those experiments or fixes are already complete.

## 7. Primary references and limits

- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/v0.22.1/configuration/optimization/), vLLM project, versioned documentation accessed2026-09-07: chunk budget trades prefill efficiency against decode interference. It is context, not installed-fork compatibility evidence.
- [vLLM Dynamic Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/), updated2026-07-07, accessed2026-09-07: full graphs require MRv2 in that feature; MRv1 supports piecewise. The current XPU full-graph deployment needs a validated port.
- [Source-backed long-context repair plan](PLAN.md) and [latest-image audit](../latest-image-audit/REPORT.md): inspected code, image provenance, historical results and unresolved causes.
- [Fresh host snapshot](evidence/latency-throughput-host-snapshot.txt): live v125 DFlash/TQ4 results and partial benchmark state; not a complete sweep or latest-experimental-image certification.

No absolute maximum tokens/s or minimum latency can be responsibly stated before the controlled sweep. The objective is the best measured operating point for each workload with stable memory and correct useful output.
