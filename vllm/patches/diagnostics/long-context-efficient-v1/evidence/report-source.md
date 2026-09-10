# Long-context efficiency: measured constraints, built candidate, and implementation gates

Date: 2026-09-08. Audience: maintainers of llm-scaler on the dual-XPU host 10.20.3.65. Scope: exact-model serving at 65,536, 131,072 and near 262,144 tokens, nonspeculative, MTP and DFlash2. This report supplements the earlier audits; it does not certify every dtype/backend combination.

## Decision

The current system does not meet the requested minor decode-speed degradation. A new single-client MTP4/TQ4-NC run measured approximately **22.0 tokens/s at 65,536 prompt tokens, 12.1 at 131,072 and 6.1 at 261,888**: declines of 45% and 72% relative to 64k. These are short, synthetic capacity probes, not a controlled quality benchmark. Prompt content and speculative acceptance can confound the context-length comparison. End-to-end results for the new candidate remain unmeasured.

| Exact prompt tokens | Output tokens | TTFT seconds | Reported decode tokens/s | Wall seconds |
|---|---:|---:|---:|---:|
| 65,536 | 128 | 37.30 | 22.03 | 43.11 |
| 131,072 | 128 | 86.32 | 12.08 | 96.91 |
| 261,888 | 128 | 228.01 | 6.08 | 249.06 |

Raw [baseline records](evidence/baseline.jsonl) preserve usage, stream-event times and generated text. Reported decode rate is completion tokens divided by time between first and last content event; it includes first-event tokens in the numerator and is therefore slightly optimistic, especially with speculative bursts. All requests completed with `finish_reason=length` and validated prompt counts. `ignore_eos=true` makes these capacity probes unsuitable for EOS or looping conclusions.

The most promising exact-model design keeps each rank's KV shard resident, decodes compressed KV inside tiled attention, reuses each KV tile across speculative query positions, and tunes split reduction by measured workload. Separately, remove draft-only conversion overhead and address collective launch costs. None of those changes makes full attention constant-time in context length. A hardware marketing bandwidth calculation cannot establish that near-flat performance is attainable.

Working acceptance target, pending a more precise service objective: no more than 15% decode-throughput loss from 64k to 128k, and from 64k to near 262k, with identical output-budget and client-count classes. Also require an absolute throughput improvement over the original 64k baseline; slowing down 64k cannot satisfy this target. Track TTFT separately. This is a target, not a demonstrated result.

## What was actually measured and built

The running container `lsv-test` remained on `llm-scaler-exp:v1.2.5`, image ID `sha256:a522bf15be2b666e7dcaab1d59557c8cc21e3bbcdf6be0a9c6bbf686b31f0ff8`. Its serving mode was MTP4, TQ4-NC, TP2, maximum length 262144, batch-token budget 8192, sequence limit 64, block size 512, FP16 target/SSM, async scheduling and prefix caching. Installed vLLM reported `0.21.1.dev0+gad7125a43.d20260826.xpu`. Newer local experimental tags are not evidence of a faster qualified release.

The root ports `ae:00.0` and `d7:00.0` report **8 GT/s x16**, rather than Gen5. The GPU-side upstream bridges also negotiated 8 GT/s. See [captured topology](evidence/topology.txt). The endpoint's internal-link reporting must not substitute for the upstream slot link. Gen3 x16 provides about 15.75 GB/s per direction before transaction overhead; the GPUs' advertised Gen5 support does not upgrade the motherboard.

A fresh two-rank XCCL benchmark reset inputs before every reduction, synchronized around measurement, and verified every output. This avoids cumulative FP16 overflow invalidating a repeated in-place benchmark. Representative per-rank median results:

| FP16 payload | Rank 0 ms | Rank 1 ms | Interpretation |
|---|---:|---:|---|
| 10,240 bytes | 0.198 | 0.225 | Small decode collectives can be latency dominated |
| 51,200 bytes | 0.384 | 0.545 | Host/synchronization and rank skew matter |
| 655,360 bytes | 0.277 | 0.356 | Microbenchmarks are not monotonic at these sizes |
| 20,971,520 bytes | 2.383 | 2.384 | About 8.8 GB/s payload rate |
| 83,886,080 bytes | 8.672 | 8.675 | About 9.7 GB/s payload rate |

These are payload/time rates, **not measured physical bus utilization**. Timings include host launch and synchronization; the serving graph may behave differently. This establishes a baseline for XCCL, not proof that all traffic used direct peer access or that this is the fastest possible backend. Raw records: [transport results](evidence/transport.jsonl), [transport log](evidence/transport.stderr), [reproducer](transport_bench.py).

Built an opt-in image `llm-scaler-exp:lce1-fp32-cache`, ID `sha256:4ebd0899a976bb74b5aa530738a04bf78e2cc58ed98fe6bd0048a045d1fa1c35`. See [build log](evidence/build.log). It adds a DFlash parameter-conversion cache, disabled by default. It was not deployed over the running server.

## Implemented candidate: preserve FP32 math, avoid repeated casts

The captured [DFlash2 model](evidence/qwen3_dflash2.py) executes `F.linear(hidden_states.float(), w.float())` inside each grouped convolution's preparation, and converts the base kernel to FP32 inside convolution. Every draft layer has attention and MLP convolution modules. Repeatedly allocating and converting static weights consumes bandwidth and allocator work unrelated to useful token generation.

[fp32_cache.py](fp32_cache.py) reuses conversions of the **already loaded and rounded** model weights. It preserves the FP32 GEMM and convolution arithmetic. Keeping BF16 draft internals and FP32 convolution math is essential: the installed model documents overflow in FP16 intermediate values. Replacing this with an FP16 GEMM is not an acceptable speed fix.

The cache checks tensor identity, version, shape, stride, dtype, device and data pointer. Normal versioned mutations and parameter replacement invalidate it. Trainable parameters and unversioned inference tensors retain the original conversion path. This fallback can make the optimization inactive if model construction creates unversioned inference parameters; instrument cache hits during full-model qualification. Mutation through `.data` that bypasses version counters is outside this immutable-inference contract and requires explicit invalidation.

Memory cost is material: a 1280×5120 FP32 projection retains 25 MiB. Two such modules per layer across five draft layers would retain approximately **250 MiB per rank**, plus base kernels. Verify the installed checkpoint's dimensions before budgeting. Account for this before KV-pool sizing and graph capture. Warm caches before capture; parameter changes require graph destruction, explicit cache clearing and graph recapture. The helper's Python invalidation does not rewrite a previously captured graph. Model migration/unload must release retained snapshots; a module device move alone does not traverse this ordinary Python dictionary.

The [generator](build_candidate.py) rejects unexpected source layouts and identical input/output paths, emits source hashes and parses generated Python. CPU and XPU helper tests passed, including mutation, replacement and unversioned fallback. Eight CPU and eight XPU grouped-convolution comparisons passed exactly across FP16/BF16, block sizes 5/8 and one/four client-shaped blocks. These tests use the captured math with a simple parameter container; they do not validate vLLM compilation, graph replay, sampling, acceptance or end-to-end speed.

At the real projection dimensions 1280×5120, the initial XPU bit-exact assertion **failed**. A diagnostic repeat established that the original GEMM also differs from itself across calls: maximum absolute repeat differences were 3.05e-5–6.10e-5. Candidate differences were 3.81e-5–6.10e-5 and passed `rtol=1e-5, atol=1e-4`. This bounds the observed discrepancy; it does not prove unchanged token decisions. Preserve both the [initial failure](evidence/cache-xpu.stderr) and [diagnostic results](evidence/cache-xpu-diagnostic.jsonl).

| Projection rows | Original median ms | Cached median ms | Microbenchmark speedup |
|---|---:|---:|---:|
| 8 | 0.1774 | 0.0716 | 2.48×; original samples show substantial warmup drift |
| 32 | 0.1273 | 0.0712 | 1.79× |
| 64 | 0.1377 | 0.0724 | 1.90× |

Six alternating-order measurements of 30 iterations followed five warmups. These are single-projection results with 25 MiB retained per weight. Do not multiply this speedup across the entire model. At the steadier sizes the saving is approximately 0.056–0.065 ms per projection, which cannot by itself explain or fix the observed hundreds-of-milliseconds long-context serving steps. The candidate remains disabled pending end-to-end qualification.

Build from the audited base using [Dockerfile](Dockerfile). Verify the base image ID above before rebuilding because a tag is mutable. Set `VLLM_LCE1_FP32_CACHE=1` only for candidate qualification, before importing the model module. Remove that setting to restore the old branch. Do not promote until the retained memory and graph tests pass. This optimization benefits DFlash only; it does not fix MTP or target attention by itself.

## Correct diagnosis of the long-context cost

For a serving step, measure target attention, target non-attention compute, collectives, draft work, verification/rejection and scheduler/host gaps. Useful speculative tokens per second are approximately committed tokens divided by draft-plus-verifier-plus-overhead time. Acceptance rate alone does not determine the best draft depth.

For exact full attention, KV reads grow with context. With 16 full-attention layers, four KV heads, head dimension 256 and TP2, the ideal raw K+V payload per rank at 262144 tokens is about 4 GiB at FP8, 8 GiB at FP16, or 2 GiB at nominal four-bit storage before scales, alignment, metadata and other caches. This is a lower-bound storage calculation, not a timing prediction. GQA reuse, speculative query reuse, rereads, reduction buffers and effective bandwidth determine actual traffic. Compare device counters against bytes per **committed token**, not peak bandwidth alone.

The official [Qwen configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json) already mixes 48 linear-attention layers with 16 full-attention layers. The linear layers' bounded decode state does not remove the remaining full-attention history. [Jamba](https://arxiv.org/abs/2403.19887) is a separately trained hybrid architecture. Replacing attention with Mamba/Jamba is a model-training and quality-evaluation project, not a runtime flag preserving the current model's outputs.

TP2 already shards four KV heads across ranks. Moving a complete KV history between GPUs each decode step would worsen the objective. [vLLM context parallelism](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) can distribute context, but adds communication and backend requirements. DCP is not automatically beneficial when there is no replicated KV head storage to eliminate. Prove compatibility with this older custom XPU fork, hybrid-state layout and speculation before considering it.

## Prioritized implementation work

| Priority | Change and code target | Why it can help | Required evidence before enablement |
|---|---|---|---|
| P0 | Add scoped per-step timing and accepted-token counters to runner/proposer; trace XCCL and device kernels | Distinguishes bandwidth, launch, acceptance and scheduling costs | Trace overhead below 2%; matching uninstrumented run |
| P1 | Compressed paged attention with tiled dequantization and online softmax; target TQ decode and MQ kernels | Avoid full-context FP16 materialization; reuse K/V tiles for query positions belonging to the same sequence and KV head | Reference attention/logit comparison, page-tail and causal-mask tests, improved measured device traffic |
| P1 | Graph-safe workload-specific split dispatch | Balance insufficient occupancy against excess partial-result traffic | Tune separate Q=1, MTP verification and DFlash verification buckets; replay with heterogeneous lengths |
| P1 | Built DFlash conversion cache | Removes repeated static parameter conversion | XPU equivalence, memory/capture lifecycle and end-to-end DFlash comparison |
| P2 | Incremental continuation prefill without whole-history dequant/concat in `turboquant_attn_v53.py::_continuation_prefill` | Reduces transient memory and repeated old-context processing | Long chunked prefill and cached continuation tests; TTFT measurement distinct from decode |
| P2 | Stable collective scratch buffers and exact local selection | Reduce allocation churn and communicated logits | Verify existing optimized paths are active before adding new code |
| P2 | Coordinated speculative-depth selection | Choose depth maximizing committed tokens/time by length and client bucket | Scheduler, proposal tensors, verifier lengths, KV/state rollback and captured graph all agree |
| P3 | Fused DFlash greedy edge scoring | Avoid constructing all K² predecessor/successor edges when only one predecessor is visited | Tie, rounding, greedy-path equivalence and real selector profile |

Split tuning must retain the existing default as control. The repository already records rejected forced 64/128 split experiments. Do not reinstate them as a universal solution. Tune tile dimensions, query grouping, warps and partial-buffer layout alongside split count. With multiple sequences, never share tiles or softmax state across request boundaries. Quantized K rotations, V transforms, scale granularity and partial-block masks must match the reference exactly.

Local vocabulary top-K plus a small padded gather is **already implemented** in the installed DFlash2 processor. MTP also has a local-argmax path. Recommending another full-vocabulary-gather replacement without checking dispatch would duplicate existing work. The fixed gather padding also deserves a boundary test for selector K above 32; the current K=16 does not exercise that case.

Dynamic depth is not a proposer-only patch. A changed number of speculative tokens can corrupt verification metadata or recurrent-state rollback if the scheduler and graph still assume the old shape. Start with fixed-depth launches: nonspec; MTP1/2/4; DFlash1/3/7. Select a qualified configuration at process startup before implementing adaptive dispatch. Never skip verifier checks to improve the apparent speed.

## GPU transport and memory policy

Keep XCCL as the measured starting point on XPU. CUDA/NCCL recommendations cannot simply be applied here. Intel's [oneCCL environment documentation](https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-14/environment-variables.html) describes topology-aware GPU paths and host-staging alternatives for that documented release. Record the actual loaded library version and effective `CCL_LOG_LEVEL=info` table before applying it to this image. `drmfd`, `pidfd` and `sockets` select IPC-handle exchange; they do not by themselves establish bulk-data bandwidth.

The current environment includes SYCL kernels enabled, P2P access requested, `drmfd` exchange, allreduce temporary buffer disabled, both GPUs visible, and expandable XPU allocator segments. Preserve this measured baseline while varying one setting at a time. Test CPU affinity separately from GPU visibility. Do not force P2P across an unsupported topology or weaken IOMMU/ACS settings based on a throughput guess.

The [oneCCL issue #212](https://github.com/uxlfoundation/oneCCL/issues/212), opened July 13, 2026, reports stale opened IPC handles after peer-buffer reallocation on dual B70. It is an external bug report, not a reproduced diagnosis of this server. Its narrow cache-disable workaround is a test candidate if allocation-churn traces reproduce the signature; disabling all caches globally may be much more expensive. Static-buffer reductions do not test this failure. Qualify request churn, cancellation and graph-size changes with device-error monitoring and bounded process recovery.

Avoid CPU KV offload for the strict decode-latency objective unless capacity makes it unavoidable. Lower KV storage can admit more clients but can also raise kernel cost. Reserve measured space for recurrent states, speculative lookahead, draft KV, graph pools, conversion snapshots and continuation scratch. Compare `gpu-memory-utilization` settings by actual capacity/preemption results; 0.9 is not inherently optimal. Never infer a leak solely from allocator-reserved memory: track allocated, reserved, cache ownership and post-drain plateaus separately.

## Qualification matrix and stop conditions

1. Inventory dtype values accepted by the installed CLI and backend dispatch. For each, record target support, draft support, kernel selected, graph support, scale source and startup outcome. Include auto/FP16/BF16 where supported, FP8 aliases including e4m3/e5m2 where implemented, and every installed TurboQuant variant. Rejection is a result; do not imply that a parsed dtype has a working kernel.
2. Establish nonspec, MTP1/2/4 and DFlash1/3/7 controls at 2k, 32k, 64k, 128k and near 262k. Separate compilation/warmup from measured runs. Preserve tokenizer, seed, checkpoint, image digest and full environment. Repeat at least five prompts per task/length; alternate candidate/control order. Exact greedy output is a diagnostic where the baseline is deterministic; sampled decoding requires distribution and quality checks.
3. Use one, two, four and eight closed-loop clients, then production arrival-rate replay. Include mixed 2k/128k clients, one near-limit request alongside short requests, cold prefixes, shared warm prefixes and distinct prefixes. Sweep sequence limits only after this, rather than treating 64 active long requests as feasible.
4. Distinguish prompt length from total context. Test 131072 prompt + 512 generation; near-limit 261632 prompt + 512 generation reaches 262144 total. Also test exact rejection above the limit, max-length truncation, page boundaries 511/512/513, and cancelled requests resumed under cache pressure. The current live near-limit probe uses 261888 + 128 = 262016, not the exact boundary.
5. Real content: repository bug localization with evidence at the beginning/middle/end; multi-document contradiction resolution; multi-hop retrieval; long multi-turn conversation; tool calls with parser validation; multilingual text and repetitive tables. Include natural EOS and separate forced-length capacity probes. Random/repeated records alone cannot certify reasoning quality or absence of looping.
6. Collect TTFT, per-request decode duration, emitted and committed tokens, inter-token latency p50/p95/p99, aggregate useful tokens/s, acceptance by draft position, preemptions, KV/state occupancy, peak allocations, collective durations/bytes and device errors. Streaming events may contain several accepted tokens; event spacing is not per-token latency. Subtract the first event's token count when calculating post-first-token decode speed where exact token accounting is available.
7. Loop checks: repeated n-grams and long repeated spans, task completion, EOS rate, parser failures and citations grounded in the input. Compare nonspec with spec using identical prompts/sampling. Examine FP16 versus FP32 recurrent state where supported. Repetition penalties can hide symptoms and alter behavior; they are not a fix for cache or state corruption.
8. Run graph capture/replay across changing batch shapes, prefix reuse, rejected speculative suffixes, cancellation, allocator churn and a multi-hour mixed-client soak. Fail on NaN/Inf, state contamination, device reset, deadlock, unexpected token mismatch or lost tool-call structure. Restore the original image on failure; stop throughput tuning until correctness is restored.

Promote only a configuration that improves the intended latency/throughput objective without degrading quality, stability or capacity beyond an explicit budget. A microbenchmark win or lower bytes stored is insufficient. The full matrix and near-flat objective remain open work; no universal fastest environment or solved long-context claim is supported by this audit.

## Evidence limits and reconciliation

The earlier v127 report's peak-bandwidth lower bounds are not attainable-throughput forecasts. Its Gen5 assumption does not describe the measured upstream links on this host. Its proposed large split counts must be reconciled with the same repository's rejection ledger. Continuation dequantization explains a prefill cost; it does not alone explain steady-state decode slowdown. This report uses fresh measurements to correct those interpretations.

Research covered the installed DFlash model, local TQ implementation and failed-split history, host link/collective behavior, official model configuration, vLLM context-parallel documentation, Intel transport documentation and a directly relevant upstream IPC issue. Broader searches stopped when these sources established the design constraints; kernel profiling, complete dtype qualification and end-to-end candidate validation remain the material evidence gaps. The Markdown artifact was structurally checked; no visual rendering was performed.
