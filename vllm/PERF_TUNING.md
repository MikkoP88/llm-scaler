# Performance Tuning: Maximum Token Generation Speed (Arc Pro B70 x2)

Measured 2026-08-24/25 on the benchmark host (2x Intel Arc Pro B70, TP=2,
`llm-scaler-vllm-adv:v6`/`v7`/`v8` images, vllm v0.21.1.dev0+gad7125a43,
torch 2.11.0+xpu). Model: qwen3.8-27b, fp8 weights, turboquant_4bit_nc KV,
`--max-model-len 262144`.

Benchmark protocol: streaming /v1/completions, `ignore_eos`, SSE chunk ==
1 token. "shallow" = 21-char prompt + 3072 gen tokens; "deep64" =
~117k-token prompt (607,747 chars) + 6144 gen tokens. `tps_steady` =
per-stream steady-state window at conc=1; at conc>1 the table reports
`tps_overall` (aggregate across streams). Fresh boot before every
campaign; engine resets => host reboot before next run.

## Locked max-speed configuration (v8 image, RECOMMENDED)

Requires an image carrying the graph-safe TurboQuant KV splits fix
(fork commit 1c41a08; host image `llm-scaler-vllm-adv:v8`). The user's
serve command gains ONE flag; no extra env vars beyond the base two:

```bash
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
VLLM_XPU_ENABLE_XPU_GRAPH=1 \
vllm serve --model /models/qwen3.8-27b \
  --tensor-parallel-size 2 --quantization fp8 \
  --kv-cache-dtype turboquant_4bit_nc --dtype float16 \
  --enable-chunked-prefill --trust-remote-code \
  --gpu-memory-utilization 0.8 --block-size 128 \
  --max-num-batched-tokens 8192 --max-model-len=262144 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

On v8 the graph-fixed KV split grid auto-selects from max_model_len
(log line `TurboQuant graph-fixed KV splits: grid=256 ...`); override
with `VLLM_TQ_GRAPH_KV_SPLITS=<n>` if needed. 256 measured optimal
(see grid sweep below).

Fallback for stock v6/v7 images (no code fix): keep env
`VLLM_XPU_ALLOW_COMM_IN_GRAPH=1 VLLM_USE_V2_MODEL_RUNNER=1` with no
compilation-config flag ("c2+c5", second-best config). Do NOT use
FULL_DECODE_ONLY on v6/v7: it disables adaptive TQ splits and costs
-25% deep throughput.

## Headline results (steady tok/s, conc=1)

| config | shallow | deep64 (~117k ctx) | TTFT |
|---|---|---|---|
| anchor (user config, v6) | 29.02 | 17.45 | 12.7 s |
| c2+c5 (stock-image best) | 32.75 (+12.9%) | 19.45 (+11.5%) | 0.23 s |
| **FULL + graph-safe splits (v8)** | **33.49 (+15.4%)** | **19.67 (+12.7%)** | **0.37 s** |

g256 repeatability: 33.49/33.48 shallow, 19.67/19.67 deep across
independent runs (fresh serve each) — run-to-run variance <0.1%.

Aggregate throughput scaling (tok/s, all streams):

| conc | anchor | c2+c5 | FULL v8 | FULL vs anchor |
|---|---|---|---|---|
| 2 (shallow) | 51.21 | 62.96 | 64.31 | +25.6% |
| 4 (shallow) | 111.71 | 117.71 | 119.83 | +7.3% |
| 8 (shallow) | 211.11 | 207.53 | 210.26 | parity (saturated) |
| 2 (deep64) | 22.61 | 22.20 | 23.01 | +1.8% |

The v8 config wins or matches every cell vs c2+c5 and beats the anchor
everywhere except saturated conc8 (parity). Latency-sensitive region
(1-2 streams) improves +15-26% with TTFT ~35x better.

## Sweep detail (conc=1, steady tok/s shallow / deep64)

Environment/flag sweep (v6/v7 images):

| config | change | shallow | deep64 | verdict |
|---|---|---|---|---|
| anchor | user's exact env+flags | 29.02 | 17.45 | baseline |
| c1 | VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 | 21.37 | 17.13 | reject (-26% shallow) |
| c2 | VLLM_XPU_ALLOW_COMM_IN_GRAPH=1 | 31.68 | 17.67 | keep (component) |
| c4 | VLLM_TQ_MAX_KV_SPLITS=512 | 29.54 | 17.46 | flat (engages >131k ctx only) |
| c5 | VLLM_USE_V2_MODEL_RUNNER=1 | 26.24 | 19.19 | keep (component) |
| c6 | CCL_ENABLE_SYCL_KERNELS=1 | ENGINE DIED | - | fatal: do not use |
| c7 | CCL_ATL_TRANSPORT=ze | init fail | - | incompatible: do not use |
| c8* | cudagraph_mode FULL_DECODE_ONLY (pre-fix code) | 32.41 | 13.10 | -25% deep: adaptive TQ splits disabled |
| c9 | --enable-prefix-caching | 25.15 | 17.45 | reject (-13% shallow, hashing overhead) |
| c10 | --max-num-batched-tokens 16384 | 26.84 | 17.50 | reject (no gain) |
| d1 | c2+c5 | 32.75 | 19.45 | best stock-image config |
| d2 | c2+c5+TQ BLOCK_KV=8 | 32.63 | 17.46 | reject (knob hurts deep) |

TQ stage-1 knob sweep on v7 image (BLOCK_KV x warps vs defaults 4/1):
16/1 -> -72% deep, 32/1 -> -83% deep, 16/2 -> -20% deep, 32/2 ->
EngineDeadError (fatal). Shipped defaults are measured-optimal on Xe2;
documented in the patch comment (commit bc1cb0b).

Full-capture grid sweep on v8 (graph-safe splits, FULL_DECODE_ONLY):

| KV split grid | shallow | deep64 | verdict |
|---|---|---|---|
| 64 | 34.06 | 18.29 | best shallow, -7% deep |
| 128 | 33.78 | 17.88 | dominated |
| **256 (auto default)** | **33.49** | **19.67** | WINNER (both axes balanced) |
| 512 (needs VLLM_TQ_MAX_KV_SPLITS=512) | 32.76 | 19.12 | dominated |

c5 (V2 runner) does not compose with FULL capture (f4: 18.06 deep vs
17.88 same-grid without it) — its eager-dispatch savings are subsumed.

## Why it works (decode profile, py-spy on healthy deep decode)

Pre-fix profile: ~51% XPU graph replay, ~16% oneCCL allreduce
dispatch+exec (~80 eager allreduces per decode step between graph
pieces), ~4.5% TQ decode launcher. Two attack routes:

1. **c2+c5 (env-only)**: `ALLOW_COMM_IN_GRAPH=1` moves collectives into
   the captured pieces; the V2 runner restructures the step loop (+10%
   deep). Leaves piecewise-graph overhead in place.
2. **v8 code fix (better)**: `FULL_DECODE_ONLY` captures the entire
   decode step, eliminating ALL eager dispatch (TQ launcher +
   allreduce) — but previously disabled TurboQuant's adaptive KV
   split-KV, collapsing deep throughput -25%.

The fix (1c41a08): both TQ Triton stages already derive split ranges
DEVICE-SIDE from `Seq_lens_ptr` (stage1 early-returns on empty splits
without writing; stage2 skips inactive splits' mid_o reduction). Only
the LAUNCH GRID must be constant during capture. So under full capture
the backend now launches a fixed oversized grid sized from
max_model_len tiers (auto: 256 for 262k) and lets the kernels
re-partition live context at every replay. Full-graph capture AND
split-KV parallelism at 117k context, simultaneously.

## DSpark/dflash speculative decode (qwen3.8-27b-fp8, 64k ctx, adv images)

Separate workload from the TQ champion above: bf16 + `--block-size 64`
+ `--max-model-len 64000` + `--async-scheduling` with the DSpark drafter
(`/models/drafter-fp8-v5`), full config in
`patches/qwen38-dflash/serve.sh` and README. Wall clock, single stream,
greedy, "Write a html car game." prompt:

| serving stack | 512 tok | 1536 tok | tok/s @512 |
|---|---|---|---|
| v9 XPU graphs, no spec | 15 s | 47 s | 34 |
| **v9 XPU graphs + dflash k=4** | **8 s** | **23 s** | **64** |
| v9 XPU graphs + dflash k=6 | 7 s | 23 s | 73 |
| rmacy v14 eager + dspark spec | 15 s | 45 s | 34 |
| rmacy v14 eager, no spec | 40 s | 114 s | 13 |

Graphs+spec is 2.0x the rmacy always-spec recipe on identical hardware
and vllm build. k-sweep: k=4 and k=6 tie (byte-identical greedy, 7-8 s
@512); k=5 is dominated (10 s/32 s, acceptance does not grow past 4
draft positions). Mean accepted length k=4: 2.77-4.17. Greedy spec
output is byte-identical to no-spec on v9.

CAUTION on adv images v4-v9: serving with compile mode +
`VLLM_XPU_ENABLE_XPU_GRAPH=0` silently corrupts TP output in BOTH
dtypes (KNOWN_ISSUES #04, introduced by 07827c0, fixed by 28ff055 in
adv:v10). Keep graphs on, or use v10+. adv:v10 and adv:v12 batteries
re-validated everything: the previously-garbage compile+graphs-off arm
is coherent (41-42 s @512 tok, ~12 tok/s — same territory as rmacy
eager; graphs remain the recommended mode at 2.7x), graphs+spec k=4 is
8 s/23-24 s with greedy output byte-identical across v9/v10/v12
(acceptance 2.8-3.6), and the TQ fp16 champion is unregressed (15 s
@512, grid=256). adv:v11 (108cfdd) was a brief experiment making the
AR custom op out-of-place via input clone — it corrupted PIECEWISE
graphs-ON serving (the alias return is load-bearing there; see
KNOWN_ISSUES #04) and is reverted; do not use the v11 image.

### adv:v14 (b19d92f): kv-cache-dtype matrix + depth-vs-rate

`--kv-cache-dtype` with dflash k=4, idle host (full matrix and failure
modes in `patches/qwen38-dflash/README.md`; KNOWN_ISSUES #05):

| dtype | 512 tok | 1536 tok | notes |
|---|---|---|---|
| none (bf16) | 8 s | 23 s | acceptance 0.69 — matches v12/v9 champion |
| fp8 | — | — | validated coherent; ~2× KV capacity |
| turboquant_3bit_nc | 40 s cold / 9 s warm | 28 s | acceptance 0.51 |
| turboquant_k8v4 eager | 105 s cold / 70 s | 213 s | graphs hang; eager 10k coherent @7.3 tok/s, acceptance ~1-2% |
| turboquant_k3v4_nc eager | — | — | graphs hang; eager 10k coherent @6.9 tok/s |
| turboquant_4bit_nc | n/a | n/a | hangs in capture; eager dies at ~4.3k — unusable |
| float16 | n/a | n/a | rejected by reshape_and_cache_flash |

Deep-generation rate (single stream, `ignore_eos`, graphs on):

| depth band | bf16 none | TQ 3bit_nc |
|---|---|---|
| 1-5k | 68 tok/s (max 72) | 55 (max 73.5) |
| 5-15k | 55-66 | 42-49 |
| 15-25k | 44-51 | 35-39 |
| 25-32k | 34-42 | died at 25.3k |

No graphs-mode arm — with or without spec — completed a 40k single
stream in the v14 battery (dflash arms died with xe DEVICE_LOST at
20-32k; the no-spec control wedged at 2.9k; KNOWN_ISSUES #05(a)). Only
eager k8v4/k3v4_nc completed their 10k gens.

REVISED 2026-08-27 (v15/v15b batteries, #05(a) RESOLVED): the v14
result was host-state accumulation (six arms chained on one boot across
engine resets, no reboots) + a random ~1/3 boot-time warmup wedge + the
`ignore_eos` early-finish leak (#05d) — NOT a graphs-mode depth limit.
On a freshly rebooted host, base recipe @ `--max-model-len 80000`,
PIECEWISE graphs, dflash k=4: a forced 40k gen completed in 700 s
(**57 tok/s average**), and a forced full-window 76k gen in 1880 s
(**40 tok/s average**), both coherent head+tail, spec acceptance 4.7-5.0
held at depth, ZERO engine resets across 116k generated tokens on one
boot, and a post-deep 512 gen still at 8 s. Deep-stream protocol:
reboot after any xe reset; retry a wedged boot (post-capture hang on
~1/3 of serves); pin `min_tokens == max_tokens` in the request (plain
`ignore_eos` returns HTTP 200 early at ~19-24k tokens on greedy
whitespace collapse). The v13 "31.5 tok/s Avg generation" report was
depth decay plus concurrent build load — not a regression. 512/1536
gens remain rock-solid everywhere.

### Sampling temperature vs dflash throughput (v16/v16b, 2026-08-27)

The dflash drafter is greedy; the target's sampling temperature
controls draft/target agreement, and deep throughput scales with it.
`/v1/chat/completions`, "Write a html car game.", forced 40k gens
(`min_tokens == max_tokens`, top_k 20 top_p 0.95 pinned), base recipe,
one boot per arm, zero resets across the whole battery (~265k tokens):

| temperature | 40k WALL | tok/s | late-window acceptance (per-position) |
|---|---|---|---|
| 0.0 greedy | 999 s | 40.0 | 4.88-4.93 (0.94-1.0 across all 4) |
| 0.3 | 1154 s | 34.7 | 4.5-4.8 (0.85-0.99) |
| 0.6 | 1114 s | 35.9 | 2.6-3.4, content-drifty |
| 0.8 | 1400 s | 28.6 | 1.9-4.5, content-drifty |
| 1.0 | 1613 s | 24.8 | 1.6-1.9 (0.37/0.15/0.07/0.01) |

Sampled acceptance is content-dependent (drifts up when the text enters
repetitive structure); WALL clock is the stable metric. Chat-endpoint
deep rates run below the v15b completions numbers (40 vs 57 tok/s greedy
@40k) — endpoint/template + cold-first-gen overhead; the curve shape is
what matters.

Default-sampling trap: with `--generation-config auto` (the default),
chat requests that omit `temperature` silently inherit the model's
generation_config.json (temp 1.0/top_k 20/top_p 0.95) — bottom of the
curve AND the PyTorch-native topk_topp sampler fallback (eager kernels
every step). Fix server-side with `--generation-config default`, or
per-request `"temperature": 0.0`. If diversity is required, temp <= 0.6
keeps deep throughput >= ~35 tok/s.

## v17/v18 battery matrix (2026-08-27, gates = bench_gen 512/1536 tok prompts)

All arms: TP=2, gmu 0.8, maxlen 64000, graphs ON (v17 ENV; spec arms on
v18 for the #03 boot-fault fix), one boot per arm, dflash k=4 where
spec. Single-stream steady tok/s; cold512 = first-request TTFT after
boot. Acceptance = draft-accepted / drafted over the gate suite.

| arm | KV | drafter | warm512 tok/s | 1536 tok/s | cold512 TTFT | notes |
|---|---|---|---|---|---|---|
| v17a2 | bf16 | no | 32.8 | 32.9 | 15.6 s | fastest target-only |
| v17a3 | fp8 | yes | 19.2 | 19.4 | 12.4 s | lowest latency; ~2.97 tok/step (2036/4136) |
| v17a4 | fp8 | no | 27.3-27.6 | 27.8 | 11.9 s | longctx 60k @ 17.29 tok/s (beats spec at depth) |
| v17a5 | k8v4 | auto-disabled | 26.5-28.8 | 25.7 | 12.1 s | graphs-ON boot FIRST EVER (#05b fix) |
| v17a6 | k8v4 | no | 30.5 | 30.7 | 12.5 s | fastest TQ arm; deep5k 5000/5000 @ 26.65 tok/s |
| v17a7 | tq4nc | no | 24.6 | 29.4 | 13.5 s | historically-unusable dtype now serves; 60k longctx PASSED @ 15.25 tok/s (100.7 s) |
| v17a8 | bf16 @ gmu 0.9 | yes | — | — | — | BOOT FAULT (pre-v18, graphs+spec; see KNOWN_ISSUES #03) |
| v17a9 (v18) | bf16 | yes | 22.0 | 21.6 | 12.9 s | FASTEST spec arm (beats a3 fp8+spec 19.2); ~2.78 tok/step (2118/4744); boot 6 min first-try — the exact config that faulted 4/4 on v17 |
| rmacy v16 own recipe | bf16 | yes | 11.9 | 12.0 | 12.4 s | conc4_512 ttft 30.5 s (maxseqs=1 serialized); acceptance 47% (3338/7104) |

adv vs baselines (matching cells): adv a3 (fp8+spec) beats rmacy v16's
own recipe 1.6× on tok/s and ~2-4× on wall for every gate; conc4 on
rmacy serializes behind maxseqs=1 (30.5 s TTFT) while adv serves
concurrency natively. b3.1 (intel/llm-scaler-vllm:0.21.0-b3.1) cannot
load the current drafter at all — its spec registry predates
`DSparkDraftModel` (has `DFlashDraftModel` only; the DFlash-era
drafter dir on the host is empty), so no b3.1 spec-parity cell exists.
b3.1 target-only (SPEC=0, gmu 0.8, both `--generation-config auto`
and `vllm`) boots healthy — KV pool 310,000 tokens, clean capture —
but the worker crashes on the FIRST /v1/completions request
(native exception in `worker_busy_loop`→`execute_model` → HTTP 500
→ engine death; 0.06 tok/s generated before death; Triton JIT of KV
kernels still running mid-inference), reproducibly across both
runs. Zero xe engine resets — a pure b3.1 software crash. NO b3.1
cell completes a single gate on this workload, spec or target-only;
every incompatibility is an adv advantage.

Long-context 60k+1536 (KV-depth stress): bf16+spec 235 s wall (a3-era;
spec loses its edge at depth), fp8 nospec 17.29 tok/s, tq4nc nospec
15.25 tok/s PASSED, k8v4 FAULTED at 60k (xe reset at end-of-prefill;
keep k8v4 out of ≈60k contexts). fp8/bf16 are the long-context KV
choices.

a9 full suite on adv:v18 (default recipe, one boot, ZERO resets
end-to-end): gates 21.6-22.1 tok/s; 60k longctx PASSED at 138.3 s wall
(ttft 28.3 s) — 1.7× faster than the a3-era graphs-off spec run (235 s);
deep10k temp 0.6 = 10000/10000 @ 69.2 tok/s and temp 1.0 @ 67.1 tok/s
(finish=length, status OK); conc2_512 21.5 tok/s, conc4_512 66.5 tok/s
aggregate @ 0.24 s ttft. Graceful teardown after the suite: zero dmesg
engine events.

## v19 car-game matrix (2026-08-28, canonical user test, adv:v19)

Prompt `"Write a html car game."`, sampling temp 0.3 / top_k 20 / top_p 0.95 /
min_p 0 / presence 0 / repetition 1.0, max_tokens 4096, streaming, one warmup
completion first. All cells: dtype float16, block-size 128, mnbt 8192,
cudagraph FULL_DECODE_ONLY, gmu 0.8. Cells: bar = the user's superior config
(tq4nc nospec, maxlen 262144); n* = same-KV nospec baselines (maxlen 262144);
c1-c4 = same-KV dflash k=4 spec (maxlen 73728 for c1-bf16, 98304 otherwise —
the drafter's UNCOMPRESSED KV pool ~11.3 GiB/GPU @262144 does not fit next to
its fp8 weights; vLLM-measured: 13.48 GiB needed vs 6.67 available; decode
speed at ~4k depth is KV-pool-size independent, so the comparison holds).
Acceptance per KV: spec cell >= matching nospec steady tok/s; headline c3 >=
bar. v19 changes under test: `VLLM_ALLOW_TQ_SPEC` default 1 (drafter serves
WITH turboquant KV — was silently dropped ≤v18) + multi-query TQ verify
kernel (one KV pass per verify step instead of 5 synthetic decodes; rollback
`VLLM_TQ_MQ_VERIFY=0`).

<!-- CARGAME_MATRIX_RESULTS_PERF -->

Results (2026-08-28; steady tok/s; P0..P3 = per-position acceptance; c3/c4
re-run on the v19b image after the #06 blind-verify-graph fix — the original
v19 c3/c4 numbers were measured on context-blind graphs and are bracketed):

| cell | KV | spec | maxlen | steady | P0..P3 |
|---|---|---|---|---|---|
| bar | tq4nc | no | 262144 | **32.79** | — |
| nbf16 | bf16 | no | 262144 | 33.10 | — |
| nfp8 | fp8_e4m3 | no | 262144 | 33.34 | — |
| c1 | bf16 | k4 | 73728 | 19.94 | .45-.55 / .20-.27 / .05-.17 / .03-.07 |
| c2 | fp8_e4m3 | k4 | 98304 | 19.69* | .47-.55 / .19-.27 / .05-.14 / .03-.06 |
| c3 | tq4nc | k4 | 98304 | 17.56 [was 21.87 blind] | .52-.76 / .24-.53 / .08-.33 / .02-.23 |
| c4 | k8v4 | k4 | 98304 | 16.10 | .41-.52 / mean len 1.67-1.98 |
| nk8v4 | k8v4 | no | 262144 | 32.61 | — |
| dspark16 | bf16 (auto) | k4 | 73728 | 11.94 | mean accept len 1.66-2.53 |

(*c2 = 4 xe engine resets mid-cell, #03 family. dspark16 = upstream baseline
image `ghcr.io/rmacy/qwen38-fp8-dspark:v16`, v16-era defaults; the identical
config on v19 (c1) is 1.67x faster — the v18/v19 graphs+guards and verify
work pay for themselves, and c1 is above nospec on true token rates.)

**v20 correction (2026-08-28):** the spec "steady" column above holds client
SSE-event rates, not token rates — the v19 bench client counted delta
*events* as tokens, and spec decode flushes ~E[len] tokens per event, so
every spec cell underreported ~1.9-2.1x. Nospec cells emit 1 token/event and
were correct. TRUE steady rates from the engine SpecDecoding windows (tail-6):
c1 **37.8** (+14% vs nbf16 33.10), c2 **36.2** (+8.6% vs nfp8 33.34), c3
**35.6** (+8.6% vs bar 32.79), c4 **32.6** (parity with nk8v4 32.61); k-curve
on tq4nc: k2 **39.8** / k4 35.6 / k6 34.9. The fixed true-token client
(v20) reproduces these. All spec cells already beat their nospec twins.

**v20 shipped-image matrix** (llm-scaler-vllm-adv:v20, 2026-08-28; TRUE
steady = engine SpecDecoding windows tail-6; client = fixed-client
steady_true; one boot per cell; all 12 gates PASS, zero resets):

| cell | KV | spec (k) | steady true | client | E[len] | ms/step | vs twin |
|---|---|---|---|---|---|---|---|
| bar | tq4nc | no | 32.78 [v19 32.79] | 32.78 | — | — | — |
| nbf16 | bf16 | no | 33.08 [33.10] | 33.08 | — | — | — |
| nfp8 | fp8_e4m3 | no | 33.36 [33.34] | 33.36 | — | — | — |
| nk8v4 | k8v4 | no | 32.60 [32.61] | 32.60 | — | — | — |
| c1 | bf16 | k4 | **44.0** | 40.66 | 2.19 | 49.6 | +32.9% vs nbf16 |
| c2 | fp8_e4m3 | k4 | **35.5** | 33.34 | 2.00 | 56.1 | +6.4% vs nfp8 |
| c3 | tq4nc | k4 | **34.6** | 33.39 | 1.99 | 57.4 | +5.5% vs bar |
| c4 | k8v4 | k4 | **35.0** | 31.63 | 2.05 | 58.9 | +7.4% vs nk8v4 |
| k2c1 | bf16 | k2 | **37.4** | 36.15 | 1.81 | 48.4 | +13.1% |
| k2c2 | fp8_e4m3 | k2 | **34.3** | 32.24 | 1.84 | 53.8 | +2.8% |
| k2c3 | tq4nc | k2 | **42.5** | 39.97 | 2.01 | 47.1 | **+29.7% vs bar** |
| k2c4 | k8v4 | k2 | **36.2** | 35.80 | 1.75 | 48.2 | +11.0% |

Headline: **k2c3 (SPEC_K=2, tq4nc) 42.5 tok/s = +29.7% over the user's bar
config**; c1 (bf16 k4) 44.0 (+32.9%). k4 remains default, `SPEC_K=2` for
max tok/s on tq4nc/k8v4. Cell notes (intermittent #05(a) re-runs, k2c2
stall windows, DHCP host moves) in the v20 patch README.

**v21 (2026-08-29): spec decode now honors `--kv-cache-dtype` in VRAM —
draft-pool parity.** v20's k2c3 config could NOT boot at the user's
`--max-model-len 262144` ("13.09 GiB KV needed vs 6.67 available"): v20's
own guard rewrote the DRAFT engine's turboquant_* dtype to "auto", forcing
an uncompressed bf16 drafter pool on top of the compressed target pool —
nospec booted, spec could not. v21 (patches/qwen38-dflash-v21) keeps the
turboquant dtype on the draft pool behind a two-regime policy (tq4nc
target: bf16 draft `<=131072`, `turboquant_k8v4` above — derived from the
measured 6.67 GiB budget; other dtypes inherit v20 exactly), plus the
non-causal draft-attention support TQ needed (DFlash draft rows attend the
full stored context; new NON_CAUSAL mode of the MQ kernel) and two
mixed-dtype pool-sizing fixes. Final image, all gates PASS, zero resets:

| cell | config | v20 gate | v21 | E[len] |
|---|---|---|---|---|
| bar | tq4nc nospec 262144 | 32.78 | **32.79** | — |
| k2c3 | tq4nc k2 98304 | 39.97 | **39.78** (−0.5%) | 2.18 |
| c3 | tq4nc k4 98304 | 33.39 | **34.08** (+2.1%) | 1.87 |
| c2 | fp8 k4 98304 | 33.34 | **34.34** (+3.0%) | 2.12 |
| t21buser | tq4nc k2 262144 (user cfg) | cannot boot | **32.93** | 1.66 |
| c2mix | fp8 + k8v4 draft override | crash | **31.42** | 2.30 |

The user's 262144 spec config now boots and edges out its nospec twin
(32.93 vs 32.79); at `<=131072` the policy keeps the v20 (bf16-draft)
behavior and v20-level throughput. Full root-cause detail and the v21d A/B
(compressed draft pool was the sole acceptance/throughput regression;
drafter-fp8-v5 swap is neutral) in the v21 patch README.

**v22 (2026-08-29): MTP with graphs now boots (eager MTP head) — 72-74
tok/s, ≈2.2x the user's live dflash k2 config.** MTP + graphs (any k, any
KV dtype, v21 and intel images) died at boot in the `eagle_head`
torch.compile warmup: the MTP head's sampling path issues oneCCL
full-vocab allgathers from inside dynamo-evaluated code →
`allgatherv_large_su_ring<half>` segfault on both ranks (KNOWN_ISSUES
#09; same #05-family eager-collectives-x-compiled-regions hazard, hitting
the drafter this time). v22 (patches/qwen38-dflash-v22) keeps the TARGET
on FULL_DECODE_ONLY decode graphs and runs only the single-layer MTP head
eager — `ignore_torch_compile()` on the three MTP head classes +
drafter `CUDAGraphMode.NONE` for `method=="mtp"` — env-gated
`VLLM_XPU_MTP_EAGER_HEAD` (default 1 on XPU; `0` = stock crash, rollback).
dflash/eagle drafters are untouched (method-gated).

Measured (completions endpoint — see environmental note below; one boot
per cell, zero resets everywhere; canonical sampling):

| cell | image | spec | endpoint | steady tok/s | tok/event | E[len] |
|---|---|---|---|---|---|---|
| mtpB2 | v22 | mtp k4 graphs | completions | **72.23** | 4.15 | — |
| mtpB3 | v22 | mtp k4 graphs | completions | **74.31** | 4.19 | — |
| engine windows | v22 | mtp k4 graphs | SpecDecoding | 66-88 (avg ~77) | — | 4.17-4.44 |
| t2 dflash k2 (user cfg) | v22 | dflash k2 graphs | chat/engine | 32.5-34.2 engine | 1.77-2.61 | 1.77-2.61 |
| t2 dflash k2 (x-check) | v22 | dflash k2 graphs | completions | 44.54-45.41 | — | — |
| v21-repro same-day | v21 | dflash k2 graphs | completions | 45.30 | — | — |
| t21buser (yesterday) | v21 | dflash k2 graphs | chat | 32.93 | 1.77 | 1.66 |
| bar | v21 | none | chat | 32.79 | 1.00 | — |

t2 no-regression gate vs 32.93 ±3%: **PASS** (engine 32.5-34.2;
completions within noise of the same-day v21-image replica 45.30) — the
`llm_base_proposer.py` edit provably left dflash untouched. Headline:
**MTP k4 with graphs ≈ 2.2x the user's live dflash k2 and ≈ 2.2x nospec**;
the eager head costs little (the win is the graphed target verify at
E[len] ~4.2).

Environmental note (2026-08-29): on the chat endpoint +
`--reasoning-parser deepseek_r1`, the model's `<think>` phase now exceeds
4096 tokens on ALL configs — v22 MTP, v21 eager MTP, v22 dflash, and a
same-day v21-image replica of the user's exact config (0 content events,
engine 30.4-33.0 tok/s ≈ v22's t2) — so chat-endpoint content-rate
benches yield 0 events and are not comparable across days. This is
pre-existing fork MTP behavior (v21 eager A/B identical long-think), not
a v22 regression; the completions endpoint (no chat template) finishes
naturally in ~3200-3500 tokens. Use completions-endpoint or engine-window
numbers for MTP-vs-dflash compares. Full detail in the v22 patch README
and KNOWN_ISSUES #09.

Conclusions:

- **The v19-era "TQ sampled acceptance = exactly 0" was NOT a sampling bug:
  the verify XPU graphs were context-blind (KNOWN_ISSUES #06).** v19b
  (build_for_cudagraph_capture forces the KV-reading continuation path into
  multi-token captures; per-row causal limits derive from the dynamic
  seq_lens buffer) restores correct text + healthy acceptance on both TQ
  dtypes: c3 8%→25.5% greedy (~2.0 tok/step gross), c4 healthy at 16.10
  (first c4 completion ever). Greedy's earlier "2.25 tok/step healthy" was
  row-0 markov coincidence — the text was garbage all along.
- **Spec DOES beat nospec at temp 0.3 in this shape (v20 correction).** The
  v19 "spec loses" reading was the client event-counting artifact (see the
  correction note above): true c3 is 35.6 vs bar 32.79 (+8.6%), and k=2
  reaches 39.8 (+21%). The measured step wall is 56-63 ms, not 95 ms — that
  figure was derived from the bad client number. The blind-graph 21.87
  remains invalid for a different reason (context-blind verify, #06). See
  the corrected step-cost decomposition below.
- **fp8 nospec needed its own fix (v19c, KNOWN_ISSUES #07):** the ESIMD
  decode fast path D2H-synced `float(layer._k_scale)` inside XPU graph
  capture — fp8 x nospec crashed at boot 4/4 while every other cell booted
  (bf16 takes the no-sync else-branch; TQ never enters flash_attn; fp8+spec
  never captures single-token decodes). Cached static scales → nfp8 boots
  first try at 33.34 (fastest nospec cell); nbf16 unchanged (33.11 vs
  33.10) on the same image.
- v19/v19b serving is otherwise sound: TQ+spec boots (silently disabled
  ≤v18), zero resets in the fixed cells, no depth-scaling collapse (the
  v18 5x-rescan pathology is gone — c3's decay tracks the MQ kernel's
  linear context scan, same as nospec decode), MQ kernel 32/32 exact.

<!-- SPEC_STEP_COST_ANALYSIS -->
### Spec step-cost decomposition — corrected in v20 (2026-08-28)

**Correction first:** v19's "95 ms/step, 3.15x overhead, spec loses to
nospec" was a measurement artifact. The v19 bench client counted SSE delta
*events* as tokens; spec decode flushes ~E[len] tokens per detokenizer event,
so every spec cell underreported ~1.9x (nospec = 1 token/event, correct).
Three independent sources agree — engine SpecDecoding counters
(`emitted = mean_accept_length x drafted/k`), `VLLM_SPEC_TIMING` step-wall
instrumentation, and a re-run with a true-token client — **all v19 spec cells
already beat their nospec twins.** The 95 ms figure was derived from the bad
client number; the true step wall is 56-63 ms.

Measured step split (v20 A2 instrumentation, `VLLM_SPEC_TIMING=1`, c3 shape =
tq4nc k4, steady state; ms/step):

| segment | host | device |
|---|---|---|
| step wall (propose-to-propose) | 60.8-63.5 | — |
| verify forward (uniform-decode replay, q=5) | 1.7 | **39.6-41.5** |
| target logits over verify rows | 0.7 | 2.3 |
| drafter propose TOTAL | 8.7 | 7.2 |
| — precompute (context-KV, <=5 tokens) | 0.7 | 0.1 |
| — drafter forward (5 layers, PIECEWISE) | 6.1 | 3.2 |
| — greedy (block head + markov loop) | 1.1 | 3.4 |
| (unattributed glue/scheduler) | ~12 | — |

Implications:

- The dominant cost is the verify replay's ~40 ms DEVICE time plus ~12 ms
  scheduler glue — NOT drafter host time. Full-graphing the propose segment
  (the previous #1 lever) targets only ~9 ms host time; with parity gates
  already met on true rates, its capture risk is no longer justified
  (retained as measured follow-up material in the v20 patch README).
- **k=2 is the BEST arm — the v19 "k=2 is the WRONG direction" claim is
  retracted.** The 3-row verify step drops to ~47.8 ms while acceptance stays
  ~1.9 tokens/step (block drafting quality is k-independent at these depths).
  True rates on tq4nc: k2 39.8 / k4 35.6 / k6 34.9 tok/s vs nospec bar 32.79.

True-rate matrix re-read from v19 serve logs (engine SpecDecoding windows,
tail-6 — the authoritative steady source):

| cell | KV | spec | TRUE steady tok/s | nospec twin | verdict |
|---|---|---|---|---|---|
| c1 | bf16 | k4 | **37.8** | 33.10 | +14% |
| c2 | fp8_e4m3 | k4 | **36.2** | 33.34 | +8.6% |
| c3 | tq4nc | k4 | **35.6** | 32.79 (bar) | +8.6% |
| c4 | k8v4 | k4 | **32.6** | 32.61 | parity |
| k2c3 | tq4nc | k2 | **39.8** | 32.79 (bar) | +21% |
| k6c3 | tq4nc | k6 | 34.9 | 32.79 (bar) | +6.5% |

Remaining reduction paths, in measured order of size (only if more headroom
is ever needed):

1. Verify replay device time (~40 ms): MQ/flash kernel width at q=5, or a
   tuned q=3 path — the k2 arm already realizes much of this (47.8 ms step)
   with zero kernel work.
2. The ~12 ms unattributed glue (scheduler/rejection bookkeeping): needs a
   host-side profile (py-spy) to split further; nothing KV-dtype-specific.
3. Drafter propose host time (~9 ms): full propose-graph capture. Hook
   documented in v20 (`VLLM_DFLASH_FULL_GRAPH`); NOT implemented — gates
   already met.

Cross-dtype safety: all remaining levers live outside the KV-dtype-specific
target attention; any captured op must preserve the seq_lens/block_table
replay-refresh contract (v19 README known limits).
<!-- /SPEC_STEP_COST_ANALYSIS -->

## LONG_CONTEXT_ANALYSIS — what hurts really-long-context tok/s and why every
other stream is dragged along (2026-08-30, adv:v22, user's exact serve flags)

Config under test (user's boot, unchanged): TP=2 (2x B70 32GB), fp8 weights
(14.73 GiB/GPU), GMU 0.8, `--dtype float16`, `--kv-cache-dtype
turboquant_4bit_nc`, `--block-size 128`, MNBT 8192, `--max-num-seqs 64`,
`--max-model-len 262144`, `--enable-prefix-caching`, `--async-scheduling`,
FULL_DECODE_ONLY graphs, MTP k=4. Model = qwen3.5-27B hybrid: 64 layers =
**48 linear-attention (GDN/mamba) + 16 full-attention** (interval 4), full-attn
GQA 24q/4kv heads x head_dim 256, mamba state 48 layers x [48,128,128] fp32,
MoE gates, vocab 248k.

### Engine-level facts (from the boot log + no-spec A/B)

| quantity | MTP k=4 arm | no-spec arm |
|---|---|---|
| Available KV memory | 9.25 GiB/GPU | 9.69 GiB/GPU |
| KV pool (tokens) | **869,550** | **1,130,964** |
| Max 262k-token requests concurrent | **3.32x** | 4.31x |
| mamba cache mode | align (forced; experimental) | align (same) |
| decode graph capture sizes | 16, max bs 128 | 11, max bs 128 |
| TQ decode-attn KV split grid | graph-fixed 256 | graph-fixed 256 |

MTP's draft KV pool costs **23% of request KV capacity** (−261k tokens) on
top of its compute effects.

### Measured: decode tok/s vs context (steady, conc=1)

| context | MTP k=4 (user arm) | MTP k=1 (2026-08-30) | no-spec |
|---|---|---|---|
| ~2k | 15.0 tok/s (TTFT 1.1 s) | 45.4 tok/s (canonical cargame) | 32.1 tok/s |
| ~32k | **WEDGED 2/2** (never returns) | **27.9-30.3 tok/s, 7/7 clean** (TTFT 14.9 s) | 27.0 tok/s |
| ~65k | — | 22.8-23.6 tok/s, ~50% WEDGE | 23.6 tok/s (filler #13 note) |
| ~131k | **WEDGED 2/2** | **~50% WEDGE** (survivor 15.1 tok/s, TTFT 158 s concurrent) | 17.9 tok/s (TTFT 66.5 s = 1956 tok/s prefill) |
| ~262k | — | not tested (wedge-gated) | **12.3 tok/s** (TTFT 170.5 s = 1526 tok/s prefill) |

k=1 detail (v25 image, block512 + mamba-fp16, canonical sampling):
acceptance E[len] 1.67-1.95, position-0 rate 0.80-0.95; cargame output
correct (`<think>` + clean HTML). k=1 matches no-spec speed at 32k and
beats it by +37% at short ctx, but retains a ~40-50% wedge rate at >=64k
(KNOWN_ISSUES #11 final update) — hence the ship posture below.

### v26 no-spec full-length battery (2026-08-30 late, adv:v26 = GDN spec kernels fixed)

The shipped v26 image (kernel OOB fix, see `patches/qwen38-dflash-v26/`)
running the ship-posture arm — no spec + FULL_DECODE_ONLY graphs, tq4nc,
block 512, mamba ssm fp16 — one boot, sequential battery, 64-token
decodes. Probe density recalibrated via `/tokenize` (true 16.0 tok/rep;
the historical 15.7443 constant reads ~2% low), so actual prompt lengths
are ~33k / ~67k / ~133k / ~262k:

| actual ctx | wedges | real decode tok/s | vs v22-era no-spec |
|---|---|---|---|
| ~33k | 0/3 | 27.75 / 27.93 | 27.0 |
| ~67k | 0/3 | 23.60 / 23.64 | 23.6 |
| ~133k | 0/2 | 18.45 / 18.49 | 17.9 (at 131k, block 128) |
| ~262k | 0/1 | 12.71 (TTFT 184.7 s) | 12.3 |

Canonical car-game correctness on the same boot: PASS (coherent HTML +
canvas, 4096 tokens / 100.3 s = 40.8 tok/s at short ctx). Zero xe
resets, zero lost requests. Confirms (a) the kernel fix changes nothing
about long-ctx speed — the decay is the bandwidth-bound full-attn KV
scan, and (b) the no-spec arm is stable across the entire 262k envelope
on the fixed image. Wedge arm matrix + the spec-x-graphs-x->=32k
verdict: KNOWN_ISSUES #11 v26 update.

### v26 late-evening arms (2026-08-30): deep-ctx tuning closed out

Three more single-boot arms on adv:v26, all REJECTED — the locked
config stands:

| arm | change | result @133k | result @262k | verdict |
|---|---|---|---|---|
| P1 | `VLLM_TQ_GRAPH_KV_SPLITS=1024` (+MAX=1024; the auto ladder wants 2048 at maxlen 262k but is capped at 256) | **8.88 tok/s** (2x worse than grid 256's 18.45) | 12.09 vs 12.71 | reject — split-combine cost grows with grid; 256 optimal across 64-1024 at every depth |
| P2 | `--max-num-batched-tokens 16384` (nospec) | TTFT 94.5 s vs 69-73 s | TTFT 198.7 s vs 184.7 s | reject — bigger chunks slow prefill (matches v6-era c10 "no gain") |
| W3 | `VLLM_XPU_ALLOW_COMM_IN_GRAPH=0` (spec k4 + FULL_DECODE_ONLY) | **WEDGE 1/1** (clean 0/5 at 32-67k) | — | reject as config (mechanism evidence: see #11); decode 9-17 tok/s @32k; short-ctx 78 tok/s (fastest single-stream number yet) |
| eager-deep | `--enforce-eager` spec k4 | 0/1 wedge, 2.37 tok/s | 0/1 wedge, 3.82 tok/s | correctness workaround only (clean across the whole envelope) |

Take-aways: (1) deep-context decode is NOT under-parallelized — raising
the KV-split grid hurts; the residual 28→12.7 tok/s decay is the
implementation-physics floor of scanning 16 full-attn layers' KV at
tq4nc on B70; (2) chunked-prefill throughput does not improve with
bigger chunks; (3) the one env that changes the wedge
(comm-out-of-graph) costs most of the decode speed and still fails at
133k. The recommended long-ctx serve remains **nospec +
FULL_DECODE_ONLY + tq4nc + block 512** exactly as locked in v8.

On the user's MTP arm the ≥32k wedge (#11) is effectively DETERMINISTIC:
4/4 requests (2x 32k, 2x 131k) wedged at the prefill→decode handoff, engine
frozen (`run=1`, gtok frozen) until client disconnect; prefix-cache retries
also wedge. So "long-context tok/s" on this arm is mostly **0 or undefined**.

On the clean no-spec arm, long-context decode decays **32.1 → 27.0 → 17.9 →
12.3 tok/s** (2k→262k, −62%). Cause: the 16 full-attn layers scan the whole
KV each step (8 KB/token/rank at tq4nc → 2.1 GB/rank read per step at 262k),
with a graph-FIXED 256-split kernel grid (adaptive ladder inactive under
FULL_DECODE_ONLY). The 48 linear-attention layers are O(1)/token and do not
grow. Cold prefill runs at only ~1.5-2.0k tok/s → a 262k prompt is **~170 s
of engine-monopolizing chunked prefill**.

### Measured: why ALL other streams suffer (no-spec arm, clean)

| stream | condition | result |
|---|---|---|
| 109-tok prompt | fired mid 131k-prefill | **TTFT 55.1 s** (waits out every 8192-tok chunk) |
| 109-tok prompt | during 131k decode (co-decode) | TTFT 0.4 s, 17.7 tok/s (idle: 32.1) |
| 109-tok prompt | decoding DURING a fresh 131k prefill | TTFT 5.4 s then **1.4 tok/s** (45x idle) |
| 131k stream | solo | 17.9 tok/s |
| 131k stream | while another 131k prefills + shorts decode | **4.9 tok/s** |
| 3x 262k streams | co-resident (pool holds 4) | **0.4 tok/s EACH** |
| 5x 262k streams | pool = 4.31x max-len | TTFT ladder 173/346/522/696 s (serialized admission; ~0.28 tok/s aggregate useful) |

Mechanisms, ranked:

1. **Chunked-prefill monopoly (the big one).** MNBT=8192 is one chunk = the
   whole per-step token budget. Each ~170-s long prefill = ~32 steps in which
   other streams' decode rides mixed steps at ~1 token per multi-second step
   (1.4 tok/s measured) or waits entirely (55-s TTFT measured). With MTP on,
   mixed steps additionally lose the k=4 draft (verify needs uniform decode
   steps), so decoders drop to E[len]=1 AND crawl.
2. **Shared step clock (batched decode).** Every running sequence advances one
   engine step at a time; each 262k-ctx seq adds a ~2 GB KV scan to the step.
   Co-decoding long+short halves the short stream (32→17.7) and co-resident
   262k streams collapse to 0.4 tok/s each.
3. **KV capacity cliff.** 869,550 tokens (MTP arm) = 3.32 max-len requests;
   the 4th+ waits (TTFT ladder above); growth preemption on a hybrid model =
   full context re-prefill (mamba state cannot be partially restored), i.e.
   ANOTHER monopoly. Prefix caching retains finished blocks (7-17% residue
   observed), further shrinking free capacity.
4. **MTP tax.** 23% KV capacity + the >=32k wedge (#11) + spec-off mixed
   steps. k=4 is net-negative here at every length (15.0 vs 32.1 tok/s at
   2k, wedged at >=32k). k=1 is a genuine win for <=32k-bounded serves
   (+37% short ctx, parity at 32k) but still wedges ~40-50% at >=64k.
   **Ship posture (2026-08-30): serve the full 262k envelope WITHOUT
   `--speculative-config`; opt into k=1 only when the envelope is capped
   at 32k.**
5. **fp16 compute + fp32 mamba state**: constant per-step costs (align-mode
   per-step `.cpu()` sync, #11 line) — context-independent, secondary.

### Degenerate long-context outputs (RESOLVED, #13 — filler artifact)

On the no-spec arm, ~65k-token highly-repetitive prompts produced degenerate
completions 4/4 tries: instant-EOS (`finish_reason=stop`, empty text,
completion_tokens=1) x3 and a literal `"!"`-loop x1. 32k hit it once (1/2);
131k/262k were clean. Not an HTTP/engine error — server-side 200 streams.

**Resolution (2026-08-30 evening):** the same instant-EOS reproduces on
no-spec AND k=1, sequential AND concurrent, distinct fillers AND
cache-hit prefills — uncorrelated with MTP or infra, and probabilistic
per request (the same varied-filler seed pair was 2/2 clean on one boot,
`"!"`-loop + instant-EOS on the next). Verdict: distribution collapse on
low-entropy pattern-dominated filler; the probe filler's CONTENT is
meaningless at >= ~32k (throughput/wedge numbers remain valid).
Correctness gates must use natural text (`realistic_probe.py`, host
/root/build, is the generator — its synthetic output still degenerates
sometimes, so treat only natural-language prompts as content-clean).

## Operational notes

- Reboot the host after any GPU engine reset before starting vLLM again
  (see KNOWN_ISSUES.md #03): post-reset boots can wedge mid-decode.
- dflash spec-decode serving (adv:v17+): `--gpu-memory-utilization 0.8`
  is safe — the #05(e) oneCCL scratch-arena pre-allocation
  (`VLLM_XPU_PREALLOC_CCL_ARENA=1`, default on) runs one worst-case
  all-reduce BEFORE KV pool sizing, restoring 0.8 with KV pool 142,317
  tokens (+26% vs the old 0.75 workaround). The pre-fix advice to run
  0.75 is retired. With XPU graphs ON (adv:v17 image ENV) plus a
  drafter, adv:v18 or newer is REQUIRED — v17's eager post-capture
  spec-shape `_dummy_run` faults the xe ccs engine 4/4 launches
  (KNOWN_ISSUES #03 root cause; `VLLM_XPU_SPEC_SAFE_WARMUP=1` default
  in v18).
- Container recipe pitfalls (2026-08-27): with
  `CCL_ZE_IPC_EXCHANGE=drmfd` the container MUST bind-mount
  `/dev/dri/by-path` (privileged alone does not populate it) or every
  worker dies at oneCCL init (`ze_fd_manager init_device_fds: opendir
  failed`). Do NOT add `CCL_ATL_SHM=1` or `CCL_ATL_TRANSPORT=mpi` on
  top of the serve.sh env — they segfault MPI GPU init
  (`MPIDI_GPU_init_mpl_global`) at first collective. Use the serve.sh
  env block verbatim.
- Mid-decode DEVICE_LOST hardening (2026-08-27, see KNOWN_ISSUES
  #05(a) addendum): serve with
  `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True` (torch 2.11 XPU
  allocator) — removes fragmentation-driven alloc stalls that can
  delay a rank past the 640 ms xe GuC preempt watchdog and reset the
  engine mid-generation.
- All fatal configs (c6, k4 32+2warps) died with 4 engine resets -
  same recovery protocol applies.
- Image lineage: v6 = stock fork build; v7 = + TQ stage-1 knob envs;
  v8 = v7 + graph-safe splits backend (turboquant_attn.py, commit
  1c41a08); v9 = v8 + ESIMD page-attn fp16+graphs gate (c10c7b2);
  v10 = v9 + all_reduce compile-bounce gate (28ff055, fixes
  KNOWN_ISSUES #04); v11 = v10 + out-of-place AR op clone (108cfdd) —
  REGRESSION, corrupted PIECEWISE graphs-on serving, do not use;
  v12 = v10 semantics restored (748b972, current recommended). v8
  piecewise-default behavior is byte-identical to v6 (k0_base
  cross-check 29.05/17.46 vs 29.02/17.45); v12 spec k=4 greedy is
  byte-identical to v9 and v10. Minor note: the graphs-on nospec arm
  reads ~18 s @512 on v10/v11/v12 vs 15 s once on v9 (reproducible,
  nospec arm only; the spec k=4 and TQ champion arms are at full v9
  parity).
- `--cudagraph-mode` is not a flag; use
  `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`.
  When passing through nested `bash -c`, keep the value single-quoted
  with backslash-escaped quotes or the inner shell strips them.
- Full raw data: `/root/bench/results.csv` in the lsv-bench/lsv-bench7/
  lsv-bench8 containers on the benchmark host; harness scripts in
  `/root` (bench_gen.py, make_longprompt.py, sweep_one*.sh, run_*.sh).
