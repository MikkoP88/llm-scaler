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
stream (dflash arms die with xe DEVICE_LOST at 20-32k; the no-spec
control wedged at 2.9k; KNOWN_ISSUES #05(a)). Only eager k8v4/k3v4_nc
completed their 10k gens. The practical planning numbers: 60-73 tok/s
only for the first ~5k tokens; 30-45 tok/s is the normal sustained deep
band; keep interactive single streams ≤1536 (rock-solid) and chunk/retry
longer ones. The v13 "31.5 tok/s Avg generation" report was this depth
decay plus concurrent build load — not a regression.

## Operational notes

- Reboot the host after any GPU engine reset before starting vLLM again
  (see KNOWN_ISSUES.md #03): post-reset boots can wedge mid-decode.
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
