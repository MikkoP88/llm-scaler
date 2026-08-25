# Performance Tuning: Maximum Token Generation Speed (Arc Pro B70 x2)

Measured 2026-08-24/25 on the benchmark host (2x Intel Arc Pro B70, TP=2,
`llm-scaler-vllm-adv:v6`/`v7` images, vllm v0.21.1.dev0+gad7125a43, torch
2.11.0+xpu). Model: qwen3.8-27b, fp8 weights, turboquant_4bit_nc KV,
`--max-model-len 262144`.

Benchmark protocol: streaming /v1/completions, `ignore_eos`, SSE chunk ==
1 token. "shallow" = 21-char prompt + 8192 (or 3072) gen tokens;
"deep64" = ~117k-token prompt (607,747 chars) + 6144 gen tokens.
`tps_steady` = per-stream steady-state window at conc=1; at conc>1 the
table reports `tps_overall` (aggregate across streams). Fresh boot
before every campaign; engine resets => host reboot before next run.

## Locked max-speed configuration

The user's serve command is unchanged. Add exactly two environment
variables:

```bash
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
VLLM_XPU_ENABLE_XPU_GRAPH=1 \
VLLM_XPU_ALLOW_COMM_IN_GRAPH=1 \      # capture oneCCL allreduce inside XPU graphs
VLLM_USE_V2_MODEL_RUNNER=1 \          # V2 model runner
vllm serve --model /models/qwen3.8-27b \
  --tensor-parallel-size 2 --quantization fp8 \
  --kv-cache-dtype turboquant_4bit_nc --dtype float16 \
  --enable-chunked-prefill --trust-remote-code \
  --gpu-memory-utilization 0.8 --block-size 128 \
  --max-num-batched-tokens 8192 --max-model-len=262144
```

No other flags/env improve throughput (see sweep below). Works on the
stock v6 image; the v7 image additionally exposes the TQ stage-1 knobs
for experiments (defaults unchanged, see "Rejected" below).

## Headline results (steady tok/s, conc=1)

| config | shallow | deep64 (~117k ctx) | TTFT |
|---|---|---|---|
| anchor (user config) | 29.02 | 17.45 | 12.7 s |
| **c2+c5 locked config** | **32.75 (+12.9%)** | **19.45 (+11.5%)** | **0.23 s** |

Aggregate throughput scaling (tok/s, all streams):

| conc | anchor | c2+c5 | delta |
|---|---|---|---|
| 2 (shallow) | 51.21 | 62.96 | +23.0% |
| 4 (shallow) | 111.71 | 117.71 | +5.4% |
| 8 (shallow) | 211.11 | 207.53 | -1.7% (saturated) |
| 2 (deep64) | 22.61 | 22.20 | parity |

The locked config wins where latency matters (1-2 streams: +13-23%,
TTFT ~50x better) and is neutral once the GPUs saturate at ~210 tok/s
aggregate (conc>=8).

## Sweep detail (conc=1, steady tok/s shallow / deep64)

| config | change | shallow | deep64 | verdict |
|---|---|---|---|---|
| anchor | user's exact env+flags | 29.02 | 17.45 | baseline |
| c1 | VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 | 21.37 | 17.13 | reject (-26% shallow) |
| c2 | VLLM_XPU_ALLOW_COMM_IN_GRAPH=1 | 31.68 | 17.67 | keep (component) |
| c4 | VLLM_TQ_MAX_KV_SPLITS=512 | 29.54 | 17.46 | flat (engages >131k ctx only) |
| c5 | VLLM_USE_V2_MODEL_RUNNER=1 | 26.24 | 19.19 | keep (component) |
| c6 | CCL_ENABLE_SYCL_KERNELS=1 | ENGINE DIED | - | fatal: do not use |
| c7 | CCL_ATL_TRANSPORT=ze | init fail | - | incompatible: do not use |
| c8 | cudagraph_mode FULL_DECODE_ONLY | 32.41 | 13.10 | reject (-25% deep; kills adaptive TQ splits) |
| c9 | --enable-prefix-caching | 25.15 | 17.45 | reject (-13% shallow, hashing overhead) |
| c10 | --max-num-batched-tokens 16384 | 26.84 | 17.50 | reject (no gain) |
| d1 | c2+c5 | 32.75 | 19.45 | LOCKED |
| d2 | c2+c5+TQ BLOCK_KV=8 | 32.63 | 17.46 | reject (knob hurts deep) |

TQ stage-1 knob sweep on v7 image (BLOCK_KV x warps vs defaults 4/1):
16/1 -> -72% deep, 32/1 -> -83% deep, 16/2 -> -20% deep, 32/2 ->
EngineDeadError (fatal). Shipped defaults are measured-optimal on Xe2;
documented in the patch comment (commit bc1cb0b).

## Why it works (decode profile, py-spy on healthy deep decode)

~51% XPU graph replay, ~16% oneCCL allreduce dispatch+exec (~80 eager
allreduces per decode step between graph pieces), ~4.5% TQ decode
launcher. `ALLOW_COMM_IN_GRAPH=1` moves the collectives into the
captured pieces (removes eager dispatch overhead); the V2 runner
restructures the step loop for better overlap (+10% deep). At high
concurrency the batched eager allreduce amortizes, so the gap closes.

## Operational notes

- Reboot the host after any GPU engine reset before starting vLLM again
  (see KNOWN_ISSUES.md #03): post-reset boots can wedge mid-decode.
- All fatal configs (c6, k4 32+2warps) died with 4 engine resets -
  same recovery protocol applies.
- Full raw data: `/root/bench/results.csv` in the lsv-bench/lsv-bench7
  containers on the benchmark host; harness scripts in `/root`
  (bench_gen.py, make_longprompt.py, sweep_one*.sh, run_*.sh).
