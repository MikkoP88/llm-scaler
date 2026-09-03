# llm-scaler-vllm-adv:v17 — reliability + memory overlay (qwen3.8-27b, 2x Arc Pro B70)

`FROM llm-scaler-vllm-adv:v14`. Five-file code overlay + supervisor + telemetry
harness. Base files were pulled from the **v14 image itself** (gpu_model_runner
carries the v14 TurboQuant padded-page stride fix; the other three were
md5-verified identical to the image before editing), so no v14 fix is reverted.

## Fixes vs v14

| Issue | File(s) | Change | Env knob (default) |
|---|---|---|---|
| #03 wedge holds dead port | gpu_worker.py | Warmup watchdog: `faulthandler.dump_traceback_later(exit=True)` around the whole compile/capture/warmup phase; dumps all thread backtraces and hard-exits on timeout | `VLLM_XPU_WARMUP_TIMEOUT_S` (1800, 0=off) |
| #03 recovery | serve_supervised.sh | `/opt` supervisor: restarts `vllm serve` after abnormal exits (watchdog kill, crash); clean exits (0/130/143) stop | `VLLM_SUPERVISE_MAX_RESTARTS` (3, 0=∞), `VLLM_SUPERVISE_COOLDOWN_S` (30) |
| #05(a) fragmentation alloc stalls | Dockerfile ENV | `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True` baked | override with `-e` |
| #05(b) TQ dtype post-capture hang | gpu_worker.py | TurboQuant KV skips the eager post-capture `_dummy_run` (never-captured 320-token shape faulted the xe queue); synthesizes hidden states so `_dummy_sampler_run` still warms the sampler | `VLLM_XPU_TQ_SAFE_WARMUP` (1) |
| #05(b) TQ + drafter useless/hang | config/vllm.py | Speculative decoding auto-disabled when `kv_cache_dtype` starts with `turboquant` (acceptance ~1-2% + wedge); serves target-only with a warning | `VLLM_ALLOW_TQ_SPEC` (0) |
| #05(d) ignore_eos early finish | v1/core/sched/utils.py | Hard guard: eos/stop-token branches can never fire with `ignore_eos=True`; FINISH_DIAG warnings on invariant violations and on early window-cap finishes (normal min==max caps stay silent) | always on |
| #05(e) startup UR39 @ gmu 0.8 | gpu_model_runner.py | One worst-case all-reduce (8192 tok x hidden) BEFORE weight load + KV pool sizing: oneCCL's scratch arena is carved from genuinely free device memory and excluded from the KV-sizing snapshot. Restores `--gpu-memory-utilization 0.8` (+22% KV pool vs the 0.75 workaround) | `VLLM_XPU_PREALLOC_CCL_ARENA` (1) |
| dflash cold-start stalls | dflash.py | Committed fix a31915f: DFLASH_STALL guard (>150 ms) + grow-only `_markov_latent_buf` scratch | `VLLM_DFLASH_STALL_MS` (150) |

Also baked: the full canonical XPU/CCL env block (see Dockerfile) as image
defaults — a bare `docker run <image> vllm serve ...` is now correctly
configured; `--generation-config default` is the serve.sh default (avoids the
model generation_config.json temp-1.0 inheritance trap).

## Files

- `Dockerfile` — overlay build + py_compile + import checks for all 5 modules
- `dflash.py`, `gpu_model_runner.py`, `gpu_worker.py`, `sched_utils.py`,
  `config_vllm.py` — the overlay set
- `serve_supervised.sh` — supervisor (baked to `/opt`)
- `monitor3.sh` — per-arm telemetry (baked to `/opt`, also deploy to host
  `/root/telemetry/`)
- `serve.sh` — host launcher: `KV_DTYPE={unset,fp8,turboquant_4bit_nc,
  turboquant_k8v4}`, `SPEC={0,1}`, `SUPERVISED={0,1}`, `GMU`, `MAXLEN`, ...

## Build (host, only while NOT serving)

```bash
scp -r vllm/patches/qwen38-dflash-v17 root@<host>:/root/build/
ssh root@<host> 'cd /root/build/qwen38-dflash-v17 && docker build -t llm-scaler-vllm-adv:v17 .'
```

## Run

```bash
# dflash k=4 champion (bf16 KV)
TARGET_DIR=/models/qwen3.8-27b-fp8 DRAFTER_DIR=/models/drafter-fp8-v5 ./serve.sh
# fp8 KV + drafter
KV_DTYPE=fp8 TARGET_DIR=... DRAFTER_DIR=... ./serve.sh
# turboquant arms (spec auto-disables -> target-only, by design)
KV_DTYPE=turboquant_4bit_nc TARGET_DIR=... ./serve.sh
# supervised (auto-restart on wedge/crash)
SUPERVISED=1 TARGET_DIR=... DRAFTER_DIR=... ./serve.sh
```

Telemetry per arm (host): copy monitor3.sh to /root/telemetry/, then
`ARM=<arm-name> ./monitor3.sh &` while serving; outputs under
`/root/telemetry/arms/<arm-name>/` (meta/health/rates/xpu-smi/metrics/issues/
dmesg + auto `capture_once.sh` forensic snapshots on engine reset, shm-broadcast
timeouts, spin-wedge, watchdog fire). meta.txt records the live env — check it
to confirm `VLLM_XPU_USE_SAMPLER_KERNEL` and `PYTORCH_XPU_ALLOC_CONF` per arm.

## Known limits (unchanged, driver-level)

- #03 root: after any xe engine reset, reboot before re-serving (post-reset
  boots can wedge mid-decode). The watchdog+supervisor turns a silent wedge
  into a fast, diagnosable restart but cannot fix the driver state.
- turboquant_4bit_nc remains unusable in graphs mode at runtime (capture-hang
  was warmup-side; the ~4.3k eager death is a kernel issue) — battery will
  re-verify on v17; k8v4/k3v4_nc now boot with graphs (warmup fixed) but stay
  slow; serve target-only.

## Validation plan (v17 battery)

1. Import/boot smoke: every {default, fp8, tq4nc, k8v4} x {SPEC 0,1} boots at
   gmu 0.8 (arena fix) with monitor3 running; zero UR39, zero wedges.
2. Perf parity gates vs v14 champions: dflash k=4 warm 512 gen == 8 s;
   greedy 40k == ~57 tok/s (completions protocol); TQ fp16 champion 15 s
   @512 grid=256.
3. Long-ctx deep gens (64k-80k windows, forced min==max), memory-limited arms
   (gmu 0.9), temp sweep 0.0-1.0.
4. Baselines under identical workloads: `intel/llm-scaler-vllm:0.21.0-b3.1`,
   `ghcr.io/rmacy/qwen38-fp8-dspark:v16` — v17 must win or tie every cell.
