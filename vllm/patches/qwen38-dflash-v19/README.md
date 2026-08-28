# llm-scaler-vllm-adv:v19 — TQ x drafter enablement + multi-query verify kernel

`FROM llm-scaler-vllm-adv:v18`. Three-file delta over v18 (which carries the
#03 boot-fault fix; see ../qwen38-dflash-v18/README.md and the v17/v14 lineage
there). Goal: make the dflash drafter serve WITH compressed KV and beat the
user's superior target-only TQ config on the canonical car-game test.

## Why (user-visible symptom)

With `--kv-cache-dtype turboquant_4bit_nc` the drafter silently never engaged
(and with fp8_e4m3 it engaged but was slow, with long-run speed drops). Two
root causes:

1. **Our own #05b guard disabled it.** v14-era `config/vllm.py` nulled
   `speculative_config` whenever `kv_cache_dtype` started with `turboquant`
   (acceptance ~1-2% + warmup wedges back then). Both causes are gone: the
   drafter's KV is forced to unquantized `auto` and the #05b/#03 warmup skips
   protect boot. fp8_e4m3 was never covered — its issue was pure perf.
2. **TQ verify was intrinsically ~5x bandwidth.** A dflash verify step
   (q_len = k+1 = 5) dispatched to `_prefill_attention` -> "synthetic decode":
   each verify token became a separate single-token decode that re-scanned
   the ENTIRE compressed context. Cost grew with depth -> the long-run
   collapse. fp8/bf16 KV verify uses multi-token flash-attn, which is why
   spec worked there.

## Changes vs v18

| Change | File(s) | What | Env knob (default) |
|---|---|---|---|
| #05b revision: allow TQ x spec | config/vllm.py | `VLLM_ALLOW_TQ_SPEC` default 0 -> 1; drafter serves WITH turboquant KV; rollback restores target-only | `VLLM_ALLOW_TQ_SPEC` (1) |
| Multi-query verify kernel (additive) | triton_turboquant_decode.py | `_tq_mq_decode_stage1` + `_tq_mq_fwd_stage2` + launcher: ONE pass over the compressed KV per (head, split) scores all q_len query rows per shared tile (flash-decoding style; per-row causal limits `q0 + j`; blind split-combine via lse=-inf zero weights). Existing kernels untouched | `VLLM_TQ_MQ_STAGE1_WARPS` (1), `VLLM_TQ_MQ_STAGE1_STAGES` (1) |
| Verify fast-path dispatch | turboquant_attn.py | Continuation branch: `1 < q_len <= 8` -> single MQ kernel call instead of q_len synthetic single-token decodes; v18 path kept verbatim as rollback | `VLLM_TQ_MQ_VERIFY` (1), `VLLM_TQ_MQ_MAX_Q` (8) |

## Numerics validation (standalone, in-container: `test_tq_mq_numerics.py`)

32/32 PASS across {MSE4/V4 (tq4nc-like), FP8-key/V4 (k8v4-like), MSE4/V4 +
norm-correction, MSE3/V3} x {cached_len 1..2000 incl. 32-split cdiv boundary
crossings} x {q_len 2..8}:
- vs the production single-query kernel called exactly like the v18 backend
  path: **max diff 0.0** (bit-identical) on every case;
- vs an independent dequant + fp32 torch attention reference: <= 5e-4
  (note: stored MSE keys are in rotated space — the reference must rotate
  queries with PiT; fp8 keys are stored raw).

Per-layer verify-attention micro-bench (q_len=5, warps=1; single = the v18
synthetic-decode call):

| KV keys | cached=2000 | cached=4096 | cached=16384 |
|---|---|---|---|
| MSE4 (tq4nc) | 1.05x | 1.12x | 1.32x |
| FP8 (k8v4) | 2.17x | 2.49x | 2.66x |

num_warps=1 measured best at depth (2/4 warps: 0.88x/0.49x on MSE) — matches
the single-query kernel lineage (wide tiles + warps spill on Xe2).

## Car-game matrix (canonical user test)

Prompt `Write a html car game.`, sampling temp 0.3 / top_k 20 / top_p 0.95 /
min_p 0 / presence 0 / repetition 1.0, max_tokens 4096, streaming; one warmup
completion first (absorbs the v18 first-request JIT trade-off). All cells use
the user's superior-config serving shape: dtype float16, block-size 128,
mnbt 8192, maxlen 262144, cudagraph_mode FULL_DECODE_ONLY
(`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`) — except
**maxlen for spec cells** (see the memory note below).

**Spec-cell maxlen note (memory wall, measured):** nospec + TQ at 262k fits
easily (bar: 9.69 GiB free KV = 1,210,665 tokens = 4.62x concurrency @262k).
With the drafter actually loaded (first time under TQ — v18's #05b guard used
to silently drop it), the wall is the **drafter's own KV pool**: dflash forces
it UNCOMPRESSED (`auto`/bf16) and sizes it by the same max_model_len —
~11.3 GiB/GPU @262144 on top of the target's ~2.1 GiB (TQ4nc) = 13.48 GiB
needed vs 6.67 GiB available (after ~3.0 GiB drafter fp8 weights). Spec cells
therefore run the largest maxlen that fits: c3/c2/c4 = 98304 (vLLM estimated
c3 max 118784), c1 = 73728 (bf16 target). Decode speed at bench depth (~4k)
is independent of KV pool size, so the nospec-vs-spec comparison stays valid.
Long-context (262k+) + spec needs the drafter KV quantized too (follow-up).

<!-- CARGAME_MATRIX_RESULTS -->

Results (steady tok/s = last-50% mean; acceptance = per-position P0..P3 from
the serve-log SpecDecoding windows; 2026-08-28, one boot per cell, monitor3):

| cell | KV | spec | maxlen | steady tok/s | acceptance P0/P1/P2/P3 | note |
|---|---|---|---|---|---|---|
| bar | turboquant_4bit_nc | no | 262144 | **32.79** | — | user's superior config: the number to beat |
| nbf16 | bf16 | no | 262144 | 33.10 | — | nospec bf16 baseline |
| nfp8 | fp8_e4m3 | no | 262144 | (banked on host — see below) | — | completed before c2 |
| c1 | bf16 | k4 | 73728 | 19.94 | .45-.55 / .20-.27 / .05-.17 / .03-.07 | healthy acceptance |
| c2 | fp8_e4m3 | k4 | 98304 | 19.69* | .47-.55 / .19-.27 / .05-.14 / .03-.06 | *4 xe engine resets mid-cell (#03 family); tail collapsed to 3 tok/s |
| c3 | turboquant_4bit_nc | k4 | 98304 | 21.87 | **0.000 / 0.000 / 0.000 / 0.000 (sampled)** | greedy warmup accepted 216/384 ≈ 2.25 tok/step |
| nk8v4 / c4 | turboquant_k8v4 | no/k4 | — | PENDING | — | host wedged in POST after the #03 protocol reboot |

**Findings:**

1. **v19 enablement works, and the verify-bandwidth fix works.** TQ+spec
   boots, serves, and shows NO long-run speed collapse (c3 buckets
   22.1 → 21.27 over 3 min; the v18-era depth-scaling rescan is gone). Zero
   resets in c3. The MQ kernel is bit-identical to the production kernel
   (32/32 numerics cases).
2. **The user bar is not met — for two independent reasons.**
   a. **Acceptance economics:** at temp 0.3 sampled decoding, healthy
      acceptance (bf16/fp8, P0 ≈ 0.5, mean length 1.85-2.0) yields ~20
      tok/s vs ~33 nospec: a spec step costs ~3x a decode step (drafter k=4
      + eager 5-row verify under FULL_DECODE_ONLY graphs), so breakeven
      needs ~65% P0. Argmax drafting under sampling tops out near P(target
      sample == drafter argmax) — this serving/test shape cannot reach it
      at k=4.
   b. **TQ-specific sampled-verify defect:** turboquant_4bit_nc + spec
      accepts EXACTLY ZERO drafted tokens under sampling (4k steps,
      per-position 0.000 across every 10 s window), while the GREEDY warmup
      on the same server accepted ≈ 2.25 tok/step (healthy). bf16/fp8
      spec at the same sampling accepts normally (~22% avg). So the TQ
      attention outputs are distributionally sound (greedy matches, text
      coherent, kernel numerics exact) but the SAMPLED comparison path
      mis-fires when the TQ backend serves verify — argmax is unaffected,
      sampling is fully broken. This reproduces the v14-era "1-2%
      acceptance" that the #05b guard papered over; it is NOT caused by the
      v19 MQ kernel (which is bit-identical to the path that showed it
      before). Bisect probe prepared (`.tmp-tq/probe_tq_sampling.sh`:
      greedy / temp 0.05 no-filter / canonical / temp 1.0 in one boot) —
      discriminating "sampled path broken" vs "topk/topp interaction".
3. **#03 recurrence on fp8+spec bar-shape:** c2 hit 4 xe engine resets
   (ccs+bcs, both GPUs) mid-generation; the matrix correctly aborted the
   remaining cells and demanded the protocol reboot. The host then hung in
   POST (unreachable >20 min; needs a power cycle) — nk8v4/c4/nfp8-numbers
   pending host recovery.

## Files

- `Dockerfile` — overlay build + grep guards + py_compile + import checks
- `config_vllm.py`, `turboquant_attn.py`, `triton_turboquant_decode.py` — overlay set
- `test_tq_mq_numerics.py` — standalone numerics + micro-bench (mount + docker exec)
- `bench_cargame.sh` + `cargame_client.py` — canonical car-game cell driver
- `cargame_matrix.sh` — host matrix runner (one boot per cell, monitor3, graceful teardown, #03 reset gate)
- `serve.sh` — host launcher (v18 knobs + `BLOCKSIZE`, `EXTRA_ARGS`,
  `VLLM_ALLOW_LONG_MODEL_LEN=1`)
- inherited from v18: `gpu_worker.py`, `gpu_model_runner.py`, `sched_utils.py`,
  `dflash.py`, `serve_supervised.sh`, `monitor3.sh`, battery scripts

## Build (host, only while NOT serving)

```bash
scp -r vllm/patches/qwen38-dflash-v19 root@<host>:/root/build/
ssh root@<host> 'cd /root/build/qwen38-dflash-v19 && docker build -t llm-scaler-vllm-adv:v19 .'
```

## Run

```bash
# TQ4nc + drafter (the v19 headline combo)
KV_DTYPE=turboquant_4bit_nc TARGET_DIR=/models/qwen3.8-27b-fp8 DRAFTER_DIR=/models/drafter-fp8-v5 ./serve.sh
# user's superior target-only shape
SPEC=0 KV_DTYPE=turboquant_4bit_nc DTYPES=float16 BLOCKSIZE=128 MNBT=8192 MAXLEN=262144 \
  EXTRA_ARGS='--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}' TARGET_DIR=... ./serve.sh
# rollback arms
VLLM_TQ_MQ_VERIFY=0 ...   # v18 synthetic-decode verify path
VLLM_ALLOW_TQ_SPEC=0 ...  # v18 target-only under turboquant
```

## Known limits

- Driver-level #03 protocol unchanged: after any xe engine reset, reboot the
  host before re-serving.
- The MQ kernel covers q_len <= 8 (dflash k<=7); larger continuation chunks
  fall back to the v128 continuation-decode / flash paths as before.
