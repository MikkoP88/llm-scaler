# llm-scaler-vllm-adv:v19 — TQ x drafter enablement + multi-query verify kernel

> **v20 correction (2026-08-28):** v19's headline "healthy TQ spec at 17.56
> tok/s loses to nospec 32.79" was **wrong** — the v19 bench client counted
> SSE delta *events* as tokens, and spec decode flushes ~E[len] tokens per
> event, underreporting every spec cell ~1.9x (nospec cells were correct).
> TRUE steady rates from the engine SpecDecoding windows: c1 37.8 / c2 36.2 /
> c3 35.6 / c4 32.6, k-curve on tq4nc k2 39.8 / k4 35.6 / k6 34.9 — **all
> spec cells beat their nospec twins.** Proof chain, fixed client, and the
> A2 step-cost decomposition: ../qwen38-dflash-v20/README.md. The engine
> itself was never at fault; only the client metric was.

`FROM llm-scaler-vllm-adv:v18`. Three-file delta over v18 (which carries the
#03 boot-fault fix; see ../qwen38-dflash-v18/README.md and the v17/v14 lineage
there). Goal: make the dflash drafter serve WITH compressed KV and beat the
user's superior target-only TQ config on the canonical car-game test.

## Why (user-visible symptom)

With `--kv-cache-dtype turboquant_4bit_nc` the drafter silently never engaged
(and with fp8_e4m3 it engaged but was slow, with long-run speed drops). Three
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
3. **v19b (2026-08-28): the verify GRAPHS were context-blind.** Under
   `FULL_DECODE_ONLY` XPU graphs, spec-verify steps are captured at bs×(k+1)
   dummy shape where `seq_lens == q_len == 5`, so `_prefill_attention`'s
   raw-KV flash fast path (`max_query_len == max_seq_len`) fired AT CAPTURE
   and got baked into the graph: replays attended ONLY the 5 in-batch
   verify tokens and never read the KV cache. Symptom: deterministic
   "user just sent a blank message" hallucinations + ~8% greedy / 0%
   sampled acceptance (the target was blind; greedy's apparent 2.25
   tok/step was row-0 markov coincidence with the drafter). This bug was
   unreachable ≤v18 (#05b guard) and explains why probe8's MQ-kernel
   rollback was bit-identical — the continuation path was never reached.
   See KNOWN_ISSUES.md #06 for the full instrumentation chain.

## Changes vs v18

| Change | File(s) | What | Env knob (default) |
|---|---|---|---|
| #05b revision: allow TQ x spec | config/vllm.py | `VLLM_ALLOW_TQ_SPEC` default 0 -> 1; drafter serves WITH turboquant KV; rollback restores target-only | `VLLM_ALLOW_TQ_SPEC` (1) |
| Multi-query verify kernel (additive) | triton_turboquant_decode.py | `_tq_mq_decode_stage1` + `_tq_mq_fwd_stage2` + launcher: ONE pass over the compressed KV per (head, split) scores all q_len query rows per shared tile (flash-decoding style; per-row causal limits `q0 + j`; blind split-combine via lse=-inf zero weights). Existing kernels untouched | `VLLM_TQ_MQ_STAGE1_WARPS` (1), `VLLM_TQ_MQ_STAGE1_STAGES` (1) |
| Verify fast-path dispatch | turboquant_attn.py | Continuation branch: `1 < q_len <= 8` -> single MQ kernel call instead of q_len synthetic single-token decodes; v18 path kept verbatim as rollback | `VLLM_TQ_MQ_VERIFY` (1), `VLLM_TQ_MQ_MAX_Q` (8) |
| v19b: replay-safe verify graphs (#06 fix) | turboquant_attn.py | `build_for_cudagraph_capture` forces the continuation path for multi-token captures (fast path can no longer be baked into verify graphs); MQ/synthetic per-row causal limits derive from the DYNAMIC `seq_lens` buffer (`seq_lens[i:i+1]-(q_len-1)` / `+arange`) instead of a static arange slice, so replayed extents track the real context | `VLLM_TQ_VERIFY_GRAPH_FIX` (1) |
| v19c: capture-safe fp8 KV scales (#07 fix) | flash_attn.py | ESIMD decode fast path cached `float(layer._k/_v_scale)` per layer (static after load) instead of a per-call D2H sync that aborted XPU graph capture for fp8 KV x nospec (boot crash, 4/4); uncached capture skips the fast path (varlen fallback) rather than bake wrong scales | `VLLM_ESIMD_F8_SCALE_FIX` (1) |

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
the serve-log SpecDecoding windows; 2026-08-28, one boot per cell, monitor3;
c3/c4 re-run on the v19b image — earlier v19 numbers for those cells were
measured on the context-blind graphs and are kept in brackets for reference):

| cell | KV | spec | maxlen | steady tok/s | acceptance P0/P1/P2/P3 | note |
|---|---|---|---|---|---|---|
| bar | turboquant_4bit_nc | no | 262144 | **32.79** | — | user's superior config: the number to beat |
| nbf16 | bf16 | no | 262144 | 33.10 | — | nospec bf16 baseline |
| nfp8 | fp8_e4m3 | no | 262144 | 33.34 | — | v19c (#07): booted first try after the scale-sync fix (was 4/4 boot crashes); fastest nospec cell; nbf16 re-run on the same image unchanged (33.11 vs 33.10) |
| c1 | bf16 | k4 | 73728 | 19.94 | .45-.55 / .20-.27 / .05-.17 / .03-.07 | healthy acceptance (no TQ, unaffected by #06) |
| c2 | fp8_e4m3 | k4 | 98304 | 19.69* | .47-.55 / .19-.27 / .05-.14 / .03-.06 | *4 xe engine resets mid-cell (#03 family); tail collapsed to 3 tok/s |
| c3 | turboquant_4bit_nc | k4 | 98304 | 17.56 [was 21.87 blind] | .52-.76 / .24-.53 / .08-.33 / .02-.23 windowed; mean len 1.87-2.85 | v19b: correct text, healthy acceptance; steady over 2115 tok |
| c4 | turboquant_k8v4 | k4 | 98304 | 16.10 | .41-.52 windowed; mean len 1.67-1.98 | v19b: first healthy c4 ever (v19 c4 aborted on 4 resets) |
| nk8v4 | turboquant_k8v4 | no | 262144 | 32.61 | — | parity with bar within noise |
| dspark16 | bf16 (auto) | k4 | 73728 | 11.94 | mean accept len 1.66-2.53 | upstream baseline image `ghcr.io/rmacy/qwen38-fp8-dspark:v16` (v16-era defaults, no graph guards): v19 c1 is 1.67x faster on the identical config |

**Findings:**

1. **v19b restores CORRECTNESS of TQ x spec (the #06 fix).** All
   deterministic arms that hallucinated identically on six boots
   (probes 3-12) now produce correct HTML-car-game text; greedy
   acceptance 8% -> 25.5% (49/192, ~2.0 tok/step gross), sampled
   21-46% windowed; zero resets; c4 (k8v4) completes healthily for the
   first time. The earlier "greedy 2.25 tok/step (healthy)" read was the
   blind target's row-0 markov coincidence — the text was garbage.
2. **The user speed bar is still NOT met — honest verify costs more
   than blind verify.** c3 fixed 17.56 vs bar 32.79 (the blind 21.87
   was "fast" only because attending 5 tokens is nearly free). With
   healthy acceptance (~2.0 tok/step gross), a TQ spec step must cost
   <= ~2x a decode step to break even; measured ~3.5-4x (MQ verify
   kernel + drafter overhead under FULL_DECODE_ONLY). The same ~3x
   step-cost ratio appears on bf16/fp8 flash verify (c1/c2 ~20 vs ~33
   nospec), so the gap is dominated by per-step spec overheads, not TQ.
   Decomposition + reduction paths: see the SPEC_STEP_COST_ANALYSIS
   section in ../../PERF_TUNING.md (headline: the dflash drafter is
   BLOCK-drafted in one eager pass under PIECEWISE cudagraphs —
   full-graphing it, batching the per-layer precompute loops, and a
   per-step sync audit project ~50-60 ms/step vs the 57 ms breakeven;
   k=6/8 is the cheap experiment, k=2 is the wrong direction for block
   drafting).
3. **#03 recurrence risk stays:** c2's 4-reset tail collapse predates
   the fix and is unrelated to it (fp8 flash path). Protocol unchanged:
   reboot after any xe reset.

## Files

- `Dockerfile` — overlay build + grep guards + py_compile + import checks
- `config_vllm.py`, `turboquant_attn.py`, `triton_turboquant_decode.py`,
  `flash_attn.py` — overlay set
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
VLLM_TQ_MQ_VERIFY=0 ...        # v18 synthetic-decode verify path
VLLM_TQ_VERIFY_GRAPH_FIX=0 ... # v19b off (NOT recommended: restores the
                               # context-blind verify graphs, KNOWN_ISSUES #06)
VLLM_ALLOW_TQ_SPEC=0 ...       # v18 target-only under turboquant
VLLM_ESIMD_F8_SCALE_FIX=0 ...  # v19c off (A/B only: fp8-nospec boot crashes
                               # return, KNOWN_ISSUES #07)
```

## Known limits

- Driver-level #03 protocol unchanged: after any xe engine reset, reboot the
  host before re-serving.
- The MQ kernel covers q_len <= 8 (dflash k<=7); larger continuation chunks
  fall back to the v128 continuation-decode / flash paths as before.
- Spec verify graphs rely on the runner refreshing `seq_lens`/`block_table`
  at replay (it does — same contract the working nospec decode graphs use);
  any future change to those buffers must preserve it.
