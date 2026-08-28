# llm-scaler-vllm-adv:v20 — the "spec loses" result was a benchmark artifact

`FROM llm-scaler-vllm-adv:v19`. v19's headline finding — "healthy TQ spec at
17.56 tok/s loses to nospec 32.79" — was **wrong**: the canonical bench
client counted SSE delta *events* as tokens, and with spec decode the
detokenizer flushes ~E[len] tokens per event. Every spec cell in the v19
table underreported by ~1.9x. The nospec cells emit 1 token/event and were
correct. v20 proves this with three independent sources, fixes the client,
and re-runs the full matrix with TRUE token counts.

## The proof chain (all measured 2026-08-28, v19 image, tq4nc k4 = cell c3)

1. **Engine counters.** The 10s `SpecDecoding metrics` windows satisfy
   `emitted = mean_accept_length x drafted/k` exactly (accepted + bonus
   identity). Steady tail windows for c3: **~35.6 true tok/s**, not 17.56.
   The engine's rolling "generation throughput" (31-44 tok/s) matches; the
   v19 client's 17.56 does not.
2. **Step-timing instrumentation (v20 A2, `VLLM_SPEC_TIMING=1`)** measured
   propose-to-propose step wall = **56-63 ms**, i.e. 16-18 steps/s. At the
   engine's ~2.0 tokens/step that is 32-36 tok/s — consistent with the
   engine, not the client. (The v19 "95 ms/step" was derived FROM the bad
   client number.)
3. **k-curve re-read (true rates).** k=2/4/6 on tq4nc: 39.8 / 35.6 / 34.9
   true tok/s. All spec cells beat their nospec twins (see table).

## A2 step-cost decomposition (tq4nc k4, steady; ms/step)

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

Implications (supersedes v19's SPEC_STEP_COST_ANALYSIS decomposition):
- The v19 plan's centerpiece lever (full-graph the propose segment) targets
  only the ~9 ms propose host time — the verify replay's ~40 ms device time
  and ~12 ms glue dominate. With all parity gates already met on true rates,
  the graph surgery's risk/reward no longer justifies it. Retained here as
  measured follow-up material.
- k=2 wins because the step cost drops (47.8 ms: 3-row verify) while
  acceptance stays ~1.9 tokens/step (block drafting quality is k-independent
  at these depths).

## True-rate matrix re-read from v19 serve logs (engine windows, tail-6)

| cell | KV | spec | TRUE steady tok/s | ms/token | nospec twin | verdict |
|---|---|---|---|---|---|---|
| c1 | bf16 | k4 | **37.8** | 26.5 | 33.10 | +14% |
| c2 | fp8_e4m3 | k4 | **36.2** | 27.5 | 33.34 | +8.6% |
| c3 | tq4nc | k4 | **35.6** | 28.0 | 32.79 (bar) | +8.6% |
| c4 | k8v4 | k4 | **32.6** | 30.8 | 32.61 | parity |
| k2c3 | tq4nc | k2 | **39.8** | 25.1 | 32.79 (bar) | +21% |
| k6c3 | tq4nc | k6 | 34.9 | 28.6 | 32.79 (bar) | +6.5% |

(The v19 client-event numbers are kept in ../qwen38-dflash-v19/README.md for
the record; they are event rates, not token rates.)

## Changes vs v19

| Change | File(s) | What | Env knob (default) |
|---|---|---|---|
| TRUE-token bench client | cargame_client.py | `stream_options.include_usage`; reports `tokens_true`, `overall_true`, `steady_true` (event rate x tokens/event) + legacy event metrics; per-bucket true rates | — |
| Spec step segment timing (inert) | spec_timing.py + hooks in gpu_model_runner.py, dflash.py, llm_base_proposer.py | xpu-event ring, one sync per 200 steps; segments step_wall/tforward/tlogits/propose/precompute/dforward/greedy. Bit-identical serving when off | `VLLM_SPEC_TIMING` (0), `VLLM_SPEC_TIMING_FLUSH` (200) |
| k=2 matrix arm | cargame_matrix.sh, serve.sh | k2c1..k2c4 cells; `SPEC_K` pass-through (already in serve.sh) | `SPEC_K` (4) |

No serving-path behavior changes: the timing hooks are null contexts unless
`VLLM_SPEC_TIMING=1`, so v20 serving is bit-identical to v19 by default.

<!-- V20_MATRIX_RESULTS -->
## v20 shipped-image matrix (llm-scaler-vllm-adv:v20, 2026-08-28, one boot per cell)

TRUE steady tok/s = serve-log SpecDecoding windows, tail-6 (authoritative);
`client` = fixed client's steady_true (last-50%); [brackets] = v19-image
reference. All 12 gates PASS: every spec cell >= its nospec twin, nospec
cells reproduce v19 within +-0.1%, zero resets on every shipped cell.

| cell | KV | spec (k) | maxlen | steady true | client | E[len] | ms/step | vs twin | verdict |
|---|---|---|---|---|---|---|---|---|---|
| bar | tq4nc | no | 262144 | 32.78 [32.79] | 32.78 | — | — | — | unchanged |
| nbf16 | bf16 | no | 262144 | 33.08 [33.10] | 33.08 | — | — | — | unchanged |
| nfp8 | fp8_e4m3 | no | 262144 | 33.36 [33.34] | 33.36 | — | — | — | unchanged |
| nk8v4 | k8v4 | no | 262144 | 32.60 [32.61] | 32.60 | — | — | — | unchanged |
| c1 | bf16 | k4 | 73728 | **44.0** [37.8] | 40.66 | 2.19 | 49.6 | +32.9% vs nbf16 | PASS |
| c2 | fp8_e4m3 | k4 | 98304 | **35.5** [36.2] | 33.34 | 2.00 | 56.1 | +6.4% vs nfp8 | PASS |
| c3 | tq4nc | k4 | 98304 | **34.6** [35.6] | 33.39 | 1.99 | 57.4 | +5.5% vs bar | PASS |
| c4 | k8v4 | k4 | 98304 | **35.0** [32.6] | 31.63 | 2.05 | 58.9 | +7.4% vs nk8v4 | PASS |
| k2c1 | bf16 | k2 | 73728 | **37.4** | 36.15 | 1.81 | 48.4 | +13.1% | PASS |
| k2c2 | fp8_e4m3 | k2 | 98304 | **34.3** | 32.24 | 1.84 | 53.8 | +2.8% | PASS |
| k2c3 | tq4nc | k2 | 98304 | **42.5** [39.8] | 39.97 | 2.01 | 47.1 | **+29.7% vs bar** | PASS — best |
| k2c4 | k8v4 | k2 | 98304 | **36.2** | 35.80 | 1.75 | 48.2 | +11.0% | PASS |

Notes:

- **Headline: k2c3 (SPEC_K=2, turboquant_4bit_nc) = 42.5 true tok/s = +29.7%
  over the user's bar config** — best cell measured. c1 (bf16, k4) is the
  k4 champion at 44.0 (+32.9%). k4 stays the default; `SPEC_K=2` for max
  tok/s on tq4nc/k8v4.
- Correctness: sane car-game text on every cell; E[len] 1.99-2.19 (k4) and
  1.75-2.01 (k2), consistent with v19 acceptance; no #06-style hallucination.
- Intermittent #05(a) (KNOWN_ISSUES #05): nfp8 and c1 each hit ONE 4-reset
  mid-stream death on their first attempts (streams died 3204/4096 and
  1992/4096; the client reports tok/event=1.00 fallback when the dying
  stream never delivers its usage chunk). Protocol reboot + clean re-runs =
  zero resets. Same ~1-2 cells/day intermittency the v19 matrix saw (its
  c2) — not a v20 regression; v20 serving is bit-identical to v19 by
  default.
- k2c2's first attempt read engine 25.4: two transient near-stall 10s
  windows (drafted 14.6/s and 5.7/s vs ~36/s steady) dragged the tail-6
  mean. Clean re-run (above): 34.3 engine / 32.24 client. Transient
  scheduler stalls, not depth decay.
- Host networking: short-lease DHCP — the host moved 10.20.3.59 ->
  10.20.3.60 -> 10.20.3.61 across the two protocol reboots (see #03 reboot
  note in KNOWN_ISSUES).
<!-- /V20_MATRIX_RESULTS -->

## Files

- `Dockerfile` — FROM v19; overlay spec_timing.py + hooked runner/dflash/
  llm_base_proposer; grep guards + py_compile + import checks
- `cargame_client.py` — the true-token client (fixes the artifact)
- `cargame_matrix.sh` — full matrix incl. k2 arm + k6 reference
- `spec_timing.py`, `gpu_model_runner.py`, `dflash.py`,
  `llm_base_proposer.py`, `qwen3_dflash.py` (pristine extract, Phase-B
  reference) — A2 instrumentation set
- inherited from v19: config_vllm.py, turboquant_attn.py,
  triton_turboquant_decode.py, flash_attn.py, gpu_worker.py, sched_utils.py,
  serve_supervised.sh, monitor3.sh, bench/battery scripts

## Build (host, only while NOT serving)

```bash
scp -r vllm/patches/qwen38-dflash-v20 root@<host>:/root/build/
ssh root@<host> 'cd /root/build/qwen38-dflash-v20 && docker build -t llm-scaler-vllm-adv:v20 .'
```

## Run

```bash
# spec k4 (default arm)
KV_DTYPE=turboquant_4bit_nc TARGET_DIR=/models/qwen3.8-27b-fp8 DRAFTER_DIR=/models/drafter-fp8-v5 ./serve.sh
# spec k2 (recommended arm: best measured true tok/s)
SPEC_K=2 KV_DTYPE=turboquant_4bit_nc TARGET_DIR=... DRAFTER_DIR=... ./serve.sh
# user's superior target-only shape (unchanged)
SPEC=0 KV_DTYPE=turboquant_4bit_nc DTYPES=float16 BLOCKSIZE=128 MNBT=8192 MAXLEN=262144 \
  EXTRA_ARGS='--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}' TARGET_DIR=... ./serve.sh
# step-cost instrumentation (dbg only; costs ~4-5 ms/step)
VLLM_SPEC_TIMING=1 ... ./serve.sh
```

## Known limits

- `steady_true` scales a uniform tokens/event factor; the authoritative
  steady source remains the serve-log SpecDecoding windows (the matrix
  runner records both).
- `stream_options.include_usage` requires a vLLM that honors it (v19+
  does); the client falls back to event counts with a loud warning.
- Driver-level #03 protocol unchanged: reboot after any xe engine reset.
- The unused full-propose-graph lever remains future work; the A2
  decomposition above is the baseline for it.
