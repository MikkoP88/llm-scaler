# wedge-endgame-v27 — wedge endgame: drafter-comm isolation (neutral),
# oneCCL reduce kernel convicted at <=32k, canonical-grade validation standard

`llm-scaler-vllm-adv:v27` = validated v26 image + one overlay: the MTP
drafter's collectives can run on a **dedicated oneCCL communicator**
(`vllm/v1/spec_decode/drafter_comm.py`, env `VLLM_XPU_DRAFTER_PG`, default
1, `=0` = stock). The overlay wraps `LLMBaseProposer.propose()` /
`dummy_run()` and swaps the TP group's `device_communicator`/`device_group`
for a second GroupCoordinator (`drafter_tp`, same TP ranks, fresh
`new_group(backend="xccl")`), so the drafter's ~6 colls/step match on a
communicator only it uses. For no-spec serving the module is unreachable
(never called); measured neutral for spec (below).

## What tonight's arms established (2026-08-30/31, host 10.20.3.65)

Probe = 64-token decode window after a full prefill (the historical
"32k:3" battery). **Canonical** = the standing 4096-token car-game
generation — ~60x the per-step exposure of a probe. Wedge per-step
probabilities below are order-of-magnitude, inferred from both.

| arm | config | probe result | canonical |
|---|---|---|---|
| stock (v26) | MTP k4, FULL_DECODE_ONLY | 32k 2/3 WEDGE | — |
| v27 | + drafter dedicated comm | 32k **1/3 WEDGE** | — |
| v27+MALLreduce | + `VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1` (k4) | **32k 7/7 clean**, 65k 1/2 W, 131k 1/2 W | — |
| v27+nukesimd | + MALLreduce + every `DISABLE_ESIMD_*=1` | 65k 2/2 clean, **131k 1/2 W** | — |
| v27+k1+MALL | k=1 + MALLreduce | 32k/65k/131k/262k **7/7 clean** | **WEDGE @ token 141 (short ctx!)** |
| v27+eager | `--enforce-eager`, MTP k4, stock comm | (v26: 0/13 incl. 131k/262k) | **PASS x3**: short-ctx x2 (48.4 tok/s, html+canvas) + 32k-ctx PASS (4.85 tok/s) |

Conclusions:

1. **Drafter collectives are exonerated** — again. The dedicated
   communicator changed nothing (1/3 vs 2/3), exactly like `draft_tensor_
   parallel_size=1` (zero draft colls, still 1/3) in the v22-era matrix.
   The v27 overlay is kept as inert capability (`VLLM_XPU_DRAFTER_PG=0`
   restores stock bit-for-bit), not as a fix.
2. **The <=32k wedge class is the eager oneCCL `all_reduce` kernel.**
   `VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1` (allgather + local add — the same
   primitive the image's NaN-repair fallback uses; already in
   `xpu_communicator.py`, default off, never A/B'd against the wedge
   before) takes k4 @32k from 1-2/3 wedge to 7/7 clean. The image's own
   comment trail ("the only TP=2 op that runs ~80x/step ... documented
   history of both intermittent NaN output and wedging") fits.
3. **A second, graphs-x-spec residual survives every lever**: comm
   placement (W3), drafter comm (v27), all ESIMD kernels off, k=1,
   allreduce-via-allgather — each still wedges at ~1e-2/step somewhere
   (>=65k for k4 arms; any context for k1 at canonical exposure: 141
   tokens). Only removing graphs entirely (enforce-eager) zeroes it.
4. **Canonical-grade safe configs** (this is the bar that matters):
   - **no spec + FULL_DECODE_ONLY graphs** — canonical PASS, 0/10
     full-envelope probes, and the fastest >=65k decode (23.6 / 18.5 /
     12.7 tok/s @ 67/133/262k). Production default, unchanged.
   - **`--enforce-eager` + MTP k4** — 3x canonical PASS (48.4 tok/s
     short-ctx, +18% over nospec canonical 40.8; 32k-ctx canonical PASS at
     4.85 tok/s) + 0/13 historical probes incl. 131k/262k. The only spec
     configuration validated at canonical exposure. Slow at deep context
     (32k 4.85, >=131k 2.4-3.8 tok/s) — use for short-ctx latency, not
     deep context.
5. The historical probe-only "clean" verdicts (incl. last session's
   comm-out-of-graph <=67k and eager 0/11) must not be read as
   per-step-zero: 64-token windows under-expose by ~60x. The v22-era
   k=1 "7/7 @32k + canonical PASS" was a lucky draw by the same logic.

## Mechanism notes (updated)

- **Long-ctx canonical filler pitfall (KNOWN_ISSUES #13, re-confirmed):**
  a filler of 2048 IDENTICAL repetitions of one sentence makes the model
  emit EOS as its first token at ~32k ctx — greedy AND sampled (v27+eager:
  `finish_reason=stop`, 0 chars). A wedge HANGS and never returns; this
  returns instantly with empty text — misread here as a serve failure
  before the #13 connection was spotted. `canonical32k.py` uses numbered
  varied sentences (passed the content gate; if a future run returns
  instant-empty, suspect #13 collapse, not a wedge).
- Post-wedge py-spy (v27, after client disconnect): both workers idle in
  `worker_busy_loop -> shm_broadcast.acquire_read`, EngineCore idle on
  its input queue, GPUs 0% — the wedged request sits "Running: 1" with
  zero throughput until the client disconnects (running -> 0), then the
  engine serves normally. During the wedge the v22-era signature
  (Compute+Copy engines 100% @ 11-22% EU, host blocked in a submit) is
  the live state; the idle stacks are the post-mortem state. Both are
  consistent with a device-side collective/spin that never retires and
  takes the request with it.
- The oneCCL reduce conviction (#2) plus the surviving graphs-x-spec
  residual (#3) together replace last session's "captured-collective
  replay vs interleaved eager drafter colls" reading: the drafter side of
  that story is disproven (DTP1, v27), the reduce-kernel side is proven
  (MALLreduce), and what remains needs graph-replay-level debugging
  (allocator/replay interaction or a captured kernel with data-dependent
  termination) — still beyond vLLM-side configuration.

## Files

- `drafter_comm.py` — the overlay module (dedicated drafter GroupCoordinator
  + `with_drafter_communicator` decorator; capture-guarded, falls back to
  stock on any init failure or TP=1).
- `llm_base_proposer.py` — v26 in-image file + 3 hunks (import + decorator
  on `propose`/`dummy_run`).
- `Dockerfile` — FROM v26, COPY + guard greps + py_compile + AST check.
- `canonical32k.py` — 32k-context canonical car game (VARIED numbered
  filler — identical-repetition fillers make the model instant-EOS at 32k
  ctx, see mechanism notes; canonical sampling, 4096 tokens) — the
  canonical-grade long-ctx probe.
- `serve.sh` — unchanged from v26.

Wedge-chase probes (2026-08-31 session; all stream, all exit 7 on a
watchdog timeout = WEDGE):

- `wedge_repro.py` — short-ctx canonical loop (90 s watchdog); the control
  probe (10/10 clean on k1+VIA => hazard <1.7e-4/step short-ctx).
- `deep_repro.py` / `deep_repro_fox.py` — ~65k probes, varied-marker vs FOX
  filler (the historical rate-raiser), 128 tokens, 120 s watchdog.
- `long_exp.py` — 2048-token ignore_eos generation @ selectable ctx
  (65k/131k/262k dict); max per-request step exposure. NOTE its printed
  "tok/s" column is actually chunks/s (steps/s); true rate = tokens/wall.
- `conc_repro.py` — 2 concurrent streams (65k + 32k), the E2-bs2 historical
  rate-raiser; first stream to time out records the WEDGE.
- `fr_watcher.sh` / `frw3.sh` — HOST-side in-flight capture: on a 50 s
  /metrics stall, grabs xpu-smi (both GPUs), py-spy dumps of both TP
  workers, engine-log tail, and the v28dbg flight-recorder tails into
  /root/build/wedge_cap/. frw3.sh is the fixed restage (fr_watcher.sh
  staging collided with a directory once; use frw3).

## Build

```
cd vllm/patches/diagnostics/wedge-endgame-v27 && docker build -t llm-scaler-vllm-adv:v27 .
```

(No wheel; pure-python overlay on the v26 image.)

## Serve

- Production (unchanged recommendation): nospec + FULL_DECODE_ONLY —
  `bash /root/build/serve_boot_nospec.sh '' '' v27serve '' 512 llm-scaler-vllm-adv:v27`
- Safe spec (short-ctx latency): `bash /root/build/serve_boot_var.sh '' ''
  v27eager '--enforce-eager' 512 llm-scaler-vllm-adv:v27`
- Spec + graphs anywhere <=32k (k4) may add
  `VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1` — it removes the reduce-kernel
  wedge class but does NOT make graphs+spec canonical-safe.
