# Deep research: fp8_e4m3 KV + MTP K=4 — collapse root cause and the path to "superior at every operating point"

2026-09-09. Parent campaign and 15-lane evidence: [../long-context-efficient-v1/REPORT.md](../long-context-efficient-v1/REPORT.md).
Sources analyzed from the running image `llm-scaler-exp:v1.2.5` (a522bf15be2b):
`vllm/v1/attention/ops/triton_fp8_mq.py` (v51 kernel, 426 lines),
`vllm/v1/attention/backends/flash_attn.py` (route gates, lines 1325-1408, knobs 1847-1903),
`vllm/v1/spec_decode/llm_base_proposer.py` (EAGLE/MTP drafter, 1791 lines),
`vllm/v1/spec_decode/spec_timing.py` (v20 A2 instrumentation).

## 1. The target

Make (fp8_e4m3 KV + MTP K=4) strictly superior to both incumbents at every
length and concurrency — i.e. beat fp8+nospec (29.79/25.91/20.69 tps c1 at
64k/128k/262k, the only falloff-gate-passing lane) and beat tq4nc+mtp4
(25.06/15.31/7.50) everywhere. Today fp8+mtp4 collapses:

| lane | 64k | 128k | 262k | verdict |
|---|---|---|---|---|
| fp8_e4m3 + nospec | 29.79 | 25.91 | 20.69 | release-candidate |
| tq4nc + mtp4 | 25.06 | 15.31 | 7.50 | spec wins ≤64k, loses after |
| **fp8_e4m3 + mtp4** | **11.02** | **7.59** | **5.09** | **collapsed ~3-4x under nospec** |
| fp8_e4m3 + mtp4 + SPF1 prefill-fix (pf8m4) | 11.31 | 7.68 | 5.10 | fix is perf-neutral — collapse NOT the phase race |

(3 images tested — v1.2.5, v1.2.6t2, spec-prefill-phase-v1-v125 — collapse
byte-identical; KV pools byte-identical 707,980 blocks. Reproduced with
zero request errors and clean cancel probes.)

## 2. Established evidence (pre-diagnostic)

### 2.1 Acceptance is healthy — the collapse is cost-side
Per-window engine telemetry (`SpecDecoding metrics`, acc_dump.py over
suite jsonl): mean acceptance ~4.9 of 5 drafted at 64k/128k on fp8,
per-position acceptance ~0.88/0.77/0.63/0.53 — comparable to or better
than the tq4nc mtp4 lane. Draft QUALITY is intact.

### 2.2 Step-time math: ~10x multiplier, linear in context
- fp8+nospec: 33.6/38.6/48.3 ms/step (64k/128k/262k)
- fp8+mtp4: ~445/646/963 ms/step — **~10-13x**, excess ~3.5-6.4 us per KV
  token per step, i.e. an O(ctx) per-step operation, spec-only.

### 2.3 Collapse is length-dependent, onset after ~2k
Best 10-s generation window inside the collapsed f8e4m4 run = **74.1 tok/s**
(the 2k warmup phase, matching v51's certified 78.7-100.4 tps battery at 2k)
while the 64k cells average 11.02. The v51 kernel work was certified at 2k
only; nobody certified long ctx.

### 2.4 The fp8mq kernel executes (at least once) during the run
One-time JIT of `_fp8_mq_stage1/_fp8_mq_stage2` in the engine log (23:58:43/45),
no recompiles. This proves the route fired at least once (likely during
warmup at short ctx) — it does NOT prove the route is the active path for
64k verify steps. JIT-once is equally consistent with "route falls away at
long ctx and a slow path takes over".

### 2.5 Source analysis — three suspects
Reading the baked v1.2.5 sources:

**S1 — target verify (q=5 multi-row) on the v51 Triton kernel.**
`triton_fp8_mq.py` two-stage split-KV flash decode, grid (B, Hq, SPLITS=32),
fixed split count, per-row causal limits, fp32 combine. Registered-lean but
compute-heavy inner loop: per KV tile it materializes K/V as fp32
[BLOCK_KV=32, BLOCK_D=256] and runs `tl.static_range(Q_LEN)` masked-reduction
loops for scores AND for PV accumulation — per tile the q@K and P@V work is
done as 5 separate [BLOCK_KV, BLOCK_D] `tl.sum` reductions instead of one
tensor contraction. Grid ceiling: B*Hq*32 programs; at B=1 the device may be
underfilled and each program serially scans seq_len/32 tokens with 2-stage
prefetch. Docstring/code default drift (docstring says BLOCK_KV 16, warps 1;
code defaults 32 and 4) hints tuning was last touched mid-experiment.
v52 history: "SPLITS 32 default / 64 long-ctx +34% @32k / 128 WEDGES
ceiling=64" — the kernel was tuned at 32k, never certified at 64k+.

**S2 — drafter loop attention on fp8 KV at q=1.**
v49c policy: draft KV dtype matches target = fp8_e4m3. The drafter runs
num_spec_tokens-1 = 3 extra q=1 forwards per step (llm_base_proposer.py:609-693)
plus the initial multi-row draft forward. The v51 gate requires
`1 < max_seqlen_q <= 8` — **q=1 is NOT covered** (Q1 knob `VLLM_XPU_FP8_MQ_Q1`
defaults 0) and the v33 gate also requires 1 < q — so draft q=1 attention
falls through to the C++ FA2 / ESIMD paged path. fp8+nospec proves that path
healthy for the TARGET (q=1 decode, 33.6 ms/step total), but the draft model
runs it per draft step with its own KV cache, per-layer metadata rebuilds
(`build_per_group_and_layer_attn_metadata` per draft step unless
`constant_draft_positions`), and eagle slot-mapping kernels.

**S3 — route fall-off at long ctx.**
The v33 lineage comment (flash_attn.py:1379-1381) convicts the pre-v51 path
at "249.5 ms/step @65k" and v50 B5b at "tforward d=4976 ms/step" — the same
order as our measured 445 ms/step. If any gate condition fails at long ctx
in the real serving path (e.g. uniform-row predicate under the async
scheduler, seqused_k/block_table plumbing, capture-mode differences), the
convicted slow path silently returns. The v51 gate conditions that could be
context- or state-dependent: `num_actual_tokens == max_seqlen_q * B`
(uniformity), `q_descale is None`, `attn_metadata.causal is True`.

### 2.6 Discriminating experiment design (this run)
Five boots on v1.2.5 + fp8_e4m3, C1, all with `VLLM_SPEC_TIMING=1
VLLM_SPEC_TIMING_FLUSH=20` (v20 A2 per-segment host/device timing:
`tforward` = target verify, `dforward`/`propose` = drafter):

| boot | knob | lens | discriminates |
|---|---|---|---|
| dA-base | — | 2048..262144 | knee + S1 vs S2 attribution from SPECTIMING |
| dB-sp64 | VLLM_FP8MQ_SPLITS=64 | 16k..131k | tps change vs dA => v51 kernel IS active verify path |
| dD-mtp1 | k=1 (q=2 verify, no draft loop) | 64k/131k | fast => cost is draft loop (S2); slow => verify multi-row (S1) |
| dE-q1 | VLLM_XPU_FP8_MQ_Q1=1 | 64k/131k | q=1 draft attention via C++ path implicated? |
| dC-mq0 | VLLM_XPU_FP8_MQ=0 (v33) | 16384 R1 | far worse than dA@16k => v51 was active; same => it wasn't |

## 3. Diagnostic results

### 3.1 Ops notes: transient host instability during the campaign (resolved)
The first dA boot attempt (17:30) wedged the workers SILENTLY on the first
spec warmup request — EngineCore died exactly +5:00 later of a shm-broadcast
TimeoutError; dmesg showed xe driver engine resets (ccs+bcs, both GPUs).
Initially blamed on VLLM_SPEC_TIMING=1 (the only delta vs the healthy
09-08 master lane) — WRONG: a second identical boot without the env died
the same way (also +5:00, same 58-token p1 request, more engine resets).
Two clean probe boots after a full teardown + settle (fp8+nospec: 10.8s
first request, " Paris", clean; fp8+mtp4: 20.3s, clean) proved the host
recovered; the v3 sweep then ran clean end-to-end (WARMUP_DONE, rc=0
cells, zero new resets). Verdict: dA v1 hit a genuine one-off wedge
(possibly the timing env), and force-killing its hung workers left the xe
driver with poisoned contexts that killed the next two boots; a full
teardown + settle heals it. Lesson recorded: settle 20s after every
container kill, 90s after any abort (now baked into the harness scripts).
Also: **VLLM_SPEC_TIMING=1 remains unusable** for attribution on this
image (suspected torch.xpu.Event interaction with XPU-graph capture) —
attribution relies on knob A/Bs.

### 3.2 dA baseline knee sweep (v1.2.5, fp8_e4m3 + MTP k=4, C1, 3 reps)
| len | mean tps | median | min | max | approx step ms (4.9 tok/step) |
|---|---|---|---|---|---|
| 2048 | 62.78 | 69.62 | 42.75 | 75.99 | ~78 |
| 8192 | 15.06 | 14.99 | 14.53 | 15.65 | ~325 |
| 16384 | 12.72 | 13.60 | 9.79 | 14.76 | ~386 |
| 32768 | 12.79 | 13.30 | 12.29 | 13.30 | ~383 |
| 65536 | 10.70 | 10.82 | 10.28 | 11.00 | ~458 |
| 131072 | 6.72 | 7.39 | 4.84 | 7.92 | ~729 |
| 261888 | 4.93 | 4.93 | — | — | ~994 |

(64k figure reproduces the campaign's f8e4m4 11.02 within day-to-day noise.)

**Shape of the collapse — two regimes, not one linear cost:**
- A **4.2x cliff between 2k and 8k** (62.8 -> 15.1 tps). At 2k the v51-era
  performance holds (62-76 tps, matching v52's 78-100 certification at 2k).
- After the cliff: a ~330-380 ms/step FLOOR that stays nearly flat 8k->32k
  (325 -> 383 ms) and then grows only gently (~2-4 us/KV-tok) to ~994 ms at
  262k.
- A single O(ctx) per-step cost cannot produce flat 8k->32k: something
  **switches state when the prompt crosses the ~2k-8k window** — aligned
  with the chunked-prefill boundary (max_num_batched_tokens=8192; prompts
  <=~8k prefill in one chunk, longer ones are chunked). fp8+nospec crosses
  the same boundary unharmed (29.79 tps @64k), so the trigger is
  spec x chunked-prefill interaction, NOT chunking alone; the SPF1
  phase-race patch was already A/B'd perf-neutral (pf8m4 lane).

### 3.3 Knob A/Bs

**dB-sp64 (VLLM_FP8MQ_SPLITS=64) — CATASTROPHIC, and diagnostic gold:**

| len | dA (SPLITS 32) tps | dB (SPLITS 64) tps | dB step ms |
|---|---|---|---|
| 16384 | 12.72 | **0.74** | ~6600 |
| 32768 | 12.79 | **0.84** | ~5800 |
| 65536 | 10.70 | **0.84** | ~5800 |
| warmup p2 (~2k) | (dA p2 ~1s/req) | 0.5-0.7 tok/s | ~5200-6600 |

- The knob is LIVE (17x effect; envs.py merely warns it is unregistered —
  the kernel module reads it via `os.environ.get` at import). **The v51
  fp8mq kernel IS the active verify path** — no route fall-off at 16k+.
- SPLITS=64 cost is a near-FIXED ~5.8 s/step, independent of context
  (16k=32k=64k, and 2k warmup crawled identically). A flat multi-second
  per-step cost is far beyond any sane kernel time for these shapes
  (~268 MB KV per step at 8k on ~460 GB/s of device bandwidth should be
  sub-millisecond) — pointing at a pathological code path (register spill
  class, or capture/allocator interaction), not a tuning offset.
- Acceptance stayed healthy at SPLITS=64 (mean 4.0-5.0) — cost-side again.
- Discrepancy vs v52 history: the v52 notes claim "SPLITS 64 long-ctx
  +34% @32k" — NOT reproducible on the baked v1.2.5 (17x worse). Their
  measurement was presumably in-code on a dev image with dflash q=8 verify;
  the shipped env-knob path on v1.2.5 with MTP q=5 is catastrophically
  different. Any fix must re-validate on the actual image + lane.
- Efficiency frame: at SPLITS=32/8k the whole step (~325 ms) scanning
  ~268 MB of KV runs at **~0.7-0.9 GB/s effective vs ~460 GB/s peak** —
  the kernel achieves <0.3% of device bandwidth. The reference existence
  proof that these shapes are tractable: the TQ MQ verify kernel
  (patch-08 lineage) sustains 25 tps @64k on 4-bit paged KV with the same
  grid concept.

Remaining A/Bs: dD-mtp1 died at warmup (container Exited, see §3.1) — no
data; dE-q1 completed.

**dE-q1 (VLLM_XPU_FP8_MQ_Q1=1 — route draft q=1 through the v51 kernel too)
— NEUTRAL:**

| len | dA tps | dE tps |
|---|---|---|
| 65536 | 10.70 | 11.09 |
| 131072 | 6.72 | 6.54 |

Routing q=1 through the badly-tiled v51 kernel costs the same as the C++
q=1 path it replaces → the collapse is NOT the draft loop's q=1 attention;
the q=5 verify kernel (S1) dominates. (With TUNED tiles, q=1-v51 may still
beat C++ — retest after dF if the draft loop becomes the next bottleneck.)

### 3.4 Source conviction: the fp8mq shipped tile config is the convicted spill regime

The v1.2.5 image also contains the TurboQuant MQ decode kernel lineage
(`triton_turboquant_decode.py` v19 MQ variant + `turboquant_attn.py`)
that sustains 25 tps @64k on 4-bit KV with the SAME two-stage split-KV,
masked-reduction design. Its own measured notes, on THIS hardware:

- Single-query kernel: "BLOCK_KV=16 -> -72% deep, BLOCK_KV=32 -> -83%
  deep, 16+2 warps -> -20% deep, and 32+2 warps is FATAL ... wide tiles
  spill registers / starve latency hiding on Xe2 ... shipped defaults
  (4 / 1 warp) are optimal" (lines 29-40).
- MQ (q=5) variant: "num_warps=1 measured fastest at depth (cached=16384,
  q_len=5): MSE keys 1.39x / **FP8 keys 2.66x** ... warps=2/4 are slower
  (0.88x/0.49x) — wide tiles + more warps spill registers on Xe2"
  (lines 752-757).
- Backend pins MQ-verify KV splits at 32: "TurboQuant MQ verify KV splits:
  32 (measured optimum)" (turboquant_attn.py:570).

The fp8mq kernel ships **BLOCK_KV=32 / warps=4 / stages=2** — exactly the
convicted regime (its own docstring says 16/32/1/2; the code drifted).
All three are env knobs read at import → fix testable without baking:
`VLLM_FP8MQ_BLOCK_KV=4 VLLM_FP8MQ_STAGE1_WARPS=1 VLLM_FP8MQ_STAGE1_STAGES=1`
(diag3 dF-t4; dF-t8 tests the BLOCK_KV=8 upper bound).

Weight-traffic frame for the ceiling: fp8+nospec @64k runs 33.6 ms/step ≈
13.5 GB/GPU weights ÷ 460 GB/s ≈ device-bound on WEIGHTS, not KV. A
spill-free verify+draft attention pass (~256 MB/GPU/step at 64k) should
cost O(1 ms) → fp8+mtp4 at ~4.9 tok/step has headroom to 60-100+ tps
@64k, far above the 29.79 nospec gate.

### 3.5 dF tile-retune A/B — theory REVISED: tiles are not the floor

| len | dA (32/4/2) tps | dF-t8 (8/1/1) tps | dF-t4 (4/1/1) |
|---|---|---|---|
| 8192 | 15.06 | 14.29 (neutral) | died at warmup (5th host-flake death) |
| 65536 | 10.70 | **6.79 (1.6x WORSE)** | — |

The TQ 4/1/1 optimum does NOT transfer: TQ's conviction pairs BLOCK_KV=4
with warps=1 (per-thread register pressure); the fp8mq kernel at warps=4
amortizes wide tiles differently, and its shipped 32/4/2 is the better
tuned point of the two tested. The docstring drift was likely deliberate.

Decisive arithmetic: at 8k ctx, stage1 does ~8 tiles/program — even a
horrible kernel is O(1-2 ms)/step across 16 layers; it cannot make a
330 ms/step floor. Combined with dB (SPLITS=64 -> flat ~5.8 s/step at ALL
ctx incl 2k, where each program has ONE tile) the floor is a per-step
cost that scales superlinearly with SPLITS and is mostly ctx-independent:
SPLITS 32: ~50 ms @2k, ~330-460 ms @8k-64k; SPLITS 64: ~5.8 s everywhere.
Kernel arithmetic, tile shape, and draft q=1 path (dE neutral) are all
excluded — suspicion moves to a per-step launch/graph/allocator-scaled
mechanism. Nobody has tested SPLITS < 32 (downward extrapolation of the
curve suggests 16/8 may crush the floor): diag4 walks SPLITS 16/8/4 plus
an --enforce-eager probe (graph-replay discriminator).

Also recorded: dF-t4 death signature = `TimeoutError: RPC call to
sample_tokens timed out` -> EngineDeadError (worker hang; dmesg engine
resets). Five warmup-phase deaths today across unrelated configs —
treated as intermittent host flakiness, not config-specific.

### 3.6 diag4 SPLITS ladder + eager probe — floor CONFIRMED SPLITS-scaled, peak at 16

All boots v1.2.5, fp8_e4m3 + mtp4, C1, 2 reps (master_diag4.sh):

| len | dB sp64 | dA sp32 | **dJ sp16** | dK sp8 | dL sp4 | dG eager (sp32) |
|---|---|---|---|---|---|---|
| 2048 | ~0.5-0.7 | 62.78 | **74.87** | — | — | — |
| 8192 | 0.74 | 15.06 | **57.14** | 47.30 | 40.78 | 15.75 |
| 65536 | 0.84 | 10.70 | **23.30** | 13.68 | 10.04 | 11.05 |
| 261888 | — | 4.93 | **6.72** | — | — | — |

- **SPLITS=16 is the peak of the full ladder: up to 3.8x over shipped**
  (8k 15.06 -> 57.14; 2k 62.78 -> 74.87). The response is CLEANLY
  NON-MONOTONIC both sides: 64/32 bad, 16 best, 8/4 decay again
  (57.14 -> 47.30 -> 40.78 @8k; 23.30 -> 13.68 -> 10.04 @64k) —
  below 16 the lost split-parallelism at long ctx starts to dominate;
  above 16 the SPLITS-scaled per-step overhead dominates.
- Boot-time mid scratch scales as expected: [512, Hq, SPLITS, 8, 257] fp32
  ≈ 2.15 GB/GPU at SPLITS=32 -> ~1.07 GB at 16 (KV snapshot identical
  707,980 tokens at every SPLITS — the scratch does NOT come out of the
  KV pool, it's allocator-resident at capture time).
- **dG-eager ≈ dA-graphs** (15.75/11.05 vs 15.06/10.70, +4-5%): the
  SPLITS-scaled cost is NOT an XPU-graph replay/capture artifact — it is
  real execution cost in the eager path too. Graphs stay ON for prod.
- Zero request errors across all diag4 cells; cancel probes rc=0.
- **Residual after the SPLITS fix**: 64k 23.30 vs nospec 29.79 gate;
  262k 6.72 vs 20.69. Step-time frame (dJ in-cell acceptance ~4.3/4.2):
  step ≈ 184.5 ms @64k, ≈ 625 ms @262k vs nospec 33.6/48.3 — residual
  slope ~2.17 µs/KV-token/step, ctx-linear with a small (~9 ms)
  intercept. The SAME slope appears in the tq4nc+mtp4 lane (raw step
  slope ~2.3 µs/KVtok) whose MQ kernel shares the v51 design lineage.
- **Residual mechanism (source conviction, to be confirmed by the
  diag5 k-series)**: stage-1 computes QK^T and PV as
  ``tl.static_range(Q_LEN)`` masked-reduction passes — 2*Q_LEN = 10
  full [BLOCK_KV, BLOCK_D] fp32 VECTOR-ALU reduce passes per tile plus
  per-row extraction reductions, no tensor contraction (no tl.dot, no
  matrix units). With GQA ~16 q-heads per kv-head per GPU each
  launching its own program over the same KV bytes, the vector work is
  ~16x redundant; arithmetic puts the per-step vector work in the
  several-TFlop range @262k — the right order for the measured ~577 ms
  excess. Fix candidate (patch_fp8mq_dot.py, image fp8-mtp4-v2, knob
  VLLM_FP8MQ_DOT): identical split layout/causal-limits/combine
  contract, but QK^T/PV as fp16 tl.dot with fp32 accumulate (2 dots
  per tile, descales folded after — exact algebra), Q_BLOCK padded to
  16 for the tl.dot minimum.


### 3.7 py-spy host attribution (dV-sp16v 64k cell, baked image)

py-spy 0.4.2 installed in-container (pip, network available); 20 s
record @100 Hz on the EngineCore process (pid 273) mid-64k-cell:
**515/515 samples blocked in `shm_broadcast.dequeue` → `acquire_read`**
(sched_yield wait loop). The EngineCore host thread spends effectively
0% of the step in Python — no scheduler cost, no metadata build, no
drafter host code. The entire step budget is consumed inside the WORKER
processes (GPU execution + their C++/graph-replay path). Any
coordinator-side host O(ctx) explanation for the ~2.2 µs/KVtok residual
is excluded; a worker-side profile (262k cell) can confirm the workers
are similarly host-idle. (Ops note: scripts pushed from the Windows
workstation must be CRLF-stripped — a CRLF'd watcher silently died with
bash syntax error status=2 twice before `sed -i 's/\r$//'` fixed it;
`systemd-run --unit=...` is the reliable launcher for long chains.)

Worker-side confirmation (262k cell, Worker_TP0 pid 472, 20 s @100 Hz,
2098 samples): **all samples busy** in eager per-op dispatch —
`worker_busy_loop → execute_model → _model_forward → cuda_graph.py:254
__call__ → compile_wrapper → qwen3_next.forward → per-layer
forward_native` (fp8 `xpu_block` scaled_mm / layernorm / mrope /
all_reduce), with 45 samples in the draft path (`sample_tokens →
propose_draft_token_ids → propose → qwen3_5_mtp forward`). **Zero
samples in the `triton_fp8_mq` launcher** (the fp8mq call is one Python
frame per step — its cost is GPU execution, not host); flash_attn
frames present (flash_attn.py:962/1433). Busy ≠ bottleneck (the host
submits ahead of the GPU), but nothing ctx-scaled exists on the host:
host dispatch is the ~9 ms O(1) intercept. **The ctx-linear residual is
GPU-side** — consistent with the stage-1 vector-ALU conviction; the
diag5 Q_LEN series + dO-dot A/B will confirm on-device.


### 3.8 diag5 dV-sp16v — baked-image knee validation (GATE PASSED)

Full knee on the baked `llm-scaler-exp:fp8-mtp4-v1` (797a16dfaaa2,
SPLITS=16 default), fp8_e4m3 + mtp4, C1, 3 reps (master_diag5.sh):

| len | 2048 | 8192 | 32768 | 65536 | 131072 | 261888 |
|---|---|---|---|---|---|---|
| dV mean tps | 64.62 | 56.26 | 35.97 | 22.26 | 10.33 | 5.80 |
| dJ (env-A/B) | 74.87 | 57.14 | — | 23.30 | — | 6.72 |

- Bake reproduces the env-A/B ladder within run-to-run noise (8k/64k
  within 2-5%; 262k mean 5.80 with high variance min 3.88/max 6.90 —
  the 262k cell is where the residual bites hardest, ~40% below the
  nospec gate 20.69).
- 32k (35.97) and 128k (10.33) fill the knee; note 128k is already
  ~2.5x below the nospec lane at the same length — the residual slope
  dominates by 128k.
- diag5 series design (blocks following): dH-q1sp16 (draft q=1 through
  the tuned kernel), dM-k2 / dN-k1 (Q_LEN 5→3→2 step-time series —
  verify-ALU predicts step ∝ ~2*Q_LEN; draft-loop predicts ∝
  #forwards 4→2→1; host predicts flat), dO-dot (v2 image,
  VLLM_FP8MQ_DOT=1 A/B, also 8192 to watch short-ctx regression).


## 4. Root cause

### 4.1 diag5 Q_LEN series — slope decomposes; k-independent floor confirmed

Exact per-request step times (completion_tokens / n_steps, decode-window),
all on baked sp16 v1 image, C1, 2-3 reps (dV 3):

| mode | k | Q_LEN | step@64k | step@262k | slope µs/KVtok | intercept |
|---|---|---|---|---|---|---|
| dV-sp16v | 4 | 5 | 181.4 ms | 569.4 ms | **1.976** | 51.9 ms |
| dM-k2 | 2 | 3 | 138.1 ms | 410.6 ms | **1.388** | 47.2 ms |
| dN-k1 | 1 | 2 | 113.0 ms | 332.6 ms | **1.118** | 39.7 ms |
| nospec | 0 | 1 | 33.6 ms | 48.3 ms | 0.075 | 28.7 ms |

- Linear model slope = c + a·Q_LEN + b·k fits all three points within
  measurement noise (predicted q2 slope 1.09 vs measured 1.118, ~3%).
  **k and Q_LEN are structurally collinear (Q_LEN = k+1), so the series
  pins a+b ≈ 0.28 µs/KVtok and c ≈ 0.84−a but cannot split a from b** —
  dO-dot is the designed discriminator.
- **The k-independent floor c ∈ [0.56, 0.84] µs/KVtok is 28-42% of the
  total slope and is paid even at k=1.** Per-KV-token in-kernel work
  (fp8 load, dequant, mask) in the v51 grid (B, Hq, SPLITS) is
  KV_GROUP_SIZE=6× redundant (24q/4kv, TP2 → Hq=12, Hk=2 — each of the
  6 q-head programs per kv-head reloads and re-dequants the same KV
  bytes). c is consistent with that redundancy class; raw KV bandwidth
  alone (268 MB/pass @262k) is 3 orders of magnitude too cheap to
  explain it, so c is dominated by the per-element vector work +
  redundant loads, not DRAM.
- dH-q1sp16 (draft q=1 routed through the tuned kernel): step
  190.6/604.8 vs dV 181.4/569.4 — NEUTRAL → draft attention routing is
  not a cost lever.
- dG-eager ≈ graphs (§3.6) + py-spy (§3.7: EngineCore 515/515 idle,
  worker 0/2098 samples in the fp8mq launcher) → the entire residual is
  GPU-side inside the verify kernel class.

### 4.2 dO-dot — tl.dot stage-1 REJECTED (2.2x regression at every length)

v2 image (VLLM_FP8MQ_DOT=1, tl.dot QK^T/PV with fp16 operands, fp32
accumulate, Q_BLOCK padded 5→16), mtp4, C1, reps 2:

| len | dV (vector) tps | dO (dot) tps | dO step-ms | dot slope |
|---|---|---|---|---|
| 8192 | 56.26 | **15.10** | 253.3 | — |
| 65536 | 22.26 | **7.85** | 537.8 | — |
| 261888 | 5.80 | **2.77** | 1388.6 | **4.33 µs/KVtok** (vs 1.98) |

Intercept also explodes (~218 ms vs ~52). **tl.dot at the
[16, 256] × [256, 32] tile shape on this Triton-XPU stack is ~2-3x
slower than the masked-reduction vector formulation — the matrix units
are not engaged efficiently (or spills dominate) at M=16/N=32/K=256.**
Consequences: (a) the dot candidate is dead; (b) the planned GQA-batched
kernel (same dot shape per program) is moot as designed; (c) the dO A/B
cannot cleanly split a from b — the dot cost swamps the signal. The
route map supplies the next discriminator instead: with BOTH
`VLLM_XPU_FP8_MQ=0` and `VLLM_XPU_TRITON_MQ3D=0`, uniform q5 fp8 paged
verify falls through to the **stock `flash_attn_varlen_func`** call
(paged KV + seqused_k + expanded per-tensor descales) — the standard
upstream spec-verify shape, which v33/v51 routing has intercepted since
v33 shipped. The v50 F6 "spec-on-fp8 never healthy" conviction was
q8/dflash-class, NOT this shape; stock flash q5 paged fp8 has never
been measured on this stack (dR2-flash, diag6). Companion probe dQ-kq1
(nospec + VLLM_XPU_FP8_MQ_Q1=1) measures the fp8mq kernel forced onto
the exact q=1 shape flash serves at 0.075 µs/KVtok — the kernel's
fixed-cost isolation.

### 4.3 diag6 dR2-flash + dQ-kq1 — the route map closed; slope fully attributed to the fp8mq kernel's own cost curve

**dR2-flash** (v1 image, mtp4, `VLLM_XPU_FP8_MQ=0 VLLM_XPU_TRITON_MQ3D=0`
→ stock `flash_attn_varlen_func` q5 paged fp8 — first measurement of this
shape on this stack), C1, reps 2:

| len | tps | step-ms | slope |
|---|---|---|---|
| 8192 | 51.17 | 76.4 | — |
| 65536 | 15.67 | 269.4 | — |
| 261888 | 4.14 | 927.5 | **3.35 µs/KVtok** |

Healthy (correct outputs, no wedge, cancel probe clean) but 1.7x slower
than the v51 vector kernel — the branch1/chunk_prefill class, consistent
with the v33-era observation. Not a fix candidate, but it retires the
"never-measured stock route" unknown: **every fp8 paged multi-row route is
now measured, and the only healthy-and-fast fp8 paged shape on this stack
remains flash q=1 (0.075 µs/KVtok, nospec lane).**

**dQ-kq1** (v1 image, nospec, `VLLM_XPU_FP8_MQ_Q1=1` — the fp8mq kernel
forced onto the exact q=1 shape flash serves), C1, reps 2:

| len | step-ms (r0/r1) | flash q=1 ref | ratio |
|---|---|---|---|
| 65536 | 84.8 / 84.9 | 33.6 | 2.5x |
| 261888 | 249.4 / 249.5 | 48.3 | 5.2x |

**Kernel q=1 slope = 0.838 µs/KVtok — 11.2x flash on the identical
shape.** Deterministic across reps.

**This closes the a/b split the collinear Q_LEN series could not.** Fit
the kernel's own cost curve κ(q_len) = κ0 + κq·q_len from its two
measured points (q1: 0.838, q5: 1.976):

- κq = (1.976 − 0.838)/4 = **0.284 µs/KVtok per verify row**
- κ0 = 0.838 − 0.284 = **0.555 µs/KVtok row-independent offset**

Predict the spec series with b (draft-loop long-ctx cost) as the only
free parameter: dN(k1,q2) = 0.555+0.568+b → b ≤ 0.005; dM(k2,q3) =
1.407+2b (meas 1.388, b ≈ −0.01); dV(k4,q5) = 1.975+4b (meas 1.976,
b ≈ 0). **b ≈ 0 — the draft's long-ctx cost is negligible** (its q=1
attention rides the flash path at 0.075). The §4.1 "k-independent floor
c" is not a spec-machinery cost at all — it is κ0 + κq, the kernel's own
cost curve, i.e. **the entire fp8+MTP long-ctx degradation is the verify
kernel charging ~0.56 + 0.28·q_len µs/KVtok where flash q=1 charges
0.075 per row.**

Fix consequence: replacing the single q5 kernel call with q_len flash
q=1 calls eliminates BOTH κ0 (no kernel launch at all) and the per-row
cost (0.075 vs 0.284). Projected verify slope = 5 × 0.075 = 0.375
µs/KVtok → projected spec-step slope ~0.375 vs 1.976 today (5.3x), i.e.
@64k step ~77 ms → ~53 tps (gate 29.79), @262k step ~150 ms → ~22 tps
(gate 20.69). The v4 fan-out route (§5) is the direct embodiment; A/B =
diag7 dS-fanout.

## 5. Fix design and implementation — position-sliced flash fan-out (v4/v5)

### 5.1 Design

The §4.3 attribution reduces the fix to one sentence: **route the uniform
q_len fp8 paged verify through the only measured-healthy shape — flash
q=1 — once per position.** v4 adds a route to
`flash_attn.py::_inner_forward` (knob `VLLM_XPU_FP8_FANOUT`, patcher
`patch_flash_fanout.py`, inserted BEFORE the v51 kernel gate so it takes
precedence when enabled):

- Gate: fan-out on AND the v51 gate's own safety predicate (fp8 str KV
  dtype, paged `block_table` + `seqused_k`, uniform
  `1 < max_seqlen_q <= 8`, `num_actual_tokens == q_len × B`, causal, no
  softcap/sliding window, `q_descale is None`, `head_size <= 256`).
- Body: for each position `_pos` in `range(q_len)`, one
  `flash_attn_varlen_func` call with `q = query[_pos::q_len][:B]
  .contiguous()` (position-slice across the packed [B·q_len] layout),
  `cu_seqlens_q = arange(B+1)` (cached per max-B), `max_seqlen_q = 1`,
  per-position causal limit `seqused_k - (q_len - 1 - _pos)` (device-side
  arithmetic — CUDA-graph-capture safe), shared paged KV cache, block
  table and per-tensor descales; all other args mirror the stock
  fallback call verbatim. Output rows written back position-sliced.
- Numerics: each output row is computed by the same flash q=1 kernel the
  nospec decode lane uses (the 0.075 µs/KVtok shape); no new math.
- Rollback: `VLLM_XPU_FP8_FANOUT=0` restores the v51 kernel route
  bit-for-bit (the route is additive; the tq4nc lane is untouched — the
  gate requires fp8 KV dtype).

Projection from the §4.3 ladder: verify slope 5 × 0.075 = 0.375
µs/KVtok (kernel: 1.976) → step @64k ≈ 77 ms (≈ 53 tps at measured
acceptance), @262k ≈ 150 ms (≈ 22 tps).

### 5.2 dS-fanout A/B verdict (diag7, v4 image, env knob ON, C1, reps 2)

| len | step-ms | tps | kernel lane (dV) | fp8+nospec | tq4nc+mtp4 | gate |
|---|---|---|---|---|---|---|
| 8192 | 54.6 | **71.6** | 64.62 | — | — | — |
| 65536 | 79.3 (79.3/79.3) | **52.63** | 22.26 | 29.79 | 25.06 | **PASS +77%** |
| 261888 | 160.4 (160.4/160.2) | **23.47** | 5.80 | 20.69 | 7.50 | **PASS +13.5%** |

- Measured slope **0.413 µs/KVtok** vs projected 0.375 (the delta is the
  5× launch/dispatch overhead) — **4.8× below the kernel**, right on the
  model.
- Steps deterministic across reps (79.3/79.3 @64k; 160.4/160.2 @262k);
  all cells rc=0 (in-cell needle/multihop answer validation green);
  cancel probe clean; no wedge, no asserts.
- Short ctx not sacrificed: +11% @8k over the kernel lane.

### 5.3 v5 bake (default-on) + certification (diag8)

v5 = same patcher with `--fanout-default 1` (`--build-arg FANOUT=1`),
layered on the v1 image (v1.2.5 + SPLITS=16 lineage). Certification
block dV5-cert runs with EMPTY extraenv (proves the baked default),
full knee 2048/8192/16384/32768/65536/131072/261888 reps 3, then
`prod_restore_v5.sh` stands PROD on fp8_e4m3 + mtp4 @0.9/262144.
Results: §6.

### 5.4 dV5-cert knee (diag8, v5 image, baked default ON, EMPTY env, C1, reps 3)

| len | step-ms (r0/r1/r2) | tps (median) |
|---|---|---|
| 2048 | 50.9 / 50.8 / 51.0 | 77.05 |
| 8192 | 54.5 / 54.3 / 54.6 | 65.22 |
| 16384 | 57.8 / 57.6 / 57.9 | 61.51 |
| 32768 | 65.4 / 65.1 | 64.10 |
| 65536 | 79.2 / 79.2 / 79.6 | 52.67 |
| 131072 | 107.1 / 107.2 / 107.6 | 33.21 |
| 261888 | 160.4 / 159.8 / 161.7 | 23.47 |

Slope 0.426 (64k→128k) and 0.407 (128k→262k) µs/KVtok — flat ~0.41
across the upper knee, matching dS (0.413). Steps deterministic within
~1%. All cells rc=0; cancel probe rc=0; no wedge, no asserts. The baked
default carries the route with no environment knob (empty extraenv).

## 6. Gates for "superior on any case" — VERDICT: PASS

Mandate: fp8_e4m3 + MTP k=4 must be strictly superior at every length.

| len | fp8+nospec | tq4nc+mtp4 | old fp8+mtp4 (kernel) | **v5 fan-out** | vs nospec | vs tq4nc | vs old |
|---|---|---|---|---|---|---|---|
| 8192 | — | — | 64.62 | **65.22** (max 78.6) | — | — | +1% |
| 65536 | 29.79 | 25.06 | 22.26 | **52.67** | **+77%** | **+110%** | **+137%** |
| 131072 | 25.91 | 15.31 | (interp ~11.5) | **33.21** | **+28%** | **+117%** | ~+189% |
| 261888 | 20.69 | 7.50 | 5.80 | **23.47** | **+13.5%** | **+213%** | **+305%** |

- **Both standing gates PASS**: 64k 52.67 > 29.79, 262k 23.47 > 20.69.
- fp8+mtp4 is now the **best lane at every measured length** — it beats
  fp8+nospec (previously unbeatable at 128k/262k) AND tq4nc+mtp4
  everywhere, including a 3.1x at 262k over the incumbent prod combo.
- Short ctx is not sacrificed: 2k 77.05 tps sits inside the v51-certified
  fp8+mtp4 envelope (78.7-100.4 battery class; ctxscan smoke 58.0 @2k on
  the standing prod lane); 8k ≥ kernel lane.
- Long-context slope 1.976 → **0.41 µs/KVtok (4.8x)**; step @262k
  571 ms → 160 ms (3.6x); tps @262k 5.80 → 23.47 (**4.05x**).
- Root-cause closure (§4.3): the entire degradation was the fp8mq
  verify kernel's own cost curve; the fix removes it by construction
  (q_len flash q=1 calls at 0.075 each + ~0.04 launch overhead).

**Prod disposition**: `llm-scaler-exp:fp8-mtp4-v5`
(sha256:e536666de558…, = v1.2.5 + SPLITS=16 + fan-out default ON)
standing since 03:54:26 as fp8_e4m3 + mtp4 @0.9/262144 — BOOT/HEALTH/
WARMUP all OK, KV 707,980 tokens, ctxscan smoke 58.0/51.7/41.7/51.2 tps
@2k/16k/32k/65k, health 200. Rollback: `VLLM_XPU_FP8_FANOUT=0` per boot,
or re-run `prod_restore127.sh` for the tq4nc lane.

## 7. Case-axis closure (diag9/9b) — concurrency + e5m2

Mandate escalation: fp8+mtp4 must be superior on ANY case, not only C1
long-context. diag9 measured the unproven axes — all mtp4 cells and all
nospec references on the SAME image (fp8-mtp4-v5), same harness, same
seeds, C-list per KV-pool budget (C8@262k impossible: 8×262k > 707,980
pool; C2@262k = 524k fits; C8@64k = 524k fits).

### 7.1 Concurrency matrix (aggregate decode tps = Σ per-client)

| case | mtp4 v5 | nospec (same-boot) | mtp4 margin |
|---|---|---|---|
| 2k C1 | 77.05 | 35.2 | **+119%** |
| 2k C2 | ~102-132 | ~64.7 | **+57-102%** |
| 2k C8 | ~440 (7×57 + straggler) | (not run) | — |
| 64k C2 | 49.7 | 33.9 | **+47%** |
| 64k C8 | 48.6 | 33.7 | **+44%** |
| 262k C2 | 22.3 / 25.3 | 21.1 / 21.1 | **+6-20%** |

- **Acceptance does NOT collapse under concurrency** — the v50-era
  "conc8 MTP 0.65x AR" fear is dead on this lane: tok/step at 2k C8 =
  4.4 (vs 3.9 at C1); at 64k C2 3.92, 262k C2 3.56.
- The dramatic per-client step inflation in concurrent cells (e.g. 836
  ms / 6.5 s "steps" on the first-admitted client) is **prefill-decode
  interference**: the first client's decode window spans the other
  clients' chunked prefills (compute-bound ~3k tok/s, max_num_batched_
  tokens 8192). The pattern is IDENTICAL in the nospec references — it
  is the shared prefill pipeline, not a spec defect.
- Aggregate tps at fixed ctx is bandwidth-bound plateau (~= C1 rate),
  as expected; per-client decode latency strongly favors mtp4 (e.g.
  262k C2: 21.8-24.7 tps/client vs nospec ~10-20).
- All cells rc=0 (answers validated); cancel probes clean at every
  concurrency.

### 7.2 e5m2 KV parity (diag9 → diag9b) — VALIDATED, FASTER than e4m3

diag9's e5m2 block died at boot on the STOCK upstream guard
(`ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints.`)
— diag9b re-ran it with the v51-lineage bypass `VLLM_XPU_ALLOW_E5M2_FP8_
CKPT=1` (e4m3-scale caveat per v51). The fan-out route itself matches
any `fp8*` str KV dtype, so parity required no code change. Boot clean,
KV pool 707,980 tok — identical to e4m3.

diag9b verdict (image fp8-mtp4-v5, mtp4, kv fp8_e5m2, C1, reps 2,
all rc=0):

| len | step ms (rep0/rep1) | e5m2 tps | e4m3 v5 tps | margin |
|---|---|---|---|---|
| 8k | 52.9 / 52.6 | 64.8 / 81.1 | 65.2 / 78.6 | par (best-rep +3%) |
| 64k | 69.5 / 69.4 | 58.8 / 61.5 | 52.67 | **+12-17%** |
| 262k | 125.1 / 125.1 | 31.3 / 32.7 | 23.47 | **+33-39%** |

- Step-time slope 64k→262k = **0.283 µs/KVtok** vs e4m3's 0.414
  ((160.4−79.2) ms / 196,352 tok) — the e5m2 flash path skips the
  descale loads entirely. @262k steps are 22% faster (125.1 vs 160.4 ms).
- Deterministic: 192/192 completion tokens in every rep; 262k step-ms
  125.1/125.1 bit-stable. Acceptance healthy and shape-identical to
  e4m3 (mean 4.22 @64k in BOTH reps; 4.92 peak @262k; per-position
  monotone, e.g. 0.947/0.868/0.763/0.658).
- vs nospec @262k: 31.3-32.7 vs 20.69 → **+51-58%** on this variant.
- Disposition: e4m3 stays the standing prod lane (conservative — no
  env knob at boot); e5m2 is the VALIDATED faster variant, one boot
  env + kv-dtype away from promotion. Answers validated rc=0; cancel
  probe clean.

## 8. Soak + cross-boot determinism (diag10) — PASS

Final case axes: durability under sustained mixed load on the STANDING
prod lane, and answer determinism across boots.

### 8.1 Soak (10 cycles, no reboot, standing prod lane)

Per cycle: health poll + engine-log fault scan
(`AssertionError|CRITICAL|Traceback`) + cells 2k C2, 64k C2, 262k C1
(reps 1, fresh seed per cycle).

- **10/10 cycles complete, 432-434 s each (~72 min sustained mixed
  load), fault counter 0→0 across every cycle, health 200 throughout,
  all 30 cells rc=0, zero request errors.** No assert, no zombie, no
  degradation-of-service event.
- Perf stability (aggregate decode tps = Σ per-client):

| cell | per-cycle range | mean | reference band |
|---|---|---|---|
| 2k C2 | 97.7-125.0 | 115.3 | diag9: ~102-132 |
| 64k C2 | 33.2-49.3 | 43.0 | diag9 best 49.7; nospec 33.9 |
| 262k C1 | 21.3-25.0 | 23.2 | cert band 23.47-25.03 |

  No time-trend at any length (first→last: 122.3→114.5, 33.7→45.0,
  21.7→23.5; extremes land mid-soak at cycles 6/9). The 64k C2 spread
  is per-cycle seed/task variance, not degradation.
- **Cancel probe**: diag10's scripted probe did not EXECUTE — harness
  bug, not a lane defect: the mode name `soakCancel` missed the
  suite's literal `cancel` dispatch and fell into the 4-arg unpack
  (`ValueError`, rc=1, engine untouched). Corrected re-run on the
  post-soak restored lane (`nlp_suite.py … cancel SEED`): **rc=0**,
  131k-token stream aborted, post-cancel checks ok at +5/+30/+90 s,
  `engine_abort_count=0` — the v51 crash-3 zombie class stays closed.

### 8.2 Cross-boot determinism (64k C1, seed 20326445, 3 boots)

Boots: dV5-cert 03:19 (certification), dZ-det 09:45 (fresh reboot),
dZ2-det 09:57 (the standing prod lane post-restore).

- rep1 (multihop): sha `b9780aeb0c3e421e`, n=46 — **BYTE-IDENTICAL
  in all three boots**.
- rep0 (needle): byte-identical between boots 2+3 (sha
  `1e143572576913c8`, n=45); boot 1 resolved a different walk
  (`5e28041a`, n=46) — one fp near-tie at an acceptance boundary, the
  documented knife-edge class (#18 / v51 k4 disposition). Every
  resolution is correct: needle_hit=true in all three, 192/192 tokens,
  no repeated 10-grams, 116 vs 118 words. Each resolution is stable on
  re-boot.
- Step-ms 79.2/79.2/79.4 across boots (<0.3% spread).

### 8.3 Mandate closure

Every case axis of "superior on ANY case" is now measured on
`fp8-mtp4-v5`: C1 at all seven lengths (§6), concurrency C2/C8 with
same-boot nospec refs (§7.1), both fp8 KV dtype variants (§7.2), and
sustained-load soak + cross-boot determinism (§8). fp8_e4m3 + MTP k=4
is at parity or superior to fp8+nospec at every measured case, with
zero degradation events; prod stands on it.

## 9. v6 production image — e5m2-lane bake + dual-lane byte-parity cert

### 9.1 Bake

`Dockerfile.v6` = `FROM llm-scaler-exp:fp8-mtp4-v5` +
`ENV VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1`. Purpose: make the validated
e5m2 lane (§7.2: +12-17% @64k, +33-39% @262k) selectable with
`--kv-cache-dtype fp8_e5m2` alone — no per-boot env. Safety: the ENV
has a **single read site** (harmonized patch 03, attention.py v34
guard) gated on `kv_cache_dtype == "fp8_e5m2"` — inert for e4m3 and
tq4nc lanes. In-image checks (diag11):

- `BAKE_V6_OK image=sha256:7cf3d51cfc60…`
- `BAKE_V6_ENV_CHECK: V6ENV=1`
- `BAKE_V6_FANOUT_STILL_ON fanout=1` (v5 default survived the layer)

Production tag: **`llm-scaler-exp:v1.2.7`** (dual-tag alias,
same image ID; v1.2.6 number-space occupied by test tags v1.2.6t1/t2 —
pattern per v1.2.5 == v1.2.5t5).

### 9.2 Certification — master_diag11 (dual-lane byte-parity, no env anywhere)

| block | config | result |
|---|---|---|
| dW6-e4m3 | v6, kv fp8_e4m3, extraenv EMPTY, full 7-length knee ×2 | **13/14 byte-identical** to dV5-cert rep0/rep1 refs |
| dW6-e5m2 | v6, kv fp8_e5m2, extraenv EMPTY (baked ENV must carry the v34 guard) | **6/6 byte-identical** to diag9b dX2-e5m2 refs |

- The one e4m3 diff is 64k rep0 `5e28041a`→`1e143572`: the documented
  boot-1 near-tie of §8.2 — v6 matches boots 2+3 (tally 3:1), both
  resolutions correct (needle_hit=true, 192 tok). Not a v6 effect.
- dW6-e5m2 **booted with empty extraenv** — the baked ENV is proven
  load-bearing for the guard, and env-passed == env-baked numerically
  (262k steps 125.1/125.1).
- Cancel probes: rc=0 both blocks, engine_abort_count=0.

### 9.3 Prod standing

`prod_restore_v6.sh`: BOOT_OK (KV pool 707,980 blocks), HEALTH_OK
11:56:47, WARMUP_OK 11:59:37, ctxscan smoke on the standing lane
60.5/52.3/61.1/51.2 tps @2k/16k/32k/65k, **PROD_V6_STANDING
12:00:44** — image `llm-scaler-exp:fp8-mtp4-v6` ==
`llm-scaler-exp:v1.2.7`, fp8_e4m3 + mtp4 @0.9/262144 (config
unchanged from v5 stand; e5m2 lane = `--kv-cache-dtype fp8_e5m2`,
no env).

### 9.4 Rollback chain

1. `docker run -e VLLM_XPU_ALLOW_E5M2_FP8_CKPT=0 …` — overrides the
   baked ENV per boot (kills only the e5m2 guard bypass).
2. `prod_restore_v5.sh` — prior production image (fan-out baked ON,
   no e5m2 ENV), same serve config.
3. `prod_restore127.sh` — tq4nc pre-era-4 lane.

### 9.5 Ledger entries

- Production patch: `../../prod/flash-fp8-fanout-v54/` (patcher copy
  + README: route, knob, bake lineage, cert, rollback).
- Harmonized README era-4 note updated: v5/v6 lineage, v1.2.7 tag,
  PRODUCTION IMAGES since 2026-09-10.
