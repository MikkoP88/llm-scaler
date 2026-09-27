# v89 — Decode-speed deep analysis + token-generation maximization plan

**Date:** 2026-09-27 · **Lane:** `lsv-test` on ainode01 (10.20.3.65:8000), image `llm-scaler-exp:v1.2.21`
**Posture:** read-only evidence collection (no instance touched, no container/serve/host modification).
**Task:** "run deep analyze current vLLM statics and why there is drop of token generations speed, do comprehensive plan to max out token generation speed configuring env and parameters and code."

---

## 1. Headline verdict

**The engine did not regress. The "drop" is demand arithmetic meeting a fixed aggregate ceiling.**

1. The GPU pair delivers a **fixed aggregate decode ceiling of ~82 tok/s** regardless of stream count.
2. Current load runs **7–8 concurrent streams** (CC fleet traffic) → each stream gets **~10–15 tok/s** by simple division (82 / 7.2 ≈ 11.4).
3. The same hardware delivers **82 tok/s solo** (23 steps/s, ~43 ms/step). Nothing broke — the bandwidth got shared.
4. Engine health is clean: **0 engine resets, 0 GPU faults, MTP acceptance healthy (84% pos0), prefix cache hitting 61%**, watchdog/f15b/v66 guards intact and quiet.
5. Live GPU telemetry proves the bound: **EU active 16% / EU stall 57%, memory read ~300 GB/s (53% of ~573 GB/s peak), engines 99.6% busy** — decode is memory-latency/bandwidth-bound, not compute-bound, and per-step cost scales ~linearly with decode batch (near-zero cross-stream amortization on this stack).

The plan below therefore works three axes: **(a) raise the ceiling where cheap** (step-cost levers), **(b) reallocate the ceiling toward decode latency** (scheduler knobs), **(c) shape demand** (concurrency/context), plus one gated unknown-upside lever (barrier-off validation leg per `PATCH_STACK_ANALYSIS.md` §3.6).

---

## 2. Evidence base (all measured 2026-09-27, lane live)

### 2.1 Throughput window (engine /metrics, 90 s double-sample)

| Metric | Value | Derived |
|---|---|---|
| Generated tokens | 7360 / 90 s | **81.8 tok/s aggregate** |
| Decode iterations | 257 / 90 s | 2.86 steps/s → **349 ms/step** |
| Prompt tokens (total) | 62071 / 90 s | 689.7 tok/s prefill load |
| — computed | 24183 | 268.7 tok/s computed |
| — cache-hit | 37888 | **61.0% prefix-cache hit in-window** |
| Draft tokens | 1840 / 90 s | effective decode batch ≈ 7.2 (matches Running 7–8) |
| Tokens / iteration | 28.6 | ≈ 3.99 per seq·step (spec verify included) |
| KV cache usage | 42.8% → 52.4% | headroom ample |

### 2.2 Lifetime latency profile (337 requests since boot 10:55Z)

| Metric | Value |
|---|---|
| Mean TTFT | **40.6 s** = 18.7 s queue + 18.8 s prefill |
| Mean decode duration | 107.2 s |
| Mean inter-token latency | 231 ms/step ≈ **65 ms/token ≈ 15 tok/s per stream** |
| Avg prompt length | **38.1k tokens** (CC context replay) |
| Avg generation | 1739 tokens; avg max_tokens 42.9k |

### 2.3 Speculative decode (MTP ×4) — healthy, not the cause

- In-window acceptance: **75.3%** (5524/7334)
- Lifetime per-position: **pos0 84.0% / pos1 68.7% / pos2 56.4% / pos3 47.1%**
- E[tokens per draft·verify] ≈ **3.56 of 5 max** — good for MTP on this model

### 2.4 Step-time vs batch (the core physics)

| Decode batch | Step time | Aggregate | Per stream |
|---|---|---|---|
| 1 (solo, Sep 25 series) | ~43 ms | ~82 tok/s @ 23 steps/s | **82 tok/s** |
| 7.2 (today) | ~349 ms | 81.8 tok/s @ 2.86 steps/s | **~11.4 tok/s** |

Step time scales ~linearly with batch (≈8× time for ≈7× batch — slightly worse than linear). Weight reads alone cannot explain linearity (they are batch-invariant); the incremental per-stream cost is **per-sequence memory traffic**: GDN/mamba state ops, KV reads over ~38k-token contexts, activations, plus 5 forward dispatches per iteration (4 draft + 1 verify) under TP2 sync.

### 2.5 Live GPU telemetry (xpu-smi, correct syntax `--metrics UTILIZATION,MEMORY --number N`)

GPU0 under load: utilization.compute 99.6% / render 99.7% / memory 99.6%; **EU active 16%, EU stall 57%, EU idle 27%**; memory read **287–305 GB/s = 50.1–53.1% of peak (~573 GB/s implied)**, write 25–26 GB/s; memory used 32536/32656 MiB.

Interpretation: engines are never idle (sync scheduler keeps them fed) but the compute units stall on memory — classic **bandwidth/latency-bound decode**. The 57% EU-stall share also indicates kernel-granularity inefficiency (small kernels, launch/sync overhead) — that is where code-level headroom lives.

### 2.6 Host/GPU health

- dmesg: no GPU faults/resets (only kernel perf sample-rate lowering); engine resets 0
- V66 FAIRFIX fired once lifetime (Sep 25, bypass=1 waiting=2 — guard working as designed)
- JIT 31 mentions (boot-warm), watchdog announcements 5, no wedge signatures

### 2.7 Traffic shape (litellm GPUNODE01, 2 h)

glm-5.3 ×211, qwen3.8-27b-fp8-haiku-plus ×189, opus-max ×37, haiku ×3 — client 10.20.3.18 ×444 (single workstation). Engine saw 337 requests in ~2.5 h. Effective concurrency 7–8 streams = parallel CC fleet sessions.

---

## 3. Operational findings (must-fix hygiene, found during analysis)

### 3.1 The current serve was launched BY HAND and has NO log

- Container `lsv-test` started 2026-09-27T10:55:42Z (host rebooted this morning per standing directive); serve pid 67 started 10:55:57.
- **pid 67 fd/1 and fd/2 → `/dev/pts/1`** — launched from an interactive session, not via `repro_bootV1221.sh` (which truncates `serve_full.log` and execs detached with file logging; its last run = Sep 25 18:34, serve_full.log mtime confirms).
- Consequence: **no throughput series exists for the current instance** — stdout goes to a dead TTY. `lane-servecap` (serve_live_capture.sh tails serve_full.log on the host) is attached but capturing nothing; log-based forensics (f15b correlation, V-series markers, throughput series) are blind until the next certified relaunch.

### 3.2 The hand-launched serve runs a NON-certified flag set

| Flag | Running (pid 67) | Certified `serve_user.sh` (baked) | Note |
|---|---|---|---|
| `--max-num-seqs` | **32** | 64 | halves concurrency cap; no effect at today's Running 7–8, but a cliff under fleet bursts |
| `--tool-call-parser` | **qwen3_coder** | qwen3_xml | tool calls have been clean all day through CC — if chosen deliberately, bake it; either way re-run the CC tool battery |
| `--quantization` | **fp8** (explicit) | — (auto-detected from fp8 checkpoint) | semantically near-noop |
| `--chat-template` | **present** | — | serve-side template is the OBSOLETE ccthink path; certified thinking control is per-tier litellm `chat_template_kwargs`. Harmless but duplicated |

Container-level protective env is intact (verified via docker inspect): BARRIER=2, V55.3 timeouts, F15B=1/45s, ALLREDUCE_VIA_ALLGATHER=1, XGRAMMAR, XPU_GRAPH=1, FP8_MQ=1, CCL tuning, ZE_AFFINITY_MASK. No V63/V64/V66 overrides → baked defaults active (contended budget 1024, interleave 2/512, starve 2.0 s). **No protection was lost — only observability and certified-flag posture.**

---

## 4. Root-cause model of the perceived speed drop (ranked)

1. **Concurrency division (dominant).** 7–8 CC streams share the fixed ~82 tok/s aggregate → 10–15 tok/s each vs 82 solo. Pure arithmetic, no defect.
2. **Long contexts raise per-step cost.** Avg 38.1k prompt → KV read per step ∝ batch × context; per-stream and aggregate degrade together as sessions grow (until /compact resets context).
3. **Prefill steals decode iterations (sync scheduler).** 689 tok/s prefill load consumes scheduler iterations; v63 caps contended chunks at 1024 tokens, v64 grants 2 decode steps per 512-token chunk. Every prefill token directly pauses all decode.
4. **Admission queue inflates perceived latency.** Mean TTFT 40.6 s (18.7 s queueing) — requests wait behind each other's admission before the first token. Distinct from decode rate, same demand>capacity cause.
5. **Bandwidth-bound step cost with weak batch amortization** (§2.4/2.5) — sets the ceiling itself.

Explicitly NOT causes: MTP acceptance (healthy), prefix cache (61% hit), GPU/host health (clean), protective-stack regressions (none), wedge-class issues (none since v88).

---

## 5. THE PLAN — maximize token generation speed

Constraints honored throughout: **never BARRIER=1**; BARRIER=2 retirement only via the `PATCH_STACK_ANALYSIS.md` §3.6 controlled leg; async scheduler forbidden; MTP ×4 + XGrammar-2 must stay supported and crash-free; only improvements allowed (no regressions to solo-cold or fairness posture); every change validated on a fresh host (reboot first) through the full ship chain.

### Tier 0 — Restore certified launch discipline (zero risk, do first)

1. **Relaunch via `repro_bootV1221.sh`** (or `serve_user.sh` direct exec) at the next agreed window. This restores: serve logging → serve_full.log + host-side servecap capture, watchdog log-visibility, the certified flag set. Requires a deliberate serve restart — **needs your go**, this analysis did not touch the lane.
2. **Consciously bake the two flag decisions** into `serve_user.sh` before relaunch: parser (`qwen3_coder` if today's CC behavior is preferred — then re-run the CC tool battery to certify it; else back to `qwen3_xml`) and `--max-num-seqs 64` (restore; 32 is a cliff under fleet bursts, 64 costs nothing at current load).
3. **Drop serve-side `--chat-template`** (litellm per-tier kwargs are the certified thinking path; keep the template file/mount lineage as-is).

Expected gain: 0 tok/s — but every subsequent A/B becomes measurable (no log = no science), and the protective stack's log-based forensics come back.

### Tier 1 — Scheduler reallocation toward decode (env knobs, A/B each)

The axis: under contention, how much prefill runs between decode batches. Current: v63 contended chunk 1024 + v64 interleave 2 decode steps per 512-token chunk.

4. **`VLLM_V63_CONTENDED_BUDGET` 1024 → 512 (A/B leg).** Halves the prefill chunk between decode batches → decode steps ~2× more frequent under load → per-stream inter-token latency improves (est. +20–40% per-stream under load); TTFT cost rises for queued heads but stays bounded by **v66 FAIRFIX** (starve bypass admits the head at full 8192 budget after 2 s). Gate: fairness drill + `ADMISSION_DONE verdict=PASS` + TTFT histogram comparison.
5. **`VLLM_V64_DECODE_INTERLEAVE` 2 → 3 (or 4) with budget 512 (A/B leg).** Same axis, decode-favoring. Certified fairness point (6.997 tok/s @ 0.05–0.06 s gaps) was measured at 2/512 — any change re-runs the fairness gate.
6. These are **tradeoff knobs, not free wins**: they shift the fixed 82 tok/s from prefill throughput to decode cadence. Choose per workload (interactive CC → decode-favoring; bulk ingestion → current).

### Tier 2 — Step-cost levers (measure first, then act)

7. **Verify XPU-graph capture sizes on the next certified boot** (log line "Capturing cudagraphs"). `cudagraph_mode FULL_DECODE_ONLY` is on; if the captured batch-size list pads today's live shapes (draft forwards batch ≈ 7, verify forward ≈ 7×5 = 35 tokens), padding waste could be ~10%+ of step time. Fix = extend capture sizes to cover {5,6,7,35,36,40}-class shapes via `--compilation-config`. Est. +5–15% if padding confirmed; 0 if already covered. **First-day check after Tier 0 relaunch.**
8. **Barrier-off validation leg (gated upside, needs permission).** Per `PATCH_STACK_ANALYSIS.md` §3.6: v33-era data showed barrier-OFF +13.7–28.5% vs mode1; mode2 (current) vs OFF was NEVER measured post-v88. Zero barrier firings ever observed → if the tax is firings-only it is µs-class; if it is per-step it is unknown. The leg: controlled barrier-OFF boot on the validation lane with the full drill (serialized 24×3, bursts, 3×14-phase, fairness, CC battery). Only retire BARRIER=2 on measurable improvement; **BARRIER=1 remains forbidden** (standing constraint, strictly dominated).
9. **Spec depth 5 probe — considered, recommend NO.** pos3 acceptance 47.1% → pos4 marginal ~35–40%; each extra draft adds a full forward on a bandwidth-bound stack → likely net-negative. Documented to close the question; revisit only if acceptance pattern shifts.

### Tier 3 — Code/kernel (weeks-scale, honest effort)

10. **Batched GDN/mamba state ops + kernel granularity.** The 57% EU-stall share and batch-linear step cost point at per-sequence state traffic and small-kernel overhead. DPC++/ESIMD work in the vllm-xpu-kernels wheel (build with `KERNELS_MAX_JOBS=52`) is the deep lever; v65 already mapped this territory for cold-start (GEMM 57.6 s residual) — steady-state decode needs its own profiling pass (py-spy/oneAPI VTune on a certified validation lane, never the live lane).
11. **Not on the table:** async scheduling (forbidden by standing constraint); touching the v31.1 inductor gate, ALLREDUCE_VIA_ALLGATHER, v55.3, f15b, WEDGEFIX-A/C/E, v58_p1/arstage/STALFIX (all load-bearing per PATCH_STACK_ANALYSIS).

### Demand-side levers (config, immediate, no engine change)

12. **litellm per-tier concurrency caps** (e.g. `max_parallel_requests` on qwen tiers at GPUNODE01 :4004): shaping admission trades aggregate fairness for per-stream speed — e.g. capping effective concurrency at 4 raises per-stream to ~20 tok/s. User-side decision; sonnet tier was raised 8→16 on 2026-09-23 (revisit if interactive speed matters more than fleet parallelism).
13. **Context hygiene:** 38.1k avg prompts are the demand-side multiplier — CC /compact cadence directly cuts per-step KV cost and prefill load. 61% cache hit already absorbs replay; smaller contexts compound with everything above.

### Validation protocol (every lever, no exceptions)

Host reboot first (standing directive) → change on the validation path (never the running lane) → solo cold parity (~101 s reference) → serialized 24×3 + burst_harsh ×3 (wedge acceptance) → 3×14-phase drill → fairness gate (`ADMISSION_DONE verdict=PASS`, interleave intact) → CC battery (thinking stream + tool shapes + AskUserQuestion selector) → dmesg resets=0 → JIT recheck. Ship only on all-green; bake discipline per standing rules (fresh bake container, content gates, no debug instrumentation).

---

## 6. Expected-gain summary

| Lever | Tier | Est. gain | Risk | Gate |
|---|---|---|---|---|
| Certified relaunch + logging (T0) | ops | 0 (enables all measurement) | none | boot gates |
| mnbs 64 restore + parser decision (T0) | ops | removes 32-cap cliff | none | CC tool battery |
| V63 budget 512 (T1) | env | +20–40% per-stream under load | TTFT ↑ (v66-bounded) | fairness + TTFT A/B |
| V64 interleave 3 (T1) | env | same axis, stacks with V63 | TTFT ↑ | fairness gate |
| Graph capture sizes (T2) | config | +5–15% if padding confirmed | low | first-day log check |
| Barrier-off leg (T2) | env+process | unmeasured; v33-era +13–28% class | crash-class → drill-gated | PATCH_STACK §3.6 |
| Spec depth 5 (T2) | config | likely negative — rejected | — | — |
| Kernel/state batching (T3) | code | largest long-term | weeks | profiling first |
| litellm concurrency caps | demand | per-stream ÷ concurrency | fleet throughput ↓ | user decision |

## 7. Evidence commands (reproducible)

- Metrics: `curl -s localhost:8000/metrics` (host) double-sample 90 s apart; rates from counter deltas
- Config: `docker inspect lsv-test` (Env, StartedAt), `docker exec lsv-test cat /root/serve_user.sh`, `docker exec lsv-test ps aux | grep vllm`
- Serve stdout target: `docker exec lsv-test ls -l /proc/<pid>/fd/1 /proc/<pid>/fd/2` (→ /dev/pts/1 = manual launch)
- GPU: `xpu-smi dump --device 0 --metrics UTILIZATION,MEMORY --number 2` (long flags only — `-m`/short combos are rejected by this build)
- Health: `dmesg | grep -iE 'xe|reset|fault'`, watchdog/servecap: `systemctl cat lane-servecap`, `head -30 /root/build/serve_live_capture.sh`
- Traffic: litellm DEBUG monitor `/root/livemodels.sh` (GPUNODE01; pollution-hardened anchors)
