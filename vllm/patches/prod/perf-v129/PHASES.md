# PHASES.md — perf-v129 (Phase-2: WS-C eviction track -> REOPEN-A regime -> REOPEN-B doc)

User directive (2026-10-01): "continue Phase 2 first after that 1 and
efter that 3" over the offered menu — execution order fixed as:

1. **WS-C** eviction/policy track (was menu item 2) — P51/P52
2. **REOPEN-A** int8 per-feature-scaled state, regime measurement
   (menu item 1) — P53
3. **REOPEN-B** e4m3 frozen-state spillover, document-not-do (menu
   item 3) — P54

Standing laws carried: no subagents; no degradation anywhere; spec
MTPx4 + XGrammar-2 crash-free; SHIP-MEASUREMENT LAW (assert prod .so
sha 1d9dcf4e before banking ship numbers); READOUT LAW; live PHASES
update per phase; ship-on-delta policy (v128 PHASES: riders fold only
into a trigger-justified bake; triggers = REOPEN-A kernel success /
WS-C baked-default policy / engine-side xgrammar fix).

---

## P51 — WS-C leg 1: live-fleet eviction measurement with perreq armed

**Goal:** characterize the P48 finding (recompute excess 18.8% of
computed; 90.1% eviction/partial-prefix-miss, ~640 tok/affected-req)
in the LIVE fleet regime, with enough structure to choose the lever:
shared-prefix pinning (image trigger) vs admission/scheduling policy
vs capacity (defer to REOPEN-A) vs no-lever.

**Instrument:** `V1227_PERREQ=1` on the certified restore boot
(`boot_v1227_restore.sh`, posture default ckpt) — the standing P48
instrument; one JSON line per finished request
(prompt/generated/cached/computed/finish) to in-container
`/root/v128_perreq.jsonl`. Telemetry stack (metrics_recorder 10 s
JSONL + stall_scope) re-armed idempotently by the same script.

**Window discipline:** host rebooted BEFORE the round per the
standing hygiene directive (lane was quiet 0/0; watchdog
stop+disable only for the reboot+arm span, re-enabled immediately
after the instrumented lane is healthy — P37 lesson honored: the
watchdog is ACTIVE for the measurement window itself).

**Legs:**
- 51a live harvest — let the CC fleet accumulate on the armed lane;
  harvest perreq JSONL + recorder kv_perc/waiting timeline; join.
- 51b controlled pressure replay — d4_replay_v128-style conversations
  at elevated concurrency to force KV pressure causally (the live
  window may be storm-free; the replay guarantees eviction signal).

**Verdict:** → P52 analysis.

Status: IN PROGRESS —
- host rebooted (uptime reset ~13:44), instrumented boot
  `BOOT_V1227_RESTORED posture=ckpt 13:51:11` with `perreq=1`
  (marker `/root/.v128_perreq` armed, KV 476,451 tok, health 200),
  watchdog re-enabled immediately after (active; P37 lesson honored)
- 51a live window: OPEN (perreq accumulating; fleet idle at leg start)
- 51b pressure replay RUNNING since 13:56 (`wsc_pressure_v129.py`:
  24 convs — 7 SHARED ~29.5k with identical ~8k head, 7 PRIV ~36k
  unique (size bands disjoint for engine-side cohort assignment),
  10 SMALL ~3k; 6 turns; barrier-paced; observed turn-1 prefill
  ~1.26k tok/s under 24-way concurrency, ttft 23.4 s on 29.5k)
- NOTE: synth text tokenized ~25% under target -> live working set
  ~1.03-1.1x pool, not 1.44x; if hit_ratios stay ~1.0 through turn
  6 the leg under-pressured -> second wave with more/bigger convs
  before any verdict. (Early engine rows DID show full eviction —
  PRIV turn-2 cached=0, SHARED turn-2 cached=5120 — so pressure
  materialized despite the tokenization shortfall.)

### P51b outcome — DEVICE_LOST under sustained oversubscription (CRASH)

- Run completed 120/144 ok (wall 4019 s) but the engine DIED under
  the pressure: 6 turn-5 client timeouts (~T+49 min), 12 turn-6
  HTTP 500s (~T+64 min), then 15:03:47
  `EngineCore fatal — RuntimeError: level_zero backend failed with
  error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` (Worker_TP0), GPU
  devcoredump on card1. Captured by the PARALLEL Q22/crashfix
  workstream (their watchdog activity: teardown 15:05:45, evidence
  /root/build/lce1/WD_crash_150545/{serve_full_Q22.log, fr_715.log,
  devcoredump_card1_Q22.bin}); watchdog relaunched the lane; excerpt
  in captures/crash_WD_150545_excerpt.txt.
- **Repro recipe (handed to the crashfix workstream):** 24-way
  concurrent prefill churn at ~1.1x KV oversubscription, 6 turns
  full-history re-send -> DEVICE_LOST at T+67 min. Same class
  recurred TODAY at 03:09 / 03:34 with no pressure driver (fleet
  regime onsets it too).
- **Cost:** the in-container /root/v128_perreq.jsonl was destroyed
  by the crash teardown -> the P52 engine-side join had no data.
- **Standing implication (elevates WS-C):** admission control /
  capacity is a STABILITY lever, not only a waste lever — sustained
  ~1.1x oversubscription at 24-way is a crash onset.

### P51c — moderated re-run with crash-proof harvesting (RUNNING)

- Host REBOOTED 15:12 (device-lost hygiene, standing directive;
  heads-up sent to the crashfix session; their on-disk captures
  untouched), instrumented boot `BOOT_P51C posture=ckpt perreq=1`
  15:27:40 / RESTORED 15:30:36, marker armed, health 200, watchdog
  re-enabled, instrument smoke-verified (row landed with full schema).
- Driver `wsc_pressure_v129b.py`: 18 convs (5 SHARED / 5 PRIV /
  8 SMALL, same synth targets), 4 turns (churn -25%, exposure -33%);
  harvester thread docker-cp's the perreq stream every 120 s and
  merges to host /root/build/lce1/wsc_perreq_v129b.jsonl — a repeat
  crash can no longer eat the data.
- Launched 15:36:18 (PID 10253).

## P52 — WS-C leg 2: cohort split analysis

**Instrument:** `wsc_split_v129.py` — joins driver rows (cohort/turn/
prompt_tokens) to perreq engine rows by exact prompt-size match in
completion order; per-conv turn deltas; verdicts HEAD_RETAINED vs
HEAD_EVICTED (SHARED-miss cached floor vs ~4.8k bar), SMALL_RESIDENT,
WASTE_PCT / MISS_TOK_PER_AFFECTED vs P48 comparands (18.8%, ~640).

**Lever question:** LRU pinning lever ALIVE (shared heads evicted ->
pinning code would help) vs DEAD (LRU already pins hot shared
prefixes -> lever = capacity REOPEN-A or admission control).

Status: CLOSED (verdict from the P51c re-run data) —

- 72/72 ok, wall 1023 s, 73 engine rows harvested continuously (a
  repeat crash can no longer eat the data); joined 72/72.
- **HEAD_RETAINED**: every SHARED-cohort miss floors at cached=5120
  (the shared head) — LRU already pins hot shared prefixes; a
  pinning lever would duplicate LRU for free behavior -> pinning
  lever DEAD.
- **PRIV fully evicted** between turns under pressure (turns 2/4
  cached=0, full ~35k recompute; turn-3 partial 13.5k retention
  during a wave lull — retention is churn-timing-dependent).
- **SMALL_PRESSURED**: even 3k convs do not survive a wave
  (turn>=2 mean hit 0.109) — recency alone protects nothing but
  the simultaneously-touched shared head.
- **WASTE_PCT=99.1** of turn>=2 computed is recompute (live-fleet
  P48 comparand: 18.8%); **MISS_TOK_PER_AFFECTED=15239** (P48:
  ~640) — under ~1.1x oversubscription the storm regime re-prefills
  entire histories.
- **LEVER VERDICT: capacity or admission control — not pinning,
  not eviction policy.** Combined with P51b (DEVICE_LOST onset in
  the same regime): oversubscription is simultaneously the waste
  and the stability problem. Follow-up candidate (own round, not
  acted on): an aggregate-KV-demand admission guard trades storm
  throughput for stability + cache retention; needs its own
  measurement before any production change.

Evidence: captures/wsc_split_v129b.json (split + per-row table),
captures/wsc_pressure_v129b.jsonl (driver), captures/
wsc_perreq_v129b.jsonl (engine perreq harvest).

## P53 — REOPEN-A: int8 per-feature scaled-state regime

**Open question (from v127 M1 rejection):** e4m3 recursing state
collapses at N~6 (6.25%/roundtrip). int8-per-feature is 16x better
(2^-8 = 0.39%/roundtrip) but the growth REGIME is set by the GDN
recurrence's decay spectrum — injected roundtrip noise propagates as
Ornstein-Uhlenbeck with steady-state amplification
sqrt(1/(1-gamma^2)) per feature.

### P53a — analytic decay-spectrum leg (CLOSED)

`reopen_a_spectrum_v129.py` (read-only, in-container, CPU-only, ran
DURING the pressure leg with zero GPU interference): read the real
checkpoint's 48 layers x 48 heads `linear_attn.A_log`/`dt_bias`,
gamma = exp(softplus(dt_bias) * -exp(A_log)).

Measured (captures/reopen_a_spectrum_v129.json):
- gamma: p50=0.973151 p90=0.999314 p99=0.999731 max=0.999962
- OU amplification: p50=4.34 p90=27.00 p99=43.09 max=114.37
- int8 SS error: **p50=1.69% p90=10.53% p99=16.81% max=44.60%**
- e4m3 comparand: p50=27.15% max=714.80% — reproduces the observed
  N~6 collapse -> model VALIDATED against the v127 measurement
- **OVER18_FRAC=0.0074** (0.74% of the 2304 heads exceed the 18%
  rejection bar); pure-sqrtN N@18% reference = 2130 steps
- over-bar heads are NOT layer-concentrated (worst layers 1, 0, 33,
  17, 49 by gamma_max) -> a per-LAYER dtype split cannot isolate the
  tail; only a per-HEAD dtype map could

**Reading:** uniform int8-per-feature everywhere clears the bar for
99.26% of heads but fails the long-memory tail (max 44.6%). The only
viable shape is a HYBRID per-head dtype map — int8 scaled state for
the ~99.3% mass, fp16 for the gamma >= ~0.99995 tail. This is
image-trigger material ONLY if P53b confirms empirically.

### P53b — empirical N-step recurrence leg (CLOSED)

`reopen_a_empirical_v129.py`: four arms share identical per-step
drives through the production spec kernel
(`esimd_gdn_conv_fused_seq_spec`, M2 shapes H=8 HV=16 K=V=128
NSD=2 NST=4, stratified pool mag 10^U(-3,1), head ladder gamma
0.95..0.99996); only SSM-pool storage differs — REF fp16, E4M3 raw
pool (kernel-internal roundtrip), I8DYN external int8
per-(hv,k)-feature roundtrip with per-step recomputed scales,
I8STATIC frozen scales with x4 headroom. Slots 0-7 dynamic
(spec_idx=arange), 8-11 static control. Ran in the THROWAWAY prod
container (v1.2.27 + `--entrypoint bash`, /so kernels + /h stage
mounts; the devel image has no torch venv — P53b infra lesson).
Run: 4096 launches = 32k decode tokens, wall 11.1 s, clean finish.
Caps `REOPEN_A_VERDICT=CLOSED`; captures/
reopen_a_empirical_v129.json.

**Measured (relative error vs fp16 reference, dynamic slots):**

- roundtrip epsilon: I8DYN N=1 eps 1.07-1.20% across heads — **3x
  the analytic 0.39% half-ULP assumption** (heavy-tailed state ->
  per-feature runmax dominates the scale; RMS error lands at
  ~full-quantum, not half).
- I8DYN growth: p50 over heads 1.8% (N=4) -> 4.1% (N=16, p90 17%,
  max 55%) -> 19% (N=64) -> **132% (N=256 ~= 2k decode tokens)**,
  saturating ~100-120% to N=4096 (max 438%). sqrt(N)-like through
  N~256, then decorrelation-saturated.
- E4M3 comparand: dyn p50 -> 144% / max 733%; first head over the
  18% bar at N=16-64 — slower than v127's N~6 because injection is
  per-LAUNCH (8 tokens) not per-token; scaling to per-token
  cadence reproduces N~6..48. Collapse physics confirmed.
- I8STATIC (the practical M0-convention variant): worst arm — dyn
  p50 -> 480% / max 862% by N=4096 with ~6% clip saturation (state
  grows past the frozen runmax+x4 headroom). Static per-request
  scales on RECURRING state are categorically dead.
- **Static-slot control (the decisive sanity anchor):** frozen
  slots show NO accumulation — E4M3 2.6% constant, I8DYN 1.2-2.2%
  ~constant across all 4096 roundtrips. Round-tripping a FIXED
  value with a recomputed scale is a deterministic fixed point:
  quantization on frozen content is idempotent. This is why fp8 KV
  (write-once) survives, and it directly validates REOPEN-B's
  no-accumulation argument for frozen spill state.

**Interpretation (with honest caveats):**

- The harness's Yule-Walker phi_hat came out compressed
  (0.76-0.99) and non-monotone vs the nominal ladder — the drive
  dominates the state trajectory, biasing the estimator low; the
  in-run OU_MODEL=INVALID verdict is an artifact of that estimator,
  NOT evidence against the physics. Refit by hand: the measured
  I8DYN curve fits OU with the TRUE per-launch spectrum
  (phi~0.9999 for the top ladder heads: eps*sqrt(4096) = 1.1%*64
  ~= 70% ~= observed saturation) — growth to the decorrelation
  horizon is the OU transient, not a new regime.
- e4m3-vs-int8 onset ratio measured 2-16x, muddier than the 32x
  the eps ratio predicts (tail spread + per-launch cadence).
- **Decision math:** rescaling the P53a analytic leg by the measured
  eps ratio (1.1% / 0.39% = 2.8x): int8 SS error p50 -> ~4.7%,
  p90 -> ~30%, p99 -> ~47%, max -> ~125%; OVER18_FRAC >= ~10%
  (vs 0.74% at the design eps). The over-bar mass is no longer a
  carveable tail — a hybrid per-head dtype map would need fp16 on
  ~10%+ of heads AND still ships the sqrt(N) transient through
  every realistic horizon (N=16 launches ~= 128 tokens already at
  p90 17%).

**P53 FINAL VERDICT: REOPEN-A STAYS CLOSED.** Both legs agree once
the measured epsilon replaces the design assumption: int8
per-feature scaled RECURRING state fails the 18% rejection bar at
realistic horizons, and the only viable static-scale variant
saturates. The burden of proof the reopen demanded is not met; no
M3 kernel program, no image delta. Reopen trigger remains a future
int8-scaled pool with demonstrated epsilon << 1% (e.g. per-step
scale re-estimation in-kernel with stochastic rounding), which
today does not exist.

## P54 — REOPEN-B: e4m3 frozen-state spillover (document-not-do)

Documented in full in `reopen_b_spill_design_v129.md`: mechanism
(v126 bridge semantics freeze/thaw at the storage layer — the
numeric core already ships), perturbation analysis (one-shot 6.25%
per thaw decays geometrically — no OU amplification — exactly the
write-once property P53b's static-slot control just validated
empirically), cost/benefit/risks, and the dependency ordering:
gated on (1) crashfix closing the DEVICE_LOST class (P51b showed
the preemption path is its trigger regime), (2) WS-C concluding
capacity — not admission control — is the required lever, and
(3) REOPEN-A resolved CLOSED (now satisfied by P53).

**Verdict: STAYS_CLOSED.** With (3) satisfied but (1) open and (2)
resolved toward admission control, the spill option is moot for
this cycle; it remains the documented fallback if a future
capacity program reopens.

---

## Round close — ship decision and lane posture

**Ship-on-delta check (v128 policy, all triggers evaluated):**

| Trigger | Fired? | Basis |
|---|---|---|
| REOPEN-A kernel success | NO | P53: CLOSED by measurement (both legs) |
| WS-C baked-default policy | NO | P52: pinning dead, verdict = config-side admission discipline, own future round |
| Engine-side xgrammar fix | NO | Not in this round's scope; recipe v2 remains client-side |

Riders (perreq telemetry knob) stay dormant boot-knobs; nothing to
fold. **DECISION: NO image bake — v1.2.28 remains not required;
production posture stays llm-scaler-exp:v1.2.27 (a7a950148fef).**
This round produced measurement closures + one handed-off crash
repro, not runtime content.

**Lane posture after the round:** host rebooted once mid-round
(after DEVICE_LOST, per hygiene directive); lane restored via the
certified chain `BOOT_P51C posture=ckpt perreq=1` (marker armed),
watchdog active, health 200, smoke-verified. Perreq marker remains
armed (telemetry is free and the standing P48 instrument).

**New standing items surfaced by this round (for future rounds):**

1. DEVICE_LOST under sustained ~1.1x oversubscription at 24-way —
   repro recipe handed to the crashfix workstream (captures/
   crash_WD_150545_excerpt.txt); WS-C's admission-control follow-up
   should be co-designed with whatever fix lands there.
2. Aggregate-KV-demand admission guard — the WS-C lever candidate;
   trades storm throughput for stability + cache retention; needs
   its own measurement round before any production change.
3. In-container telemetry fragility law — any in-container jsonl
   must be harvested continuously (wsc_pressure_v129b.py's
   harvester thread pattern: docker cp + rid/ts dedupe every 120 s).
4. P53b infra lesson: GPU throwaway containers run the PROD image
   with `--entrypoint bash` (venv torch-xpu works, kernels load);
   the devel image has no /opt/venv and no torch.

**Status: perf-v129 round CLOSED (P51a/b/c, P52, P53a/b, P54
complete).**

