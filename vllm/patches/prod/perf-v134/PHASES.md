# PHASES.md — perf-v134 (storm-stability: the guc ccs engine-reset class)

Round opened 2026-10-02 as the direct continuation of the exhausted
surgical-churn program: the remaining production defect with
accumulated evidence is the GPU crash class that voids measurement
legs — ccs engine reset under storm capacity pressure.

User directive (standing): deep improvement + fixing + testing
suite, bake into production image using best values; spec MTPx4 +
XGrammar-2 supported and crash-free; **no subagents**; commit+push.
Host-reboot directive (standing 2026-10-02): reboot after ANY crash
before the next test leg.

Standing laws carried: SHIP-MEASUREMENT LAW (.so sha 1d9dcf4e
before banking ship numbers); READOUT LAW; RIDER LAW (watchdog
re-create wipes in-container patches); ORPHANED-ENGINE RESTART LAW
(full-tree pkill + emptiness verify); GATE LAW; no degradation
(hardened 2026-10-02: degradation forbidden, ONLY improvements
allowed); async ON; live PHASES update per phase; ship-on-delta.

## The evidence base (banked before this round)

| Datum | When | Card | Tree | Onset | Signature |
|---|---|---|---|---|---|
| v129 P51b | 2026-09-30 | card1 | single (24-way) | T+67min | level_zero error 20 DEVICE_LOST, devcoredump |
| v132 P65 leg A | 2026-10-02 05:27 | **card2** (`da:00.0`) | single | storm T+8min, turn-4 | `Tile0 GT0 ccs reset, logical_mask 0x1, guc_id=62` → devcoredump; EngineCore fatal 05:31:39 |
| v133-A1 | 2026-10-02 06:32 | **card1** (`b1:00.0`) | TWO (confound) | T+4.6min | IDENTICAL: `Tile0 GT0 ccs, logical_mask 0x1, guc_id=62` → bcs reset → `guc_exec_queue_timedout_job` → full GT reset 06:32:35 |

**P71 opening fact:** the v132 and v133-A1 resets carry the SAME
engine instance fingerprint (ccs, logical_mask 0x1, guc_id=62) on
DIFFERENT cards. Hardware flakiness does not reproduce the same guc
context across cards — this is a deterministic software class: a
ccs work submission that stops retiring until the guc job timeout
resets the engine, killing every process on the card. crashfix-v55
precedent (2026-09-11 v55.2) is a DIFFERENT class (one flaky card,
reboot-cured); this one recurs across cards.

Hypotheses to discriminate in P71-P73:
- H1 KERNEL HANG (leading): an ESIMD/custom kernel launch (lgrf /
  GDN conv / attention) deadlocks at a specific extreme shape —
  turn-4 = peak occupancy + largest chunked-prefill resends. guc
  timeout → engine reset. Predicts: hung batch identifiable in the
  devcoredump; repro concentrates at turn-4 shapes.
- H2 DRIVER/SCHED: guc submission starvation under sustained
  pressure (queue depth), independent of any single kernel.
  Predicts: no single hung batch; timeouts spread across ops.
- H3 CARD WEAR (v55.2 class): predicts different fingerprints per
  card and no cross-card repro. ALREADY disfavored by the identical
  fingerprint.

## P71 — forensic triage of the banked crashes (read-only)

Status: **CLOSED 2026-10-02** — the class is CHRONIC, predates every
recent image, and the death mechanism is sealed to the op AFTER a
completed AR/propose inside a deep-context target forward. Evidence
below (journal persistent: boots back to 10-01; 52 WD_crash_* capture
dirs on host, 27 carry the ccs reset).

**1. CHRONIC CENSUS (27 resets, 2026-09-15 → 10-02, both cards):**
`Engine reset: engine_class=ccs, logical_mask: 0x1, guc_id={22,32}`
pairs dominate (Sept 15/18/19/20/21/23/24 + Oct 1 15:05); Oct 2 =
guc_id 62 ×2; Sept 20 = guc_id 112. guc slot = allocation epoch, not
a stable identity — SAME CLASS. Every devcoredump header reads
`Reason: LR job cleanup`, `GuC version: 70.72.1 (wanted 70.49.4)`
(standing fw/kernel mismatch since ≤Sept 15), `Process: python3
[worker pid]`. WD_crash_170230's Q25crash_summary references a Sept
15 17:50:58 card1 guc_id=22 event — the class PREDATES and SURVIVED
crashfix-v55 (v1.2.8/v1.2.9) and every image since. v129's Sept 30
crash (084554, card1, guc_id=22) = this class. H3 (card wear) DEAD
beyond appeal: both cards, 17 days, multiple guc slots, identical
reason string.

**2. DEATH MECHANISM (sealed, both Oct 2 crashes):**
worker submits target-forward kernels at deep context on the ccs
LR (long-running) context → a kernel stops retiring (journal shows
ZERO pre-reset warnings; telemetry freezes ~30-60 s BEFORE the reset
line = worker stopped stepping first, guc job timeout fires after
its interval) → "LR job cleanup" ccs engine reset + devcoredump →
kernel-submitted bcs job (guc_id=0, `in no process [-1]`) times out
6 s later → `guc_exec_queue_timedout_job` → full GT reset →
DEVICE_LOST (UR error 20) cascade: Worker_TP0 sample_tokens →
EngineCore fatal → HTTP 500s.

**3. DYING-STEP SHAPES (dump_input census across captures):**
| Capture | Date | sched | num_computed | shape |
|---|---|---|---|---|
| 033400/034704 | 09-24 | 7847 | [2576, 28160] | giant chunk + deep |
| 053251 (v132) | 10-02 | 7768 | [33792, 7168] | giant chunk + deep |
| 063354 (A1) | 10-02 | 7173 | [29500, 11264] | giant chunk + deep |
| 150545 | 10-01 | 7178 | [35901, 35711, 22528] | giant chunk + 3 deep |
| 230954 | 09-20 | 10 | [24431, 200] | DEEP decode |
| 235557 | 09-20 | 30 | [37632…32130] ×6 | 6× deep decode (24-way shape) |
| 084554 (v129) | 09-30 | — | — | (fr ends after propose) |
| 085644/093937/170230 | 09-24/18 | 8-18 | shallow | multi-tree-night outliers |

Unifying variable = **batch contains request(s) with num_computed
≥ ~22k (observed max 37.6k)** — giant-chunk correlation was an
artifact of turn-4 storms (deep context + big resends co-occur).
kv usage at death 0.15-0.17, running=2 — the old "turn-4 capacity
pressure" framing is RETIRED: this is a DEPTH shape, not occupancy.

**4. fr AR-TRACE DEATH POINTS (f15b instrument, riding since v55):**
v132 final ops = 4× `AR numel=39772160` (= 7768 sched × 5120 hidden
— the dying step's own per-layer ARs) then silence → hang a few
layers INTO the giant-chunk forward. v133-A1 + v129 final ops =
3× tiny decode ARs (10240/15360 = 2-3 × 5120) + `propose end` then
silence → hang at the START of the target-verify forward of a deep
decode request (MTP draft had just completed). NEVER mid-AR, never
mid-propose → the hung kernel is the NON-collective compute in the
target forward at deep context: attention over ≥20k KV (prefill or
verify) or the GDN linear scan at depth ≥22 boundaries. H1 (kernel
hang at extreme shape) SURVIVES, sharpened to deep-context forward;
H2 (driver/guc: fw mismatch + LR-context cleanup path) survives as
the alternative — discriminated only by the FULL devcoredump (our
captures truncate at 512 KB mid-GuC-log; `[LOG].length: 0x115000`
≈ 1.1 MB — engine-state/LRC section never captured).

**5. Capture gap for P72:** pull the FULL
`/sys/class/drm/card*/device/devcoredump/data` (text, can be MBs —
the LRC head names the hung kernel). The truncated .bin captures
cannot answer H1-vs-H2.

## P71a — telemetry freeze confirmation

Frozen-rows analysis banked in P71: v132 metrics froze
running=2/waiting=6/kv=0.149 with counters static from ~05:26:4x
through capture; v133-A1 froze running=2/waiting=4/kv=0.155,
prompt_tokens_total=29500 = exactly the dying request (TTFT 226 s,
1 token generated at death). Freeze precedes reset = worker-side
hang first, driver timeout second.

Status: CLOSED 2026-10-02 (folded into P71 synthesis above).

## P72 — repro-rate matrix (lane, fresh host) + full-dump capture

The crash is intermittent at the standard storm shape (recent
rate: v132 leg A T+8min, A1 T+4.6min crashed; v133-A2 clean — but
the class is CHRONIC per P71: 27 banked resets since Sept 15, so
the per-leg rate is the unknown, not the existence). N=3+
back-to-back standard storms, single tree (full-tree restart law
between legs), host reboot before leg 1 and after any crash
(directive), perreq telemetry riding.

**CAPTURE UPGRADE (mandatory): the watchdog/capture path saves only
the first 512 KB of each devcoredump — the engine-state/LRC section
that names the hung kernel is never captured. Fix the capture to
pull the FULL /sys/class/drm/card*/device/devcoredump/data before
it expires (default 10 min retention; verify the sysfs read window
on next crash).** On any crash: bank full dump + fr tails + dmesg
window, THEN host reboot.

Outcome: crash-rate estimate + (on crash) the LRC-head evidence
P73 needs. If a clean N=3 passes, escalate depth (6-turn storm /
+2 convs → num_computed ≥ 30k) toward the onset envelope.

Status: IN FLIGHT 2026-10-02 —
- Capture upgrade DEPLOYED pre-reboot: q34_crash_capture.sh (q22
  + fr tails + debugfs clients/internal_clients + xe param
  snapshot + all-engine Timedout-job lines + journal -15min
  window); watchdog repointed (line 70); syntax-checked on host.
  Full devcoredump section confirmed UNOBTAINABLE (driver-capped
  ~500 KB at source — no xe module param raises it; cp already
  reads to EOF). P73's LRC path replaced by: force_execlist A/B
  (xe param exists on host), microbench replay, q34 clients lists.
- Host rebooted 08:04 per directive; watchdog restored lane
  (re-create 08:16:46); posture asserted: image v1.2.27, .so sha
  1d9dcf4e7a1c8db6, 0 patch markers, 1 startup line, health 200.
- DECISION: legs run on the BYTE-PRISTINE lane (no perreq patcher
  — RIDER skip): the repro surface must match production posture;
  prometheus jsonl + q34 dump_input cover the crash evidence
  (perreq adds only per-request totals; HARVEST_ROWS=0 expected).
- LEG 1 launched 08:26 UTC: wsc_pressure_v134_leg1.py (v129b
  moderated standard storm: 18 convs x 4 turns = 72 reqs, temp
  0.6, max_tokens 300, full history re-send; the same shape that
  crashed v132 leg A T+8min and A1 T+4.6min, clean in A2).
  Prior exposures at this shape: 2 crash / 1 clean.
- LEG 1 CLEAN: `PRESSUREB_DONE ok=72/72 wall=1312s HARVEST_ROWS=0`
  (harvest 0 = expected, pristine lane). 13 dmesg polls over the
  run: ZERO Engine resets during the leg. Exposure tally: 2 crash /
  2 clean.
- SURVIVED-RESET DATUM (new class member, benign form): 08:49:52 —
  lone `ccs reset, logical_mask 0x1, guc_id=52` on card1 at T+~70s
  AFTER leg-1 completion (engine idle, tree alive). NO devcoredump
  created (sysfs checked — none), NO bcs timeout, NO GT reset, no
  escalation of any kind through the following boots. LAW: the
  "LR job cleanup" ccs reset has a benign Lone form; death REQUIRES
  the kernel-submitted bcs job (guc_id=0, `in no process`) timing
  out behind the wedged context within ~6 s (as in A1's 06:32:29 →
  06:32:35 sequence). The reset alone is survivable; the bcs
  timeout behind it is the killer. Evidence limits: the 08:16
  container re-create gave a fresh /tmp (leg-1 tree wrote fr rings
  into pid slots never listed again) and the 08:52 boot truncated
  serve_full.log — dmesg is the sole record of this event.
- LEG 2 launched 08:57 UTC (same shape, full-tree restart between
  legs per ORPHANED-ENGINE LAW). Leg-2 live fr rings observed at
  ring cap (fr_686/fr_692, 2039520 B each). Baseline for reset
  monitoring: 1 (the 08:49:52 line) — watch NEW lines only.
- LEG 2 CLEAN: `PRESSUREB_DONE ok=72/72 wall=1095s HARVEST_ROWS=0`,
  zero new resets (baseline 1 held through 6 polls), health 200
  throughout. Both prior crash onsets (T+4.6/T+8 min) passed clean.
  Exposure tally: 2 crash / 3 clean.
- LEG 3 launched ~09:24 UTC after full-tree restart (pkill -9 -f
  [v]llm; leftover resource_tracker 486 killed by pid; new tree
  pid 1825 + own tracker 2097; health 200 at T+~3.5 min).
- LEG 3 CLEAN: `PRESSUREB_DONE ok=72/72 wall=865s HARVEST_ROWS=0`,
  zero new resets. MATRIX CLOSED — N=3 back-to-back standard
  storms, all clean:

  | Leg | ok | wall | new resets |
  |---|---|---|---|
  | 1 | 72/72 | 1312 s | 0 |
  | 2 | 72/72 | 1095 s | 0 |
  | 3 | 72/72 | 865 s | 0 |

**P72 SYNTHESIS (rate matrix + survived-reset law):**
1. Per-leg crash rate at the standard shape THIS round: 0/3;
   lifetime at this shape 2 crash / 5 clean — INTERMITTENT, not
   deterministic. The v132/A1 crashes are the tail of a low-rate
   chronic class, not a per-leg certainty.
2. The 08:49:52 SURVIVED reset sharpens the mechanism: the ccs
   "LR job cleanup" reset fired on an ENGINE-IDLE context
   (T+~70 s after a clean ok=72/72 completion) with zero
   escalation. Model: the root event is an unretired ccs LR job
   (kernel from the storm that never retired — H1 — OR guc context
   bookkeeping — H2); the reset itself is survivable when nothing
   is queued behind; DEATH requires a kernel-submitted bcs job
   landing in the ~6 s window behind the wedged context
   (A1: 06:32:29 ccs → 06:32:35 bcs). Crash = reset ×
   P(busy bcs behind). Fr evidence for this event was lost (fresh
   /tmp from the 08:16 re-create + serve-log truncation), so H1
   vs H2 stays open for P73's force_execlist A/B + microbench.
3. Depth escalation (6-turn / +2 convs → num_computed ≥ 30k) is
   NOT run pre-bake: the lane now executes the P75 cleanup+bake
   directive; the escalation shape folds into the post-bake
   verification matrix (one depth-escalated leg among the ≥3,
   serving both the P74 stability gate and this escalation).

Status: **CLOSED 2026-10-02** — N=3 clean matrix + survived-reset
law banked; capture upgrade (q34) armed for any future crash.

## P75 — surface cleanup + v1.2.28 bake (user directive 2026-10-02)

Directive (verbatim intent): clean ALL older-patch leftover loggings
on every surface; the production image keeps ONLY the fork's
default-style logging — no patch-version-prefixed (v*) artifacts,
no experimental file loggers; then BAKE a new production image.

Inventory (classify KEEP-default vs STRIP-patch-leftover):
1. **IMAGE/build surface (bake input):**
   - STRIP: f15b/fr AR-trace instrument (riding since crashfix-v55;
     writes fr_*.log rings + f15b_dump_*), v128 perreq jsonl patch +
     /root/.v127_m0live dormant M0 collector, image-baked stale
     /tmp/fr_* files, any v*-prefixed log writers. Sequencing: the
     fr instrument rides THROUGH P72 completion (it is the
     death-point instrument for the open crash class), then removed
     from the build surface before the bake.
   - KEEP: default vLLM logging + standard /metrics; ALL functional
     posture — fp8 checkpoint + baked template, MTPx4 + XGrammar-2,
     async scheduling, barriers 0, genconfig folds, litellm bridge.
     Function survives; only experimental logging goes.
2. **Host /root/build:** stale wsc_*.out / v*-prefixed outputs from
   closed rounds; KEEP captures (lce1/* = evidence), watchdog,
   restore chain, q34 (its fr steps no-op gracefully post-strip).
   Ops-script NAMES (boot_v1227_restore.sh, V1227_IMAGE lineage)
   are infrastructure, not logging — lineage pointer updates to
   v1.2.28 at bake, naming stays.
3. **Repo tree:** untracked .tmp-* litter, `nul`, stray outputs
   (git clean dry-run list first, .tmp-* only); KEEP all
   patches/*/ history (audit trail — cleanup targets artifacts,
   not records).
4. **BAKE:** rebuild only what the strip touches (wheel rebuild
   with KERNELS_MAX_JOBS=52 iff .so changes) → llm-scaler-exp:
   v1.2.28 → restore-chain rebase to the new image + md5 re-cert →
   posture certification (MTPx4 + XGrammar-2 + async + barriers 0 +
   health 200 + 0 patch markers + NEW .so sha banked into
   SHIP-MEASUREMENT LAW) → storm matrix ≥3 clean legs on v1.2.28,
   doubling as the P74 stability gate on the cleaned surface.
5. Constraint — ZERO-DEGRADATION LAW (user directive 2026-10-02,
   hardened): "any performance degradation is not allowed, only
   improvements are allowed." Measurable gate: v1.2.28 storm legs
   vs the v1.2.27 baseline banked THIS round (72/72 ×3, walls
   1312/1095/865 s; single-run noise ±~6%) — every leg ok=72/72,
   wall within-or-better than the baseline band, totals
   (computed tokens) not worse. Removing the per-AR evmark hook
   (1 Event record + import lookup + ring append per all_reduce)
   is expected to measure as a STRICT improvement; any consistent
   regression = NO BAKE. Functional posture (MTPx4, XGrammar-2,
   async, barriers 0, fp8, fences, Fix L/M) carries unchanged.

Status: PLANNED — executes after the P72 matrix closes.

## P73 — root-cause discrimination (H1 deep-context kernel vs H2 guc/LR)

Evidence paths, in order:
1. FULL devcoredump from a P72 crash: LRC head/batch descriptors on
   ccs guc_id=N → names the hung kernel (H1 direct hit; if the
   kernel is in custom_esimd_kernels_lgrf .so → OUR surface; if
   stock attention/GDN from ipex/oneDNN/vllm-xpu → upstream kernel,
   still fixable in our wheel).
2. No single hung batch + guc queue starvation state → H2 (driver/
   fw; guc 70.72.1 vs wanted 70.49.4 mismatch standing since ≤Sept
   15) → host/OS surface: document + handoff, mitigation shifts to
   workaround class (P74c).
3. If P72 legs run clean (no new dump): standalone microbench
   replay of the two death shapes (deep giant-chunk forward;
   deep-decode MTP verify loop) on a throwaway container — the
   minimal repro for upstream handoff either way.
Deliverable: named culprit (kernel + shape, or driver condition).

Status: pending P72.

## P74 — mitigation + verdict

Candidate mitigations by culprit class:
a. KERNEL FIX (if the hung kernel is ours — custom_esimd_kernels
   ESIMD source, KERNELS_MAX_JOBS=52 build): fix + rebuild + storm
   matrix ≥3 clean + parity → ship-on-delta bake.
b. SHAPE MITIGATION at scheduler (chunk-size/verify batching at
   num_computed ≥ threshold): MUST NOT degrade totals (GATE LAW);
   only acceptable with measured parity or better.
c. DRIVER/WORKAROUND (guc/job-timeout tuning, fw alignment): last
   resort, needs evidence + host-OS surface ownership; document +
   hand off with the repro recipe.
Gate: mitigation survives the P72 matrix (≥3 storms clean) with
parity + no degradation; ship-on-delta bake only on a certified
fix. If the culprit is outside this repo's surface (upstream
xe/kernel/firmware), document + hand off with the repro recipe and
this round closes no-bake.

Status: pending.
