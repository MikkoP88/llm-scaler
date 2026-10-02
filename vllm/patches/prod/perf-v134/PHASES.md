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

Status: **CLOSED 2026-10-02** — v1.2.28 (`15168e205816`) STANDING.
Surface stripped + baked, chain rebased (md5 `941fc03b0d1e25b328febf87fc0bbca5`),
RESTORE-CHAIN RIDER killed and proven in production fire (crash-cycle
containers instrument-free), standard-shape parity 1089/1029/1042 vs
v1.2.27 band 865–1312 (parity-or-better, ZERO-DEGRADATION LAW met).
Depth leg crashed on-class 2/2 (intermittent death class, predates the
strip — sandwich-proven); its full devcoredump is BANKED for P73.

Execution log (2026-10-02):
- **STRIP certified 2×** — `p75_strip_logging.py` (16 anchor pairs,
  two-phase verify-first): run on the live lane container AND on a
  fresh bake container from v1.2.27; both `P75_STRIP_OK` (py_compile +
  keep-asserts + forbidden-grep on all touched files).
- **RESTORE-CHAIN RIDER LAW (new):** the instrument is baked INTO
  image v1.2.27 itself (apply logs: `marker present, skipping` +
  `module installed` on every fresh container) AND the restore chain
  re-applies `patch_f15b.py`/`patch_v58_p1.py` per boot. The first
  in-container strip was reverted at 09:48 when the watchdog's full
  restore `docker rm -f`'d the stripped container mid-restart (apply
  logs + Created timestamp prove the path; no in-place re-applier
  exists). Durable fix = bake + chain rebase, both done below.
- **v1.2.28 BAKED** = `15168e205816` (24.8 GB, 2026-10-02 10:07Z):
  bake container from v1.2.27 + strip + surface cleanup — /root
  pruned 65→4 entries (57 f15b_*.log dump/pyspy/dmesg/xpusmi,
  serve_full* / fp8mq logs, dbg_sampler_v60f.py, t2_*/repro_v88/
  v51_utils/dt_warmup scripts, 12 staged patch_*.py copies,
  gdn_cap_input.pt debug capture, v88out/, /tmp v60g_*/x*.txt junk);
  KEPT: kernel whl, dflash2 staging, all `.llm_scaler_exp_v12*_baked`
  lineage markers, serve_user.sh. **.so sha UNCHANGED**
  (`1d9dcf4e…` verified on image) — python-only bake, no wheel
  rebuild, KERNELS_MAX_JOBS=52 not needed. Known-inert: image Env
  carries `VLLM_F15B=1`/`VLLM_F15B_STALL_S=45` from the commit
  lineage — zero readers exist (module deleted, forbidden-grep
  proof); chain no longer passes them.
- **CHAIN REBASED** (`boot_v1227_restore.sh`, both copies identical,
  **md5 `941fc03b0d1e25b328febf87fc0bbca5`** — retires
  `e2f0b00d…`; no script enforces the md5, certification-record
  only): `V1227_IMAGE` default → v1.2.28; patcher steps removed =
  patch_f15b.py, patch_v58_p1.py (instrument) + patch_v55_3.py
  (FAILs noisily on the absent `_f15b.py`; its pipe-masking `| tee`
  hid the rc — fences are baked, the chain keeps the bare
  `grep -c "llm-scaler v55.3"` assert); `-e VLLM_F15B=1 -e
  VLLM_F15B_STALL_S=45` env tokens dropped from docker run.
- **ARSTAGE PATCHER v2.1** (`patch_arstage.py`, host + this repo):
  the chain reinstalled `_arstage.py` WITH the census f15b feed
  every boot (only `_f15b` residue tree-wide after the first chain
  test). Feed + `_MARK_EVERY` removed from the embedded module at
  source; census counters/`_summary`/`stats_lines` retained
  (decision census = functional, not logging). Reinstalled →
  `NO_F15B_RESIDUE` tree-wide, compiles, Fix L marker intact.
- **LEG-4 PARITY PASS (pre-commit, stripped live lane):**
  `ok=72/72 wall=1089s HARVEST_ROWS=0`, zero `/tmp/fr_*.log`
  mid-storm — band 865–1312 (median 1095): parity-or-better,
  ZERO-DEGRADATION LAW met.
- **CHAIN END-TO-END TEST PASS (TEST75):** fresh container from
  v1.2.28 through the rebased chain — all functional patchers skip
  clean (v60/v62b/v63/v64/sed ALREADY-APPLIED), serve boots, HEALTH
  200 at ~150s, posture ckpt verified on live cmdline, KV
  476,451 tok / concurrency 1.82×.
- Remaining: legs 5/6 standard + leg 7 depth (N_TURNS 6 →
  num_computed ≥ 30k, P72-escalation fold-in) on the watchdog-managed
  v1.2.28 lane; repo commit + push; memory update.

Matrix run 1 (10:35–10:59) + crash #3 + recovery:
- **leg5 PASS**: `ok=72/72 wall=1029s` — best-band wall (median 1095,
  band 865–1312). v1.2.28 ≥ v1.2.27 confirmed on the standard shape.
- **leg6 = crash #3 of the intermittent class** (NOT a strip
  regression — leg5 ran the identical shape clean minutes earlier):
  wedge mid-storm → 68 request fails → `ok=4/72 wall=364s`; dmesg
  10:58:52 `ccs Engine reset guc_id=52` card b1 (same LR-cleanup
  signature, same guc_id as crash #1/#2; a second reset guc_id=62 on
  da at 11:00:20 rode the teardown). Lifetime: 3 crash / 6 clean.
  No Q34 capture fired (no capture dir); dmesg = evidence of record
  — fr rings no longer exist BY DESIGN (the P75 evidence tradeoff).
- **leg7_depth VOID**: launched 10:58:52 into the dead engine,
  `ok=0/108 wall=1s` — rerun required post-recovery.
- **Watchdog auto-recovery through the REBASED chain: perfect.**
  WD fired 11:00:22 → fresh container from `15168e205816` (v1.2.28)
  → chain ran WITHOUT the instrument patchers (f15b apply log
  untouched at 09:48) → `BOOT_V1227_RESTORED posture=ckpt 11:02:47`,
  health 200, KV 476,451 / 1.82× parity, and the recovered container
  is **instrument-free** (fr=0, F15B_GONE, NO_RESIDUE) — the exact
  property v1.2.27 lacked (RESTORE-CHAIN RIDER dead, proven in
  production fire).
- **Host rebooted 11:05** per the standing post-crash directive;
  legs 6′ and 7′ rerun after watchdog recovery.

Matrix run 2 (11:22–12:16, post-reboot, matrix2 runner):
- **leg6′ PASS**: `ok=72/72 wall=1042s` (11:22:45→11:40:07) — in-band
  (865–1312, median 1095). Crash #3 is now SANDWICHED by clean
  passes on the identical shape (leg5 1029 / leg6′ 1042) —
  intermittent class, definitively not a surface regression.
- **leg7_depth = crash #4, ON-CLASS (depth shape)** — the known
  dying shape per P71 (num_computed ≥ 22k = depth, not occupancy):
  12:09:57 card da (card2) `ccs Engine reset guc_id=32` MID-STORM
  **with devcoredump created**; engine LIMPED ~6.5 min (health still
  200 at 12:11:02) while 47 requests failed → `ok=61/108 wall=2060s`
  (done 12:14:28); 12:16:29 full death cascade — b1 `ccs guc_id=22` +
  `bcs guc_id=26`, da `bcs guc_id=36` (both cards, bcs timing out
  behind the wedged context) → health 000 → WD fired `boot_WD_121631`,
  container recreated from v1.2.28 (instrument-free, chain correct).
  **Law refinement (SURVIVED-RESET):** P72's survivable reset was at
  engine-IDLE; a mid-storm ccs LR reset = PRE-DEATH (limp window then
  cascade), not survivable.
- **DEVDUMP-DISPLACEMENT LAW:** the 12:16:29 cascade's later resets
  CLOBBERED the 12:09:57 devcoredump (card2 data file 0 bytes, mtime
  12:16) — a mid-storm dump survives only the ~6-min limp window;
  Q34 auto-capture did NOT fire (2nd consecutive miss, #3 and #4) →
  P73 must make capture instantaneous (udev rule on devcoredump
  create, not post-hoc polling). Evidence of record: dmesg timeline
  saved `lce1/WD_crash_120957_dmesg.txt` + matrix2 log + driver
  terminal line.
- Lifetime: **4 crash / 7 clean.** Standard-shape parity on v1.2.28:
  1089 / 1029 / 1042 (leg4/5/6′) vs v1.2.27 N=3 1312/1095/865 —
  parity-or-better, ZERO-DEGRADATION LAW met on the gated shape.
- **Host rebooted 12:19** per the standing post-crash directive;
  leg7_depth rerun after watchdog recovery.

Matrix run 3 (12:39–13:25) + crash #5 + FIRST FULL DEVCOREDUMP:
- **leg7_depth rerun: `ok=105/108 wall=2740s`** — 97% complete, death
  struck at the storm tail: 13:24:00 card da (card2) `ccs Engine reset
  guc_id=32` (same class/signature as crash #4), engine died without a
  second-reset cascade this time; 3 tail requests lost; WD fired
  `boot_WD_132634`. Depth shape is now a 2/2 reliable trigger for the
  death class (#4 at 61/108 mid-storm, #5 at 105/108 tail).
- **WATCHER VOID + SYSFS-DEVDUMP-SIZE-0 LAW:** devdump_watch.sh polled
  correctly but never fired — the sysfs devcoredump `data` node
  ALWAYS stats 0 bytes (bin_attr size unknown); readability exists
  only ON READ (`cat data | wc -c` = 512,555 while `stat -c%s` = 0).
  Every `[ -s ]`/stat-size-gated capture (incl. the Q34 non-fires on
  #3/#4) is structurally blind on this driver. Crash #4's dump was
  likely readable in its limp window and lost to dismissal-by-cascade,
  not displacement alone.
- **MANUAL HARVEST BANKED:** `lce1/WD_crash_132400_card2.devcoredump`
  (512,555 B, md5 `9eaf8fa7c3bebf9af7ca240f40e93be9`) + dmesg
  `WD_crash_132400_dmesg.txt`. Header: `Reason: LR job cleanup,
  guc_id=32`, kernel 6.17.0-1010-intel, `Process: python3 [8290]`,
  PCI 0xe223. Sections present: GuC Log, GuC CT, Contexts, Job,
  HW Engines, VM state.
- **Forensic core (P73 input, from the dump):** context `ccs32`
  carries an UNFINISHED job (`Job: seqno=2509, fence=0, finished=0`;
  context Seqno 2508 → current job never completed); engine ccs0
  `RING_INSTDONE 0xffdefffe` (units not retired; SAMPLER_INSTDONE all
  done, ROW_INSTDONE 0x0); **`ACTHD/RING_BBADDR 0x0000d5569db23a04`**
  = the hung batch's active-head host VA — nameable against VM state
  page tables + a same-moment `/proc/<pid>/maps` (watcher v2 adds
  this); `RING_HEAD 0x03202fd4` >> `RING_TAIL 0x3028` (deep queued
  workload behind the wedge).
- Lifetime: **5 crash / 7 clean.** Watcher v2 (read-probe harvest +
  dismiss + pid maps snapshot, 100 KB completeness threshold) staged
  for the P73 era.
- **Host rebooted 13:31** per the standing post-crash directive.

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

Status: **READY 2026-10-02** (P75 handoff) — path 1 UNLOCKED: the
first readable full devcoredump of the mid-storm death class is banked
(`lce1/WD_crash_132400_card2.devcoredump`, 512,555 B, md5
`9eaf8fa7c3bebf9af7ca240f40e93be9`): unfinished ccs32 job seqno=2509,
ACTHD/RING_BBADDR `0x0000d5569db23a04`, full GuC log/CT/VM state.
Opening moves: (a) map ACTHD through the dump's VM state page tables
to the owning allocation; (b) decode the ccs32 HWCTX image (LRC head
12168); (c) GuC-log tail around seqno 2508→2509; (d) deploy watcher v2
(read-probe + `/proc/<pid>/maps` snapshot) so the NEXT crash pairs the
hung VA with the live process map; (e) if the kernel is ours
(custom_esimd_kernels_lgrf .so) → P74a, else upstream → P74c.

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
