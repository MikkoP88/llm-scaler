# PHASES.md — perf-v130 (WS-C follow-up: aggregate-KV-demand admission guard)

**Round CLOSED 2026-10-01 (~19:20): WS-C admission lever measured
DEAD at the production storm shape — structural null (Legs A/B/C
all dead-center in controls); the waste lives in between-turn
cyclic-LRU thrash; NO image bake (v1.2.27 stands). Full verdict in
P57/P58 below.**

Round opened 2026-10-01 (~17:10) as the direct continuation of
perf-v129's named follow-up: *"aggregate-KV-demand admission guard —
the WS-C lever candidate; trades storm throughput for stability +
cache retention; needs its own measurement round before any
production change."*

User directive (standing): deep improvement + fixing + testing
suite, bake into production image using best values; spec MTPx4 +
XGrammar-2 supported and crash-free; **no subagents**; commit+push.

Standing laws carried: SHIP-MEASUREMENT LAW (assert prod .so sha
1d9dcf4e before banking ship numbers); READOUT LAW (no device syncs
in the forward path — scheduler-side block tables are HOST state,
legal to read); no degradation anywhere; barriers 0; async ON;
live PHASES update per phase; ship-on-delta (a certified WS-C
baked-default policy IS an official bake trigger from the v128
policy list).

## Why this round (from v129 measurements)

- P52: under ~1.1x oversubscription, 99.1% of turn>=2 computed
  tokens are eviction recompute; full-history misses ~15.2k
  tok/affected; even 3k convs evicted. LRU already pins shared
  heads (pinning lever dead) — the lever is WHAT GETS ADMITTED.
- P51b: the same regime is a DEVICE_LOST crash onset (24-way,
  T+67 min; spontaneous fleet recurrences 03:09/03:34 same day).
- Physics: storm demand (1.03-1.44x pool) > capacity. No policy
  prevents ALL eviction when admitted demand > capacity; policy
  chooses the sacrifice. Baseline (admit-all) sacrifices every
  resident's cache every turn (99% waste) + stability. A demand
  cap sacrifices the EXCESS conversations' turn latency (queued
  until budget) while the admitted majority stays cached.

## P55 — design + code recon (this phase)

Insertion point (lane v1.2.27, `v1/core/sched/scheduler.py`):
the WAITING-admission loop inside `schedule()` (starts ~L854):
per-candidate, after `num_new_tokens` is computed and before
`kv_cache_manager.allocate_slots` — hold (skip WITHOUT evicting)
when projected demand exceeds the cap:

    hold iff run_footprint_tok + cand_new_tokens > HIGH x pool_tok

where `run_footprint_tok` = sum of allocated blocks over RUNNING
requests (host-side block tables — READOUT-LEGAL), and cache-only
blocks are deliberately EXCLUDED (they are the evictable buffer;
counting them would deadlock behind dead conversations).

- **Liveness proof:** a lone candidate admits iff its own demand
  <= HIGH x pool; max_model_len 262,144 < 0.92 x 476,451 = 438k
  -> true for any HIGH >= 0.56 -> no permanent hold at the
  measured operating points.
- **skip uses `continue`, not `break`:** smaller waiting requests
  can still admit under the same cap (best-fit filling — the
  SMALL cohort should survive waves, unlike P52's 0.109 hit).
- **v66 FAIRFIX interplay:** `_v66_bypass_now` (starved waiting
  head > 2 s) overrides the hold for the HEAD candidate only —
  starvation always wins (v66 law), but the throttle still paces
  the rest of the wave one head at a time.
- **Knob:** `VLLM_V130_ADMIT_HIGH` (float, default 0 = OFF ->
  constant-false, byte-identical behavior). Read at module top in
  the V63/V64/V66 pattern (scheduler runs in EngineCore — env
  reaches it; the marker-file law applies only to TP workers).
- **Announcements:** `V130_ADMGUARD_ACTIVE high=... run_tok=...
  cand=... waiting=...` once + rate-limited; hold counter on self.
- Solo/quiet regimes never reach the cap (W1 fleet kv usage p50
  0.42 / max 0.71; single 100k conv = 0.21 of pool) -> guard
  inert exactly where the no-degradation law lives.

Status: CLOSED — anchors identified (waiting loop ~L854-990,
`allocate_slots` call ~L1033, v66 flag ~L578).

## P56 — patcher + knob-off inertness certification

`patch_v130_admitguard.py` (staged /root/build/v130_stage/):
anchors through EOL, idempotent (marker comment + V130_ADMGUARD_OK
caps line), .bak backup, py-compile check. Applies to the LIVE lane
container after a quiet-check; env via `sed 1i export` in
serve_user.sh + serve relaunch (certified env-change path; the
boot script itself is md5-guarded and untouched).

Gates (knob OFF, patched file):
- py-compile + V130_ADMGUARD_OK apply marker
- solo spot (guard inactive — usage far below cap)
- serialized 24/24
- knob-off pressure spot: behavior within noise of the banked P51c
  control (the guard must not change scheduling when HIGH=0)

Status: CLOSED — ALL GATES GREEN —

| Gate | Result |
|---|---|
| py-compile + apply marker | `V130_ADMGUARD_OK markers=4` (dry copy first, then live) |
| solo spot | PASS (2.0 s, clean "OK"; perreq row landed, full schema) |
| serial24 | PASS 24/24, wall 15.3 s |
| knob-off pressure vs banked P51c | 72/72 ok; WASTE_PCT 99.2 (control 99.1); HEAD_RETAINED both (SHARED miss floor med=5120); SMALL_PRESSURED both (0.093 vs 0.109); MISS_TOK 16839 (control 15239) |

- Patcher `patch_v130_admitguard.py` written + dry-certified on a
  copy (S1 knob block / S2 per-step state / S3 waiting-loop hold
  before `allocate_slots`; anchors through EOL; idempotence marker;
  .v130bak; temp-write -> py-compile -> atomic replace; --restore
  mode round-trips byte-identical). DRY FINDING fixed en route: the
  S3 anchor needs the sum block at 20/24/20 spaces (nested inside
  the encoder-decoder `if`), not the 16/20/16 first sketched.
- LIVE APPLY after quiet-check (no fleet traffic since 15:53):
  watchdog paused, serve SIGTERM'd (PID 236), relaunched detached
  via `serve_user.sh` — VERIFIED byte-identical to the boot
  generated `serve_user_v1227.sh`, so the invocation IS the
  certified P51C one (perreq comes from the marker file, not env).
  Healthy after 150 s; KV pool 476,451 tok; zero V130 log lines;
  watchdog resumed. The only pre-run Tracebacks in the log are
  pid=236 lines from 17:06 (pre-restart, pre-run) — run-clean.
- SHIP-MEASUREMENT LAW asserted: prod lgrf .so sha256
  1d9dcf4e...85dff exact.
- **Inertness argument for the wall delta** (1318 s vs 1023 s):
  per-turn walls knob-off/P51c = 154/185, 108/97, 194/160, 248/200 —
  turn-1 (eviction-free, where any inserted-code cost would appear;
  schedule() runs hottest during 18-way prefill) ran FASTER, and the
  knob-off hot-path delta is 3 assignments + 1 constant-false test
  per candidate (us-scale, physically incapable of 40 s/turn). The
  later-turn slowdowns sit exactly where eviction-churn cascades
  dominate. Structural scheduling verdicts identical -> scheduling
  behavior byte-identical; wall/miss deltas = storm-regime variance.
- **Control-variance bar for P57 (this run is a 2nd control
  sample):** walls 1023/1318 s (~±15%), MISS_TOK 15239/16839 (~±10%)
  — the treatment effect must exceed this spread to count.
- **Harvester stale-row law (new):** the driver's harvester `seen`
  set starts EMPTY and dedupes only across its own passes — the
  "snapshot" is merely the docker-cp download target. Any rows
  pre-existing in the container jsonl at T0 merge in (98 stale rows
  this run; P51c was immune only because its container file was
  born empty). Discipline for every future leg: TRUNCATE the
  in-container jsonl at T0 AND ts-filter the harvest in analysis.
- **Watchdog caveat for P57:** lane-watchdog fires
  `boot_v1227_restore.sh` on sustained health-000, which
  RE-CREATES the container from the image — silently erasing the
  in-container v130 patch (safe-revert). Any armed leg must verify
  `V130_ADMGUARD_ARMED` in the serve log after watchdog activity.
- Knob-off leg driver: `wsc_pressure_v130.py`
  (`wsc_pressure_v129b.py` verbatim, 4 output constants redirected
  lce1 -> v130_stage; banked P51c artifacts left immutable).

## P57 — A/B storm legs (the measurement)

Driver: `wsc_pressure_v129b.py` verbatim (18 convs / 4 turns /
barrier pacing / harvester thread — crash-proof perreq harvest).
Legs on the prod posture (fresh certified boot; prod .so sha
asserted before banking):

- CONTROL: TWO banked samples now — P51c (72/72, WASTE_PCT 99.1,
  MISS_TOK 15239, wall 1023 s) and the P56 knob-off cert run (99.2 /
  16839 / 1318 s). Control spread: walls ~±15%, MISS_TOK ~±10%.
- TREATMENT LEGS (recalibrated mid-round, see below):
  - **Leg A: VLLM_V130_ADMIT_HIGH=0.92** — armed 17:56
    (`V130_ADMGUARD_ARMED high=0.92` verified, markers intact,
    armed-path solo smoke PASS, watchdog resumed, perreq truncated
    at T0). Launched 17:59.
  - **Leg B: VLLM_V130_ADMIT_HIGH=0.62** — the real waste-leg
    treatment (rationale below).

  **Threshold recalibration (from conv-mix arithmetic):** the wave's
  uncached demand = 5x562 + 5x336 + 8x47 = 4866 blk ~= 311k tok =
  **0.65x pool**. The guard's budget is RUNNING-only (cache
  deliberately excluded), so HIGH=0.92 (and 0.85) can NEVER engage
  at the P51c storm shape — Leg A is expected inert-by-arithmetic at
  this shape and banks (i) threshold-insensitivity + no-harm at the
  P51c shape and (ii) the 0.92 point for the P51b 24-way shape
  (~0.91-1.0x demand), where the DEVICE_LOST stability story lives.
  The waste lever at THIS shape needs HIGH in (0.56 liveness floor,
  0.65 wave demand) -> **0.62**: the wave tail holds, the held
  convs' cold histories become the eviction victims instead of
  everyone's, admitted convs re-ref their own prefixes at admission
  (LRU promotion) -> waste should collapse from 99% toward the
  held-tail share (~44% analytic floor: 5x36k victims vs
  13x17.5k deltas + 5x36k recompute), at the cost of held convs'
  turn latency (the accepted trade). Liveness holds (0.62 > 0.56 ->
  a lone 262k request still admits).

Metrics (split via wsc_split_v129.py):
1. WASTE_PCT and MISS_TOK_PER_AFFECTED (target: large reduction)
2. SHARED head retention (must stay = 5120 floor; LRU untouched)
3. ok/N completion parity (target 72/72), turn walls
4. TTFT/queue cost of held conversations (the accepted trade —
   bounded, vs P51b's total engine death)
5. No DEVICE_LOST / no 500s / no timeouts
6. Recorder join: waiting depth + kv_perc timeline during waves;
   V130/V66 announcement counts (starve-bypass frequency under
   the guard = the fairness stress signal)

### P57 leg results

- **Leg A (HIGH=0.92): NULL, as the arithmetic predicted.** 72/72,
  wall 995 s, **0 holds**, WASTE 99.0, MISS 14032 — dead-center in
  the control distribution (99.0-99.2 / 14032-16839 / 995-1318 s).
  Banks threshold-insensitivity + no-harm at the P51c shape and the
  0.92 point for the 24-way DEVICE_LOST shape (never engaged there
  either — no storm was run at that shape this round).
- **Leg B (HIGH=0.62): NULL — 1 hold, and the hold exposed the unit
  bug.** 72/72, wall 1161 s, exactly one hold (18:22:33):
  `V130_ADMGUARD_ACTIVE high=0.62 run_blk=324 adm_blk=3 cand_blk=3
  total_blk=497 waiting=2 holds=1`; WASTE 99.1, MISS 14367 — null
  dead-center in controls.
- **Diagnosis (why both armed legs were inert):**
  - The hybrid boot pads pages byte-equal —
    `interface.py:645/669` set attention block size 896 -> (XPU)
    -> 1024 tokens with mamba page padded to 1 MiB — and
    `resolve_kv_cache_block_sizes` (kv_cache_utils.py:570) makes the
    scheduler's `self.block_size` the **LCM of group page sizes =
    1024**. So `total_blk=497` is 1024-**token pages** (not 64-tok
    blocks) and `cand_blk` was page units all along — those two were
    right.
  - The bug: `run_blk` **SUMMED all KV cache groups' block-id
    lists**, and the live engine has **groups=4** (hybrid mirrors
    page allocations across groups) — a 4x-mirrored run term of
    ambiguous sign vs the wave arithmetic. Combined with the
    mid-wave waiting loop breaking on token budget BEFORE the
    allocate (the guard only evaluates candidates the budget would
    admit), the net engagement was 1 hold in a drain moment.
- **Patch v2 (Leg C):** run term = per-request **MAX over groups**
  (correct under mirroring AND under O(1)-mamba layouts; the
  attention footprint dominates), plus a one-shot
  `V130_ADMGUARD_ANAT` anatomy line at first armed check. Dry-cert
  green (apply markers=4, restore round-trip byte-identical), live
  v1 restored + v2 applied, knob **HIGH=0.55** (page-unit
  recalibration: wave ~304-479 pages = 0.61-0.96x of 497; liveness
  floor 262,144 tok = 256 pages -> HIGH >= 0.52).
- **Leg C (HIGH=0.55, v2): RUNNING since 18:49.** Boot verified:
  `V130_ADMGUARD_ARMED high=0.55`,
  `V130_ADMGUARD_ANAT bs=1024 total_blk=497 groups=4 run_blk=0` —
  the anatomy line confirms the page-units model live. Solo smoke
  PASS (2.0 s). Watchdog active; perreq truncated at T0 (18:49:03).
  (Early check ~2 min in: no holds yet — the only ACTIVE line in
  the log was the stale 18:22:33 v1 one. The guard evaluates only
  candidates the token budget would admit; early-wave chunked
  prefills hadn't reached the check yet.)

Status: CLOSED — all three legs NULL; see Leg C result below.

### P57 Leg C result — the decisive null

- **72/72, wall 1059 s, ZERO holds** (corrected units, HIGH=0.55),
  WASTE_PCT 99.1, MISS_TOK 14946, HEAD_RETAINED (floor med 5120),
  SMALL_PRESSURED (0.177) — every metric dead-center in the
  3-sample control distribution (WASTE 99.0/99.1/99.2; MISS
  14032/15239/16839; walls 995/1023/1318 s).
- **Mechanism isolated (why zero holds):** engine samples during
  the wave show Running 1-3, Waiting 5-14, GPU KV usage 8-19% —
  chunked prefill (MNBT 8192) admits big prefills 2-3 at a time, so
  the CONCURRENT running footprint is structurally ~15% of the
  pool; a running-only demand cap at any HIGH >= 0.52 can never
  engage. The one v1 hold (run_blk=324) was the 4-group-sum
  inflation artifact crossing a low threshold, not real demand.
- **WS-C LEVER VERDICT: the aggregate-KV-demand admission guard is
  DEAD at the production storm shape — by construction, not by
  tuning.** The 99% waste is SEQUENTIAL CYCLIC LRU THRASH: all 18
  convs re-send full histories every turn; the aggregate cached
  working set (~490k tok ~= 0.96x pool) exceeds the pool, and
  cyclic access is LRU's worst case (every access misses). The
  damage flows through cache replacement between turns, which an
  admission-time running-demand check cannot see. (The guard could
  still matter at the P51b 24-way CONCURRENT oversubscription shape
  — a stability lever there — but that shape kills the engine and
  was not run.)
- **Where the lever actually lives (recorded for the next round):**
  EVICTION POLICY, size-aware. P52/P57 splits consistently show the
  sacrifice hierarchy under LRU: PRIV tails die (cached=0), SMALL
  dies (hit 0.11-0.28), SHARED heads survive (5120 floor, free via
  simultaneous touch). A size-aware evictor (protect blocks under a
  ~4k-token footprint; evict PRIV tails first) costs ~24k tok of
  pool (8 SMALL convs) and would take SMALL hit -> ~1.0 while PRIV
  thrash continues: total recompute mass drops ~60% (5x35k PRIV
  victims remain vs everyone's histories today). NOTE the metric:
  WASTE_PCT (share of computed that is recompute) stays ~98% under
  that policy because it is a RATIO — the mover is total computed
  tokens / wall; size the next round's gates accordingly.

## P58 — verdict + ship decision

**VERDICT: CLOSE BY MEASUREMENT — NO BAKE.** All three treatment
legs (0.92 / 0.62 / 0.55-v2) are nulls dead-center in the
3-sample control distribution, and the Leg C mechanism analysis
shows the null is structural (running-only demand ~15% under
chunked-prefill pacing; the waste is between-turn cyclic-LRU
thrash). The official v128 trigger "WS-C baked-default policy"
does NOT fire. Production stays **llm-scaler-exp:v1.2.27**;
v1.2.28 remains not required.

Lane posture restored and certified at close:
- scheduler.py restored pristine (`--restore`, markers_left=0,
  on-disk count 0); `VLLM_V130_ADMIT_HIGH` line removed from
  serve_user.sh; relaunched via the certified path
  (`docker exec -d bash /root/serve_user.sh`, APIServer pid 8919);
  health 200; all 8 historical V130 log lines pre-date the close
  restart (none from the new boot); solo smoke PASS 2.0 s;
  watchdog active; prod lgrf .so sha256 asserted exact
  (1d9dcf4e...85dff).
- Patchers banked in-repo for the record: `patch_v130_admitguard.py`
  (v2 = page-units-correct; v1's group-sum bug documented in its
  docstring), restore path proven round-trip byte-identical.
- Captures: captures/wsc_{pressure,perreq}_v129b-{knoboff,armed092,
  armed062,legc055}.jsonl + wsc_v130_*.out + wsc_split_v130_legc.txt
  + wsc_perreq_v130knoboff.jsonl (ts-filtered 72-row set).

Standing items handed forward:

1. **Size-aware eviction policy** (the measured-live WS-C lever:
   protect small footprints, evict PRIV tails first; gates must
   track total computed tokens / wall, not WASTE_PCT — see P57).
   Own round; eviction.py surgery (FreeQueue LRU -> two-tier).
2. The admission-guard patch is inert-but-certified at the P51c
   shape; revisit ONLY at a genuinely concurrent-oversubscription
   posture (e.g. after a crashfix that stabilizes 24-way), as a
   STABILITY lever (P51b class), not a waste lever.
3. Hybrid-anatomy law (new, banked): scheduler `self.block_size`
   = LCM of KV-group page sizes = 1024 tok on this lane (pages
   padded byte-equal across attention/mamba); `num_gpu_blocks`=498
   counts 1024-tok pages; **4 KV cache groups** mirror allocations
   — any host-side footprint sum MUST take max-over-groups, and
   `V130_ADMGUARD_ANAT` (banked in the patcher) is the one-line
   way to re-verify anatomy on any future boot.

Status: CLOSED.
