# FIX_AND_TEST_PLAN.md — v127 comprehensive stall-fix + improvement program (rev 2)

**GOAL (user, 2026-09-30): FIX ALL OF IT. These are MAJOR issues; weeks of
work is acceptable — issues have to be FIXED, not documented.** Every issue
identified below ends in either a root fix shipped to a certified image or
a measured rejection with the measurement named. Nothing is left as
"known-broken" except the explicitly banked format-impossibility verdicts
(P29: raw-fp8 SSM arithmetic — which this program routes AROUND via
scaled-space).

Standing constraints (non-negotiable, carried): no subagents this round;
**no degradation anywhere**; spec MTP×4 + XGrammar-2 supported and
crash-free on every image; barriers default 0, never BARRIER=1;
`--async-scheduling` required (V1212-lineage boots FORBIDDEN);
SHIP-MEASUREMENT LAW (assert prod .so sha `1d9dcf4e…` before banking
numbers); READOUT LAW (never read device state from the forward path);
KERNELS_MAX_JOBS=52 on every wheel/kernel build; host reboot after big
changes / before test rounds; live PHASES.md update after every phase;
bake from -raw, never the running lane; build JSON bodies with python
json.dumps; litellm always via `Authorization: Bearer sk-dummy` at :4000;
litellm-proxy is `unless-stopped` — never re-create it casually; live z.ai
key in litellm configs — never spread it.

---

## STANDING TELEMETRY DIRECTIVE (user, 2026-09-30 — applies to every phase)

> **Every phase ships with its telemetry capture points defined BEFORE the
> change lands. Root issues must be capturable at the moment they happen —
> from recordings, not from forensics.** The P32 stall had to be
> reconstructed hours later from cumulative counters and `.bash_history`;
> that failure mode is itself a defect this program fixes first.

Instrumentation layers (all host-side, read-only, WARN-only — nothing here
may kill/restart the lane or touch the engine forward path):

| Layer | Artifact | What it captures | What it convicts |
|---|---|---|---|
| Continuous recorder | `metrics_recorder.py` (10 s JSONL at `/root/build/telemetry/metrics_*.jsonl`) | running/waiting/waiting_by_reason, `kv_cache_usage_perc`, preemptions, prefix hits/queries, `prompt_tokens_by_source`, prompt/TTFT/TPOT bucket vectors | eviction-pressure cycles (kv sawtooth), re-prefill storms (local_compute deltas), admission backlog onset, TTFT tail growth in real time |
| Stall scoper | `stall_scope.sh --watch` (waiting>3 for 30 s, or kv>0.97 for 60 s; 300 s cooldown) | full /metrics snapshot + serve-log tail + py-spy dumps (EngineCore+workers) + xpu-smi ×2 + dmesg tail + litellm 30 m + recorder tail → timestamped tarball `/root/build/captures/` | WHICH stall class an episode is, with stack+GPU evidence at the moment of the stall |
| Boot baseline | `boot_v1227_restore.sh` evidence block | boot duration, KV pool size, max concurrency, posture assertions | posture drift at boot (Cause A recurrence), capacity regressions per image |
| Replay gate | `cc_fleet_replay.py` | per-request TTFT/TPOT/e2e + re-prefill volume under a synthetic CC-fleet | whether a leg actually fixes the tail (PASS = p99<20 s, max<45 s, no re-prefills beyond first turns) |
| Serve log | chain-launched lanes only | `V66_FAIRFIX_ACTIVE`, preemption/retraction/JIT lines, block-pool warnings | scheduler-leg engagement; config-path divergence |

**Phase rule:** a phase is not COMPLETE until its capture layer has recorded
at least one full replay + one live-fleet interval, and the PHASES.md entry
names the file that holds the evidence. A leg whose gain does not show in
the recorder/replay numbers is a REJECTED leg regardless of microbenchmarks.

---

## Situation (fresh capture 2026-09-30 ~16:05 UTC, lane = designated swift posture, uncertified)

314 requests since 08:56 boot; prompt mix 49% in 50k–100k, 44% in 20k–50k;
prefix hit 91.4% (14.33M cached / 15.68M total); `local_compute` 1.357M
tokens cumulative. **TTFT: p50 ≈ 5 s, ≤20 s for 82% — but 29 requests
20–40 s, 10 requests 40–80 s, 10 requests 80–160 s, and 8 requests
160–640 s.** Decode healthy throughout (TPOT p99 < 0.75 s/token; no request
averaged above it), preemptions 0, dmesg Engine resets 0 (current boot),
litellm errors 0 in 3 h, spec acceptance healthy. The "hard stall" = the
TTFT tail, full stop. Everything below attacks exactly that tail.

## Root-cause hierarchy (code-anchored, each verified this round)

**RC1 — CAPACITY GAP (structural, the #1 driver).** Working set of the
fleet ≈ N_sessions × ~100k tokens ≥ 800k; KV pool ≈ 493k fp8_e4m3 tokens
(gmu 0.8) fits ~4–5 resident 100k sessions. Verified in
`vllm/v1/core/block_pool.py`: blocks of a RUNNING request are unevictable
(`ref_cnt>0`); at request finish `free_blocks()` appends the whole prefix
to the free-queue TAIL (evict-last); `get_new_blocks()` pops from the HEAD
(LRU-first). Between turns — while an agent does tool work for
seconds-to-minutes — its entire 100k prefix sits evictable; the
oldest-idle sessions lose their prefix to newer prefills; their next turn
re-prefills ~100k tokens. LRU is near-optimal for this access pattern —
**the miss rate is set by capacity, not by policy**, which fixes the
priorities below.

**RC2 — MISS PENALTY: prefill throughput (structural, #2 driver).** A full
100k re-prefill at the measured contended effective rate (~1.1–2.5k tok/s)
costs 40–90+ s; chunked prefill (mnbt 8192 → 13 chunks) also steals
compute from every other stream while it runs (v64 interleave keeps decode
alive — TPOT survived — but other sessions' PREFILLS queue behind it).
v65 analysis already bounded this: prefill is FA2-attention + GEMM bound
(solo-cold decomposition 40.1 s attn / 57.6 s GEMM at 34k ctx); only
weeks-scale DPC++/ESIMD kernel work moves it. That work is IN this program
(WS-E), not deferred.

**RC3 — STORM SCHEDULING (tuning, #3 driver).** v63 (contended prefill
budget 1024), v64 (decode interleave K=2/budget 512), v66 (waiting-head
bypass after 2 s) were all tuned and frozen on ≤1024-token contexts. Under
a 100k re-prefill storm their operating point is unmeasured (is
`V66_FAIRFIX_ACTIVE` even firing? — recorder + serve-log capture answers
this in W1).

**RC4 — GDN/SSM POOL MEMORY (capacity contributor).** The GDN state pool
is fp16 (P29 certified endgame; scaled-space M2/M3 halves it → frees GPU
memory for KV at the same gmu — direct RC1 relief). Exact pool byte split
KV-vs-SSM gets measured and recorded at W1 boot (mamba_utils logging is
already switched to info in the boot chain).

**RC5 — BLINDNESS + UNCERTIFIED ROLLOUT (the Cause A residue).** The
08:56 lane ran the designated config outside the boot chain (no patcher
verification, no marker gates) with stdout to a dead pts — zero engine
telemetry. Fixed by: certified boot chain (WS-A), watchdog v2 drift guard
(WS-A), recorder + scoper (WS-B, running from NOW).

**RC6 — AMPLIFIERS (audit items).** (a) Client retry/timeout behavior: a
client that aborts a slow TTFT and re-sends DOUBLES the prefill load —
litellm retry policy for local entries must be audited and pinned
(WS-F). (b) GDN prefix-cache granularity is 4096 tokens (banked v65
note) → up to 4k tokens recompute per turn on drift; measure its real
share from recorder deltas before deciding whether a kernel-granularity
work package exists (WS-D D4).

**Explicitly NOT causes** (exonerated by capture; keep recorded so future
rounds don't re-litigate): GPU wedge (resets 0; the f15b_dmesg reset lines
are historical baked content); decode starvation (TPOT < 0.75 s
everywhere); preemption/retraction storms (0); litellm faults (0 in 3 h);
spec decode (acceptance healthy); the engine being "broken" (decode
aggregate throughput matches the banked async ceiling class).

---

## STATUS LEDGER (live — updated 2026-10-01)

| Item | State | Evidence |
|---|---|---|
| WS-B B1/B2 telemetry | **LIVE since 2026-09-30** (recorder 10 s JSONL + scoper watch, host-side, WARN-only) | `/root/build/telemetry/metrics_*.jsonl`, `/root/build/captures/` |
| WS-A A1/A2/A3 (W1 recert) | **DONE 2026-09-30, CERTIFIED 2026-10-01 (P38)** — full battery green under live CC storm (0 wedges/resets/tracebacks); quiet re-run set CLOSED on reverted gmu-0.8 posture: admission PASS (TTFT ≤8.67 s), serial 24/24 ×3 = 72/72, solo genspeed 73.71 = bank-exact, async 4×1024 173.71 = new quiet record, xgrammar crash-free zero FSM errors → v1.2.26 swift-fp8 @ 0.8 = certified production posture | PHASES P34+P38, lce1/quiet_* logs |
| WS-C C1 pool split | **DONE (W1)** — concurrency cap is the SSM pool, NOT KV: waiting reason=capacity fires at KV p50 0.42 | PHASES P34 envelope |
| WS-C C2 gmu 0.85 | **NOT-EXECUTE — USER DIRECTIVE 2026-10-01 (P38)**: wedge-crash attribution (the 22:13 v88-class wedge landed on the 0.85 posture at +1 h 38 m) — leg forbidden; live lane REVERTED to certified 0.8 (KV 497,499); historical measurement retained for the record (+18.45 % KV while it ran); gmu 0.88 leg MOOT (same attribution, higher pressure); C3 scaled-space remains the capacity path | PHASES P35/P37/P38 |
| WS-F F1 retry=0 | **DONE 2026-09-30** — `num_retries: 0` pinned on the 5 local litellm entries | `wsf_litellm_local_retry0.py`, litellm config |
| WS-E M0 offline | **DONE** — 87.5 % of features over the raw e4m3 ceiling → 0 clipped after scaling; clamp-policy delta recorded (drop `max=1.0` at M3) | `calibrate_offline.py`, M2_DESIGN.md §2 |
| WS-E M1 bridge | **MEASURED + REJECTED 2026-10-01 (P43)** — leg ran clean (engaged both ranks, KV +3.4 %); MATH 0/80+0/80 (representability FIXED — P29 share gone) but TOOL 30/30 salad + counting collapses at decode step ~6 (`1\n2\n3\n!!!…`); fp16 control identical-probe PERFECT (tool_calls @45 tok, clean 1..60). ROOT CAUSE = RECURRENCE-COMPOUNDING LAW: e4m3 mantissa floor ~2⁻⁴/roundtrip compounds per decode step through the GDN recurrence (KV survives fp8 because it is NOT recurrent); scaling cannot fix a floating-point relative floor → **fp16 is the minimum viable recurrent-state storage** | PHASES P43, lce1/m1_forensics.log vs ctrl_forensics.log |
| WS-E M2 kernel | **CLOSED 2026-10-01 (P43 consequence)** — kernel itself was BUILT + HARNESS-PASSED (P36, v132 .so sha `efb53bef…`, gates A-D incl. stratified superiority), but it stores the same e4m3 format the recurrence-compounding law damns: harness superiority was on stratified STORAGE error, not on compounding-through-recurrence decode quality. Parked as a documented-rejected artifact; NOT on any lane and never ships | perf-v127/scaledspace/, PHASES P36+P43 |
| XGrammar degeneration under load | attribution REFINED (P38 quiet sample): stochastic at the probe's DEFAULT temperature 1.0 — 4/11 quiet-lane runs degenerate (grammar-legal unbounded number emission, finish=length, ZERO FSM/backend errors); load-function component from W1 stands, load-independent sampling component now measured → WS-D fix leg carries the rate | lce1/quiet_extra_REV08Q.log, PHASES P38 |
| W2-B (D1 mnbt 16384) | **MEASURED + REJECTED 2026-10-01 (P39)** — 4×50k×2 A/B: TTFT p50 +15.5 %, TPOT p50 +86 % (265 vs 142 ms), worst-gap 2.75×, KV −14.6 % (424 788 vs 497 499); mechanism = 16k chunk blocks decode ~13 s. **mnbt stays 8192** | lce1/d1_*.log, PHASES P39 |
| D2 v63 budget 4096 | **MEASURED + REJECTED 2026-10-01 (P40)** — TPOT p50 +20.2 % (171 vs 142 ms), TTFT p50 +5.7 %, nothing improved; prefill counts byte-identical to control; brackets 1024 from above (v89 T1 bracketed from below). **baked 1024 stays** | lce1/d2_v63_4096.log, PHASES P40 |
| v66 sweep 2.0/5.0 | **MEASURED + REJECTED 2026-10-01 (P41)** — 5.0: TTFT p50 +6.7 %, TPOT +11.2 % (fair-fix just fires 3 s later); `VLLM_V66_PREFILL_STARVE_S` passthrough added to boot script (V1227_V66_STARVE). **baked 2.0 stays — D-knob program CLOSED: certified posture = measured optimum from both directions** | lce1/d2_v66_50.log, PHASES P41 |
| M0-live boot, C4 | **DONE 2026-10-01 (P42)** — collector built+debugged (anchor at dtype-neutral entry: the fp8-only gather line never fires on fp16 lanes); harvest: F=393 216, p50=45.6 / p100=6 924, 7.63 % of live features over the raw e4m3 ceiling (P29 confirmed live), cross-rank ratio p50=1.60 / max=63.45 → shared merged scales decided (e4m3 relative precision is scale-invariant; only denormal tail at the 63× end); calibrated no-clamp-up 0 clipped, staged `v127_stage/v127_scales_e4m3.pt` | PHASES P42, lce1/m0live_calibrate.log |
| WS-D D4 GDN granularity | **MEASURED 2026-10-01 (P37)** — recompute excess 24.7 % of computed ≥ 15 % bar; storm/quiet computed-per-req 2138.6 vs 400.6; GDN-snap vs eviction split unresolved ⇒ per-request split leg BEFORE any weeks-scale kernel commitment | `d4_d2_measurement_20261001.json` |
| WS-D D2 measurement half | **DONE 2026-10-01 (P37)** — v66 firing CONFIRMED (81 fires @ starve 2.0 s); waiting 100 % capacity-class (max 67); TTFT: 21 % >40 s, 10.6 % >160 s, median ≈5 s | P37 + serve_live.log |
| LANE INCIDENT 2026-09-30 22:13 | **RESTORED 2026-10-01 04:49 (P37)** — v88-class GPU wedge (execute_model 601 s → EngineDeadError → XPU drain wedge); lane dark ~6 h 20 m (watchdog was window-disabled = detection gap); host reboot + `boot_v1227_restore.sh INC01` swift gmu 0.85 → KV 589,345 ✓ | P37 |
| lane-watchdog | **RE-ARMED 2026-10-01 04:50** (`systemctl enable --now` → active) — window ended de facto at the wedge; disabling the watchdog during a "no-stop" window inverts its purpose | P37 |

---

# PARTITIONS (workstreams) — each owns phases, gates, and evidence

## WS-A — STABILIZE & CERTIFY (the designated posture, properly)

| Phase | Change | Gate | Rollback |
|---|---|---|---|
| **A1 (W1)** | `boot_v1227_restore.sh` default `V1227_POSTURE=swift`: certified v1226 chain (image v1.2.26, full patcher sequence, marker+DBN checks) + designated serve (`--model /models/swift --served-model-name qwen3.8-27b-fp8 --quantization fp8 --chat-template /root/chat_template_qwen38_high.jinja`, template staged from `/root/build/v127_stage/` — NOT in the image) + telemetry to `/root/serve_full.log` + 25-min health ceiling (52 GB bf16 on-the-fly quant) | `BOOT_V1227_RESTORED posture=swift`; live-cmdline posture assertions (exits 12–15); boot duration + KV pool + max-concurrency recorded to PHASES | `V1227_POSTURE=legacy` one command (exact v1226 posture) |
| **A2 (W1)** | Full certification battery on the designated posture: (1) sanity/admission/v1226-posture subset, fence drills 3/3, serial24 ×3 24/24, bursts 108/108, resets 0; (2) **CC quality battery through litellm :4000 under the NEW template — tool calls (qwen3_coder parser), thinking blocks, no template echo, spec + XGrammar-2** — the highest-risk item, a template that breaks tool-call format is a FAIL regardless of speed; (3) speed spot-check vs banked 74 tok/s-class solo; (4) `cc_fleet_replay.py` baseline | `V1227_RECERT_CLEAN` in PHASES P33; any red → named failing gate + rollback decision | rollback above; template issues get a template-fix sub-phase, not a waiver |
| **A3 (W1)** | Wire `watchdog_v2_drift_check.sh` into lane_watchdog 60 s loop (WARN-only): image, swift mount binds, serve flags (model/served-name/quant/template/async/fp8-KV/fp16-SSM/spec), template-present-in-container (absence = relaunch outside the chain), serve-log freshness | `V1227_DRIFT_NONE` lines in watchdog log; live-fire test passes on current lane | unwrap the one appended line |

## WS-B — TELEMETRY (running from NOW; zero lane impact)

| Phase | Change | Gate |
|---|---|---|
| **B1 (NOW)** | `metrics_recorder.py` daemon live on host (10 s JSONL, daily files) | first JSONL lines verified; survives container recreates; re-armed by boot script |
| **B2 (NOW)** | `stall_scope.sh --watch` daemon live (waiting/kv signatures → evidence tarballs to `/root/build/captures/`) | `--now` manual capture produces a complete tarball; watch loop logs its thresholds |
| **B3 (W1+)** | boot-baseline capture block in every window (duration, pool sizes, JIT lines, first-token latency) — recorder keeps the post-boot interval | PHASES entry cites the file |
| **B4 (every phase)** | capture-checklist discipline per the TELEMETRY DIRECTIVE above | phase not complete without its evidence file named |

## WS-C — CAPACITY (attacks RC1+RC4, the #1 driver)

| Phase | Change | Hypothesis / math | Gate | Risk |
|---|---|---|---|---|
| **C1** | Measure + record the KV-vs-SSM pool byte split at W1 boot (mamba_utils info logs; boot evidence block) | tells exactly how much KV M2 frees and what gmu legs buy in tokens | numbers in PHASES; feeds C2/C3 decisions | none (measurement) |
| **C2** | `V1227_GMU` legs 0.85 / 0.88 — **NOT-EXECUTE (user directive 2026-10-01, P38: wedge-crash attribution)** | (+6%/+10% pool — historical only; 0.85 ran +18.45 % KV before the directive) | n/a — forbidden | n/a — forbidden |
| **C3** | scaled-space M2/M3 (WS-E) shipped → GDN pool halved → KV grows by the freed bytes at same gmu | direct RC1 relief; the structural fix | WS-E ship gates + replay + battery | WS-E's own gates |
| **C4** | sticky-prefix eviction experiment (patch on `block_pool.py`: TTL-protect prefixes of sessions active within N min) | honest expectation: LRU is near-optimal; at a FULL pool protection is zero-sum (picks a different victim). Ship ONLY if the replay gate shows a real tail win; otherwise REJECT BY MEASUREMENT and record | replay gate | wasted weeks if the math is right — hence last in the partition and measurement-gated |

## WS-D — PREFILL & SCHEDULING (attacks RC2+RC3, the #3 driver; RC2's kernel half lives in WS-E)

| Phase | Change | Hypothesis | Gate | Risk |
|---|---|---|---|---|
| **D1** | `V1227_MNBT=16384` leg | halves chunk count of a 100k re-prefill (13→7); fewer decode pauses per storm | replay gate (TTFT tail + TPOT p99) | decode cadence dip; capture-failure abort |
| **D2** | `V1227_V63_BUDGET=4096` leg (+ v66 threshold sweep 2.0/5.0 s using recorder's `V66_FAIRFIX_ACTIVE` + waiting-by-reason traces) | contended prefill budget 1024 was frozen on ≤1k ctx; 4096 lets storms actually drain; is v66 even firing at 2.0 s? | replay gate; v66 firing frequency recorded before/after | small-prompt TTFT during storms |
| **D3** | re-prefill head-of-line priority: a request whose prefix-hit ratio is LOW (full re-prefill) gets admission priority over incremental prefills once waiting > threshold (scheduler patch, v127 marker, env-gated) | the tail requests ARE the re-prefills; queueing behind incremental turns is pure added stall | replay gate + admission battery (v66 `ADMISSION_DONE verdict=PASS`) | starvation of small prompts — bounded by v66 machinery |
| **D4** | GDN 4096-token granularity measurement from recorder deltas (hit-vs-computed per turn boundary) | quantifies up-to-4k-tok/turn recompute share; only escalates to a kernel work package if ≥15% of computed tokens | **MEASURED P37: 24.7 % ≥ bar** — next: per-request split leg (GDN-snap vs eviction attribution) before kernel commitment | none (measurement first) |
| **D5** | FA2 prefill + GEMM kernel program (weeks-scale; v65 tier list) — merged into WS-E's kernel track as K2 | moves the 40–90 s miss penalty itself; only weeks-scale work moves it | bitwise/correctness harness + superiority matrix, same law as K1 | weeks of effort — accepted by directive |

## WS-E — SCALED-SPACE fp8 SSM KERNEL PROGRAM — **CLOSED 2026-10-01 (P43: recurrence-compounding law)**

- **M0 — DONE (offline P29M-prior + LIVE harvest P42).** `calibrate_offline.py`
  + `m0_live_collect_v127.py`/`m0_harvest_v127.py`: live F=393 216,
  p100=6 924, 7.63 % over-ceiling → 0 clipped after no-clamp-up scaling,
  stored max 439/448; shared cross-rank scales (ratio p50 1.60 / max 63.45,
  safe direction — e4m3 relative precision is scale-invariant).
- **M1 — MEASURED + REJECTED (P43).** `patch_scaled_space_v127.py` ran
  armed on the M1LEG boot: representability fixed (MATH 0/80+0/80, 0
  flips — vs v126 raw-fp8's concurrent-wrong class) BUT structured
  decode collapses at step ~6 (tools 30/30 salad, counting `1\n2\n3\n!!!`;
  identical probe on the fp16 control = perfect). ROOT CAUSE: e4m3
  mantissa floor ~2⁻⁴ per write-read roundtrip COMPOUNDS through the
  recurrent GDN state (~N·eps ⇒ visible at N≈6); scaling repairs the
  ceiling, not the relative floor. fp8 KV survives because KV is not
  recurrent. **fp16 = minimum viable recurrent-state storage.**
- **M2 — CLOSED (P43 consequence).** Kernel built + harness-passed (P36)
  but it stores the same e4m3 format the law damns; the harness measured
  stratified STORAGE error, not compounding decode quality. Parked,
  documented-rejected; the v132 .so stays staged but unshipped.
- **M3 — CLOSED** (was conditional on M2 superiority; never reached).
  v1.2.28 scaled-space bake cancelled; `V1227_EXPECT_SSM` watchdog
  switch stays at its fp16 default permanently.
- **REOPEN CONDITIONS (documented 2026-10-01, user question; NOT scheduled —
  both lose to the no-degradation law at the current +3.4 % KV prize):**
  - **A — per-feature-scaled int8 pool.** int8 spends all 8 bits on mantissa
    (exponent handled by the M0/P42 scale vector, already harvested) →
    ~2⁻⁸ ≈ 0.39 %/roundtrip = 16× better than e4m3, same 1 byte, same
    capacity win. Moves collapse from N≈6 to N≈100 (linear) / N≈1500 (√N).
    GATE ZERO before any kernel work: measure which accumulation regime the
    live recurrence actually has (one forensics-style probe on the M1 data:
    collapse-at-6 with ε=6.25 % bounds the tolerance at ~15 %; the same
    probe design gives the √N-vs-linear slope). Verdict framework: √N +
    tolerance ≥ 3× the 2048-token bound (0.39 %·√2048 ≈ 17.6 %) = still
    FAILS → int8 stays closed; only a measured tolerance ≫ 18 % reopens it.
    Cost if reopened: full M2-scale int8 SYCL kernel program.
  - **B — fp8 spillover for FROZEN states only.** The law damns states that
    keep recursing; a preempted/parked request's state does not step.
    Quantize once on park, dequantize once on resume = a single ~6.25 %
    roundtrip (~1 decode-step of noise — survivable; math tolerated ~10
    steps at that rate in the M1 leg). Converts SSM-paging into checkpoint
    storage, the same class as KV where fp8 is proven. Payoff = resume-
    without-recompute latency (2 preemptions observed in the W1 storm),
    ZERO capacity gain; cost = deep scheduler surgery on the hybrid-mamba
    preemption path. Reopen only as an explicit scheduler project.
- **K2 — prefill kernels (D5 merged here).** Same build discipline
  (KERNELS_MAX_JOBS=52, `//` patch marks, wheel version-gating on install).

## WS-F — FLEET/PROXY HYGIENE (attacks RC6 amplifiers)

| Phase | Change | Gate |
|---|---|---|
| **F1** | litellm audit for the local entries: retry count, timeout, fallback chains — pin retry=0 (or timeout ≥ worst-case TTFT) for LOCAL models so an aborting client cannot double-queue a 100k prefill; record in the config note (config edits need a litellm restart to take effect — coordinate with the user; proxy is `unless-stopped`, use the standing recreate script only if needed) | config diff + one observed no-retry episode in litellm logs |
| **F2** | CC-fleet note (user-side, informational): context-editing/compaction reduces resent history; engine-side fixes must NOT depend on this | note in PHASES |

## WS-G — SHIP DISCIPLINE

- Bake v1.2.27 (capability deltas ONLY — the dormant M0-live collector;
  the D-knob program closed P39-P41 with zero winners, so no knob deltas;
  the scaled-space patch is EXCLUDED: known-broken-when-armed, P43) —
  v1.2.28 cancelled (WS-E closed). Bake ONLY
  from -raw containers; full gate battery re-run on the committed image;
  caps-marker convention on every patch/gate; watchdog repointed + armed;
  host reboot before test rounds; SHIP-MEASUREMENT LAW on every number.
- Every ship adds its capture points to the boot evidence block (B3).

---

## Window / phase schedule (each window = one lane restart unless noted)

| Window | Contents | Evidence produced |
|---|---|---|
| **NOW (no restart)** | WS-B live (recorder + scoper armed on the current lane); this plan; PHASES P33 | telemetry JSONL accumulating; manual `stall_scope --now` baseline tarball |
| **W1** | A1 recert boot (swift posture) + A2 full battery + A3 watchdog wiring + C1 pool-split measurement + M0 live telemetry leg + replay baseline | boot evidence, battery logs, pool split, real scales, `cc_replay_W1.json` |
| **W2+** (one leg per window, order by expected leverage) | ~~C2 gmu 0.85~~ **REMOVED by user directive P38 (wedge attribution)** → D1 mnbt → D2 v63/v66 sweep → ~~C2 gmu 0.88~~ **MOOT (same attribution)** → D3 HoL priority → C4 sticky-prefix (only if D legs leave a tail) | per-leg replay JSONs + recorder intervals; keep/rollback decision per leg |
| **Weeks track (parallel, no lane impact until validation legs)** | M1 validation leg → M2 kernel build+bench (≥v132 .so, bitwise harness) → M3 bake v1.2.28; K2 prefill kernels | M-gate reports; superiority matrices; ship gates |
| **Final** | bake v1.2.27 (tuning winners) → v1.2.28 (scaled-space) → full-fleet soak with WS-B running; program closed when the replay gate passes against the LIVE fleet regime (p99<20 s, max<45 s, re-prefills ≈ first-turns only) | closing PHASES entry + round write-up |

## Acceptance = the definition of FIXED

1. `cc_fleet_replay` PASS (TTFT p99 < 20 s, max < 45 s, re-prefill volume ≈
   first-turn totals) on the shipped image, at 8 streams × 50k base ctx.
2. Live-fleet recorder interval (≥24 h) shows: TTFT tail (>40 s) requests
   → ~0, kv sawtooth amplitude reduced or capacity raised, preemptions 0,
   no wedge/reset lines.
3. Full v1226-class crash battery green on the shipped image; spec MTP×4 +
   XGrammar-2 + qwen3_coder parser intact; CC battery through litellm
   green under the certified template.
4. Watchdog v2 + recorder + scoper armed and verified firing on drift.
5. Every phase's evidence file named in PHASES.md (TELEMETRY DIRECTIVE).

## Notes — laws, traps, risk register

- **Laws carried:** READOUT (timer-thread only); marker-file (env doesn't
  reach TP workers); caps-marker final-line convention; SHIP-MEASUREMENT
  (.so sha first); quoting-layer ($() and awk-fields never inside plink
  double-quotes; awk in HOST scripts is fine); file-transfer via pscp only
  (no heredocs through plink); `docker exec -d` needs redirect; build JSON
  bodies with python; grep -c double-print trap; gate stdout is tail-80 —
  grep full logs.
- **Risk register:** (1) gmu legs on the 52 GB swift load → capture OOM
  (abort criteria + legacy rollback ready); (2) template/quality battery
  failure at W1 → template-fix sub-phase, never a waiver; (3) M2
  non-superiority → stays off prod (fp16 pool remains certified endgame);
  (4) sticky-prefix zero-sum at full pool (expected — measurement-gated);
  (5) fleet load varies between legs → every leg re-runs the replay gate
  back-to-back with its control, decisions never from live-fleet
  anecdotes; (6) recorder gaps during host reboots → boot script re-arms.
- **Open diagnostic debt (tracked, ship-unaffected):** P29H strict-subset
  wedge micro-mechanism; non-spec seq kernel sibling-hazard audit; cosmetic
  carryovers (V1225 caps literals, V1212 usage line, watchdog "100 s"
  comment). These ride the program's documentation phases, not the lane.
- **Partition ownership note (fleet coordination):** multiple agents manage
  this lane. The watchdog drift guard + boot-chain posture assertions are
  the enforcement that a manual relaunch cannot silently detune the lane
  again; any agent taking a window MUST use the window scripts, never a
  hand-pasted docker run.
