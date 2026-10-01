# PHASES.md — perf-v131 (WS-C waste lever, act 2: size-aware eviction — or whatever P59 finds)

Round opened 2026-10-01 (~19:30) as the direct continuation of
perf-v130's handed-forward lever: *size-aware eviction policy —
protect small footprints, evict PRIV tails first; gates must track
TOTAL COMPUTED tokens / wall, not WASTE_PCT.*

User directive (standing): deep improvement + fixing + testing
suite, bake into production image using best values; spec MTPx4 +
XGrammar-2 supported and crash-free; **no subagents**; commit+push.
This round's ask: implement the improvements/fixes from the v130
findings and continue.

Standing laws carried: SHIP-MEASUREMENT LAW (assert prod .so sha
1d9dcf4e before banking ship numbers); READOUT LAW (host-state
reads only in scheduler-side code; no device syncs); no
degradation anywhere; barriers 0; async ON; live PHASES update per
phase; ship-on-delta (a certified WS-C baked-default policy IS an
official bake trigger); v130's HYBRID ANATOMY LAW (self.block_size
= 1024-tok LCM pages; num_gpu_blocks counts pages; 4 mirroring KV
groups -> footprint sums take MAX-over-groups); harvester
truncate-at-T0 + ts-filter; watchdog re-create erases in-container
patches (verify arming after any watchdog activity); 3-sample
control spread for gates (walls ±15%, MISS ±10%).

## The open question P59 must answer (from v130)

The v130 close attributed the 99% waste to "cyclic LRU thrash" —
but the arithmetic does not close: the 18-conv working set is
5x29.5k + 5x36k + 8x3k = 351.5k tok ~= 344 pages, which FITS the
pool (~465-498 pages). Plain FIFO/LRU should NOT thrash a working
set smaller than the cache. Yet PRIV turn>=2 lands cached=0 (full
~35k recompute) in every control and treatment leg, deterministically.

Candidate mechanisms (to be discriminated by measurement, not argued):

- (a) EVICTED-BEFORE-RESEND: the free-block FIFO evicts PRIV caches
  during admission bursts (victim = oldest-freed; `_maybe_evict_
  cached_block` destroys the hash mapping at `get_new_blocks` pop
  time). Policy lever alive -> P60 two-tier evictor.
- (b) MATCH-FAILURE: blocks are resident but `find_longest_cache_
  hit` returns ~0 — mamba-align checkpoint unavailability collapsing
  the hybrid fixed-point, or hash-chain divergence. Lever is
  match-side, NOT eviction policy.
- (c) CAPACITY-ILLUSION: real concurrent demand (draft/spec blocks,
  align rows, partial pages, in-flight prefills) is much larger
  than the WS arithmetic -> true oversubscription after all.

## P59 — admission-match + eviction trace (instrumentation, read-only)

Two patchers, knob-gated `VLLM_V131_TRACE` (default 0 = zero-cost):

- `patch_v131_mtrace.py` (kv_cache_manager.py): hook AFTER the
  `find_longest_cache_hit` call in `get_computed_blocks` — per
  admission log `V131_MATCH rid ptok hit kmiss freeq` where
  `kmiss` = first index into `request.block_hashes` whose hash is
  ABSENT from `block_pool.cached_block_hash_to_block` (bounded
  probe), `freeq` = free-queue length (includes reclaimable
  cached). Discriminates (b): hit=0 with kmiss>0 = present-but-
  unmatched; kmiss=0 = absent (evicted or never-inserted).
- `patch_v131_etrace.py` (block_pool.py): counters on
  `_maybe_evict_cached_block`(True) + `get_new_blocks`; rate-limited
  `V131_EVICT ev_total freeq new` lines. Discriminates (a)/(c):
  eviction bursts on the storm timeline vs near-zero evictions.

Legs: one instrumented storm run (18-conv wsc driver verbatim) on
the armed lane; join MATCH/EVICT timeline against driver turns.

Status: CLOSED 2026-10-01 (~20:10) — mechanism found; it is NOT
classic capacity thrash and NOT a broken match.

**Build/cert:** both patchers dry-certified (markers 2/3, restore
round-trip md5-identical) then live-applied; lane relaunched with
`VLLM_V131_TRACE=1` (sed 1i in serve_user.sh), health 200, both
V131_*_ARMED lines in EngineCore, prod .so sha asserted
(1d9dcf4e...). One restart hiccup: stale APIServer survived
SIGTERM alongside the new boot — SIGKILLed before port/GPU
collision; solo traffic clean after.

**Leg (instrumented storm):** 72/72 ok, wall 876 s (controls
995/1023/1318 — inside spread; trace is read-only). Analyzer
v131_p59_join.py v2 (labels MATCH lines by exact driver
prompt_tokens -> turn+cohort; per-rid FIRST admission = true
boundary; EVICT lines bucketed by driver completion windows).
Artifacts: v131_stage/{v131_match,v131_evict,v131_trace}.txt,
wsc_pressure_v131p59.jsonl, wsc_v131p59.out.

**Findings (first-admission hits, per turn x cohort):**
- t2-t4 SHARED: ALWAYS exactly 5120 (the common head, 5 pages) —
  never their own ~24k bodies.
- t2/t4 PRIV: exactly ONE rid per wave matches FULLY (33792 =
  its whole chain); the other four get 0. t3 PRIV: all five 0.
- t2+ SMALL: 0 (max 1024 once).
- All-or-nothing per rid — chains never partially match.
- EVICT: 1505 total (~375/wave), min freeq 298-323 — mappings die
  while hundreds of blocks remain free. Alloc sizes at evict
  moments: new=5 dominates (19/47 lines) = the MTP/mamba-spec
  setup allocations (2 + num_speculative_blocks).
- kmiss probe INVALID as built (required the hash in ALL 4 groups
  incl. mamba; the real attention match consults only the
  attention-group subset) — it reports "absent" even for
  admissions the engine matched at 5120/33792. Documented, not
  re-run: engine-side hit values + evict counters already
  discriminate the hypotheses.

**P59 LAWS (the round's payload):**
1. CHURN-DESTRUCTION LAW: cached mappings die at free-queue pop
   time in proportion to ALLOCATION EVENT COUNT, not capacity
   shortfall — `get_new_blocks` pops the queue front and
   `_maybe_evict_cached_block` destroys whatever hash-bearing
   block sits there, even with freeq ~300/498. The per-request
   mamba-align rotation + MTP draft setup (new=5-6 pops each,
   continuously during decode/prefill) is the guillotine.
2. WHOLESALE CHAIN DEATH: a conv's blocks free contiguously (one
   free_blocks call), so front-pops kill entire chains — hence
   all-or-nothing per-rid hits (0 or full), never partial.
3. SURVIVORS: the common head is re-referenced every wave by all
   5 SHARED convs (ref_cnt held until the last finishes, then
   touched again on every match) — immortal under LRU (v129
   HEAD_RETAINED, now with exact mechanism). The first-admitted
   big request per wave matches before the wave's own allocations
   destroy the rest (the lone 33792 PRIV).
4. Match machinery WORKS (full deep hits occur every wave) —
   hypothesis (b) rejected; hypothesis (a)/(c) blend: eviction
   pressure is event-driven, not pool-driven.

**P60 consequence:** the lever is not queue tiers but POP-TIME
PROTECTION — soft-pop: when freeq minus demand >= SLACK, rotate
hash-bearing front blocks to the tail instead of evicting them;
stock hard front-pop only under genuine pressure. Design moved to
P60 below.

## P60 — SOFT-POP policy patch (redesigned on P59 evidence)

Surgery: ONE site in block_pool.get_new_blocks. When
`VLLM_V131_SOFTPOP_SLACK` > 0 AND enable_caching AND
freeq - num_blocks >= SLACK: pop one block at a time; if the head
block is hash-bearing (cached), `append()` it to the queue TAIL
(rotate — mapping survives) and continue; else keep it. Guard
4*num+1024 skips then hard-pop the remainder (pool entirely
hash-bearing corner). Under genuine pressure (freeq - num <
SLACK): stock front-pop path, byte-identical.

Self-organization: expendable non-hash state blocks (mamba
rotation, draft setup) never rotate and stay consumable; cached
blocks migrate tailward under slack; genuine exhaustion still
evicts LRU-first. freeq never drops below SLACK during soft-pop
(pops-minus-appends <= num_blocks), so the existing ValueError
guard semantics are unchanged.

Knob `VLLM_V131_SOFTPOP_SLACK` (default 0 = OFF = byte-identical;
certify inertness knob-off like every prior lever). Keep
MTRACE/ETRACE armed through the legs — hit/EVICT distributions are
the treatment readout. One-shot V131_SOFTPOP_ARMED line; rotation
counter logged every 4096 rotations (V131_SOFTPOP rot=... hard=...).

Status: CLOSED 2026-10-01 (~21:15) — implemented, applied, armed
(SLACK=64), measured in P61 leg A. patch_v131_softpop.py: dual S1
anchor (post-ETRACE tail, pristine fallback for watchdog-recreate
order), S2 soft-pop block with the annotated stock line as the
else branch; backups .v131sbak (restore order softpop -> etrace ->
pristine). Dry-certified markers 2, compile-checked, restore
round-trip.

## P61 — A/B/C legs + split (gates: TOTAL COMPUTED / wall)

Controls: the three banked v130 runs (99.0/99.1/99.2 WASTE;
14032/15239/16839 MISS; 995/1023/1318 s walls) + total-computed
derivations. Treatment legs at certified threshold(s).

Primary gate: TOTAL COMPUTED tokens per run (sum over perreq
computed) — the metric that actually moves under a retention
policy (WASTE_PCT stays ~98% BY CONSTRUCTION: it is a ratio whose
numerator and denominator shrink together). Secondary: SMALL/SHARED
turn>=2 hit rates, cohort walls, ok/N parity, HEAD_RETAINED floor
5120, no 500s/timeouts/DEVICE_LOST.

Status: CLOSED 2026-10-01 (~21:40) — leg A (SLACK=64) REJECTED by
measurement, decisively, on every gate.

**Leg A (soft-pop SLACK=64):** 72/72 ok, wall 1307 s vs control
876 s (+49%). Gate: total_computed 1,324,003 vs control 1,168,355
(+13.3% WORSE — recompute increased). WASTE_PCT 99.3 (vs 99.1);
MISS_TOK_PER_AFFECTED 18,540 (vs 15,372). Split: HEAD_EVICTED
(SHARED cached floor min=0 med=3072 — the always-immortal head
DIED), SMALL_PRESSURED (mean hit 0.177). Counters: rot=684,033
rotations, hard=645 guard-trip fallbacks. Forensics
(v131_p59_join.py on the boot-scoped trace, 1125 MATCH lines):
the clean all-or-nothing control world became scattered partials —
SHARED t2 firsthit max 11,264 / min 0 (some rids lost the head
entirely), PRIV t4 max 6,144, SMALL max 2,048-3,072 — and TOTAL
evictions ROSE to 1,568 (vs control 1,505). Artifacts:
v131_stage/{v131_traceA_full.txt, wsc_perreq_v131p61a.jsonl,
wsc_pressure_v131p61a.jsonl, wsc_v131p61a.out}.

**Failure mechanism (understood, structural):** with the working
set ~0.7x pool, the free queue front is dominated by hash-bearing
blocks — every soft scan must chew through hundreds of cached
blocks (guard 4n+1024) to find ~5 expendable ones; the guard trips
645 times and each trip's fallback popleft_n evicts whatever
cached chain sits at the scan front, in VISIT order, not recency
order. Rotation-to-tail reorders the queue by hash-presence, which
destroys the LRU recency semantics that were the head's ONLY
protection (match -> re-reference -> free-to-tail immortality).

**RECENCY-PROTECTION LAW (the round's second payload):** the LRU
order IS the protection. Any pop-time policy that reorders the
free queue by hash-presence (soft-pop rotation, two-tier evictors,
visit-order scans) converts reliable head/full-chain survivors
into scattered partials AND raises total mapping destruction.
Slack tuning cannot rescue it (SLACK=8 = more rotations = worse;
large SLACK = off = control); the harm scales with soft-path
activity, not slack magnitude. Pop-time segregation of the free
queue is a dead lever in this abstraction.

**Handed forward (surviving designs, v132 candidates):**
1. CHURN-EVENT ELIMINATION at the source — the mamba-align
   rotation frees block N-2 then allocates N via the free queue;
   making that alloc REUSE the just-freed block (bypass the queue
   for that pair) removes the guillotine allocations P59
   identified, without touching queue order. MambaManager-scope
   surgery.
2. Two-free-list segregation (hash-bearing vs expendable) — a
   real FreeKVCacheBlockQueue redesign, out of surgical scope
   this round.

## P62 — verdict + ship decision

Win (total computed down materially, parity, quiet-inert): bake
v1.2.28 with the policy default-ON (official "WS-C baked-default
policy" trigger). Lose: close by measurement, no bake, v1.2.27
stands.

Status: CLOSED 2026-10-01 — **LOSE. No bake; v1.2.27 stands.**
Soft-pop measured +13.3% total_computed / +49% wall / HEAD_EVICTED
(P61 leg A). No ship trigger fired; lane restored to pristine
v1.2.27 posture (softpop -> etrace -> mtrace restored, env lines
stripped, markers 0, .so sha re-asserted, watchdog resumed).
Round payload: CHURN-DESTRUCTION LAW + RECENCY-PROTECTION LAW +
soft-pop/two-tier pop-time policies measured dead + handed-forward
mamba-rotation-reuse design.
