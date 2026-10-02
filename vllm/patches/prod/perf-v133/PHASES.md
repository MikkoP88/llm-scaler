# PHASES.md — perf-v133 (churn-event elimination: alloc-side rotation reuse)

Round opened 2026-10-02 as the direct continuation of perf-v132's
handed-forward lever — the only live surgical candidate after the
law chain closed both pop-time reorder (v131 RECENCY-PROTECTION)
and free-side segregation (v132 RING-STARVATION): *ALLOC-side
rotation reuse — MambaManager-scope BLOCK-REUSE, making the
per-step rotation alloc bypass get_new_blocks so it never pops the
free-queue front.*

User directive (standing): deep improvement + fixing + testing
suite, bake into production image using best values; spec MTPx4 +
XGrammar-2 supported and crash-free; **no subagents**; commit+push.

Standing laws carried: SHIP-MEASUREMENT LAW (assert prod .so sha
1d9dcf4e before banking ship numbers); READOUT LAW (host-state
reads only; no device syncs); no degradation anywhere; barriers 0;
async ON; live PHASES update per phase; ship-on-delta (certified
TOTAL-COMPUTED win with parity = official bake trigger v1.2.28);
CHURN-DESTRUCTION LAW (mapping death ∝ alloc EVENT count) +
WHOLESALE CHAIN DEATH + SURVIVORS + RECENCY-PROTECTION LAW (v131);
RING-STARVATION LAW + HYBRID ANATOMY LAW (v132/v130); GATE LAW
(track TOTAL COMPUTED / wall, not WASTE_PCT); RIDER LAW (watchdog
re-create wipes in-container patches INCLUDING the v128 perreq
telemetry — re-apply before gate legs); harvester truncate-at-T0 +
ts-filter; 3-sample control spread; host reboot before test legs;
plink traps ($-expansion, no pscp brace expansion, tail -n +N).

## Why this lever is the survivor

The v131/v132 law chain says the guillotine is the ALLOC side:
~650k free-queue pops per storm, driven by the per-step rotation
(steady new=1/step/request) + first alloc (new=1+spec=5); each pop
runs `_maybe_evict_cached_block` on whatever hash-bearing chain
block sits at the queue front. Both policy moves that keep the pops
are dead:

- pop-time REORDER (soft-pop): +13.3% computed, HEAD_EVICTED —
  RECENCY-PROTECTION LAW.
- free-side SEGREGATION (scratch ring): zero hash-free non-null
  frees exist to segregate — RING-STARVATION LAW.

The remaining move: the churn allocs stop consulting the queue
entirely. The rotation frees block N-x and allocates block N in the
same step; if the alloc reuses a block the rotation itself just
released (held MambaManager-side, outside the free queue), the
pop count — the CHURN-DESTRUCTION driver — drops by the rotation
mass without touching LRU order anywhere.

## Design constraint P67 must settle (before any patch)

1. HASH-LIVENESS: steady-state churn frees are hash-bearing
   (snapshot-hashed at aligned boundaries) or null-danced
   (RING-STARVATION LAW). Reuse must not leave a stale mapping:
   a hash-bearing reuse target needs a deliberate, targeted
   eviction of its OWN spent mapping (known churn, about to be
   overwritten) — never an innocent front-pop. If the steady
   currency is null-danced blocks (never in the queue, no hash
   machinery), the reuse target is even cleaner. P67 reads the
   align/snapshot/match machinery and seals which.
2. REQUEST-END HYGIENE: any MambaManager-held reuse blocks must
   return to the normal free path at request end (no leaks across
   requests; free(rid) must sweep them).
3. PREEMPTION/SWAP: audit the paths that free or re-allocate mamba
   blocks outside the steady step (preemption, recompute,
   finished-request teardown) — held blocks must not orphan there.
4. ACCOUNTING: get_num_free_blocks must include held reuse blocks
   (same accounting class as the v132 ring — every capacity check
   keeps stock semantics).
5. FIRST-ALLOC (new=5) SCOPE: the 1+spec first alloc is a different
   event class (draft setup). P67 decides whether the reuse pool
   serves it too or only the steady new=1 rotation.

## P67 — rotation alloc/free anatomy (read in-container)

Read the MambaManager rotation sites in
vllm/v1/core/single_type_kv_cache_manager.py (+ kv_cache_manager /
kv_cache_coordinator call seams + block_pool handoff): exact free
site (remove_skipped_blocks), exact alloc site(s) and their call
chain to block_pool.get_new_blocks, hash state at free time, the
null-block dance mechanics, request-end/preemption frees. Seal the
design: surgery sites, knob, invariants, telemetry.

Status: CLOSED 2026-10-02 — anatomy fully verified from the
container sources (single_type_kv_cache_manager.py,
kv_cache_manager.py, kv_cache_coordinator.py, block_pool.py,
kv_cache_utils.py pulled to captures/ + v133_stage/); design sealed
as the SELF-SERVICE ROTATION PAIR (5 sites, MambaManager scope).

**Verified anatomy (v1.2.27 lane, align mode + MTPx4):**
- GEOMETRY: attention block_size = 1024 tok (boot log: forced 896
  -> 1024 for XPU 64-multiple); mamba page = ONE state slot padded
  to 1 MiB (917504 -> 1048576 B) = one attention page. MambaSpec
  align-mode memory bound = page x (2 + num_speculative_blocks) —
  ~6 REAL blocks per request; the skipped prefix is the SHARED null
  block. Chain shape: [null x (k-1), state, spec x 4].
- THE ROTATION PAIR, one seam each: FREE =
  MambaManager.remove_skipped_blocks L884 (frees the recorded
  last_state_block_idx once the window advances past it). ALLOC =
  MambaManager.allocate_new_blocks L1033 ->
  block_pool.get_new_blocks(num_new). The spec-block dance
  (L1010-1027) absorbs multi-token chunks: nulls extend, the 4 spec
  blocks recycle forward, and num_new collapses to <= 1 per
  allocate_slots call REGARDLESS of chunk size — the
  blocks_allocated assert (L1029) proves it. So the armed
  num_new==1 branch covers essentially every steady/continuation
  mamba alloc; only first allocs (1+spec=5, fresh rid) are
  genuinely new scratch.
- ORDERING (no interleaving): kv_cache_manager.allocate_slots runs
  remove_skipped_blocks (L362) -> coordinator capacity check
  (L376) -> allocate_new_blocks fan-out (L393) inside ONE
  synchronous call per request — the park/consume window is
  atomic. Single-threaded scheduler: other requests' matches or
  allocs cannot interleave.
- CAPACITY SUM: coordinator.get_num_blocks_to_allocate SUMS across
  all managers before the check; the fan-out pops total <= demand
  <= queue length, and the parked block sits at the queue TAIL
  (last append; FullAttention remove_skipped is a no-op on this
  lane) -> no other manager's popleft_n can reach it. A
  membership+ref_cnt guard at consume (fall back to stock) covers
  any exotic residual case (e.g. a future SWA group appending
  after the park).
- STOCK POP TREATMENT (block_pool.get_new_blocks L368-377):
  popleft_n front -> per block: _maybe_evict_cached_block (destroys
  the popped block's own mapping), assert ref_cnt==0, ref_cnt+=1,
  metrics on_block_allocated. The queue's remove(block) (L285-303,
  the primitive touch() uses) is O(1), unlink-only, loud
  RuntimeError on non-members; membership testable via
  prev_free_block/next_free_block not None.
- HASH STATE AT FREE: RING-STARVATION LAW (v132 measurement)
  already settles it — every non-null free in the storm was
  hash-bearing (zero hash-free frees at any cap). The rotation
  freed state blocks carry live mappings into the queue.
- MATCH-ECONOMY SAFETY (the design's key argument): resend matches
  need mamba state mappings alive at the LAST aligned boundary —
  those blocks free at REQUEST END via the wholesale path (base
  free, untouched by this patch). The rotation-freed INTERMEDIATE
  states are spent currency (a resend matches at the final
  boundary, never at intermediates), so consuming them one step
  early — destroying their OWN mapping instead of an innocent
  front block's — cannot degrade the match economy. If a parked
  block IS matched (touched) before consumption, the guard sees
  ref_cnt != 0 and falls back to stock — the match wins.

**The sealed design — SELF-SERVICE ROTATION PAIR (5 sites,
single_type_kv_cache_manager.py, knob VLLM_V133_ROTREUSE):**
- M1 module knob + logger (after imports). M2 MambaManager.__init__
  align branch: per-request parked dict + counters + ARMED line.
- M3 park (remove_skipped_blocks L884): stock free_blocks VERBATIM
  (block enters the queue tail as a completely ordinary free
  member, ref_cnt 0, mapping alive) + park the block ref.
- M4 consume (allocate_new_blocks L1033): when armed +
  blocks_allocated + num_new==1 + parked entry valid (ref_cnt==0,
  live queue member, not null): free_block_queue.remove(block) by
  identity -> _maybe_evict_cached_block (own spent mapping — the
  exact stock pop-time treatment) -> assert ref_cnt==0 ->
  ref_cnt+=1 -> metrics on_block_allocated. Byte-identical block
  lifecycle to stock; the ONLY delta is WHICH mapping dies (own
  spent vs innocent front) and that no front pop happens. Any
  guard failure or first-alloc (1+spec) falls back to the stock
  call. Telemetry: V133_ROT_ARMED + V133_ROT reused/stock_first/
  stock_guard/parked/freeq every 256 reuses (first line = reuse#1
  = arming proof within one alloc step of the first park).
- M5 free (request end / preemption): pop the dict entry; the
  block itself is already a normal free member — zero pool-side
  cleanup, no held ref_cnt anywhere.
- ACCOUNTING: untouched — while parked the block IS a queue member
  (counted exactly as stock); get_num_free_blocks stock at every
  instant. No v132-ring-style accounting surgery needed.

**Expected mass (to be measured, not argued):** mamba alloc blocks
~2.4k/storm (72 reqs x ~34 1024-crossings incl. chunked-prefill
continuation) + 360 first-alloc blocks; attention pages ~2.4k. The
lever eliminates the ~2.4k steady mamba pops = roughly half of all
popped blocks -> ceiling ~40-45% of the ~1505 mapping deaths/storm.
The reused counter measures the true mass in leg A regardless.

## P68 — rotation-reuse patcher (knob-gated, dry-cert)

patch_v133_rotreuse.py: knob VLLM_V133_ROTREUSE (default 0 = OFF =
stock lines verbatim in else-branches). Pristine-file anchors
(watchdog-recreate order), marker count, .v133rbak backups,
compile-checked temp-write → os.replace, --restore round-trip,
idempotent. Dry-cert on host against pulled image files.

Status: CLOSED 2026-10-02 — patch_v133_rotreuse.py written (5 sites
M1-M5, marker "# llm-scaler v133 ROTREUSE", EXPECTED_MARKERS=5,
.v133rbak, py_compile temp-write -> os.replace, --restore,
idempotent, CRLF-normalized anchors) and DRY-CERTED on host against
the pristine staged image file (single_type_kv_cache_manager.py md5
eae6cd07d08dd807c60324bc3a5a4a71):
- PASS1 patch: markers=5, md5 -> 82375df2640d1882ba2f2514e8a18d30,
  51594 -> 57105 bytes, compile-checked, atomic replace.
- PASS2 re-run: ALREADY markers=5 (idempotent).
- PASS3 --restore: md5 back to pristine eae6cd07... (exact).
- PASS4 re-patch: byte-deterministic (same md5 82375df2...) + grep
  markers=5.
- Hunk review: all five inserts placed exactly per the sealed
  design; stock free/alloc lines preserved verbatim inside the
  patched branches; counters/telemetry quiet when knob off (one
  extra `if` per site).
- RIDER asset located for P69: /root/build/v128_stage/
  patch_perreq_v128.py (perreq telemetry patcher, host-side, ready
  to re-apply after the watchdog-recreate wipe).

## P69 — A/B legs (gates: TOTAL COMPUTED / wall / parity / pop count)

RIDER LAW: re-apply the v128 perreq metrics_recorder patcher BEFORE
the legs (the 05:32:51 watchdog re-create wiped it — engine-side
computed numbers are otherwise unavailable; HARVEST_ROWS=0 trap).
Host reboot before the legs. Controls: v131 instrumented control
(total_computed 1,168,355, wall 876 s) + v130 spread walls
995/1023/1318. Gates: total_computed materially DOWN; wall in
spread; ok/N parity 72/72; no 500s / timeouts / DEVICE_LOST;
HEAD_RETAINED floor 5120; treatment readout = reuse counters
(pops avoided) + MTRACE/EVICT-class distributions if re-armed.

ROUND RISK (pre-declared): the v132 leg A crash — GPU ccs-engine
reset at storm turn-4 capacity pressure, T+8min, NEW earlier onset
(capture lce1/WD_crash_053251). Legs may crash intermittently;
policy: one re-run per crashed leg; if a CONTROL leg crashes the
same way, the round PAUSES and hands the lane to the crashfix
workstream (measurement base unstable).

Status: CLOSED 2026-10-02 — leg A2 ran CLEAN to completion after the
A1 void (host reboot per crash directive, patches re-applied
byte-exact: markers=5 md5 82375df2, perreq markers=2, knob armed,
FIXED full-tree restart `pkill -9 -f [v]llm` + emptiness verify +
detached relaunch; boot 07:03:58, ARMED x3 EngineCore pid 1531,
health 200, serve_full.log scope line 240+).

**LEG A2 RESULTS (treatment, VLLM_V133_ROTREUSE=1):**
- PRESSUREB_DONE ok=72/72 wall=1070s HARVEST_ROWS=72; all finish=stop;
  client completion_tokens 5168, prompt 1,399,779 — parity exact.
- Engine-side (v128 perreq): TOTAL computed 1,234,915 / generated
  5,168 / cached 164,864; ttft p50 126.8s p90 259.4s.
- GATE VERDICT: computed vs v131 control 1,168,355 = **+5.7% —
  GATE FAIL** (gate: materially DOWN). wall 1070s inside the v130
  spread (995/1023/1318), above the v131 control point 876s.
- STABILITY: clean through BOTH prior crash windows (A1 T+4.6min,
  v132 T+8min); 0 HTTP 500, 0 ERROR lines, no DEVICE_LOST — the
  cleanest treatment leg of the round.

**ROTATION-MASS LAW (the round's measured discovery):** no manager
ever crossed reused=256 (no second V133_ROT checkpoint line all
storm; final mass <= 768 blocks/storm, each manager <= 255). The
P67 ceiling estimate (~2.4k steady mamba allocs -> 40-45% of ~1505
mapping deaths) was WRONG by ~3 orders: the per-1024-crossing
rotation is a rounding error against the recompute mass (1.2M
computed tokens driven by chunked-prefill page churn — v130's
between-turn CYCLIC LRU THRASH, not rotation pops). The +5.7%
computed delta is single-run noise on an immaterial lever (n=1 vs
n=1; v130 wall spread alone is +/-15%).
- Mechanism proof retained: park->consume worked from the first
  event (reused=1 parks=1 within one alloc step of first park,
  freeq ~462 stable), zero exceptions from any v133 site, guards
  quiet (stock_guard=1 early, then silence) — the design was sound;
  the mass was not there.

**LEG A1 = VOID (confounded, not lever-caused).** ok=0/72 wall=277s:
all 18 turn-1 requests HTTP500 at T+276s; watchdog captured
(lce1/WD_crash_063354: devcoredump_card1, dmesg ccs engine reset
06:32:29 -> GT reset 06:32:35) and re-created the lane pristine.
ROOT CAUSE (ps.txt + cumulative log analysis): the 06:22 knob-arm
restart used `pkill -9 -f vllm.entrypoints`, which killed ONLY the
APIServer — the old EngineCore+Workers tree (06:16 watchdog boot:
pids 485/684/690) survived ORPHANED on the GPUs while the new tree
(1525/1724/1730) booted; TWO engines drove the same cards until the
ccs reset (both trees logged DEVICE_LOST at 06:32:35-36; two
"Application startup complete" lines in one log; ps elapsed 16:30
vs 10:46). The lever is UNIMPEACHED: V133_ROT telemetry healthy
(reused/parks flowing, freeq ~462) 3 min pre-death, death was
device-side in sample_tokens, ZERO exceptions from any v133 site.
RESTART FIX (standing, replaces the old SIGKILL recipe): kill the
ENTIRE tree — `docker exec lsv-test bash -c "pkill -9 -f vllm"`
then VERIFY `pgrep -af python` is empty (or none serving) BEFORE
`docker exec -d lsv-test bash /root/serve_user.sh`; narrow
entrypoint-pattern kills orphan the EngineCore/Workers.
USER DIRECTIVE (2026-10-02, standing): host reboot after ANY crash
before the next test leg. A2 = clean re-run after reboot+restore+
re-apply (one re-run per crashed leg policy).

## P70 — verdict + ship decision

Win (total computed down materially, parity, quiet-inert knob-off):
bake v1.2.28 with reuse default-ON (ship-on-delta trigger). Lose:
close by measurement, no bake, v1.2.27 stands.

Status: CLOSED 2026-10-02 — **LOSE / REJECTED BY MEASUREMENT, no
bake; v1.2.27 stands.** Leg A2 gate fail (+5.7% computed, single-run
noise on an immaterial lever) + ROTATION-MASS LAW (<=768 reusable
blocks/storm vs 1.2M computed-token mass). No ship trigger fired.
Lane restored pristine: patcher --restore (md5 eae6cd07 exact),
perreq marker removed (patch dormant), knob line stripped from
serve_user.sh, full-tree restart + sha assert. Hand forward to the
crashfix workstream: WD_crash_063354 (card1 ccs reset under the A1
orphaned-two-engine confound — distinct from v132's card2 single-
tree T+8min onset; both devcoredumps banked) + the A1 lesson
(narrow `pkill -f vllm.entrypoints` orphans EngineCore/Workers —
restart recipe is now the full-tree kill + emptiness verify).
Surgical-churn program status after v131/v132/v133: pop-time
reorder DEAD (law), free-side segregation DEAD (law), alloc-side
reuse FUNCTIONAL-but-immaterial (mass law) — the remaining live
levers are structural (two-free-list queue redesign) or the v130
size-aware eviction / capacity direction; none are surgical.
