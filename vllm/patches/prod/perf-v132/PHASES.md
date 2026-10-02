# PHASES.md — perf-v132 (churn-event elimination: scratch ring segregation)

Round opened 2026-10-02 as the direct continuation of perf-v131's
handed-forward lever: *mamba-rotation BLOCK-REUSE / churn-event
elimination at the source* — the surviving v132 candidate after
soft-pop was rejected by measurement.

User directive (standing): deep improvement + fixing + testing
suite, bake into production image using best values; spec MTPx4 +
XGrammar-2 supported and crash-free; **no subagents**; commit+push.

Standing laws carried: SHIP-MEASUREMENT LAW (assert prod .so sha
1d9dcf4e before banking ship numbers); READOUT LAW (host-state
reads only; no device syncs); no degradation anywhere; barriers 0;
async ON; live PHASES update per phase; ship-on-delta (a certified
TOTAL-COMPUTED win with parity = official bake trigger for
v1.2.28); CHURN-DESTRUCTION LAW + RECENCY-PROTECTION LAW (v131);
HYBRID ANATOMY LAW (v130); harvester truncate-at-T0 + ts-filter;
watchdog re-create erases in-container patches (verify arming
after any watchdog activity; patchers need pristine-file anchors);
3-sample control spread for gates; host reboot before test legs.

## The design (and why it does not violate the RECENCY-PROTECTION LAW)

v131 leg A proved pop-time REORDERING of the free queue is dead.
The v132 lever therefore does not reorder anything — it SEGREGATES:

- The per-step churn (mamba rotation + MTP draft setup, the
  new=5-6 allocs P59 identified as the guillotine) cycles the same
  scratch blocks: freed each step (hash-free), re-allocated the
  next. Under stock, every cycle pops the free-queue FRONT — the
  conveyor that eventually carries cached chains into the popper
  (~650k pops over one storm, ~1500 of them destroying mappings).
- SCRATCH RING: hash-free freed blocks park in a bounded ring
  OUTSIDE the free queue (default cap 128); get_new_blocks drains
  the ring FIRST; only the overflow pops the queue. The free queue
  keeps pure LRU semantics for everything hash-bearing (arrivals in
  free order, pops from front, touch on hit) — the queue ORDER is
  never touched by the ring.
- Accounting: get_num_free_blocks = queue length + ring length, so
  every existing capacity check / ValueError guard / preemption
  decision keeps exact semantics.
- Safety invariant: ONLY hash-free blocks may enter the ring. A
  hash-bearing block parked outside the queue would break touch()
  (doubly-linked-list remove of a non-member) and destroy its
  mapping implicitly. Hash-bearing frees always take the queue.

Predicted win mechanism (to be measured, not argued): admit-window
bursts drain the ring before touching the queue, so sibling chains
survive between a wave's first admission (which touches/saves its
own chain) and their own matches — the deterministic 4/5-PRIV death
pattern of P59. Churn blocks never ride the queue conveyor.

Predicted failure modes (gates must discriminate):
- If the churn frees turn out HASH-BEARING (mamba snapshots are
  hashed), the ring starves -> null result, close by measurement.
- If ring supply < admit demand, overflow pops = stock behavior
  (partial mitigation at best).

## P63 — churn free/alloc path verification (read + instrument)

Read in-container: block_pool free paths (free/free_blocks),
kv_cache_manager allocate_slots/free, the MambaManager rotation
site, the scheduler's spec-draft alloc site ("2 +
num_speculative_blocks"). Verify: (1) which function the churn
frees go through; (2) whether the per-step freed blocks are
hash-free at free time (draft/spec transient blocks vs hashed
mamba snapshots); (3) where the size-5-6 allocs originate.
If P59 evidence already answers a point, bank it and move on.

Status: CLOSED 2026-10-02 (~09:40) — every seam verified in-container,
design sealed.

**Verified anatomy (block_pool.py + single_type_kv_cache_manager.py
on the pristine v1.2.27 lane):**
- FREE PATH IS ONE SEAM: `block_pool.free_blocks` (L440) — callers
  = base `free(request_id)` (request end, tail-first reversed chain),
  L426 trim path, and `MambaManager.remove_skipped_blocks` (L884,
  the per-step rolling-state free: `free_blocks([blocks[last_
  state_block_idx]])`, 1 block/step/request — the CHURN-DESTRUCTION
  guillotine member).
- ALLOC PATH IS ONE SEAM: `block_pool.get_new_blocks` (L354).
  MambaManager align-mode decode: steady alloc = exactly 1
  block/step/request (`assert num_new_blocks <= 1`; the
  num_speculative_blocks spec blocks are REUSED in place via the
  null-block dance — no pool alloc); first alloc = 1 + spec = 5
  (the P59 new=5 class). Attention decode ≈ 1 page per ~200 steps
  at the 1024-tok page size.
- HASH STATUS: per-block, queryable at free time
  (`block.block_hash is None`). Blocks gain hashes only in
  cache_full_blocks on ALLOCATED (ref_cnt>0) blocks; rolling state
  blocks are hash-free until snapshotted at an aligned boundary.
  Park decision is per-block — no caller knowledge needed.
- ACCOUNTING IS ONE METHOD: `get_num_free_blocks` (L510) — the
  L365 ValueError guard, L484 used-count, and kv_cache_manager
  L348/L376 capacity checks all route through it. Ring-inclusive
  count keeps all consistent; with num_blocks <= queue+ring and
  ring_served >= num_blocks - queue, the remainder pop never
  exceeds queue length (algebraic no-assert proof).
- SAFETY: `touch` (L423) removes ref_cnt-0 HIT blocks from the
  queue — ring blocks are hash-free, never in cached_block_hash_
  to_block, never matched, hence never touched (by construction).
  `_maybe_evict_cached_block` no-ops on hash-None (stock pops them
  daily). `reset_prefix_cache` walks `self.blocks`, not the queue.
- DIRECT QUEUE ACCESS outside block_pool/kv_cache_utils: only
  `SinkFullAttentionManager.__init__` (class unused by this hybrid)
  and the `simple_kv_offload` cursor walk (dormant: no offload flag
  in serve_user.sh, no call sites in v1/). Both inert on this lane.

**P63 verdict:** the ring is implementable as exactly 4 sites in
block_pool.py (knob, park, drain, accounting) with stock lines
verbatim on the knob-off path. No free-trace leg needed — the
patch carries its own supply/demand telemetry (parked/served/
capfull counters + rate-limited V132_RING lines), and the park
condition is per-block so supply questions affect only magnitude,
not correctness.

## P64 — scratch-ring patch (block_pool.py, knob-gated)

patch_v132_ring.py: S1 knob VLLM_V132_RING (0=OFF byte-identical;
1..4096 = ring cap), S2 free-path park (hash-free only, ring cap),
S3 get_new_blocks ring-first drain. get_num_free_blocks includes
ring. Pristine-file anchors (watchdog-recreate order). Dry-cert,
inert certification knob-off, restore round-trip.

Status: CLOSED 2026-10-02 (~10:15) — patcher written, dry-certified
on the host against the pulled image file.

**Build/cert:** block_pool.py pulled from the container to
/root/build/v132_stage/block_pool_pristine.py (21,476 B; only
prior llm-scaler content = the v50 hash-starvation clamp baked in
the image itself — no anchor collision). patch_v132_ring.py
dry-cert: apply → V132_RING_OK markers=4 (all 4 anchors count=1),
idempotent (V132_RING_ALREADY marker_count=4), py_compile OK,
--restore → markers_left=0, md5 round-trip identical
(14bf0787301e8b15d78820c9e20d9b2b). Backups .v132rbak; knob-off
path = stock comprehension (S2 else), stock popleft_n line (S3
else), stock return (S4 fall-through) — the established
one-extra-if inertness class (same certification basis as
MTRACE/ETRACE/SOFTPOP).

## P65 — A/B legs (gates: TOTAL COMPUTED / wall / parity)

Controls: v131 instrumented control (total_computed 1,168,355,
wall 876 s) + v130 spread walls 995/1023/1318. Host reboot before
the legs (standing hygiene). Leg A ring=128; leg B only if A is
close (cap sensitivity 64/256). Gates: total_computed vs 1,168,355
materially down; wall in spread; ok/N parity 72/72; no 500s /
timeouts / DEVICE_LOST; HEAD_RETAINED floor 5120 (the head must
stay immortal — a head loss is an automatic fail per v131).

Status: PENDING.

## P66 — verdict + ship decision

Win (total computed down materially, parity, quiet-inert): bake
v1.2.28 with the ring default-ON (ship-on-delta trigger). Lose:
close by measurement, no bake, v1.2.27 stands.

Status: PENDING.
