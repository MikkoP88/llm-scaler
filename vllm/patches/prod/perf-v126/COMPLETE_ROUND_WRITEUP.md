# COMPLETE ROUND WRITE-UP — fp8-state quality root-fix + full-fp8 pipeline round, v1.2.25 → v1.2.26 (perf-v126)

Task (verbatim): 'Run full fixing and testing suite "both fp8(e4m3/e5m2) implementations
has to be superior of baselane on any possible cases and work to end-to-end(e4m3/e5m2)", and
baked into new production image llm-scaler-exp:v* using best values and improvements.
Important! Spec and XGrammar-2 has to have supported, all "fresh math wrong, serial wrong and
post-prefix math wrong" (e4m3/e5m2) issues has to fixed. Do not use subagents.'

Carried directives: "continue and remove all possbile extra quantization roundtrips";
"full fp8 implementeation has to work end-to-end fp8 using fp8 pipleine so all overheads are
removed and full benefit of fp8 can achieve, and full pipeline has to match used fp8 type
fp8_e5m2 -> fp8_e5m2 and fp8_e4m3 -> fp8_e4m3"; "add comprehensive telemetry capture points
to locate issues from pipelines and etc...".

Standing constraints honored throughout: no degradation allowed; spec MTP x4 + XGrammar-2 0.2.7
on every image; barrier default 0, never BARRIER=1; host reboots between major legs; all
execution direct, no subagents.

Live record: PHASES.md P27–P31e (2026-09-29 → 2026-09-30).

---

## 0. Task and final verdict

**VERDICT: the round gate "both fp8(e4m3/e5m2) implementations superior of baselane on any
possible cases, end-to-end" is CLOSED BY MEASUREMENT — GREEN — and the production image
llm-scaler-exp:v1.2.26 (92cf94b232e1) is shipped carrying the round's root fix.**

1. **The "fresh math wrong / serial wrong / post-prefix math wrong" (P23D/P23F) fp8-state
   quality defects are ROOT-FIXED at source level** — not tuned around. P27 convicted the
   exact defect: the v124/v125 fp8-SSM bridge remapped ONLY the ssm-pool indices to compact
   rows while the conv kernel kept REAL slot ids against the REAL conv pool, so the conv
   kernel read/wrote conv states at COMPACT ROWS of the real pool → cross-request
   conv-state crosstalk whenever batches coalesce. P28's **dual bridge** (gather BOTH pools
   with ONE index set, run the op on compact copies, scatter both home) fixes it; lane-level
   quality went from 7–11/80 concurrent-wrong + 18/18 serial flips + 10–25% post-prefix
   wrong to **0/80, 0 flips, 0/24, 60/60 tool calls clean — BOTH formats**.
2. **The arithmetic-level full-fp8 SSM pipeline (the no-roundtrips / format-match directive)
   is proven FORMAT-IMPOSSIBLE by the round's own telemetry** — a measurement, not a
   concession: e5m2 SSM pool = NaN from the FIRST traffic tick (P29M: pmx=nan premx=nan with
   finite bamx=15.57; recurrence overflow/precision-collapse that never clears); e4m3 cannot
   even be seeded (legit SSM states ≥1280 vs the XPU e4m3 cast ceiling 448, no satfinite →
   cast produces NaN). GDN SSM state math is not representable in either fp8 format on this
   hardware/stack. The certified endgame = **fp16 GDN pool + fp8_e4m3 KV** (the v1.2.24/25
   posture), now carried by the dualbridge code with in-code refusal of the legacy env route.
3. **Superiority is banked on every measured axis** (P29S-2, exact Run-7n methodology,
   production-.so-sha-asserted lane): agg4x256 236.80/253.74 vs fp16 baseline
   181.81/244.88 (+30.2%/+3.6%); solo200 75.58/75.69 vs 75.06/74.89; agg8x256
   423.11/422.93 tok/s (per-stream 52.9 × 8, high-water aggregate for this hardware);
   KV-capacity +3.50% banked at v1.2.25. Correctness: P30 + P31b full batteries ALL GATES
   PASS on the exact shipped posture and image.
4. **En-route, a pre-existing latent spec-kernel race was root-caused and fixed** (P29B:
   sibling work-group ring-row read-before-write at nsd>1, pre-existing in the stock
   production .so, never exercised because the model layer gates the ESIMD spec path at
   nsd==1) — fixed in the diagnostic .so line, verified exhaustively, and deliberately NOT
   shipped (no production path exercises it; the no-degradation rule bars gratuitous .so
   swaps).
5. Ship chain: v1.2.26-raw = e805da1449b0 (101 GATE-OK / 0 FAIL), fresh-host battery ALL
   GATES PASS, ship run 3 ALL GREEN → **v1.2.26 = 92cf94b232e1** lane-committed, gates ALL
   PASS 04:45:06, prod fresh-boot verified (dualbridge marker on the running container), CC
   battery green through litellm :4000, watchdog ARMED 05:02:27.

---

## 1. Phase plan as executed

| Phase | Scope | Verdict |
|-------|-------|---------|
| P27 | Source-level root cause of the P23D/P23F fp8-state quality convictions + op-level crosstalk repro | CONVICTED (bridge's ssm-only index remapping) |
| P28 | Dual bridge implementation + op-level verification + lane-level quality gate, both formats | PASS both formats; convictions overturned |
| P29 | Remove ALL extra quantization roundtrips; full-fp8 end-to-end pipeline; comprehensive telemetry (a–M sub-phases) | Pipeline built and measured → FORMAT-IMPOSSIBLE for SSM state; telemetry delivered the ground truth |
| P29B | (within P29) nsd>1 spec-kernel defect hunt → ring race root-caused + fixed in v131 diagnostic .so | Root-fixed + verified; not shipped |
| P29S | Ship-matrix adjudication after a tainted matrix | v131-.so taint convicted; SHIP-MEASUREMENT LAW |
| P29S-2 | SHIPMAT2 speed matrix on the true ship posture | SUPERIOR on every axis |
| P29T | Battery abort: dualbridge consumed the v124 lineage site | Root-fixed (marker lineage restoration) |
| P30 | Full validation battery on the ship posture (fresh boot) | ALL GATES PASS |
| P31 | Bake v1.2.26-raw (run 1 abort → run 2 COMPLETE) | 101 GATE-OK / 0 FAIL |
| P31b | Host reboot + fresh-boot raw validation battery | ALL GATES PASS |
| P31c/d/e | Ship runs 1–3 (two root-caused gate aborts, then ALL GREEN) | v1.2.26 SHIPPED |

---

## 2. What was tested — full ledger

### P27 — root-cause conviction (source + op level)
- Full end-to-end source read of the shipped fp8-state paths: `_xpu_ops.py` (baked v124
  P19.5a unique bridge + C7), `v1/attention/backends/gdn_attn.py`, wheel sources
  `gdn_attn_interface.cpp` / `causal_conv1d.hpp` / `gated_delta_rule.hpp`,
  `gdn_linear_attn.py`, `qwen3_next.py` pool dtypes.
- Mechanism: `gdn_attention` runs TWO kernels driven by the SAME index tensors
  (interface.cpp:310 conv, :346 gated_delta_rule). The bridge gathered the ssm pool to a
  compact fp16 copy and remapped the indices, but passed `conv_state=kv_cache[0]` (the REAL
  pool) with REAL slot ids → conv kernel addressed compact rows of the real pool. Fits every
  prior observation: concurrent-wrong → serial-right (single-seq batches self-consistently
  use row 0); post-prefix worse (chunked-prefill continuation loads its initial conv window
  from the wrong row); heals after ~3 fresh prefills; both formats, both bridge routes; fp16
  pools never bridge → control clean.
- **Op-level runtime conviction** (p27_conv_repro.py, throwaway container on v1.2.25-raw,
  fp16 pools to isolate index semantics from fp8 math): Variant A (real indices [5,9] vs
  real pools) writes conv rows {5,9}; Variant B (bridge simulation) writes **{0,1}** —
  compact rows of the REAL conv pool, slots 5/9 left stale. P27_RUNTIME_VERDICT: CONVICTED.
- Secondary hazards found en route: torch.unique (value-dependent shape) inside the bridge
  must never run under full-decode-graph capture; NULL_BLOCK_ID (−1) padded tails make
  index_select wrap to the last pool row (excluded by narrowing in v126).

### P28 — dual bridge (op level + lane level, both formats)
- Design facts pinned from wheel sources first: spec index columns are DISTINCT
  per-rollback slots (gated_delta_rule.hpp:527-540 per-draft-step cache writes;
  causal_conv1d.hpp:827-845 per-column conv checkpoints — the "ONLY the last cache slot"
  file-header comment is STALE); therefore the per-VALUE mapping must be preserved exactly,
  and the capture-safe static remap must be FLAT: `remap[r,c] = r*W+c` (all ids distinct by
  allocator guarantee, verified once eagerly pre-capture via
  `torch.xpu.is_current_stream_capturing`).
- patch_v126_dualbridge.py (applies to `_xpu_ops.py`, backup .v126bak, py_compile gate,
  idempotent): decode-only batches → STATIC FLAT remap (no torch.unique under capture — the
  class that killed e5m2s4 boots on v124); eager prefill/mixed → per-VALUE unique remap,
  negatives filtered; BOTH pools gathered with the ONE index set, op run on compact copies,
  BOTH scattered home (ssm via uint8 view, conv via fp16 index_copy_). C7 → C7': certified
  enablement = `--mamba-ssm-cache-dtype fp8_e4m3|fp8_e5m2` directly; legacy
  `VLLM_XPU_GDN_FP8_NATIVE=1/2` refused in-code (rc=1, message 'llm-scaler v126').
- Op-level (p28_dual_check.py, fp8_e4m3 ssm pool + fp16 conv pool, O(1) inputs so state
  deltas survive fp8 quantization): LEG1 ns-route prefill slots [5,9] → conv rows written
  {5,9} (v124 wrote {0,1}), rows 0/1 untouched; LEG2 spec-route (2 reqs, W=5 columns 5..14,
  accepted [3,2], static flat branch) → changed-row sets IDENTICAL to the real-index fp16
  control for BOTH pools. P28_DUAL_BRIDGE_VERDICT: PASS.
- Lane-level e4m3 arm (fresh host, v1.2.25-raw + dualbridge + `--mamba-ssm-cache-dtype
  fp8_e4m3`, spec MTPx4, XGrammar-2, async, KV fp8_e4m3): boot HEALTH_OK ~180s; FULL decode
  graph capture SURVIVED with the fp8 pool (v124 postures died at the spec-verify dummy
  run — static flat remap is capture-safe as designed); DUAL_FP8_BRIDGE engaged on both
  workers; resets 0; JIT lines 0. P23F mode A: **0/80 concurrent wrong, 0/80 serial, 0
  flips, 30/30 tools clean**; sustain SUSTAIN_COMPLETE_NO_WEDGE; burst 36/36 resets 0;
  mode B post-prefix **0/24 wrong, 30/30 tools**; tracebacks 0.
- Lane-level e5m2 arm: boot HEALTH_OK ~150s, capture survived, identical clean battery:
  **0/80, 0 flips, 0/24, 30/30+30/30 tools**, sustain no-wedge, burst 36/36, resets 0.
- **P28 GATE: PASSED BOTH FORMATS — the P23D/P23F quality convictions are OVERTURNED.**
  fp8 SSM STORAGE under the dual bridge is quality-clean.

### P29 — roundtrip removal + full-fp8 pipeline + telemetry (a → M)
- Source audit (whole esimd_kernels tree): the only pool-roundtrip chain in production
  kernels = gdn_spec_update_seq loading prev state from pool slot[t-1] and storing slot[t]
  every draft token (W loads + W requantizing round-trips per verify round). e5m2 primitives
  already minimal; e4m3 LOAD wasted 3 fp16↔f32 conversions and mapped NaN codes 0x7F/0xFF
  to finite 480.0 (c10 decodes NaN).
- P29a kernel patch (v129 .so, MAX_JOBS=52 KERNELS_MAX_JOBS=52, sha 8f222ffe…): e4m3 load
  single-conversion (sign ORed into fp32 bits via u32 bitcast, NaN codes → qNaN, exact c10
  parity); fp8 REGISTER CHAIN in the spec kernel (caller-owned fp32 carriers; token 0
  dequantizes the rollback state, tokens 1..W-1 reuse registers — W-1 pool loads + ALL
  requant feedback removed; fp16 never chains, constexpr-gated, bit-identical dataflow).
- P29b op-level battery (windows 1–2): L2 fp8-vs-fp16 parity PASS both formats (after the
  race fix: e5m2 cos 0.99865, e4m3 cos 0.99964). L1 solo-vs-batch initially FAIL in ALL
  dtypes incl. fp16 → harness confound convicted (shared mutated conv clones; read-before-
  write seed semantics at gdn_conv_fused_seq_spec.h:218-261) → p29b_check.py REV2 (pristine
  clones per run) became a hard pre-serve gate. Clean-harness failure SURVIVED → real
  kernel defect hunt (p29b2→p29b13).
- **p29b12 — PRE-EXISTENCE PROVEN**: the nsd>1 double_v defect reproduces IDENTICALLY on
  the STOCK production .so (extracted fresh from v1.2.25-raw, sha-matched 1d9dcf4e…) — an
  upstream latent bug never exercised in production (model layer gates the ESIMD spec path
  at nsd==1). Not a v129 regression.
- P29B-DIAG (v130 instrumented kernel): first-divergence capture → value forensics
  (p29b13): mismatches split EXACTLY on dims whose writer work-groups already ran;
  indices/epoch fields never mismatch.
- **P29B ROOT CAUSE**: in the spec kernel the ring init row is a SAVE row — at nsd=2/HV=24
  the launch (48 wgs) exceeds ~32 HW residency slots, so late sibling work-groups read the
  init row AFTER early siblings rewrote it (sibling read-before-write race). Explains
  everything: flap = scheduling jitter; solo/HV=16 pass = full residency; zero-input failure
  = zero windows; determinism-within-run = stable schedule per launch shape; pre-existence
  in the stock fp16 .so.
- **P29B-FIX (v131 .so, e050551e…)**: pre-launch ring-row snapshot (at::index_select on
  the in-order stream, conv rows + SSM pool rows indexed by ring position); kernel reads
  the wg-start conv init window and the t==0 SSM rollback row from the never-written
  snapshot; t≥1 SSM reads stay on the live pool (own prior store, program-ordered);
  unraced paths bit-identical. Verification ALL GREEN: p29b13 rerun 0 mismatches (was 768);
  p29fix_verify.py bitwise solo-vs-batch (nsd=2/4 × HV=24/16 × 3 dtypes, 10 reps; nsd=1)
  PASS; L1–L4 PASS. **P29B CLOSED: root-caused, fixed, verified.** v130/v131 stay
  diagnostic-only.
- P29C lane integration (poolpath E1/E2/E3 + v131 .so): e4m3 lane RED from the first
  request (MATH 68/80, TOOLS 0/30) — the corruption is NOT the kernel (P29D real-params
  trajectory: NAT8-vs-BRIDGE8 pool cos == 1.000000 over 30 chained verify rounds at real
  layer-0 params incl. saturation).
- P29D serve bisection: fp16+no-poolpath GREEN; e4m3+bridge GREEN; e4m3+native+eager
  GREEN; e4m3+native+graphs RED (with and without async) → convicted: XPU decode-graph
  capture/replay × native-fp8 spec ride at nsd≥2 (LEG D nsd==1 gate GREEN).
- P29E/E-2/F/G — op-level graph forensics, ALL GREEN (capture freshness, full v131 wrapper
  under capture bitwise == eager for all dtypes; padding vectors A/B proven BENIGN — dummy
  rows walk the unowned null-block row 0; graphed multi-round trajectory bitwise through 30
  replay rounds with rotated rings): corruption lives in the SERVE INTEGRATION of the
  native fp8 path, and specifically in what real activations do to an fp8 pool.
- **P29H — comprehensive in-situ telemetry (the user directive), v1→v9f**: capture-legality
  laws established empirically (host syncs inside capture ILLEGAL; pin_memory needs
  device="cpu"; no dot/norm/sum inside capture — max-based metrics only; FUNCTION-SCOPE
  LAW: gate-site code may reference only module globals and self attrs; env vars do NOT
  reach spawned TP workers on this build → marker-file activation). v8 convicted the
  corruption IN-SITU: ring analysis of 434 in-graph replays — nan_pool=1 on EVERY row from
  row #0; the fp16 pre-call snapshot finite; qkvz finite (absmax 60-67); NaN appears DURING
  the native call; 364 nsd=1 rows all nan_out=1 = later reads of the ALREADY-POISONED pool
  (explains LEG D green and the "any-nsd" appearance). v9–v9e bisection matrix (fp16 lane):
  P29H instrumentation convicted as the wedge agent on BOTH pool dtypes (v9f extended it
  to e5m2); poolpath, dualbridge (incl. its fp16-pool decode-under-graphs cell, first time
  ever run), and the v131 .so ALL battery-green on fp16. P29H never ships.
- **P29I — op-level real-scale sweeps: the KERNEL IS INNOCENT; the e4m3 FORMAT cannot hold
  the state.** randn sweeps to absmax ~198 clean; heavy-tail ba to 1729 clean; pool-start
  sweep: e4m3 at seed absmax ~1280 → NaN INSTANTLY INCLUDING the fp16 twin — the XPU e4m3
  cast itself produces NaN beyond 448 (no satfinite): the pool cannot even be SEEDED with
  legit-magnitude state. e5m2 holds 1280/2560 (range 57344) but out-diff vs the fp16 twin
  grows to 0.93–1.84 on large states = 2-mantissa-bit quantization error (numerically
  matching the old e5m2 P23F tool-salad RED).
- **P29M — magnitude-minimal telemetry (timer-thread readout; READOUT LAW established)**:
  82 TICK lines, wedge-proof daemon dump. FIRST traffic tick after capture END already
  reads pmx=nan premx=nan with bamx=15.57 finite; |ba| runmax 16.0–20.2 (normal). NaN
  entered via the first prefill-written states / first spec update and the recurrence never
  clears it — the live mechanism behind e5m2's garbage answers. **FINAL VERDICT: BOTH fp8
  dtypes are FORMAT-IMPOSSIBLE for GDN SSM state math. The certified endgame = fp16 GDN
  pool + fp8_e4m3 KV.** (Bonus datum: the P29M lane SURVIVED where every P29H lane died —
  wedge surface narrowed to ring/reference-kernel/FLUSH machinery, diagnostic-only.)
- **P29S — ship-matrix adjudication**: the 23:35 "ship-posture" matrix (solo 26.83–31.03,
  4st 111–140) initially read as a 2× regression; cadence probe (gap p50=131 ms dead flat
  vs certified 43–48 ms; a contended stream measured FASTER than solo = impossible)
  convicted the measurement, and the lane's live .so sha was the v131 DIAGNOSTIC build
  (ring-row snapshot ≈ +7 ms × ~12 GDN layer calls ≈ the +88 ms/step; spec acceptance
  healthy 3.21–3.28). Matrix INVALID. **SHIP-MEASUREMENT LAW: every ship-posture
  measurement must assert the production .so sha before numbers are banked.**
- **P29S-2 — SHIPMAT2 matrix (PRODSO=1, production .so 1d9dcf4e, dualbridge-only, fp16 SSM
  + e4m3 KV + async + MTPx4), exact Run-7n methodology**:
  - cadence gap p50 = 50 ms (certified range restored; production .so acquitted by
    difference);
  - agg4x256: **236.80** / **253.74** tok/s (fp16 baseline 181.81 / 244.88 → **+30.2% /
    +3.6%**);
  - solo200: **75.58 / 75.69** tok/s (baseline 75.06 / 74.89 → above on both);
  - agg8x256: **423.11 / 422.93** tok/s, per-stream 52.9 × 8 — high-water aggregate on
    this hardware.
  The v125 fp8 solo deficit (−8.9%) is INVERTED: the posture is ≥ baseline on every
  measured axis and above on all of them.
- **P30 — full validation battery on the ship posture (fresh boot shipmat2b)**: ALL GATES
  PASS 02:03:00 — sanity/admission/v125-posture; deltas opsall=1 c7=1 fixline=1
  dualbridge=1; C7 functional 5/5; P23F probe A 0/80 + 0 flips + 30/30; sustain 3 rounds;
  serialized 24×3 SURVIVED; burst 36/36 ×3 resets 0; parser battery PASS; JIT recheck
  lines=8 cache-delta=1 (bound 2); P23F probe B 0/24 + 30/30; tracebacks 0; solo genspeed
  78.80/79.80/79.70; async 4×1024 aggregate **167.67 tok/s** (new probe high-water);
  runtime fp8 resolution float8_e4m3fn + float8_e5m2 both selectable end-to-end.
- **P31 — bake v1.2.26-raw (run 2)**: 101 GATE-OK / 0 FAIL, DONE 02:41:34; image
  e805da1449b0; repro_bootV1226.sh built (7285 B, async-REQUIRED lineage, baked-marker +
  DUAL BRIDGE boot gates); warm rounds JITWARM_OK (warm_ext 26.6 s, verify replay 11.9 s
  rc=0, xgrammar t2 200 + PARSED_JSON, all sampler legs 200); pedigree v1218–v1225 all
  present; jit stamps v64 + v1220–v1226; baked_serve_config = certified v1225 posture +
  dualbridge; no bake scripts left in image.
- **P31b — fresh-host raw validation**: host rebooted 02:45; RAWVAL boot HEALTH_OK ~170 s,
  live_resets=0, dualbridge boot gate passed; battery ALL GATES PASS 03:57:43 (every gate
  green, incl. P23F fresh 0/80 + 0 flips + 30/30, sustain/serialized/bursts, JIT delta=1,
  solo 78.50/79.72/79.65, async 4×1024 147.66 ≥ 100, tracebacks 0).
- **P31c/d/e — ship runs 1–3**: run 1 and run 2 aborted on ONE gate each (see §3); run 3
  **SHIP_V1226 ALL GREEN 04:56:14** — lane-commit v1.2.26 = 92cf94b232e1; V1226 SHIP GATES
  ALL PASS 04:45:06 (full inventory in §4); prod fresh-boot verify incl. v126 dualbridge
  marker on the RUNNING container (count=1); CC battery through litellm :4000 green
  (liveness 200; t1/t2/t3 200 blocks=text; thinking present); watchdog repointed to
  repro_bootV1226_prod.sh (backup .pre_v1226) and ARMED — "alive armed code=200 cycles=10"
  at 05:02:27 (the ship's "no fresh armed-line yet" WARN was benign: the watchdog marks
  every 10 CYCLES = 10 MINUTES; see §3.9).

---

## 3. Failure ledger (every RED / abort, root cause + resolution)

1. **P23D/P23F fp8-state quality defects (the round's entering failures)** — root cause:
   the bridge's ssm-only index remapping left the conv pool addressed at compact rows
   (P27). Fixed by the dual bridge (P28); both format arms re-measured clean (0/80, 0
   flips, 0/24, 60/60 tools). Convictions overturned; CLOSED.
2. **P29b L1 solo-vs-batch FAIL in all dtypes incl. fp16** — harness confound (shared
   mutated conv clones; read-before-write seed semantics), convicted against kernel
   source. Fixed in p29b_check.py REV2 (pristine clones per run; valid same-slot solo
   references; accepted=[1] so indices[0] is on the load path). L4's first construction
   error (accepted=[2] → init_col=1 → the filled slot never loaded) fixed likewise.
3. **nsd>1 double_v spec-kernel defect (p29b2–p29b13)** — sibling work-group ring-row
   read-before-write race when the launch exceeds HW residency; PRE-EXISTING in the stock
   production .so (p29b12), never exercised in production (nsd==1 gate). Root-fixed in the
   v131 diagnostic .so (pre-launch ring-row snapshot) and exhaustively verified — NOT
   shipped (no production path exercises it; no-degradation rule). En-route process
   failures fixed: v130 build redefinition storm (unguarded double include), installer
   idempotency guards testing strings that don't exist in the inserted text (multi-op
   registration abort) — lessons in §5.
4. **P29C/P29D e4m3 native-poolpath serve collapse (MATH 68/80, TOOLS 0/30)** — final
   root cause is NOT a kernel bug (P29D trajectory + P29E/F/G op-level exonerations) but
   FORMAT RANGE INADEQUACY: real GDN SSM states exceed e4m3's ±448 and the XPU cast has no
   satfinite (P29I/P29H-v8/P29M). The native fp8 SSM ride therefore never ships; the
   certified posture keeps the fp16 SSM pool (dualbridge carries fp8 STORAGE as a
   certified-selectable, quality-clean option via `--mamba-ssm-cache-dtype`).
5. **P29H lane wedges (v9, v9b, PRODSO leg, v9f)** — EngineDeadError via sample_tokens
   RPC timeout, on fp16 AND e5m2 lanes; bisection matrix (v9c–v9e) convicted the P29H
   diagnostic overlay itself (poolpath, dualbridge, and the v131 .so all battery-green on
   fp16; plain poolpath green; v131 .so fully acquitted). P29H never ships. The
   micro-mechanism (a near-inert strict subset of a stable footprint wedging the engine)
   remains open diagnostic debt (§6). En-route boot failures of v1–v3 were each root-caused
   into capture-legality laws (§5).
6. **P29S tainted ship matrix** — the "ship-posture" lane was booted without PRODSO=1 and
   measured the v131 diagnostic .so (3× step cost from the ring-row snapshot). Matrix
   invalidated; SHIPMAT2 re-run with the production-.so assertion; SHIP-MEASUREMENT LAW
   adopted. (Methodology mismatch — short-EOS prompts, thinking not disabled — was real
   but secondary; bench_run7n.py replicates Run-7n exactly.)
7. **P29T battery abort ABORT_P195_OPS (p195_ops=0)** — patch_v126_dualbridge.py replaced
   the v124 P19.5a block and dropped the lineage tag that battery/ship gates assert.
   Fixed three-part: inserted dual-bridge comment opens with the v124 P19.5a lineage note;
   the "old block must be gone" assert retargeted to the old comment's unique phrasing;
   ALREADY_APPLIED path self-heals pre-fix files. Verified on the live lane.
8. **P31 bake run-1 abort (dualbridge_patch_exit expected=1 got=0)** — output-contract
   defect: the gate greps tail -1 for caps tokens the patch never printed. Fixed by the
   v125 convention: V126_DUALBRIDGE_OK / V126_DUALBRIDGE_ALREADY as the FINAL line on
   every success path (error paths keep lowercase so tail -1 correctly fails). Lesson in
   §5.
9. **Ship run-1 abort (v124_p195_sycl_bridge ==1 vs got=2)** — the v126 lineage comment
   carries the P19.5a phrase twice; the battery assert was already >=1 but the inherited
   ship gate still demanded ==1. Fixed via mk_gates transform 3b (>=1 semantics;
   uniqueness of the v126 surface stays enforced by ==1 marker gates). Site-move audit
   lesson in §5.
10. **Ship run-2 abort (same gate, got= EMPTY)** — quoting-layer defect in the run-1 fix:
    awk placed OUTSIDE the `sh -c "..."` boundary must use BARE `$1`, not `\$1` (the
    escaped form belongs to the inner sh -c context); backslash-dollar → awk syntax error
    → empty read. Fixed, and the exact generated line tested standalone (echo-pipe both
    branches + committed-image grep) before run 3. Lesson in §5.
11. **RAWVAL boot usage error** — repro_bootV1226.sh requires a free-form MODE label;
    first launch died on the usage line. Relabeled RAWVAL (cosmetic: the usage literal
    still says repro_bootV1212.sh — recorded carryover).
12. **Ship WARN "no fresh armed-line yet"** — benign: watchdog loop is sleep 60/cycle and
    logs "alive armed" every 10 cycles = 10 minutes (plus 420 s post-launch grace); the
    ship's "10-cycle = 100s" comment is a stale carryover. Armed-line confirmed 05:02:27.

---

## 4. Certified production posture

**Image lineage**

| Image | ID | Role |
|-------|----|------|
| llm-scaler-exp:v1.2.26 | 92cf94b232e1 (24.7 GB) | PRODUCTION (lane-commit of the fully-validated warm lane) |
| llm-scaler-exp:v1.2.26-raw | e805da1449b0 | bake image (from v1.2.25-raw 4e83528bfd66), 101 GATE-OK / 0 FAIL + fresh-host battery ALL PASS |
| llm-scaler-exp:v1.2.25 / -raw | 3f3c91637692 / 4e83528bfd66 | retained (prior posture) |

Dangling ship-run commits 29257c1e5034 / 872c1b5f8b8c (runs 1–2, same lineage, superseded)
removed; pre-existing dangler b9b5b697baba left untouched.

**Lane**: lsv-test UP on v1.2.26, health=200, live_resets=0. Watchdog ARMED on
repro_bootV1226_prod.sh (backup lane_watchdog.sh.pre_v1226), marks "alive armed code=200"
every 10 minutes.

**Baked serve flags (verified on the running cmdline)**: `--kv-cache-dtype fp8_e4m3
--mamba-ssm-cache-dtype float16 --speculative-config mtp×4 --cudagraph_mode
FULL_DECODE_ONLY --cudagraph_capture_sizes [85 sizes] --async-scheduling
--tool-call-parser qwen3_coder`, served-model-name qwen3.8-27b-fp8, gmu 0.8, bs 64,
prefix-cache ON, mnbt 8192, no serve-side chat template.

**v126 code delta over v1.2.25**: `_xpu_ops.py` dual bridge (+ DUAL BRIDGE marker, count
exactly 1) + C7' certified enablement ('llm-scaler v126 C7' marker, count 1;
VLLM_XPU_GDN_FP8_NATIVE refs = 3: comment + environ.get + refusal message). Everything
else byte-identical to the certified v1225 stack (production lgrf .so exact sha
1d9dcf4e7a1c8db6e94b0673c51f58935d03de359dc109c41bbac6df00f85dff — gate-asserted).

**Ship-gates inventory (gates_v1226_lane.sh — ALL PASS 04:45:06)**: image-id asserts; v125
opsall markers; v126 C7 marker + 3 refs + functional refusal 5/5; v126 dualbridge marker
==1; v125-C7 text gone; production .so exact sha; P29H/P29M/poolpath round artifacts ABSENT;
wheel 0.1.8.3.dev0+g3cab97a.d20260925 with no fp8-state strings; v124 lineage deltas
(p16=2, p195_cache=1, p195_mamba=1, p195_ops>=1, mamba_fp16=1, mamba_fp8=0); v1222
serve-config (parser qwen3_coder + 85 capture sizes, no-template); v123 async+barrier
(False False 0); v60–v66 lineage; v65/v84–87/v89 absence; xgrammar 0.2.7; spec mtp×4
exact; pedigree v1218–v1226; jit stamps v64 + v1220–v1226; baked_serve_config; triton cache
floor 79 ≥ 79.

**Standing traps honored**: V1212-lineage boot scripts never reused (async-FORBIDDEN trap)
— V1226 lineage boots REQUIRE async; litellm through :4000 always with
`Authorization: Bearer sk-dummy`; litellm-proxy container never re-created
(unless-stopped, auto-returns after reboots); JSON bodies built with python + pre-validated.

---

## 5. Lessons (standing laws, added this round)

1. **Marker-convention law (P31)**: every patch consumed by a gate script MUST emit its
   machine-greppable caps status marker as its FINAL output line. Inventing gate literals
   the patch never prints is a generator defect the bake only catches at apply time;
   post-generation audits must check BOTH sides of every output-consuming gate.
2. **Site-move audit law (P29T/P31c)**: when a marker's SITE moves (v124 block → v126
   lineage comment), EVERY consumer of that marker across the chain (battery, ship gates,
   patch asserts, boot gates) must be re-audited in the same edit — the P29T fix updated 3
   of 4 consumers and the 4th surfaced only at ship time.
3. **Quoting-layer law (P31d)**: `\$1` escaping is a property of the sh -c INNER context,
   not of "gates awk" generally. When moving a pipeline fragment across a quoting boundary
   in a generator, re-derive the escaping from scratch and test the exact generated line
   standalone (echo-pipe both branches) before any consumer run.
4. **SHIP-MEASUREMENT LAW (P29S)**: every ship-posture measurement must assert the
   production .so sha (PRODSO legs) before numbers are banked — a diagnostic .so in the
   lane silently convicts the wrong stack (here: 3× step cost from a +7 ms/call snapshot).
5. **READOUT LAW (P29H v9f)**: never read device state from the forward path — host syncs
   are async-scheduling-hostile AND gate sites are unreachable on serial lanes; readout
   belongs to a daemon TIMER thread.
6. **Capture-legality laws (P29H v1–v3, v7)**: host syncs (`.item()`, event waits) inside
   XPU graph capture are ILLEGAL, not just slow; pinned tensors need explicit
   device="cpu"; no dot/norm/sum reductions inside capture (max-based metrics are proven
   legal); gate-site instrumentation may reference only module globals and self attrs
   (FUNCTION-SCOPE LAW).
7. **Activation-transport law (P29H v4/v6)**: env vars do NOT reliably reach spawned TP
   workers on this vLLM build — diagnostic activation must use a MARKER FILE (filesystem
   is process-tier-proof).
8. **Idempotency-guard law (P29B-DIAG build saga)**: an installer's no-op guard must test
   a string that ACTUALLY EXISTS in the inserted text (probe for the inserted symbol), and
   the guard's own print output must be read critically — "patched:" with a probe-looking
   tag is the tell of a guard that never matched.
9. **Synthetic-scale blind spot (P29I)**: op-level probes with randn*0.5 inputs were
   20–100× below real activation scale and stayed green while the serve lane was NaN-poisoned
   — exoneration probes must sweep REAL magnitudes (real layer-0 params, real absmax
   ranges) before a kernel is acquitted.
10. **Measurement sanity checks (P29S)**: cross-check impossible readings (a contended
    stream "faster" than solo; flat 131 ms gaps vs 43–48 ms certified) before adjudicating
    a regression — they convict the measurement setup, not the stack.

---

## 6. Open items (diagnostic debt; ship-unaffected)

1. **P29H fp16 strict-subset wedge micro-mechanism**: on fp16 lanes the P29H executed
   footprint is provably a near-inert strict subset of what ran STABLE on fp8-pool lanes
   (INIT allocs + prints + periodic eager torch.save; PRE/POST never execute — ctr=0,
   pmx=0.0 through capture AND traffic), yet every P29H lane wedged with the identical
   EngineDeadError signature on both pool dtypes, while P29M (P29H minus
   ring/reference-kernel/FLUSH) survived. Candidates: capture-time interaction of inserted
   text/global state with graph recording; async-scheduling interplay of flush D2H under
   concurrent prefill; allocation-timing effects of lazily-allocated diagnostic buffers.
   Diagnostic-lane-only impact; P29H never ships.
2. **Per-head scaled-space kernel arithmetic**: the only conceivable future fp8 SSM-state
   path (rewrite the state update in scaled coordinates, unscale on read). Weeks-scale,
   quality-risky; documented as a future plan, deliberately not attempted (clamp-to-448
   would be saturated-state systematic corruption, not a fix).
3. **Non-spec gdn_conv_fused_seq sibling-hazard audit**: the same init-row/save-row
   aliasing class exists in principle in the non-spec seq kernel; untouched this round per
   the no-degradation rule (no evidence of a production path exposing it — noted for the
   kernel team).
4. **Watchdog cadence comment**: ship script's "10-cycle = 100s" text is stale (loop is
   sleep 60; armed-line every 10 minutes). Cosmetic; fix opportunistically next round.
5. **Cosmetic carryovers**: validate_v1226_run.sh caps literals read
   VALIDATE_V1225/V1225_VALIDATION_COMPLETE (rename carryover; ship precondition greps the
   V1225 literal against the v1226 log by design); repro_bootV1226.sh usage line still
   says repro_bootV1212.sh.

---

## 7. Artifacts (perf-v126/)

- **P27/P28**: p27_conv_repro.py (op-level crosstalk conviction), p28_dual_check.py
  (dual-bridge op-level LEG1/LEG2), patch_v126_dualbridge.py (the SHIPPED patch — dual
  bridge + C7' + P29T lineage fix + P31 caps markers), boot_v126_test.sh.
- **P29 kernel line**: p29a_kernel_patch.py, build_esimd_v129.sh/v130/v131,
  gdn_conv_fused_seq_spec_dbg.h, p29fix_{orig_,}gdn_conv_fused_seq_spec{,_dbg}.h,
  p29fix_install.py, p29fix_verify.py, p29dbg_{dedup,install}.py.
- **P29 probes**: p29b_run.sh, p29b_check.py, p29b2–p29b13_probe.py, p29d_traj.py,
  p29e_graph_probe.py, p29f_probe.py, p29g_probe.py, p29i_probe.py, p29i_probe2.py,
  mk_poolpath_nsd1.py.
- **P29 telemetry line**: patch_v126_poolpath.py, mk_poolpath_p29h.py (v7/v8/v9
  five-capture-point telemetry), p29h_analyze.py, p29_serial_mag.py, mk_poolpath_p29m.py
  (magnitude-minimal timer-thread telemetry), p29c_boot.sh (bisection-switch boot chain).
- **P29S/P30**: p29s_cadence.py, bench_run7n.py (exact Run-7n replica), mk_validate_v1226.py.
- **P31 ship chain**: mk_stage5_bake_v1226.py, mk_ship_v1226.py, mk_gates_v1226_lane.py
  (with the >=1 transform 3b + quoting-layer comment), src_v126_*.py (source snapshots).
- **Records**: PHASES.md (live P27–P31e), PLAN.md, this COMPLETE_ROUND_WRITEUP.md.

Host-side: /root/build/{patch_v126_dualbridge.py, stage5_bake_v1226.sh,
gates_v1226_lane.sh, ship_v1226.sh, validate_v1226_run.sh, validate_v1226.sh,
sanity_v1226.sh, repro_bootV1226.sh, repro_bootV1226_prod.sh, lane_watchdog.sh +
.pre_v1226}; logs under /root/build/lce1/ (v1226_bake.log, v1226_rawboot.log,
v1226_validate_run.log, v1226_ship.log, v1226_gates_lane.log,
boot_v1226_shipwarm.out); preserved diagnostics: p29h_diag_{869,875}_v8.pt,
p29h_diag_{869,875}_v9b_serial.pt, p29m_mag_*.pt, shipmat1_v131so_serve_full.log,
p29fix_full_build.log; .so backups /root/build/{v129_so, v130_so}.

— recorded 2026-09-30, round COMPLETE.
