# v125 round — live investigation record

Round task: "Start write full end-to-end plan of implementation, when plan
is ready run full implementation and testing suite, goal is fix all possible
issues and bottlenecks to archive a big improvement, bake new production
image llm-scaler-exp:v* using best values and improvements. Important! Spec
and XGrammar-2 has to have supported and also e5m2 and e4m3 are supported
end-to-end. Do not use subagents." (all execution direct, autonomous chain)

Plan: `PLAN.md` (same dir). Phase chain P0..P26. This file is the live
record — updated after every phase per standing directive.

---

## P0–P21 (condensed; full details in commit history / T3_ROUND_REPORT)

- **v1.2.24 SHIPPED** (image `b13f56543057`): certified fp16 spec-4 posture
  from the v1224 ship chain (`ship_v1224_cont.sh`, `fix_gates_v1224.py`,
  `cc_battery_fix_v1224.sh`). Lane standing on it since; watchdog armed.
- Round analysis (PLAN.md §2): decode step remains memory-bound; dominant
  per-stream state bytes = mamba/GDN SSM state reads (fp16) over 38k
  contexts; 51% EU stall / 55%-of-peak read profile from v89 telemetry.
- **P22 scoped** as the round's kernel lever: fp8-native SSM state
  (e4m3 + e5m2) on BOTH GPU surfaces:
  - Surface A = ESIMD fused kernels (`custom_esimd_kernels_vllm`),
    non-spec + spec-1 decode;
  - Surface B = SYCL `torch.ops._xpu_C.gdn_attention` decode kernel
    (python dispatch in `vllm/_xpu_ops.py`).

## P22-B — SYCL surface, xe2 C++ route: EVALUATED-BLOCKED, reverted

- Attempt: extend the xe2 chunk kernel's fp16 state template
  (`chunk_gated_delta_rule_kernels_xe2.hpp`) to c10 fp8 element types.
- **Blocker (build-time, 12 errors)**: the cutlass/cute subbyte reorder
  helper (`upcast_subbyte_t`) eagerly instantiates
  `cutlass::platform::numeric_limits<c10::Float8_e4m3fn/e5m2>` which is
  **undefined in cutlass** — no header-local fix; would require vendoring
  numeric_limits for both fp8 types into the cutlass copy. Cost/benefit
  rejected for this round (python-side split achieves the decode win).
- Tree restored to pre-patch (`p22b_xe2_revert.py`; original reject message
  at :1737 back, KERNEL_LAUNCHER count 4, marker gone). Verified clean.

## P22-B2 — SYCL surface, python decode-only split: SHIPPED, legs GREEN

- Change (`p22b2_python_patch.py`, applied in lsv-test): in
  `vllm/_xpu_ops.py`, native fp8 routing requires
  `attn_metadata.num_prefills == 0` — decode/spec-only batches take the
  native fp8 decode kernel; any prefill-containing batch takes the
  certified fp16 bridge (gather→fp16→scatter). Single
  `torch.ops._xpu_C.gdn_attention(...)` op internally routes
  prefill→xe2 chunk kernel (fp8-incapable), decode→fp8-native decode.
- **Why this shape**: the original B2 gate (pool fp8 + native enabled)
  also routed PREFILL batches into the fp8 decode kernel → boot death.
  The split keeps prefill on the certified bridge forever; decode keeps
  the fp8-native fast path. ENGAGED log on first decode-only batch (during
  graph capture at boot).
- Legs (spec-4 serves, `p22b_run2_v125.sh`, restored 18:10:52):
  - **e4m3n**: ENGAGED ✓, `391` math ✓, coherent prose, acceptance
    0.814/0.729/0.607/0.529 (mean 3.68–3.75), solo 70.06/71.12/71.09,
    4×1024 agg 121.02, tracebacks 0.
  - **e5m2n**: ENGAGED ✓, acceptance 0.894–0.904/0.745–0.776/0.63/
    0.504–0.519 (mean 3.35–3.82), solo 67.47/70.37/71.12, agg 87.45
    (one slow stream — P23 re-check candidate), tracebacks 0.
  - Both vs stage-A bridge ≈ 5× degradation → **bridge-penalty removal
    proven**; e4m3 near-parity with fp16 solo.
- PROBE_LIST/PROBE_MATH `TypeError: object of type 'NoneType' has no
  len()` = known qwen3 content=None quirk, not a kernel signal.

## P22-A — ESIMD surface: build clean; smoke gates doing their job

- Kernel patch `p22a_kernel_patch.py` (+`p22a_fix2.py` on the live tree):
  StateT-templating of `gdn_conv_fused{,_seq,_seq_spec}` state pointers +
  fp8 load/store overload sets in `utils.h` (e5m2: byte<<8 exact load /
  RNE-truncate store; e4m3: exact load incl. subnormals / c10-exact store
  with denorm magic-add 141<<23). Build fixes folded: (1) `#pragma once`
  utils.h (no include guard → redefinitions), (2) `bit_cast_view` is
  `&`-qualified → named `magic_bits` lvalue, (3) four `fp16* sr0/sr1`
  row-pointer sites in spec header → `StateT*`.
- Build (`build_esimd_v125.sh`, esimd-inc, KERNELS_MAX_JOBS=52): CLEAN,
  `.so` 97,447,152 B, sha `5f9b0378ad01...`, 3 `esimd_gdn_conv_fused`
  symbols. NOT yet production-swapped.
- Routing fact (settles leg design): under spec×4 ALL decode is spec
  batches → SYCL surface; ESIMD surface engages only non-spec decode
  (and spec-1). P22-A legs therefore run spec-stripped serves.
- Gate ordering hazard honored: the extended `_gdn_conv_state_fp16_ok`
  gate (`p22a_gate_patch.py`, admits fp8 SSM pool under
  `VLLM_XPU_GDN_FP8_NATIVE=1`) is valid ONLY with the new `.so` —
  old `.so` + fp8 pool = fp8-bytes-as-fp16 NaN cascade. Runner order:
  swap → import → smoke → gate → legs.

### P22-A smoke v1 (18:25) — ABORTED CLEAN, and it caught two real things

Runner `p22a_run_v125.sh`: pause watchdog → backup `.so` (.v125bak) →
swap → import → op-level smoke → gate → legs → restore. Smoke failure
aborted BEFORE any serve leg; original `.so` restored, certified serve
back up, watchdog re-armed (18:25:51). Zero lane impact.

Findings from v1 failures (14 z_out fails, core_attn_out PERFECT 0.0
everywhere — fp8 loads exact):

1. **`esimd_gdn_conv_fused` (interleaved) is structurally H≤4-only —
   pre-existing, NOT P22A.** WG_SIZE=32 hardcoded; `double_v = HV >
   (32-4*H)/2`; useful threads = `4*H + HV`. At our production H=8/HV=24:
   `4*H = 32` alone exhausts the WG → z_out rows never written (both
   fp8 dtypes AND fp16 fail bitwise-identically = dtype-independent).
   Upstream `test_gdn_conv_fused_tp_fix.py` covers only H=2/H=4 and its
   docstring is literally about z_out + adjacent-buffer stomp detection.
   Our model never dispatches it (sequential layout; vLLM spec-eligibility
   pins H==8/HV==24 sequential) → out of scope for this lane; documented.
   Caveat recorded: a future Qwen3-Next-80B (interleaved) lane on ≥TP4
   with H≤4 is its only valid deployment shape.
2. **seq op (`esimd_gdn_conv_fused_seq`) PASSes 100%** for e4m3+e5m2,
   N∈{1,4,8}, prod shape H=8/HV=24 — the fp8 state path is correct on
   the non-spec surface this lane actually uses.

### P22-A smoke v2 (18:37) — ABORTED CLEAN; defect isolated to spec+e4m3

v2 (`p22a_smoke.py` rewrite): dropped fused cells (documented above),
added T0 fp16-vs-fp16 harness control, added spec-1 differential
(`esimd_gdn_conv_fused_seq_spec`, nsd=1/nst=5 mirroring the production
call site), zero-filled out/z buffers (v1 lesson: torch.empty +
partial-write = spurious cross-run garbage diffs).

- ALL PASS except ONE case trio — **spec op + e4m3**: T1 z_out 0.027,
  T2 z_out 22.5, T3 updated-pool NaN. core_attn_out PASSES both T1/T2.
- spec + e5m2 PASSES 100% (z diffs 1.5e-5/1.28e-4 — but z_out is a RAW
  qkvz copy so ANY nonzero diff = stomp; magnitude differs only by how
  fp8 byte pairs decode as fp16 → **both dtypes stomp, e4m3 worse**).
- Host dispatch (`.sycl`) verified symmetric across dtypes — not the
  asymmetry source.
- **Debug window** (`p22a_dbg_spec.py` + `p22a_dbg_run.sh`, window 2
  18:47): z compared EXACTLY against the raw-copy contract; out 64-half
  coverage counts; pool NaN slot map; canary tensor; variant sweep
  (same-slot vs distinct slots, nst 1..5, num_accepted 1..5, both
  dtypes). Window 1 lost to a format-string TypeError; window 2 completed.

### P22-A ROOT CAUSE — harness bug, kernel exonerated (pending re-verify)

The window-2 table flipped the diagnosis. In EVERY variant (both dtypes,
all slot/nst/accepted combos): `out` ALL-ZERO (nz lo/hi=0/0 — never
written), `z` ALL-ZERO (zbad=full tensor, d=0.1149 = the expected value
itself), pool WRITTEN (e4m3: 12 NaN bytes exactly at the last-saved slot
— slot 7 same-slot, slot 11 distinct-slots; e5m2: NaN impossible at these
magnitudes so invisible to a NaN-only scan). This also un-masked smoke v2
as partially vacuous: its spec `core_attn_out` PASSes were two unwritten
zero buffers comparing equal; only the z cross-run diffs carried signal.
The smoke-vs-dbg z discrepancy (stomped there, clean here) matched the
allocation-order adjacency (smoke: conv→out→z contiguous → overflow lands
in z; dbg: conv allocated last → overflow lands in an untracked gap).

**The bug was in both harnesses, not the kernel.** Kernel contract
(`gdn_conv_fused_seq_spec.h` :228/:237/:361/:368/:376):
`global_t = token_indx_ptr[state_row + t]` indexes the **qkvz row, the ba
row, the output row AND the z_out row** — i.e. `token_indx` = PACKED-ROW
POSITIONS `0..nsd*nst-1` (production call site asserts
`qkvz.size(0) == nsd*nst`; upstream gathers via
`mixed_qkv.index_select(0, spec_token_indx)` then scatters back with
`index_copy_`). `spec_state_indices` carries the pool SLOTS — a different
tensor for a different purpose. Both harnesses passed the SLOT value (7)
as `token_indx`: with 5-row tensors, every out/z write landed at row 7 —
past the buffer end — which is exactly the smoke z-stomp (contiguous
buffers) and the dbg unwritten-z signature (writes in the gap), and OOB
qkvz/ba row-7 reads (garbage in smoke's exact-size allocations → the
22.5 / NaN magnitudes; valid storage in dbg's 8-row alloc → only the
residual 12-NaN e4m3 signal, expected to vanish with correct indices).
No upstream test exists for the spec op (`grep seq_spec tests/` empty) —
the kernel source is the only contract; read it before the next harness.

Fix: `token_indx = torch.arange(rows)` in both harnesses; dbg also gained
a pool-nonzero-byte count (e5m2-write visibility). Both staged; full
round relaunched (`p22a_run_v125.sh`, run 3).

### Run 3 (18:57) — harness fix verified; REAL kernel defect isolated

With correct indices: **z_out exactly 0.000000 both dtypes** (stomp gone,
raw-copy contract now exact), **e5m2 spec PASSES 100%** (T1 core 1.5e-5,
T2 4.65e-4 = legitimate quantization scale). Remaining FAILs, spec+e4m3
only: T1 core 0.881 (from IDENTICAL zero pools!), T2 core 30.98, T3 pool
image NaN. Aborted+restored clean (18:57:49). Math fact that ruled out
precision drift: the update map h → exp_g·h + beta·(v − kn·h)·kn is
CONTRACTING (eigenvalues exp_g, exp_g−beta, both <1) — h cannot reach
480/NaN from σ=0.1 inputs with correct arithmetic ⇒ wrong VALUES, not
accumulated rounding.

### P22-A ROOT CAUSE #2 — `esimd_e4m3_store` wrong magic constant (REAL)

`utils.h` `esimd_e4m3_store`: `norm = f_bits + 0xF887FFFFu` — the comment
itself says `((7-127)<<23)+0x7FFFF` which is **0xC407FFFF**. Sanity:
f=1.0 → 0x3F800000+0xC407FFFF = 0x03807FFF, >>20 = 0x38 = e4m3(1.0) ✓;
with 0xF887FFFF → 0x380 → low byte 0x80 = **−0.0**. With the built
constant, 0.03 encodes to a byte decoding as ~256 — per-save ~300×
amplification. Smoking-gun evidence already in the logs: **seq e4m3 T3
pool image max_abs = 80/88** (N=1/4/8) vs e5m2's 0.25 — the seq saves
were corrupted all along; T1/T2 never saw it (single-call ops exercise
the LOAD path only; saves don't feed back within one call) and T3's
sanity bound (100) let 88 pass. The spec op reloads its own saves every
token → amplification compounds over the 5-token chain → 480 cap → NaN
codes → e4m3 load aliases NaN codes to ±480 (no-NaN format) → finite but
huge out (T3 core 30.98, nan=False) — every number explained. e5m2 store
(self-contained RNE top-bits trick, no magic constant) immune ✓.

Fixes: (1) `p22a_fix3_store_const.sh` — 0xF887FFFF→0xC407FFFF in the
host tree utils.h (applied, verified in place); (2) same constant fixed
in host + local canonical `p22a_kernel_patch.py` (both occurrences:
docstring + code) so the idempotent build patch can't regress it; (3)
smoke gained **T1b pool-image differential** (fp16-run saves vs fp8-run
saves decoded, atol 0.1) in seq AND spec cases — the check that would
have caught this instantly; sanity bound 100 stays only as a backstop.
Incremental rebuild launched (KERNELS_MAX_JOBS=52, standing directive).

### Rebuild (19:11→19:14, 2m48s incremental) + Run 4 (19:17)

Fixed .so: 96,808,176 B, sha `7c5fdcdda0be…`, 3 `esimd_gdn_conv_fused*`
symbols, lane healthy pre-swap. **Run 4 smoke: SMOKE_FAILS=0,
P22A_SMOKE_PASS — both dtypes, seq+spec, including the new T1b
pool-image differential at 0.001–0.004** (true e4m3/e5m2 quantization
scale; the 80/88 corruption is gone — root cause #2 fix verified at op
level). Run 4 then aborted at the GATE step: my own
`p22a_gate_patch.py` post-replace assert demanded
`_gdn_fp8_native_enabled` count == 1, but NEW adds it TWICE (import +
call) — the assert could never pass; unreached until now because every
prior attempt aborted at smoke. Assert corrected to == 2, restaged,
**run 5 launched** (idempotent: backup exists, gate marker absent).

### Run 5 (19:26) — smoke+gate GREEN; all legs FAILED on a REGRESSED LANE

Smoke + gate passed cleanly at last (`P22A_SMOKE_PASS`, `P22A_GATE_PATCHED`),
then every leg failed at boot:

- **ctrl-ns**: EngineCore death in `_arstage.py` under dynamo
  (`Unsupported hasattr call` on `getattr(s, "xpu_stream")`) — engine
  init, before any traffic.
- **e4m3ns / e5m2ns**: `vllm serve: error: argument --mamba-ssm-cache-dtype:
  invalid choice: 'fp8_e4m3' (choose from 'auto', 'float16', 'float32')`.

### ROOT CAUSE #3 — the lane had silently REGRESSED to v1.2.23 (watchdog service stale-parse)

`docker inspect lsv-test`: Created **18:49:22**, image
`23a07b1e5ff6…` = **v1.2.23** — NOT v1.2.24 (`b13f56543057`). The lane
was recreated mid-round during P22-A debug window 2 (serve down for the
harness) when the containment watchdog fired its capture+relaunch:

1. **Why v1.2.23**: the lane-watchdog **service process** started
   **15:06:10** — BEFORE the 17:07 repoint to `repro_bootV1224_prod.sh`.
   Bash parses the whole `while` loop at first execution — **editing
   `lane_watchdog.sh` on disk never updates the running process**. The
   18:49:21 trigger fired the in-memory V1223 boot script → `docker rm
   -f` + run of tag v1.2.23. (Its own startup log still says
   `lineage=repro_bootV1221.sh`; service restart is REQUIRED after every
   watchdog repoint — new standing trap.)
2. **Cascade**: the recreate wiped ALL in-container round state — the
   P22B2 python patch (which also defines `_gdn_fp8_native_enabled`),
   B2 serve logs, and the original `.v125bak` (the current one backs up
   v1.2.23's stock .so).
3. **Leg failures fully explained by the wrong image**: v1.2.23 lacks
   the baked fp8 choices — v1.2.24 (b13f) HAS them
   (`MambaDType = Literal["auto","float32","float16","fp8_e4m3","fp8_e5m2"]`,
   verified via `docker run --rm` grep). The B2 legs at 18:01/18:05 ran
   on the pre-recreate v1.2.24 container (host log shows
   `'mamba_ssm_cache_dtype': 'fp8_e4m3'` accepted). ctrl-ns's
   `_arstage` dynamo death is likewise a v1.2.23 artifact, not a P22A
   signal. Run-5 leg results are VOID — image regression artifacts.
4. Runs 3/4/5 smoke results remain VALID (op-level, only the custom
   ESIMD module + GPU — no vllm serve involvement).

**Repair executed (19:58→20:19), two refinements discovered en route:**

1. Run 5's own restore had FAILED (`ABORT_RESTORE_BOOT`) — the v1.2.23
   container carried the gate patch importing
   `_gdn_fp8_native_enabled`, but the recreate had wiped the B2 patch
   that defines it → engine death at first decode. So the runner exited
   without DONE; the repair precondition was relaxed to accept the
   abort marker (runner-gone is what matters — no boot races).
2. `p22b2_python_patch.py` alone is NOT the full B2 stack — it builds
   on the BASE `p22b_python_patch.py` (module resolver
   `_gdn_fp8_native_enabled` + the `FP8_NATIVE ENGAGED` gate rework of
   the baked P19.5a bridge). The pre-recreate container had the base
   from run-1 (17:13); a fresh image needs **p22b → p22b2 in order**
   (`p22a_repair2_v125.sh`, both PATCHED, syntax + helper import OK).
3. Repair outcome: watchdog service RESTARTED (fresh parse = V1224
   lineage; the 15:06 process was the stale one), full
   `repro_bootV1224_prod.sh` boot — **image id verified
   b13f56543057**, HEALTH_OK ~150 s, posture async ✓ spec ✓ ssm fp16 ✓,
   baked `MambaDType` fp8 choices ✓, PROBE_MATH **391** ✓ (content=None
   quirk needs `or ""` + reasoning_content fallback — probe bug, not a
   serve defect). Watchdog re-armed 20:18. Note: boot reported
   live_resets=1 (stale dmesg from today's chaos; P23 host reboot
   clears it — watch resets=0 there).
4. **Run 6 launched 20:19** on the correct image (fresh container:
   `.v125bak` = b13f stock .so, gate marker absent, fp8 CLI admitted).

## Run 6 (20:19→20:36) — legs COMPLETE on the correct image; verdict: parity speed, e4m3 quality BROKEN, capacity win NOT realized

All three non-spec legs booted and served clean (health OK, tracebacks
0, markers exact: dtype/env/spec per leg). Solo 1x1024 x3 / async
4x1024 (tok/s aggregate):

| leg | solo r0/r1/r2 | async agg | notes |
|-----|----------------|-----------|-------|
| ctrl-ns fp16 | 30.13/30.18/30.19 | 47.62 | no ESIMD_FP8_ELIGIBLE line = .so+gate neutral ✓ |
| e4m3ns | 30.02/30.09/28.57 | 47.86 | ESIMD_FP8_ELIGIBLE fired |
| e5m2ns | 30.10/30.17/30.15 | 46.83 | ESIMD_FP8_ELIGIBLE fired |

**Numerics (corrected probes, mt>=250 + content-only reads):**
- e4m3ns: MATH600 **391 ✓** but LIST300 degenerate (`"1, 22, 333,
  4444444"`) and PROSE250 **repetition collapse** (`"...contains all
  all all all..."`, 956 chars) → **e4m3 ESIMD is NOT end-to-end
  working**. Single-op differentials (0.001-0.004) did not predict
  this: quantization error compounds through the state RECURRENCE over
  ~250 steps. The SYCL surface stayed coherent with e4m3 — its
  quantization semantics is the reference to port.
- e5m2ns: NOT captured — probe watcher v1 had a timeout-arithmetic bug
  (`while [ $i -lt 160 ]` with `i+=15` = ~160 s total, NOT the intended
  40 min; died 20:32:37 before the 20:33 window), manual rescue fired 1
  min after teardown. Fixed in `p22a_probe600_v125.sh` v2 (2400 s
  budget, plain-var bookkeeping). Re-capture owed in P22-C.
- ctrl-ns numerics: never captured (watcher only targeted fp8 tags) —
  attribution gap: cannot yet fully separate fp8-state drift from
  spec-stripped behavior. P22-C must probe the fp16 non-spec leg too.

**Capacity finding (from saved per-leg serve logs):**
`Available KV cache memory: 8.59 GiB` — **byte-identical across
fp16/e4m3/e5m2**. The fp8 SSM pool's memory saving is NOT being
converted into more KV cache; the sizing/profiling path apparently
doesn't account for pool dtype. Defect + the biggest untapped
superiority lever.

**User directive (20:4x, raises the bar):** "both fp8(e4m3/e5m2)
implementations has to be superior of baseline on any cases and work
end-to-end" + bake into new production image. Current state does NOT
meet it (SYCL e4m3 70-71 vs 72-74 spec-4; ESIMD parity 30.1 with e4m3
prose broken; KV budget identical). Nothing bakes until P22-C closes
the gap.

## P22-C (inserted before P23) — make fp8 SUPERIOR, not parity

**Refined user directive (run 6 aftermath):** "both fp8(e4m3/e5m2)
implementations has to be superior of baseline on any possible cases
but **degradation is not allowed** and has to work end-to-end" + bake
into new production image. Ship-gate matrix this imposes:

| dimension | requirement for fp8 to be the SHIPPED default |
|-----------|----------------------------------------------|
| speed (solo, agg, contention) | >= fp16 baseline in every measured case |
| quality (prose/list/math, long-gen) | coherent, no repetition collapse — equal-grade to fp16 |
| capacity (KV blocks / context) | > fp16 (this is the expected superiority case) |
| end-to-end support | CLI -> serve -> spec-4 (SYCL) AND non-spec (ESIMD) paths both selectable+tested on every image |
| degradation | NONE allowed anywhere; a single regressing case blocks the default flip |

If a format cannot clear the matrix it stays SELECTABLE+certified (never
removed — e4m3/e5m2 end-to-end support is standing) but the default
serve dtype stays whichever clears it. Decision by measurement in P24.

- **C1 numerics attribution + root-fix** (RUNNING): first the H2
  discriminator — probe the fp16 non-spec control with corrected
  max_tokens (p22c1_ctrl_probe.sh). If fp16ns collapses too, run-6's
  e4m3 "failure" was a non-spec-path defect, not fp8. Then per verdict:
  recurrent-state trajectory differential (fp64 ref vs SYCL vs ESIMD,
  ~64 steps, production shapes); SYCL stayed coherent with e4m3 — its
  quantization semantics is the reference. NOTE: utils.h audit shows
  BOTH stores are already RNE-unbiased (e5m2: RNE bit-trick; e4m3:
  magic-add subnormals + corrected 0xC407FFFF normals; loads exact) —
  so if H1 holds, the compounding mechanism is precision (3-bit
  mantissa random walk), and the fix direction is WHERE quantization
  sits in the recurrence (e.g. keep-decay in higher precision), not the
  store's rounding.
- **C2 capacity realization**: trace the KV-budget calculation
  (gpu_worker.py:483 "Available KV cache memory" / profile pass) —
  make the fp8 pool shrink profiled usage -> more KV blocks. Measure
  delta; convert into context or concurrency headroom. This is
  superiority case #1 (more capacity at identical memory).
- **C3 op-level perf**: microbench fp16 vs fp8 ESIMD kernels (wall +
  bytes moved, bs 1..64). If conversion-bound: vectorized 8-wide fp8
  ld/st, remove staging buffers. Target fp8 kernel < 0.7x fp16; then
  step-level re-measure incl. high-bs contention where state bandwidth
  matters most. If solo decode stays parity (transformer dominates
  step time), the honest superior cases are capacity + high-bs
  regimes — extract contention wins, no degradation anywhere.
- **C4 re-legs + full gate matrix**: all three non-spec legs + SYCL
  spec-4 legs, corrected probes on EVERY leg incl. the fp16 control.
  Gate to P23: the matrix above fully measured, both formats green on
  end-to-end, default-candidate identified.

## Run 7 (P22-C C1, 2026-09-28 late) — H2 CONFIRMED: fp16 non-spec control collapses too; run-6 e4m3 "failure" was NOT fp8

**Instrument history (why v3):** v1 watcher concatenated
content+reasoning (e4m3ns collapse measured on CONCAT); v2 probe
(content-only) returned EMPTY on the non-spec serve (qwen3 keeps
everything in reasoning_content at mt=600); v3 = the only comparable
instrument — `chat_template_kwargs: {"enable_thinking": False}` + dual
content/reasoning capture. Also fixed in v2's shell: pkill ERE trap
(`\|` = literal pipe = matches nothing — certified serve never died,
v1 probes measured the WRONG serve; v2+ kill with plain patterns +
port-down proof loops, fail-closed).

**v3 verdict on ctrl16ns (fp16 pool, spec stripped, rebuilt .so
sha 7c5fdcd — verified in place; .v125bak = stock 1d9dcf4e preserved;
for fp16 pools stock and rebuild run identical code paths):**

| probe | content head | degeneration | finish |
|-------|-------------|--------------|--------|
| MATH600 | `'3!!!` | `!` lockup to len 600 | length |
| LIST300 | `'1!!!` | `!` lockup to len 300 | length |
| PROSE250 | `'The!!!` | `!` lockup to len 252 | length |
| STORY400 | `'Unit!!!` | `!` lockup to len 403 | length |

VERDICT math391=False list5=False. **The fp16 non-spec control
degenerates on every long greedy probe — same failure class as run-6's
e4m3ns (different flavor: pure `!` lockup vs word repetition). H2
CONFIRMED: the collapse is a property of the spec-stripped serve
surface (or the no-think template on it), NOT the pool dtype. fp8 is
exonerated at serve level.**

Last discriminator cell (p22c1_chain.sh, running): the same no-think
probes against the RESTORED certified spec-4 fp16 serve (cert16s4):
- cert16s4 coherent -> broken surface = spec-stripped serving (or its
  ESIMD non-spec kernel path); no-think template fine; fp8 quality
  gates must be judged under spec-4 posture only (B2 e4m3 already
  coherent there, think-on).
- cert16s4 also `!!!` -> no-think+greedy on this hybrid model is the
  artifact itself; all no-think probes are invalid instruments, quality
  judged think-on (content+reasoning concat, probe600 v1 pattern).

Chain then runs p22c1_traj.py with serve down: op-level recurrent
trajectory differential (T=64 single-token calls of
esimd_gdn_conv_fused_seq, pool carried in place = production solo
decode; cells fp16/e4m3/e5m2 pool, identical inputs+seeds, fp16 params
in all cells; classification BOUNDED / RANDOM-WALK(sqrt(64)=8x floor,
threshold 16x) / AMPLIFIED; + spec-op section nsd=1 nst=5 cycling
num_accepted 1..4). Even with H2 confirmed this matters: an AMPLIFIED
fp8 trajectory would be an op-level defect to fix before C4 spec legs;
BOUNDED/RANDOM-WALK + healthy states = op-level clean.

Routing insight recorded during this pass (affects C4 design): the
certified spec-4 lane runs decode through the ESIMD SPEC kernel
(esimd_gdn_conv_fused_seq_spec, gdn_linear_attn.py ~1063) — the wheel
op (_xpu_ops.py) is prefill/mixed/fallback. B2's coherent fp8 legs
measured the WHEEL path (stock .so, gate False on fp8 pool). The
ESIMD-spec+fp8 combination has NEVER been leg-tested — C4 must add
e4m3s/e5m2s legs with the v125 gate env ON (spec-4 + fp8 pool + rebuilt
.so routes spec decode through ESIMD fp8).

## Run 7b (P22-C C1 CLOSED + C2 root finding, 2026-09-28 21:08)

**C1 CLOSED — fp8 exonerated on quality, three independent ways:**

1. **cert16s4 COHERENT** (chain step B, certified spec-4 fp16 + the same
   no-think instrument): MATH600 '391' ✓ stop, LIST '1, 2, 3, 4, 5' ✓,
   PROSE Rayleigh-coherent 378c, STORY coherent 633c — verdict
   math391=True list5=True. The no-think template/instrument is VALID;
   only the spec-stripped surface degenerates.
2. **ctrl16ns DEGENERATE** (fp16 pool): one plausible token then `!!!!`
   lockup on all four probes (MATH '3!!!…', LIST '1!!!…', PROSE
   'The!!!…', STORY 'Unit!!!…'), all finish=length.
3. **traj PASS** (p22c1_traj.py, serve down, rebuilt .so): T=64
   single-token recurrent calls + spec-op T=32 nsd=1 nst=5 cycling
   num_accepted 1..4; determinism exact (cross-run dS=0); e4m3/e5m2
   pools BOUNDED on BOTH ops (seq d_state ≤0.225 vs floor 0.033, non-
   growing, ref_drift 13.7 = recurrence active and contractive; spec
   d_state ≤0.185). No NaN, states healthy. Note: my synthetic A_log
   scale (±0.1 randn) is NOT production scale — real-weights traj only
   if serve-level evidence demands it.

**Attribution: run-6's e4m3ns collapse = the spec-stripped SERVE
surface, broken for fp16 identically — pre-existing (first non-spec
serve legs ever run on this lane), NOT fp8, NOT the rebuild .so (fp16
spec path runs the same .so coherent in cert16s4).** Under the standing
"any degradation is not allowed" directive this is a BLOCKING defect on
an end-to-end-selectable surface and must be root-fixed.

**Code inspection of the non-spec path (no defect found at inspect
level):** dispatch correct (sequential layout → esimd_gdn_conv_fused_seq
at gdn_linear_attn.py:1101 — the H≤4-broken interleaved kernel is
Qwen3-Next-only and NOT dispatched here); non-spec invocation
(:1105-1126) mirrors the smoke/traj-proven pattern (state_idx passed as
both conv and ssm indices, n_dec ≤128); non_spec_state_indices_tensor
populated for pure-decode batches (gdn_attn.py:120,207,279,412);
serve diff vs certified = exactly the --speculative-config line + log
path (clean sed).

**Running (p22c1_nsdisc.sh):** cert-vs-ctrl paired battery — short-mt
sweep (2/4/8/16/32/64) with top-5 logprobs (localize first corrupt
step), think-ON mt=600 dual capture (the cell v2 lost: if non-spec
reasoning coherent → only no-think+non-spec combination broken).

**C2 root finding (from run-6 leg logs, no new boot):** the hybrid
allocator PEGS mamba page == attention page (interface.py:645 sizes attn
block so its page ≥ mamba page; :669 pads mamba page UP to exactly the
attention page; xpu.py:637 then rounds attn block to multiple-of-64 and
re-pads mamba). Net: fp16 851,968→1,048,576 B (+23.1% pad), fp8
458,752→524,288 B (+14.3% pad); group page 2048 B/token-slot for BOTH
dtypes — today's +1.25% capacity win is ONLY the 512-vs-1024 block
granularity effect, not the dtype halving. **Removing/relaxing the
equal-page constraint = ~+6.7% class capacity lever for fp8 (1920 vs
2048 B/token-slot), ~+10% for fp16.** Both specs already carry
page_size_padded >= real machinery — the constraint lives in the hybrid
merge/XPU layer. Design + risk assessment (flat-pool stride math,
_xpu_ops page-table mapping) queued behind the C1 root fix.

## Run 7c (P22-C C1 root-cause narrowing, 2026-09-28 21:16)

**nsdisc paired batteries (cert16s4 vs ctrl16ns, same prompts, greedy,
top-5 logprobs):**

| | pos0 | pos1 | pos2 | pos3 |
|---|---|---|---|---|
| cert16s4 | '3':-0.00 | '9':-0.00 | '1':-0.00 | '<|im_end|>':-0.00 → "391" stop |
| ctrl16ns | '3':-0.00 (**identical dist to cert**) | '!' (ALL top: -12.42) | same flat | same flat |

- ctrl pos1+ top-3 logprobs are EXACTLY EQUAL ('!':-12.42, '"':-12.42,
  '#':-12.42) ≈ ln(1/250000) = −12.43 = **uniform over the whole vocab**.
- THINK600/THINKPROSE on ctrl: content_len=0 reasoning_len=0
  finish=length — 600 tokens sampled from the uniform distribution are
  all invisible/special tokens (nothing renderable).
- Boundary is EXACT: prefill + first decode correct (pos0
  bit-consistent with certified), decode step 2+ broken. Solo request,
  single KV block throughout (~22 tokens) — block allocation NOT
  involved.

**Interpretation refinement:** perfectly FLAT logits ≈ constant
logit vector ≈ zero/constant residual at the LM head OR a ZEROED
logits tensor being read for sampling+logprobs. Corrupt-but-real state
would produce STRUCTURED-but-wrong logits, not exact uniform. So the
leading class is an OUTPUT-side buffer defect on the non-spec decode
path — candidates: (a) cudagraph replay output wiring for the dense
bs=1..N decode graphs (v89 85-size set; NEVER replayed by the
certified lane because mixed batches reclassify non-spec decodes as
prefills, gdn_attn.py ~266 — run 6 was their first exercise ever),
(b) the v1.2.23 async-scheduling handoff
(AsyncGPUModelRunnerOutput, built+validated under spec-4 verify
batches) on non-spec decode output. State-corruption (GDN slots)
remains possible but is now second-class. op-level traj PASS supports
the model computing correctly internally.

**Running (p22c1_eager.sh):** ctrl config + --enforce-eager
(compilation-config stripped), same battery. Coherent → graph replay
wiring; still uniform → async handoff (eager keeps async scheduling);
follow-up matrix legs (graphs+no-async, eager+no-async) only as needed.

## Run 7d — C1 ROOT CAUSE FOUND (eager leg + v60g log): no-draft FULL
graph replay never materializes output rows (KNOWN fork defect,
wedgefix-v60 E, sanitizer-not-fix)

**Eager leg (p22c1_eager.sh, DONE 21:22:35): FULLY COHERENT — graphs
convicted, async exonerated.** eager16ns (ctrl config + --enforce-eager,
--async-scheduling KEPT): MT8/MT64 logprobs sharp at every position
('3':-0.00, '9':-0.00, '1':-0.00, '<|im_end|>':-0.00 with structured
runners-up — bit-consistent with cert16s4's fingerprints: '1':-13.16
'<|im_end|>':-12.86 identical), MATH600=391, LIST300='1, 2, 3, 4, 5',
PROSE250=378 chars real Rayleigh prose, STORY400=735 chars, THINK600
coherent ('\n\n391'). So: graphs OFF + async ON = coherent; graphs ON +
async ON (ctrl16ns) = degenerate from pos1. The async handoff and the
non-spec forward itself are both innocent. Mechanism = cudagraph replay
of no-draft decode batches.

**Smoking gun in the ctrl serve log (grep v60g):** serve_p22c1_ctrl.log
fired, on BOTH TP ranks, from the first battery probes (21:13:06+):

    llm-scaler v60g NO-DRAFT DEGENERATE-ROW SANITIZED #1..#N rows=1/1 —
    NaN logits from unmaterialized nospec graph output replaced with
    uniform over grammar-allowed tokens (root: fork nospec
    FULL_DECODE_ONLY output wiring, wedgefix-v60 E)

= the exact-uniform -12.42 ≈ ln(1/250000) signature in the nsdisc
logprobs was NEVER a zeros-logits read — it is the **v60g sanitizer
substituting uniform** for NaN logits rows. The fork's own wedgefix-v60
investigation had already root-caused this class (gpu_model_runner.py
:4944-4990 comment): "the nospec FULL_DECODE_ONLY graph never writes the
hidden rows the deferred tlogits gather reads, so the logits row is
zeros-or-NaN"; v60e's lm_head recompute over sample_hidden_states was
PROVEN INSUFFICIENT (hidden states themselves are NaN). On the MTP lane
this fired only on rare no-draft steps (post-rejection-tail bare 1-token
steps) and the sanitize kept XGrammar-2 from 500ing — so it shipped as a
mitigation, root never fixed. Run 6 was the first ever NON-SPEC serve
surface: EVERY decode step is a no-draft step → every step sanitized →
'Unit!!!'/'!' degenerate text. The current serve_full.log has 0 v60g
firings (spec-4 verify batches always carry drafts; probes on the
restored lane take the spec path).

**Boundary re-derivation (corrects Run 7c):** pos0 was correct NOT
because "first decode materializes" but because **the first generated
token is sampled from the PREFILL forward** (eager under
FULL_DECODE_ONLY). The first DECODE step (producing pos1) already
replayed the broken graph. Boot log shows "Capturing CUDA graphs
(decode, FULL): 27/27" — size-1 graphs pre-captured at boot, so the very
first live decode was a replay, not a lazy capture. Every decode replay
= unmaterialized rows.

**Config confound eliminated:** ctrl serve (compilation-config
stripped) resolved to the SAME platform defaults as certified —
cudagraph_mode=FULL_DECODE_ONLY, the 85-size capture list, 27 graphs
captured at boot (serve_p22c1_ctrl.log line 8/31/125). No mode
difference between ctrl and certified; only spec on/off differs.

**CUDAGraphWrapper contract read (compilation/cuda_graph.py:145-370):**
replay returns `entry.output` = weak_ref_tensors of the CAPTURE-time
output (`torch.ops._C.weak_ref_tensor`); input copy-in is done outside
the wrapper. Spec-shaped graphs materialize correctly on replay (months
of certified operation); no-draft-shaped graphs do not (fork-known).
Root fix lives at DISPATCH level, not kernel level.

**ROOT FIX (designed, next):** exclude no-draft decode batches from
FULL-graph dispatch — `_determine_batch_execution_and_padding` already
receives `disable_full` (use_cascade_attn or has_encoder_output,
:4349-4351) and execute_model computes `spec_decode_metadata` at :4626
BEFORE calling it at :4647. Add: no-spec batch → disable_full →
dispatcher returns NONE → eager forward (proven coherent by eager16ns).
Env-guard VLLM_V125_NOSPEC_NO_FULL_GRAPH=1 default-on; v60g stays as
tripwire (must fire 0 after fix). This ALSO fixes a live production
defect: rare no-draft steps on the certified MTP lane currently emit
sanitized-uniform GARBAGE tokens (real quality win, tiny perf cost on
those rare steps). Spec verify batches keep graphs — certified posture
untouched. PIECEWISE-nospec remains unvalidated/unscoped (lane runs
FULL_DECODE_ONLY only).

## Run 7e — C2 VERDICT: equal-page peg NOT removable this round (design-documented); fixval round-1 ABORT was a stale staged patcher (stray colon)

**C2 (capacity peg) closed as DESIGN-DOCUMENTED, NOT SHIPPED.** The peg
sits in `platforms/interface.py:_align_hybrid_block_size` (+ xpu.py:620
multiple-of-64 attn rounding): it raises `block_size` and pads
`mamba_page_size_padded = block_size × attn_page_size_1_token` so
attention page == mamba page EXACTLY ("Padding mamba page size by
%.2f%%"). Why it exists — the consumer side makes it an INVARIANT, not
accounting: `mamba_utils.preprocess_mamba` derives mamba state slots
FROM BLOCK-TABLE INDEX ARITHMETIC on the single shared block_size
(`prev_state_idx = (num_computed_tokens - 1) // block_size`;
`num_blocks = cdiv(computed+scheduled, block_size) + num_spec_blocks`;
`curr_state_idx = num_blocks - 1 - num_spec_blocks` — the speculative
state-carry corner case in its comment is all block_size division),
`copy_bufs` gathers use `mamba_state_idx` as block indices, and
`abstract.py:47 → MambaSpec(page_size_padded=...)` makes the padded page
the flat-pool byte STRIDE the ESIMD kernels address slots by. Unpegging
(attn page ≠ mamba page) would fork: hybrid allocator grouping, both
index formulas, the ESIMD slot addressing, AND prefix-cache block
identity (a cached prefix block must resolve to one attn block AND one
mamba state slot — equal pages are what makes it one table). That is a
weeks-scale allocator fork for +6.7% (fp8, 1920 B/slot vs 2048) / ~+10%
(fp16) capacity — NOT contained late-round. Cost today is bounded and
known: fp16 851,968→1,048,576 B/slot (+23.1%, block 1024), e4m3
458,752→524,288 (+14.3%, block 512); the run-6 +1.25% granularity win
is banked and the capacity gate is ALREADY cleared at 2.10–2.12× by fp8
attention KV. Future-work note: any unpeg attempt needs its own
allocator fork + full validation leg (spec carry, prefix identity,
preemption resume — the mamba_state_idx pop/retain logic at
preprocess_mamba:176-183 is exactly the class that regresses).

**fixval round-1 ABORT_PATCH root cause (process lesson):** the staged
`patch_v125_nospec_no_full_graph.py` on the host was a STALE revision
with a stray trailing `:` after the already-paren-balanced line
`(EDIT3_OLD, EDIT3_NEW))):` → container SyntaxError → clean ABORT at
step 1 (before watchdog pause — lane stayed certified and armed). Fix =
one-character local edit + re-stage + a NEW pre-launch gate: host-side
`python3 -m py_compile` on every staged .py before launch (COMPILE_GATE_OK
now required). Round-2 relaunched 21:5x: V125_C1_PATCH_OK with all three
EDIT_OK lines, backup .pre_v125_c1, ns serve booting.

## Run 7f — fixval round-2: patch engaged statically but FULL replay persisted → instrumented diag launched (run 7e's C2 verdict stands)

**fixval round-2 (21:39-21:49):** V125_C1_PATCH_OK (all 3 edits, py_compile
clean, backup .pre_v125_c1), ns serve booted (spec_lines=0, capture line
present), battery STILL DEGENERATE — fix16ns '3!!!!!!!' / 'Unit!!!...'
with the exact uniform signature (pos1+ every top = -12.42, pos0 sharp
'3':-0.00) and v60g fired 20× (rows=1/1, both TP ranks) → ABORT_V60G.
Certified restore side ALL GREEN (spec_lines=2, MT8/MT64 = '391' with
sharp per-position logprobs identical to the cert16s4 fingerprints —
the C1 patch does NOT disturb the spec-4 lane).

**Static re-audit after the failure (all checks PASS, mechanism sound
on paper):** (1) marker present ×2 in the loaded file, workers booted
post-patch; (2) a SECOND dispatch_cudagraph call exists at :4406 but is
the DP>1 re-dispatch — inert on this TP2 lane; :3364 is kv-sharing
fast-prefill — also inert; (3) the ONLY live caller of
_determine_batch_execution_and_padding is execute_model:4670 with the
patched disable_full_nospec=spec_decode_metadata is None (LHS unpacks
directly into cudagraph_mode/batch_desc, no rename); (4) dispatcher
semantics confirmed: CUDAGraphMode runtime values are the concrete
{NONE,PIECEWISE,FULL} (FULL_DECODE_ONLY=(FULL,NONE) is a config
composite), invalid_modes={FULL} subtracts FULL, PIECEWISE keys are
empty under FULL_DECODE_ONLY → dispatch MUST return NONE; (5)
set_forward_context:4816 feeds that same cudagraph_mode;
CUDAGraphWrapper.__call__ reads forward_context.cudagraph_runtime_mode
and falls to runnable (eager) on mode mismatch. Every link verified —
yet the graph still replayed. Conclusion: an assumption is factually
wrong at runtime; stop static analysis, instrument.

**Instrument (patch_v125_diag.py + p22c1_diag.sh, launched 21:5x):**
D1 logs first 8 live dispatches (nospec flag, env flag, tokens,
RETURNED mode, padded descriptor, cascade/enc) at the C1 dispatch site;
D2 logs first 8 CUDAGraphWrapper.__call__ entries (ctx mode vs wrapper
runtime mode, match, descriptor, entry count). One instrumented ns
boot + two probes will show whether the exclusion engages, what mode
comes back, and which mode the wrapper actually sees. Restore +
re-arm in the same chain.

### Run 7g — 2026-09-28 22:08-22:13 diag round-3: C1 ENGAGED, graphs
### theory DEAD — poisoning is global to the graphs boot (branch C)

Round-3 launched while round-2's aborting chain was still inside its
480 s health-wait: TWO writers held the same log fd at different
offsets and round-2's abort bytes landed mid-line into round-3's
output (`…/gpu_moABORT_NS_HEALTH`, pid-22191 traceback fragment).
Round-3's own data was intact. LESSON: before relaunching a chain,
kill the prior chain by name (`pkill -f p22c1_diag.sh`), not just its
poller — a script in a silent health loop is still a live writer.

Findings, in order of weight:

1. **The C1 exclusion DID engage at runtime — the output is STILL
   degenerate.** D2 (wrapper `__call__`, budget 40/worker) used
   exactly 27 entries during capture (one per size) and had 13 slots
   left; ZERO live entries. The D2 site sits AFTER `__call__`'s early
   return (`ctx_mode == NONE or ctx_mode != wrapper runtime_mode →
   return self.runnable(...)`) — so live no-draft decodes took that
   early return: dispatch returned NONE, the wrapper fell to the
   runnable, i.e. an EAGER forward ran. Probes still degenerate
   (MT12 `'3!!!!!!!!!!!'`, LIST40 `'1!!!…'`, v60g ×20). Branch (C)
   confirmed: eager-under-runtime-NONE on a graphs boot still NaNs.
2. **D1 budget arithmetic (why no nospec=True lines):** boot consumes
   ~56 dispatches/worker (profile 8192-NONE + warmup-64-NONE + 27
   sizes × (NONE warmup + FULL capture)) > budget 40 — exhausted
   before first live decode. Budgets must exceed ~60 for any
   live-inclusive window.
3. **Inductor hypothesis DEAD:** both ns and cert boots log
   `TORCH_COMPILE_DISABLE is set, disabling torch.compile` (vllm.py:984)
   AND the xpu.py:349 gate ("v31.1 #11 + v124 P16: spec-off dynamo
   cannot trace arstage"). The runnable is the UNcompiled model — the
   same forward enforce-eager runs.
4. **NaN originates in the forward output.** Logits pipeline (read in
   full): `hidden_states[logits_indices]` → `compute_logits` inside
   execute_model (spec_seg `tlogits`); `hidden_states` +
   `sample_hidden_states` ride the ephemeral state into
   `sample_tokens` where v60g sanitizes. v60e already proved
   `sample_hidden_states` NaN — the hidden rows themselves are NaN.
5. Full `CUDAGraphWrapper.__call__` read: no early-replay branch
   before the assert; the wrapper is entered only on FULL match. The
   fork's split-phase execute confirmed: `execute_model` returns None
   and stashes `ExecuteModelState`; `sample_tokens` (collective_rpc)
   finalizes — v60g sits there (:5016). The ":6405 fifth call site"
   of `_determine_batch_execution_and_padding` is the capture
   machinery's dummy-run (`force_eager=is_profile or mode==NONE`,
   `force_uniform_decode`) — boot-only, accounted.
6. **Elimination matrix (graphs boot, ns serve):** FULL replay =
   degenerate (run 6/7); NONE → runnable eager = degenerate
   (round-3); enforce-eager boot = coherent (eager16ns,
   bit-consistent with cert16s4). Script diff ctrl vs eager verified:
   ONLY `--enforce-eager` + the `cudagraph_mode=FULL_DECODE_ONLY` +
   85-size compilation-config differ. The poisoning is GLOBAL to the
   graphs-captured boot, upstream of any dispatch decision.
7. **H1/H2 discriminator launched** (p22c1_h12.sh, 22:2x): boot =
   ctrl + `cudagraph_mode=NONE` (graphs never captured, NOT
   enforce-eager — compile wrapper, persistent buffers, profile run,
   async all present). Coherent → H1 capture/boot-side pollution
   (persistent buffers, mamba copy bufs, capture dummy runs);
   degenerate → H2 any non-enforce-eager boot poisons spec-off
   decode → root fix = enforce-eager-equivalent semantics for
   spec-off serves. D1 will show live nospec=True dispatches in this
   boot (boot uses ~2 dispatches, budget survives).

## Pending (next)

- Verify run-6 restore terminal state (P22A_V125_DONE, jitwarm,
  watchdog re-arm).
- P22-C C1-C4 as above (new user directive: fp8 must be SUPERIOR).
- P23 full E2E battery both formats on fresh host (reboot 10.20.3.65
  first per standing directive); e5m2 87.45-agg slow-stream re-check.
- P24 full wheel build (KERNELS_MAX_JOBS=52) + default-serve decision by
  measurement+quality. P25 bake v1.2.25 + ship chain + watchdog repoint.
- P26 EU telemetry re-measure, parse_output share, COMPLETE_ROUND_WRITEUP
  + git commit.

## Run 7h — H2 CONFIRMED (H12) + TWO-PASS resolution discovery: the
compile/posture theory rebuilt on log evidence; 3-leg D3 discriminator
chain launched (2026-09-29)

1. **H12 verdict (p22c1_h12.sh): H2 CONFIRMED.** Boot = ctrl ns with
   `cudagraph_mode=NONE` only: zero `Capturing` lines (graphs never
   captured), zero D2 wrapper entries, live D1 lines
   `nospec=True env_flag=True mode=NONE` for prefill (ntok_pad=27) and
   decode (ntok_pad=1, uniform=True). Probes degenerate ('3!!!'-class,
   v60g NO-DRAFT x20). ⇒ ANY non-enforce-eager boot poisons spec-off
   decode; poisoning is upstream of cudagraph dispatch entirely. H1
   (capture/persistent-buffer pollution) falsified.
2. **RETRACTION — "inductor DEAD as cause" (Run 7g item 3) was
   WRONG.** TORCH_COMPILE_DISABLE is NOT in the container env, NOT in
   any serve script (verified docker inspect + grep). What Run 7g saw
   was real but mis-read: the vllm.py:984 warning fires only in the
   LATE second pass (below), and the xpu.py:349 gate fires in every
   process — those two lines got conflated into "env already set".
3. **TWO-PASS resolution (the actual mechanism, from boot-log
   timelines):** pass 1 constructs the config with
   `optimization_level=O2` → `mode=VLLM_COMPILE` → custom_ops append
   `'none'` → `ir_enable_torch_wrap=True` → kernel_config
   rms_norm/fused_add = `['native']`. LATER, `check_and_update_config`
   (platforms/xpu.py:313; TP>1 unsafe gate v31.1 #11 + v124 P16) sets
   `TORCH_COMPILE_DISABLE=1` in that process → the vllm.py:985 branch
   fires → `mode=NONE` — but custom_ops stays STALE at
   `['+quant_fp8','none','+quant_fp8']` and rms stays `['native']`
   (the :1032 append rule sees 'none' already present and appends
   nothing; set_platform_defaults already ran). `enforce_eager` wins
   at PASS 1 (vllm.py:975 → mode NONE + cudagraph NONE) → ops
   `'all'` + rms `['xpu_kernels','native']` + warmups 0 — the coherent
   posture. Evidence: nograph EngineCore init print 22:20:41
   (VLLM_COMPILE/'none'/native, `cudagraph_num_of_warmups: 1`, worker
   "torch.compile and initial profiling/warmup run together took
   0.96 s" at 22:21:15) vs its own vllm.py:984 warning only at
   22:21:38; eager EngineCore init print already NONE/'all'/xpu.
   Whether the 0.96 s window truly traced/compiled anything (env
   timing differs per process — nograph workers never printed :984)
   is UNRESOLVED; it is a measured pass-1 difference regardless.
4. **Cert lane shares the degenerate pass-1 posture.** serve_full.log
   init print: `mode=VLLM_COMPILE`, `custom_ops=['+quant_fp8','none',
   '+quant_fp8']`, `ir_enable_torch_wrap=True`, rms `['native']`,
   enforce_eager=False — IDENTICAL to nograph, and coherent (spec-4).
   ⇒ the pass-1 compiled posture alone is NOT the poison; the poison
   is the CONJUNCTION (pass-1 compiled/'none' posture) × (spec-off).
   What survives as candidate factors: (a) the compile window /
   VLLM_COMPILE-machinery artifact for no-draft forwards, (b)
   custom_ops 'none' = op substitution OFF (native rms_norm/native op
   impls NaN on ns decode while the spec path — different kernels /
   verify layout — tolerates them), (c) any other enforce_eager-gated
   runner machinery (the only literal runner refs are spec-config
   gates at :5726/:6625; gpu_worker/warmup.py fork code also gates).
5. **Lane hygiene finding:** the certified serve restored after H12
   was booted with the C1+diag venv still applied (v125DIAG dispatch
   lines present in serve_full.log 22:24 boot, `nospec=False
   env_flag=<unset>` — C1 inert for spec-on, D1 log-only past budget
   40, zero functional effect). The D3 chain's step 1 restores the
   CLEAN venv (.pre_v125_c1 runner + .pre_v125_diag cuda_graph) and
   the final restore re-boots serve_user.sh on clean code, closing
   this gap. LESSON: the h12-style chains restore the serve script
   but not the venv — every diag chain must restore BOTH.
6. **D3 instrument (patch_v125_d3.py):** NaN slice at the tlogits
   gather + lm_head (`hs_nan` rows / total, `shs_nan`, `logits_nan`,
   `li_max`, `li_tail`), budget 80 (boot ~2 + MT12 12 + LIST40 40),
   on top of the clean venv. Applied once; present in all three legs.
7. **3-leg discriminator chain LAUNCHED (p22c1_d3.sh, pid 211738,
   log /root/build/lce1/p22c1_d3.log)** — all legs ns serve, cudagraph
   NONE, D3 in venv, single-variable compilation-config changes:
   - A `nograph`  `{"cudagraph_mode":"NONE"}` — known-bad baseline;
     D3 answers NaN NATURE (wholesale forward vs gathered rows only).
   - B `modenone` `{"mode":"NONE","cudagraph_mode":"NONE"}` — pass-1
     eager op-posture (ops 'all', xpu_kernels rms, no compile window)
     with enforce_eager STILL False.
   - C `opsall`   `{"cudagraph_mode":"NONE","custom_ops":["all"]}` —
     compile window kept, op substitution forced ON.
   Verdict map: B coherent → poison inside pass-1 posture; then C
   coherent → custom_ops 'none' is the poison (fix = ops 'all' for
   spec-off); C degenerate → compile window is the poison (fix = mode
   NONE at pass 1). B degenerate → poison is enforce_eager-gated
   runner machinery, next probe = D3 NaN pattern + runner gates.
   Each leg: capture=0 check, pass-1 posture extract, MT12/LIST40
   probes, v60g count, D3 slice. EXIT trap restores the certified
   lane on abort (fixes the h12 abort-path gap). Terminal
   P22C1_D3_DONE.

## Run 7i — CONVICTION + ROOT FIX + FIXVAL (2026-09-28 22:38-23:09)

1. **D3 3-leg matrix COMPLETE (p22c1_d3.sh, all legs ran, lane restored
   cert spec_lines=2 instr_lines=0, re-armed 22:48:25):**

   | Leg | config (ns, cudagraph NONE) | mode | custom_ops | rms | compile window | verdict |
   |-----|------------------------------|------|------------|-----|----------------|---------|
   | A nograph  | `{"cudagraph_mode":"NONE"}` | VLLM_COMPILE | `['+quant_fp8','none','+quant_fp8']` | native | 0.93 s | **DEGENERATE** (MT12 '3!!!', LIST40 '1!!!...', v60g x20) |
   | B modenone | `{"mode":"NONE","cudagraph_mode":"NONE"}` | NONE | `'all'` | xpu_kernels | none | **COHERENT** (MT12 '391', LIST40 '1, 2, 3, 4, 5', v60g 0) |
   | C opsall   | `{"cudagraph_mode":"NONE","custom_ops":["all"]}` | VLLM_COMPILE | `'all'` | native | 0.88 s | **COHERENT** (MT12 '391', v60g 0) |

   A↔C differ ONLY in custom_ops — single-variable conviction. Mode,
   rms priorities, compile window, graphs, enforce_eager runner gates
   are ALL exonerated (B and C coherent despite differing in each).
2. **D3 NaN nature (leg A):** prefill forwards CLEAN (hs_nan 0/27,
   0/28); EVERY decode forward wholesale NaN — hs_nan=1/1 from the
   FIRST decode step (entire hidden state, not just gathered rows),
   shs_nan=1/1, logits_nan=1/1, li_max=0; the NEXT prefill after 11
   NaN decodes clean again → no persistent state/weight corruption;
   the plain 1-token decode forward itself computes NaN. Points at
   the decode-side native op fallback (GDN/mamba recurrent decode
   path); NOT rms_norm (leg C ran native rms fine).
3. **ROOT CAUSE (CONVICTED):** stale `custom_ops='none'` from pass-1
   resolution poisons spec-off decode. Sequence: `O2` → pass 1
   resolves `mode=VLLM_COMPILE` → append rule (vllm/config/vllm.py:
   1032-1039) adds `'none'` (op substitution OFF); LATER the TP>1
   gate (platforms/xpu.py:347-349, v31.1 #11 + v124 P16) sets
   `TORCH_COMPILE_DISABLE=1` → vllm.py:985 flips mode to NONE but
   custom_ops stays stale `'none'`. With op substitution off, the
   native fallback of the registered custom-op set NaNs single-token
   GDN decode forwards. enforce_eager wins at pass 1 (:975 → NONE →
   ops 'all') — why eager boots were coherent. The spec-4 cert lane
   survives 'none' only because verify steps are multi-token
   (different kernel path). Elimination matrix closed: graphs-boot +
   FULL replay degenerate; graphs-boot + C1 runnable degenerate;
   NO-graphs degenerate (H12); enforce-eager coherent; pass-1 NONE
   coherent (B); ops 'all' coherent (C).
4. **Process incident (double launch):** the first fixval launch
   attempt returned plink exit 128 AFTER printing STAGE_GATE_OK but
   without the LAUNCHED echo — it HAD started (log 22:51:22); a
   verification call then launched a SECOND instance → two-writer
   log + interleaved boots; the first instance's EXIT trap killed
   the second's Boot-A serve → health-timeout abort → trap cascade.
   Cleanup: killed survivors with self-match-proof patterns, waited
   out traps, verified converged lane (one cert serve 219272/31887
   host/container view, health 200, fingerprints pass, venv clean,
   pause flag removed), archived the corrupted log as
   `p22c1_fix2.log.attempt1`, relaunched ONCE (pid 222334).
   LESSONS: (a) verify a launch actually started (log exists / pid)
   BEFORE relaunching — plink exit code after `;`-chains ending in
   pkill/pgrep is unreliable; (b) fix2's restore_lane omits the
   xpu.py restore (only R + CGP) — restore functions must cover
   EVERY file the chain touches; (c) `pgrep -af` matches its own
   wrapper when the pattern appears in an embedded path argument
   (log filename), not only the bracket-broken pattern itself.
5. **ROOT FIX (patch_v125_opsall_fix.py):** at the same xpu.py point
   the gate forces compile off, restore the intended compile-off
   posture — `custom_ops 'none' -> 'all'` (guarded: only when 'none'
   present and 'all' absent, so explicit user configs are never
   overridden) + warning `v125 root fix: custom_ops 'none' -> 'all'`.
   Unconditional for TP>1 boots; VLLM_XPU_ALLOW_UNSAFE_SPEC_TP_GRAPH=1
   bypass skips the whole gate (unchanged).
6. **FIXVAL (p22c1_fix2.sh, P22C1_FIX2_DONE 23:08:46):**
   - **Boot A PASS — the previously-DEGENERATE config now coherent:**
     serve_p22c1_nograph (cudagraph NONE, NO explicit ops) on the
     fixed venv: MT12 '391', LIST40 '1, 2, 3, 4, 5', MT8 '144',
     LIST60 full, THINK '57', GEN100 coherent 8.9 s; v60g=0;
     captures=0; pass-1 posture VLLM_COMPILE +
     `['+quant_fp8','all','+quant_fp8']`; v125-fix warning FIRED —
     the venv fix propagates through the natural boot path (no
     explicit ops needed anywhere).
   - **Boot B PASS — cert spec-4 + FULL 85-size graphs on fixed venv:**
     all 6 probes coherent (identical fingerprints), capture 64/64,
     spec_lines=2, v60g=0, fix warning fired (cert moved from
     accidental-'none' to 'all'), GEN100 3.7 s (spec+graphs ~2.4x
     faster than ns 8.9 s as expected). Structural check only —
     full battery is P23.
   - **Restore verified:** REVERT_CLEAN_OK; cert serve on clean venv
     (spec_lines=2, fix-marker-lines=0), health 150 s, watchdog
     re-armed 23:08:46, no pause flag.
   ⇒ the ns (spec-off) end-to-end surface is UNBLOCKED with the fix
   staged in /root/build; C4 legs + P23 battery run on the fixed
   posture (fix re-applied per chain), P24 bakes it into the wheel.

## Run 7j — C4 six-leg matrix: ns legs + cert GREEN, e4m3s4 BOOT DEATH
root-caused to TWO layers (wheel vintage + routing); P22C5 static spec
bridge designed+applied (2026-09-28 23:09 → 2026-09-29 00:21)

1. **C4 legs chain (p22c4_legs.sh, banked in p22c4_legs.log, lane
   trap-restored + re-armed 23:50:41):** six legs = 3 non-spec
   (fix16ns / e4m3ns / e5m2ns, cudagraph NONE, opsall-fix venv) +
   cert16s4 control + e4m3s4 / e5m2s4 (spec-4 fp8 pools). Banked:
   - fix16ns solo 15.02/15.06 tok/s; e4m3ns 14.89/14.97; e5m2ns 14.67
     — parity within noise, all math391=True list5=True v60g=0
     v125fix=1 (the opsall fix fires on every ns boot as designed).
   - cert16s4 solo 74.12/75.36 tok/s, capture 64/64, spec_lines=2 —
     certified posture reproduced on the patched venv.
   - **e4m3s4 ABORT at capture 0/64** — both TP workers died in the
     spec-verify dummy run: `RuntimeError: ssm_state dtype must be
     float32/float16/bfloat16, but got Float8_e4m3fn` raised by
     `torch.ops._xpu_C.gdn_attention` (compiled guard, serve log
     /root/serve_p22c4_e4m3s4.log). e5m2s4 never reached (chain
     order). Same failure class as the pre-P22B predictions.
   - agg 4x256 probe had a python bug (o[2] on one-element list →
     IndexError after solo#2) — corrected probe re-run planned in the
     reworked agg chain (all six legs incl. the two s4 legs).
2. **ROOT CAUSE — TWO LAYERS, both convicted with source evidence:**
   - **Layer 1 (routing, _xpu_ops.py):** `_native_ok` under
     VLLM_XPU_GDN_FP8_NATIVE=1 passed the fp8 pool STRAIGHT to the
     SYCL op for ANY `num_prefills == 0` batch — including spec-verify
     batches. Multi-request spec can never take the ESIMD spec op
     (`_gdn_conv_decode` gate: `num_spec_decodes != 1` → decline), so
     spec fp8 ALWAYS lands on the SYCL path → the wheel guard fires.
   - **Layer 2 (wheel vintage):** the installed wheel
     vllm_xpu_kernels-0.1.8.3.dev0+g3cab97a.d20260925 has NO fp8 state
     dispatch anywhere in the SYCL path — `strings _xpu_C.abi3.so |
     grep -c "fp8_e4m3fn/fp8_e5m2"` = 0 (only the OLD reject literal).
     The P22B patch (written 2026-09-28 15:50 into /root/build/vxk)
     was never built+installed: /root/build/vxk has NO wheel
     artifacts, /root/build/wheels holds only d20260830.
   - **Why ns fp8 legs were blind to it:** their decode (≤128
     tok/batch) runs 100% on the ESIMD fused kernels (P22A IS
     installed — 582 fp8-related strings in the lgrf .so) and
     prefill batches take the fp16 unique bridge; the SYCL fp8-native
     surface was NEVER exercised. Latent hazard: any fp8 decode-only
     SYCL call (decode batch >128) would crash identically on this
     wheel; all current tests stay ≤128 (bursts 108) so it stayed
     latent until spec fp8.
   - **Wheel routing map (gdn_attn_interface.cpp 320-500):**
     `spec_token > 0` → gdn::causal_conv1d + gdn::gated_delta_rule
     (plain SYCL spec kernel — state via `cache_indices[batch_id*
     stride + init_col]`, `init_col = num_accepted-1` clamped);
     non-spec with `num_prefills > 0` → xe2 chunk path (cutlass
     reorder machinery cannot host c10 fp8 StateT — stays bridged);
     else NATIVE_LAUNCHER decode. P22B's DISPATCH_STATE_DTYPE
     (gated_delta_rule.hpp:822, fp8 branches :833-838) wraps the
     KERNEL_LAUNCHER macro INCLUDING the is_spec branch → a P22B-built
     wheel gives fp8-native SPEC too (P24 lever).
3. **Metadata contract verified for the bridge design (gdn_attn.py):**
   num_decodes and num_spec_decodes mutually exclusive (assert :352);
   pure-spec batch → num_decodes==0; spec_state_indices_tensor =
   `block_table[spec_masks, :num_spec+1]` shape [n, W=5]; the padded
   cudagraph path refreshes a persistent buffer via `copy_` (:368) and
   NULL-block-fills the tail → recorded gather ops read FRESH slot ids
   on every replay (the foundation the static bridge builds on).
4. **FIX (P22C5, patch_v125_s4fp8_bridge.py — applied V125_C5_OK
   00:21, backup .pre_v125_c5, py_compile clean):** capture-safe
   STATIC fp8 bridge in `_gdn_attention_core_xpu_impl`:
   - `_native_ok` now requires `VLLM_XPU_GDN_FP8_NATIVE=2` (the
     explicit "P22B wheel installed" claim; =1 keeps ESIMD decode
     eligible via the unchanged `_gdn_fp8_native_enabled`, used by
     gdn_linear_attn's P22A gate — ns legs unaffected).
   - Under =1/unset, EVERY decode-only fp8 SYCL call (spec verify or
     pure decode >128 ESIMD-declined) takes the static bridge: gather
     the narrowed [n, W] flat rows to a fp16 copy
     (`index_select.reshape(-1).to(int64)` → `.to(float16)`), remap =
     CONSTANT positional arange `remap[r,c] = r*W + c` (the gather
     order DEFINES the remap — no value lookup), kernel on the copy
     with remapped spec/non-spec indices, scatter home via the uint8
     view. Every op fixed-shape → FULL decode graphs can replay it
     (torch.unique is FORBIDDEN inside capture: value-dependent output
     shape; prefill/mixed keep the certified v124 unique bridge —
     prefills are never captured).
   - Bridge semantics identical to the certified prefill bridge
     (fp8→fp16 exact widening, state math in fp16, quantize on
     scatter-back); kernel-untouched gathered rows scatter home as
     no-op byte writes.
   - Honest overhead estimate: n*W rows × ~768KB fp16 × ~36 GDN
     layers gather+scatter per spec step → solo (n=1, 5 rows) ~6% tax
     expected (~68-70 vs cert 74-75 tok/s); measured in fixval/agg;
     the P24 P22B wheel (=2, pool-straight native fp8 spec) removes
     the tax entirely — that is the path to fp8 ≥ fp16 EVERYWHERE.
5. **FIXVAL (p22c5_fixval.sh):** boots e4m3s4 + e5m2s4 on the patched
   venv (opsall + C5), gates + battery MT8/MATH/LIST/THINK + solo ×2
   per leg (speed RECORDED not gated — feeds the P24 decision), then
   teardown → 4-file clean-venv restore → cert serve → markers=0 →
   re-arm.
   - **Attempt-1 (00:21): THE FIX IS PROVEN AT THE CRASH SITE.** The
     e4m3s4 leg — which died at capture 0/64 before the fix — BOOTED
     to healthy on the patched venv: `Capturing CUDA graphs (decode,
     FULL): 100%|...| 64/64` + `Graph capturing finished in 38 secs,
     took 2.96 GiB`; C5 marker fired on BOTH TP workers DURING
     capture (`v125 P22C5 STATIC_FP8_BRIDGE ssm_dtype=torch.float8_e4m3fn`)
     — the static bridge's capture-safety held through all 64 graph
     sizes on both ranks; spec_lines=2, v125fix=1, P22A eligible.
   - The chain still ABORT_e4m3s4_CAPTURE — a GATE bug, not a serve
     bug: `grep -c 'Capturing'` counts LINES and tqdm carriage returns
     collapse the whole bar onto 3 lines (3 ≠ 64). The legs chain had
     this calibrated (cert leg banner count = 1). Gate re-anchored to
     `grep -c 'Graph capturing finished' ≥ 1` (the runner's own
     post-capture line; worker death is already gated by health).
     LESSON: never gate on progress-bar line counts — anchor gates on
     terminal status lines the runner prints once it is done.
   - Attempt-2 (00:29-00:38, DONE): **BOTH s4 fp8 legs boot, capture
     64/64, and serve end-to-end — the crash class is ELIMINATED.**
     | Leg | gates | battery | solo tok/s |
     |-----|-------|---------|------------|
     | e4m3s4 | capture_finished=1 spec=2 fix=1 C5=2 P22A=96 — ALL PASS | math391 T list5 T | **80.18 / 80.43** |
     | e5m2s4 | capture_finished=1 spec=2 fix=1 C5=2 — ALL PASS | math391 T list5 T | 71.96 / 72.25 |
     cert fp16 control (legs chain): 74.12/75.36. e4m3 spec-4 solo is
     **+7-8% over cert** — at n=1 the ESIMD single-request spec op runs
     fp8-NATIVE (P22A; `_gdn_conv_decode` accepts num_spec_decodes==1,
     no bridge) — first real fp8-native speed WIN. e5m2 ~parity (-3%).
     NOTE: solo numbers are the ESIMD path, not the bridge — the
     bridge's concurrency cost shows in the 4x256 agg numbers (pending).
   - Battery anomalies adjudicated:
     - `think=False` = PROBE BUG (both legs, deterministic). This API
       surfaces reasoning as `message.reasoning` (stream deltas; empty
       in non-streaming) NOT `reasoning_content`. Cert serve probed:
       content `'\n\n391'` with 53 completion tokens = thinking worked;
       message keys include `reasoning`. Battery check corrected in the
       diag chain (content + token-burn check).
     - `mt8=7/8` (both legs) = UNDER DIAGNOSIS. MT8 is the ONLY battery
       probe that exercises n>1 STATIC-BRIDGE batches (solo/math/list
       run at n=1 → ESIMD spec op, no bridge). Warm cert lane probes
       8/8 `'144'` @ 4 tok clean. Two hypotheses: (a) cold-first-batch
       warmup flake (battery MT8 ran first-after-capture; cert check
       ran warm), (b) real bridge defect at n>1 (e.g. duplicate-slot
       scatter loss). p22c5_diag.sh (launched 00:45): MT8 x4 rounds
       with per-stream dumps + 8-way identical x10 crosstalk + pair x20
       persistence + corrected think check. agg 4x256 numbers are ONLY
       valid after this resolves clean.
6. **Restore verified both attempts:** REVERT_CLEAN_OK (4 files), cert
   serve spec_lines=2 markers=0, watchdog re-armed 00:38:49.


## Run 7k — P22C5 diag CONVICTION: static bridge corrupts at n>1; n=1
clean runs never touched the bridge at all (2026-09-29 00:45-00:57)

1. **diag chain results (p22c5_diag.sh, DONE 00:49:40, lane restored +
   re-armed):**
   - mt8 rounds: r1 7/8 (BAD stream 3); r2/r3/r4 5/8 — **BAD streams
     4,5,6 STABLE across rounds**. Corruption is slot/order-stable within
     a probe, not random flake. Bad streams emit `'1!!!!!!!...'`
     (finish=length) — first token right, then explosion.
   - x8-identical x10: **0/10 clean rounds**; 5/8 degenerate EVERY round
     (`'9!!!...'` word salad), varying which slots.
   - pair20: 10/20 rounds both-correct; either stream can corrupt; r4
     both. n=2 corrupts too.
   - think corrected check: content `'391'`, finish=stop, 33 tokens → OK
     (confirms the fixval think=False adjudication: probe field bug).
     All n=1 probes clean — and NOT persistent (single-stream probes
     after corrupting rounds pass; the pool itself is fine).
2. **VERDICT: REAL bridge defect at n≥2** — deterministic-ish, affects
   a stable subset of streams per probe, pattern = recurrent-state
   corruption after a correct first token.
3. **KEY ROUTING INSIGHT (changes the evidence reading):** n=1 spec
   batches NEVER take the static bridge — `_gdn_conv_decode` (ESIMD)
   accepts `num_spec_decodes == 1`, so every "clean n=1" probe validated
   the ESIMD fp8-native path ONLY. The SYCL-kernel + gather/arange-remap
   contract has NEVER run clean at ANY n: the bridge's only exercised
   regime is exactly the corrupting regime. (Run 7j's "kernel semantics
   preserved" claim remains a design argument, not a measurement.)
4. **Kernel contract fully read (gated_delta_rule.hpp):** spec kernel
   init read `cache_indices[batch_id*stride + init_col]`,
   `init_col = num_accepted-1` clamped to [0, W-1] (both clamps are the
   #11 wedge fixes); per-token writeback
   `cache_indices[batch_id*stride + t_local]`, `t_local = t -
   query_start_loc[batch_id]` — all strides taken from the passed
   tensors, protocol fully row-local and consistent with gathered copy +
   arange remap. Pure analysis cannot split the remaining hypotheses.
5. **diag2 discriminators (p22c5_diag2.sh, launched 00:57, eager health
   OK 120s):** ONE eager boot (--enforce-eager, everything else identical
   to e4m3s4) with paired probe families —
   - eager vs capture: clean eager ⇒ capture-replay interaction (prime
     capture-side suspect: the narrowed persistent spec_state_indices
     buffer under FULL replay); corrupt eager ⇒ bridge/kernel contract
     itself.
   - identical vs DISTINCT prompts: every corrupting probe so far used
     IDENTICAL prompts (prefix-cache slot-sharing territory);
     x8-distinct/pair-distinct falsify or confirm the sharing mechanism.
   - C5DUPD once-print (patch_v125_c5_dupdiag.py) at the first n>1
     gather: n, W, dup count, slot ids — capture-guarded, fires
     eager-only.
6. **Strategic: P22B wheel build running in parallel**
   (build_vxk_v125_buildonly.sh, launched ~01:00 in the vxk-inc builder;
   BUILD-ONLY — the lsv-test swap is deliberately deferred so a
   non-certified wheel never sits under the certified serve mid-chain).
   The wheel path (=2 pool-straight fp8 spec) makes the kernel own state
   updates on the real pool with real cache_indices — byte-identical
   protocol to certified fp16, only the StateT dtype differs — the
   highest-confidence fix for BOTH correctness and the bridge tax.
   p22c4_agg.sh stays HELD until the s4 path is correctness-clean.


## Run 7l — diag2 EAGER verdict: bridge contract broken (not capture);
## dup=0 kills duplicate-slot theory; wheel BUILT; =2 validation launched
(2026-09-29 01:00-01:17)

1. **P22B wheel BUILDONLY DONE 01:06:55** (~7 min warm incremental,
   ninja -j 52): .so at /root/build/vxk/build/temp/_xpu_C.abi3.so
   md5 0fb4e5b8dedd438a629d04f034cb6871, strings proof fp8_e4m3fn=1
   fp8_e5m2=1 — the patched dispatch is in the binary. Swap deferred
   (certified-serve hygiene).
2. **diag2 attempt-1 ABORT was a gate miscalibration (my error):** the
   v125fix/C5 markers print at FIRST INFERENCE CALL — under captured
   legs they fired during capture (dummy batch = inference); eager has
   no capture, so they cannot exist 15s after health. Gates reordered:
   boot gates = config only (spec_lines, fp8_e4m3 cfg); marker gates
   AFTER probes. LESSON: marker gates must be scheduled where the
   workload that triggers them has actually run.
3. **diag2 attempt-2 DONE 01:12:21 (lane restored, re-armed).** EAGER
   boot (--enforce-eager, otherwise identical to e4m3s4) health 120s;
   C5 fired (C5_static=2) — the static bridge ran eagerly.
   - n1-math '391' OK; **mt8-identical r1+r2 both 8/8 CLEAN in eager**
     (vs 5-7/8 captured).
   - x8-identical 5/6 rounds clean (only r6 6/8).
   - **x8-DISTINCT 0/6 rounds** — every round exactly 3/8 corrupt, BAD
     positions stable: stream 6 bad 6/6 rounds, stream 0 4/6, stream 5
     4/6, stream 4 2/6, stream 1 1/6; **streams 2,3 NEVER bad**.
   - pair-identical 3/10, pair-distinct 10/10.
   - **Every corrupt stream emits the CORRECT first digit then explodes**
     ('76'→'7!!!...', '184'→'1!!!...'): the first verify step is right,
     the recurrent state is corrupted for subsequent steps.
   - C5DUPD (first n>1 gather): **n=6 W=5 rows=30 uniq=30 dup=0**,
     slots = consecutive per-request groups [129-133],[113-117],[49-53],
     [65-69],... identical on both TP ranks.
4. **VERDICT:** (a) eager corrupts ⇒ **NOT a capture interaction — the
   bridge/kernel contract itself is defective at n>1** (capture merely
   amplifies: x8id 0/10 captured vs 5/6 eager). (b) dup=0 ⇒ duplicate-
   slot scatter loss DEAD. (c) identical-prompt runs mostly clean while
   distinct-prompt runs corrupt with stable positional subsets ⇒ the
   defect behaves like CROSS-REQUEST STATE CROSSTALK that identical
   states mask (crossed states are equal bytes); stream-position-
   dependent (2,3 never; 6 always in x8di). (d) v125fix=0 in eager is
   expected (the opsall warning is compile/capture-path; informational).
5. **=2 wheel validation launched 01:17:14 (p22c5_n2val.sh):** wheel
   swapped into lsv-test (.v125bak backup; WHEEL_LOAD_OK; strings=1),
   =2 serve scripts generated from the s4 leg scripts, both legs gate on
   capture_finished + spec_lines=2 + FP8_NATIVE ENGAGED + C5-static
   ABSENT (bridge must not engage under =2), then the SAME probe
   families that convicted the bridge. Chain ends with wheel REVERT
   (strings back to 0 = old wheel proof) + 4-file venv revert + cert
   serve + markers=0 + re-arm. Decision fork: =2 clean ⇒ ship =2 as the
   spec-fp8 posture, bridge demoted to documented-defective fallback;
   =2 corrupts ⇒ the fp8 StateT spec kernel path itself is defective
   (investigation goes kernel-side, P22B branches under review).


## Run 7m — N2VAL ALL-CLEAN BOTH FORMATS: =2 + P22B wheel is THE
## spec-fp8 path; the static bridge is the convicted defect
(2026-09-29 01:17-01:27)

1. **n2e4m3s4 ALL_CLEAN:** ENGAGED fired on BOTH TP workers during
   capture (`v125 P22B FP8_NATIVE ENGAGED ssm_dtype=torch.float8_e4m3fn
   ... asserted via =2`), capture 64/64 in 34s, C5 static marker ABSENT
   (bridge correctly not engaged). Probes: n1-math OK, mt8id 8/8+8/8,
   x8id 6/6, **x8di 6/6** (bridge: 0/6), pid 10/10, pdi 10/10, solo
   69.77/85.88 tok/s (n=1 = ESIMD path; wide warmup variance).
2. **n2e5m2s4 ALL_CLEAN:** same gates pass, mt8id 8/8, x8id 6/6, x8di
   6/6, pid 10/10, pdi 10/10, solo 66.84/75.60.
3. **CONVICTION FINAL:** the defect was the static bridge's
   gather/copy/remap/scatter machinery — NOT the SYCL spec kernel, NOT
   capture, NOT fp8 state semantics, NOT the metadata contract. The
   kernel with REAL pool + REAL cache_indices + fp8 StateT (pool-straight
   via the P22B wheel under =2) is correctness-clean at every n and both
   formats. The bridge's exact defect mechanism remains
   design-analysis-opaque (contract reads correct arg-by-arg; dup=0;
   row-local kernel writes) — it is RETIRED, not fixed: production never
   ships a correctness-defective path when a clean one exists.
4. **Posture decision (feeds P24/P25):**
   - SPEC fp8 (either format) ⇒ VLLM_XPU_GDN_FP8_NATIVE=2 + P22B wheel
     (pool-straight native). VALIDATED.
   - ns fp8 (either format) ⇒ =1 (ESIMD decode + unique-bridge
     prefill). VALIDATED (C4 legs).
   - The static bridge (=1, decode-only, no ESIMD) must NEVER engage
     silently: P22C6 patch converts any would-be static-bridge
     engagement into a loud RuntimeError pointing at =2 — never silent
     corruption (world-class quality law). Applies before bake.
   - Wheel revert verified by binary proof (fp8_strings=0) + venv
     4-file revert + cert serve markers=0 + re-arm 01:27:01.
5. **Next:** p22c5_agg2.sh — the agg 4×256 matrix on the validated
   posture (wheel swapped in; s4 legs under =2 with ENGAGED gate; ns
   legs + cert control unchanged) + solo ×2 per leg for the uniform
   speed table; then P22C6; then C3 microbench window, P23 full battery
   (host reboot first), P25 bake v1.2.25.

## Run 7n — AGG2 matrix complete: uniform speed/capacity table on the
## validated posture; P22B wheel does NOT regress fp16; fp8 speed
## verdict is BELOW fp16 in every family → P24 default-fp16 posture
(2026-09-29 01:37-02:00)

Chain p22c5_agg2.sh (md5 51b4c4fc…): all six legs booted clean on the
UNIFORM posture (clean venv → opsall + C5 → P22B wheel swapped in for
ALL legs incl. the fp16 control → per-leg serve). Gates per leg:
spec_lines, v125fix ≥1; s4 fp8 legs: ENGAGED ≥1 AND C5 static == 0;
cert control: ENGAGED == 0 AND C5 == 0. All PASS. Teardown: wheel
revert (fp8_strings=0 binary proof), 4-file venv revert, cert serve
spec_lines=2 markers=0, watchdog re-armed 02:00:03.

1. **Speed matrix (agg 4×256 ×2, solo ×2, tok/s):**

   | leg | agg#1 | agg#2 | solo | solo vs fp16 |
   |---|---|---|---|---|
   | cert16s4 (fp16 s4) | 181.81 | 244.88 | 75.06 / 74.89 | — |
   | n2e4m3s4 (=2) | 163.80 | 228.71 | 68.39 / 68.44 | **−8.9%** |
   | n2e5m2s4 (=2) | 157.96 | 210.85 | 65.73 / 65.06 | **−12.5%** |
   | fix16ns (fp16 ns) | 38.12 | 43.72 | 15.12 / 15.11 | — |
   | e4m3ns (=1) | 37.13 | 42.51 | 14.71 / 14.51 | −2.7 / −4.0% |
   | e5m2ns (=1) | 38.06 | 43.17 | 14.90 / 14.75 | −1.5 / −2.4% |

2. **CRITICAL CONTROL RESULT:** cert16s4 solo 75.06/74.89 on the P22B
   wheel == banked cert 74.12/75.36 on the production wheel → the
   wheel's fp16 path is byte-equivalent in performance; shipping it
   imposes NO fp16 degradation (the patch-stack "no degradation" gate).
   Today's fp8 solos are TIGHT (68.39/68.44; 65.73/65.06) — the legs
   round's 85.88 outlier was warmup variance, not headroom.
3. **Capacity matrix (KV tokens @ 262144 ctx, from boot logs):**
   s4: cert 476,451 → fp8 493,127 (+3.50%); ns: fp16 554,304 → fp8
   560,725 (+1.16%, both formats identical — the freed SSM bytes are
   dtype-symmetric). Concurrency @ max ctx: s4 1.82x → 1.88x; ns 2.11x
   → 2.14x.
4. **SPEED VERDICT (feeds P24):** fp8 is BELOW fp16 in EVERY family
   (solo −2.4% … −12.5%; agg −2.6% … −13.9%) — the "≥ fp16 on speed in
   every measured case" gate is NOT cleared by either format. The
   e2e+correctness gate IS cleared (=2 ALL_CLEAN + this matrix's clean
   agg runs, all streams ok 4/4). Unless P23/P24 surface contrary
   fresh-host evidence, the ship decision routes to: **both formats
   SELECTABLE + CERTIFIED (selectable dtype, tested posture, =2 for
   spec / =1 for ns), default serve dtype stays fp16** — per the
   decision rule "if a format can't clear it, default stays fp16".
   The capacity gain (+1.2-3.5% KV) does not offset a consistent
   −3…−12% speed loss for the default posture.
5. **e5m2 87.45-agg slow-stream re-check: NOT reproduced** — all four
   streams uniform (4.7-4.9 s each in agg#2; 6.4/6.5/5.6/6.4 s in
   agg#1). The legs-round observation was a one-off (transient), not a
   format property.
6. **Artifacts staged this window:** patch_v125_c6_staticraise.py
   (retirement guard; raises on any would-be static-bridge engagement,
   message names =2+wheel posture; applied+validated in P23),
   p22c3_run.sh (C3 microbench wrapper, serve-down window),
   p23_battery_v125.sh (P23 full battery: C6 posture checks + 6-leg
   battery incl. cert-with-opsall full validation; lsv-test docker-start
   guard for the post-reboot stopped container).
7. **Next (in order):** C3 ESIMD op bench (launched in the post-agg2
   serve-down window) → host reboot 10.20.3.65 (standing directive) →
   `docker start lsv-test` → p23_battery_v125.sh → P24 decision record
   → P25 bake v1.2.25.

## Run 7o — C3 op microbench: fp8 ESIMD op-level verdict is
## format-asymmetric — e5m2 clean everywhere, e4m3 has bounded
## regressions; the s4 tax lives in the =2 SYCL path
(2026-09-29 02:02-02:06)

p22c3_run.sh in the post-agg2 serve-down window (teardown → bench on
idle XPU → cert restore health OK, markers=0, re-arm 02:06:28).
esimd_gdn_conv_fused_seq + _seq_spec, pool dtype the single variable,
n sweep 1..64, PASS bound ≤1.10× fp16. P22C3_FAILS=5 (all e4m3).

1. **e5m2: ALL CELLS OK both ops, every n** — seq ratios 0.611-0.995;
   spec ratios 0.724-1.032 (spec n≥8 WINS 15-28%: 0.724/0.802/0.851/
   0.854). Op-level, e5m2 is a strictly-safe fp8 surface.
2. **e4m3: 5 REGRESS cells** — seq n=16 (1.367) + n=32 (1.134) (fp16
   seq hits 681 GB/s at n=16 vs e4m3 261 GB/s — a fp16 vectorization
   window e4m3 lacks); spec n=1 (1.176), n=2 (1.389), n=4 (1.389),
   converging to ≤1.099 by n≥8.
3. **Consistency with serve numbers:** ns serve legs (seq op IS the
   decode op): e5m2 −1.5/−2.4% vs e4m3 −2.7/−4.0% — matches e5m2's
   cleaner op profile. The s4 serve deficit (−8.9% e4m3 / −12.5% e5m2)
   lives in the =2 POOL-STRAIGHT SYCL path (P22B wheel fp8 StateT
   branches — correctness-clean, perf untuned), which C3 does not
   measure; ESIMD spec cells here are the =1 n=1 surface.
4. **Feeds P24:** op-level evidence closes the "is parity luck?"
   question — no: fp8 ESIMD op cost is real, format-dependent, and
   bounded; SYCL =2 tax is the dominant s4 cost. Default-fp16 posture
   stands unless P23 contradicts.
5. **Next:** host reboot → docker start lsv-test → p23_battery_v125.sh
   (C6 posture checks + 6-leg full battery).

## Run 7p — P23 fresh-host battery (partial) + P23B parser diag:
## raise-leg PASS, cert16s4 control fully green; n2e4m3s4 green through
## bursts then parser T1/T2 fail — which does NOT reproduce on fresh
## boot of the same posture (context-dependent)
(2026-09-29 02:22-04:17)

Host rebooted (standing directive), lsv-test restarted, p23_battery_v125.sh
launched 02:22:28 on the fresh host (dmesg resets=0).

1. **Raise-leg PASS (first live C6 exercise):** e4m3s4 under =1 →
   health_up=0, c6_message=6 — the retired bridge raises loudly across
   engine retries; no silent corruption path remains.
2. **cert16s4 LEG PASS 03:30:47** — fp16 control on the full stack
   (opsall + C5 + C6 venv + P22B wheel): math OK, x8 8/8, drill 3/3,
   serialized 24×3 all 24/24, bursts SURVIVED, parser T1/T2/T3 PASS,
   V66 fairness PASS, tracebacks=0, resets_delta=0. The wheel+patch
   stack imposes NO fp16 degradation end-to-end.
3. **n2e4m3s4 leg green through the heavy prefix:** engaged=2, c5=0,
   c6=0, math OK, x8 8/8, drill 2/2 SUSTAIN_COMPLETE_NO_WEDGE,
   serialized 2×24/24, bursts 2× SURVIVED (36/36, resets=0) — then
   **PARSER T1/T2 FAIL** (name=None args=None; questions NoneType;
   T3 plain-reply PASS). Chain aborted; trap restored cert serve,
   re-armed 04:03:37.
4. **P23B diag (04:07-04:16) — the failure is CONTEXT-DEPENDENT:**
   re-booted the SAME =2 e4m3 posture fresh, re-sent the EXACT failing
   bodies → T1 stock PASS (comp_tok=74, perfect run_command call),
   T2 stock PASS (comp_tok=112, perfect ask_user questions array),
   T3 PASS; long/nothink variants identical; e5m2s4 (=2) ALL CLEAN
   (T1 74 tok, T2 112 tok — token-for-token the same outputs as the
   fp16-cert lane). So: fresh-boot tool-call quality on BOTH fp8
   formats equals fp16; the P23 failure appeared only after the
   sustain+serialized+burst prefix on one instance.
5. **Discrimination design (P23C):** remaining 5 legs re-run with
   cc_parser_battery_v125b.py = v89 assertions + full raw dump on fail
   + one 20 s settle-retry (verdict FAIL only on double-fail) + serve
   -log structured-output grep on abort. If the failure reproduces
   post-prefix → accumulated-state defect (ship-gate blocker); if it
   passes or RETRIED-PASSes → transient signature (one-off class,
   like the e5m2 87.45 slow-stream).
6. **Artifacts:** p23b_diag.py + p23b_parser_diag.sh (executed clean,
   teardown verified strings=0 markers=0), cc_parser_battery_v125b.py,
   p23c_battery_resume.sh (launched 04:2x).

## Run 7q — P23C resume battery: e4m3 re-run LEG PASS (parser PASS
## first-try, same context as the P23 failure) — then e5m2 caught
## RED-HANDED by the instrumented probe: reasoning-channel '!'
## salad + grammar_matcher desync — the P23 mystery is a real,
## reproducible-in-context fp8-spec defect class
(2026-09-29 04:20-05:24)

1. **n2e4m3s4 re-run LEG PASS 04:52:49** — identical context to the
   P23 failure (sustain 2/2, serialized 2×24/24, bursts 2× SURVIVED):
   parser T1/T2/T3 PASS FIRST-TRY, no retry; tracebacks=0,
   resets_delta=0; short-gen solo 76.0-79.9 tok/s, agg4 155.3.
   dmesg at chain start =0 → the P23 parser failure was NOT an engine
   reset artifact.
2. **n2e5m2s4 leg — the instrumented probe caught the defect live:**
   - T1 attempt1: finish=length, comp_tok=300, reasoning =
     'The' + 297×'!' — degenerate-repetition SALAD, no tool call
   - T1 attempt2 (after 20 s settle): IDENTICAL salad (deterministic)
   - T2 attempt1: same salad at 400 tokens
   - T2 attempt2: RETRIED-PASS — clean ask_user call, same instance
   - T3 plain-reply: PASS
   - BATTERY FAIL T1 (double-fail) → LEG_E5M2S4_FAILED, trap
     restored cert serve, re-armed 05:24:01
3. **Serve-log correlation:** `grammar_matcher.cc:1221 Warning: The
   matcher has terminated after accepting the stop token, but is
   trying to accept new token with id 198` — one per failing request
   (04:59:33→T1a, 05:00:35→T2a). CAVEAT: the e4m3 PASS leg ALSO has
   2 matcher warnings → the warning is context noise, the SALAD is
   the discriminator.
4. **Mechanistic picture:** the P23 e4m3 'name=None' failure is now
   explained — same salad consumed the 300-token budget in the
   reasoning channel before any tool call; v89 printed only
   name=None because it never showed reasoning. Failure signature:
   thinking-phase degenerate attractor ('!' forever, temp 0,
   deterministic within instance), grammar/xgrammar matcher desync
   alongside, plain replies unaffected (T3 always passes), recovery
   possible without reboot (T2b). Distribution so far: 2 failures in
   4 post-prefix parser runs across BOTH =2 formats; 0 in any
   fresh-boot probe; 0 ever on fp16.
5. **P23D launched (salad-rate A/B):** 3 arms (cert16s4 fp16 control /
   n2e4m3s4 / n2e5m2s4), identical prefix (1 sustain + 1 burst) then
   30 alternating T1/T2 requests with raw capture; per-arm salad
   count + grammar_matcher deltas. Turns anecdotes into rates and
   tests whether the fp16 control is truly immune. Remaining ns legs
   deferred until after P23D (their parser verdicts need this
   context).
6. **Artifacts:** p23c_battery.log + _p23c_n2e4m3s4.txt (PASS),
   _p23c_n2e5m2s4.txt (SALAD raw dumps), p23d_salad_ab.sh,
   p23d_probe.py.

## Run 7r — P23D salad A/B: the =2 spec-fp8 postures CONVICTED —
## 20% post-prefix tool-call salad on BOTH formats, fp16 control 0/30;
## poisoned-state windows (~3 consecutive requests) that heal; grammar
## matcher decoupled
(2026-09-29 05:32-06:14)

Standardized design: per arm, boot + posture gates → prefix (1 sustain
round + 1 burst) → 30 alternating T1/T2 tool-call requests (exact
battery bodies, thinking ON, temp 0) with raw capture + per-arm
grammar_matcher deltas. All arms on the SAME venv (opsall+C5+C6) and
P22B wheel — the only delta is pool dtype/env.

1. **cert16s4 (fp16 control): 30/30 CLEAN, 0 salad, 0 matcher
   warnings, 0 tracebacks** (38 s for 30 requests).
2. **n2e4m3s4 (=2): 24 clean / 6 SALAD (20%)** — indices 02,04,05 +
   16,17,18; every salad = finish=length, 'The'+'!'×N in reasoning,
   no tool call.
3. **n2e5m2s4 (=2): 24 clean / 6 SALAD (20%)** — indices 09,10,11 +
   18,19,20 (consecutive triplets). Same signature, same rate,
   different windows.
4. **Interpretation:** the trigger is a POISONED-STATE WINDOW — bursts
   of ~3 consecutive affected requests that then heal without reboot
   (matches P23C's T2b RETRIED-PASS after 20 s). Rate identical across
   formats → format-independent mechanism in the shared =2
   pool-straight fp8 path (SYCL kernel fp8 state handling), triggered
   by post-prefix batch/scheduler state; thinking-phase degenerate
   attractor at temp 0. grammar_matcher delta 0 during the 6-salad
   probes → the matcher warnings in P23C were CORRELATED NOISE, not
   the cause; plain replies (T3) never affected.
5. **Attribution airtight:** fp16 arm ran the identical patched venv +
   wheel (only pool dtype differs) → clean. Third instance of the
   GDN-state-slot-corruption smell: v88 int32 slot bug (fixed) →
   static-bridge cross-request crosstalk (Run 7k/7l, retired) →
   pool-straight fp8 salad (this run). Root-cause = kernel-round
   scope (fp8 DISPATCH_STATE_DTYPE branches in _xpu_C.abi3.so);
   deterministic-within-arm repro recorded (indices above) as the
   handoff.
6. **Ship-gate impact:** spec-fp8 (=2) now DOUBLE-disqualified —
   speed (Run 7n: −8.9%/−12.5%) AND quality (20% post-prefix salad).
   Neither s4 fp8 format can be certified even as selectable until
   the kernel fix. ns-fp8 (=1) fate = P23E (running).
7. **P23E launched:** e4m3ns / e5m2ns / fix16ns (fp16 ns control)
   full battery + v125b parser + the 30-request salad probe appended.
8. **Artifacts:** p23d_salad.log, _p23d_{cert16s4,n2e4m3s4,n2e5m2s4}_
   {probe,sustain,burst}.txt, p23d_salad_ab.sh, p23d_probe.py.


## Run 7s — P23E aborts at its FIRST gate: e4m3ns (=1 ESIMD fp8)
## returns a WRONG concurrent-math answer (x8d 7/8) on a FRESH boot,
## no prefix — first direct ns-fp8 quality failure; contradicts the
## clean Run-7m n2val evidence (intermittent); P23F rate study launched
(2026-09-29 06:15-06:20)

1. **Abort:** `p23e_ns_legs.sh` (launched 06:15:10) died at leg 1,
   minutes in: posture gates passed (spec_lines=0, v125fix=1,
   engaged=0 — correct for =1, the ENGAGED marker is =2-only; c5=0,
   c6_raise=0), `n1-math OK` (123+268=391), then **`x8d 7/8 -> BAD`** —
   one of 8 concurrent distinct-multiplication streams (enable_thinking
   False, mt 32, temp 0) returned a product other than the exact value.
   Wrong ARITHMETIC, not formatting — a harder failure class than the
   =2 reasoning-channel salad. Chain aborted by design (ABORT_e4m3ns_
   MATH); trap restored certified lane (wheel reverted, venv clean,
   watchdog re-armed 06:20:10); e5m2ns + fix16ns legs never ran.
2. **Why this is significant:** the trigger differs from =2 — fired on
   a FRESH boot with NO prefix (the =2 salad needs post-prefix batch
   state). And the ns path had previously passed every probe: Run 7m
   n2val clean, agg2 ns legs green through the same math gate hours
   earlier. So ns-fp8 corruption is INTERMITTENT and the 8-request
   x8d is a 1-sample exposure — the true rate is unmeasured. Suspect
   #1: the v124 unique-bridge PREFILL under 8-way concurrency (8
   prefills in flight when =2's decode-phase pool path is not engaged).
3. **Instrumentation gap, again:** math_probe printed only the count —
   no dump of WHICH question failed or WHAT it answered (same gap that
   hid the =2 salad until v125b). Fixed for P23F: p23f_probe.py prints
   every wrong answer verbatim + a serial re-ask of all 80 questions to
   split concurrency-crosstalk (concurrent-wrong/serial-right) from
   numerics (wrong both ways) + concurrent-vs-serial FLIP detection.
4. **P23F launched (06:33:20):** rate study, same stack as P23E
   (opsall+C5+C6+P22B wheel). Arms fix16ns (fp16 ns control) → e4m3ns
   → e5m2ns; per arm PHASE A fresh-boot (10 concurrent rounds = 80 reqs
   + 80-question serial control + 30 tool calls) → prefix (the P23D
   prefix: 1 sustain + 1 burst) → PHASE B (3 rounds + 30 tool calls);
   grammar_matcher deltas + tracebacks per phase. Quality failures DO
   NOT abort — measurement only; only infra failures abort.
5. **Artifacts:** p23e_battery.log (abort record), p23f_quality.log,
   _p23f_{arm}_{A,B}.txt, p23f_ns_quality.sh, p23f_probe.py.


## Run 7t — P23F ns quality matrix COMPLETE: BOTH ns-fp8 formats
## CONVICTED — 8.75-13.75% fresh-boot WRONG-ANSWER rate exploding to
## 25-42% post-prefix; ALL serial re-asks correct (pure concurrency
## crosstalk); fp8-KV + fp16 control 0/104; every fp8 GDN-state path
## on BOTH kernel routes (=1 ESIMD+bridge AND =2 SYCL pool) corrupts
(2026-09-29 06:33-08:11)

| arm (posture)            | fresh math | serial | post-prefix math | fresh tools | post-prefix tools |
|--------------------------|-----------|--------|------------------|-------------|-------------------|
| fix16ns (fp16 + fp8 KV)  | 0/80      | 0/80   | 0/24             | 30/30       | 30/30             |
| e4m3ns  (=1 ESIMD)       | 7/80      | 0/80   | **10/24**        | 30/30       | 30/30             |
| e5m2ns  (=1 ESIMD)       | 11/80     | 0/80   | 6/24             | 30/30       | 27+2S+1O          |

1. **Wrong-answer taxonomy (identical across formats AND across =1/=2
   routes):** leading digits correct then degenerate truncation
   (`84→'8!!!…!'`, `207→'20'`, `126→'1'`), digit duplication
   (`112→'1112'`, `66→'666'`, `114→'1114'`), attractor runs in
   long-form (`999…`, `!!!…`), and one `want=135 got='13\n\n135'` —
   wrong prefix emitted then FULL MID-GENERATION RECOVERY (state
   partially poisoned, heals during decode). Math uses
   enable_thinking False mt 32 temp 0 — corruption lands in the
   content channel here vs the reasoning channel in P23D (=2): the
   channel is wherever the poisoned state drives generation.
2. **Pure concurrency crosstalk, NOT numerics:** every one of the 18
   wrong concurrent answers was CORRECT on serial re-ask (18/18 flips
   concurrent-wrong→serial-right; serial wrong 0/160 across arms).
   Victim positions cluster at the HEAD of each 8-request burst
   (req0/req1, spreading to req2/req4/req5 post-prefix as the rate
   rises) — consistent with per-request GDN/SSM state-slot mapping
   corruption when prefills coalesce into one batch. Same root family
   as v88 int32 slot bug and the static-bridge crosstalk (Run 7k/7l).
3. **Rate escalates with batch-state:** e4m3 8.75% fresh → 41.7%
   post-prefix; e5m2 13.75% → 25%. Post-prefix e5m2 also produced the
   first ns-path TOOL salad (2 SALAD + 1 OTHER /30 — `Theo999…`
   attractor) — the =2 P23D signature reproduced on =1.
4. **Exonerated:** fp8 KV cache (fix16ns runs --kv-cache-dtype
   fp8_e4m3: 0/104 wrong, 60/60 tools clean — KV fp8 is quality-safe);
   fp16 GDN pool (control arm clean through the heaviest probe);
   grammar_matcher warnings (delta 0 through every phase incl. all
   corruption events); engine health (tracebacks 0, resets_delta 0 on
   all arms — corruption is SILENT).
5. **Why earlier rounds missed it:** Run 7m n2val and agg2 legs passed
   the same x8d gate — a single 8-request exposure at these rates
   (8.75-13.75%) fails with p≈0.6-0.7 per burst; the 80-request phase
   A + 24-request phase B + serial control is what converts the
   anecdote into a rate. (P23E's abort was the lucky catch.)
6. **P23F verdict: every fp8 GDN/SSM-state path is quality-BROKEN
   under concurrency on both formats and both kernel routes.** With
   Run 7n (speed below fp16 everywhere) fp8-state is
   double-disqualified on BOTH axes; the only surviving fp8 surface is
   the KV cache (clean, +1.16-3.5% capacity, already shippable via
   --kv-cache-dtype which the certified lane ALREADY uses — fp8 KV
   has been in production since v1.2.24).
7. **Artifacts:** p23f_quality.log (full matrix + teardown proof),
   _p23f_{fix16ns,e4m3ns,e5m2ns}_{A,B}.txt (per-request wrong answers
   + flips), _p23f_*_sustain/_burst.txt. Restore verified: wheel
   fp8_strings=0, spec_lines=2, markers=0, watchdog re-armed 08:10:51,
   lane health 200.


## Run 7u — P24 ROUTING DECISION (by measurement, pre-registered gate)

1. **Default serve posture = fp16 GDN pool + spec MTP×4 + fp8 KV
   (e4m3) — the certified v1.2.24 lane.** The only posture that is
   both faster (Run 7n) and quality-clean (P23D 0/30, P23F fix16ns
   0/104+60/60) on this silicon. fp8 KV remains the shipped capacity
   lever (+KV capacity at zero measured quality cost).
2. **VLLM_XPU_GDN_FP8_NATIVE =2 and =1 postures: NOT certifiable —
   documented known-broken, not exposed as options.** Quality:
   20% post-prefix tool salad (=2, P23D), 8.75-41.7% concurrent
   wrong answers (=1, P23F), both formats, both routes. Speed: below
   fp16 solo AND aggregate (Run 7n). "Superior on any case,
   degradation not allowed" fails on both axes — a 9-42% wrong-answer
   rate is not a warning-label selectable, it is a defect. The env
   knobs remain physically present for the kernel round but are
   documented as BROKEN in the image README + KNOWN_ISSUES.
3. **Root-cause handoff = kernel round (vllm_xpu_kernels):** fp8
   DISPATCH_STATE_DTYPE branches in _xpu_C.abi3.so — per-request
   GDN/SSM state-slot handling under concurrent prefill batches.
   Repro kit recorded: P23F probe (10-round concurrent + serial
   control + flip detection), deterministic-within-arm indices
   (P23D), rate table (P23F). The P22B wheel at
   /root/build/vxk/build/temp/_xpu_C.abi3.so is the artifact under
   test; do NOT bake it.
4. **Ship gate for any future fp8-state claim:** P23F probe phase A
   0/80 wrong + 0 flips + P23D 30-probe 0 salad post-prefix + full
   battery + Run-7n speed parity, on BOTH formats, before any
   certification talk.
5. **P25 bake v1.2.25 = v1.2.24 lineage + opsall fix + C5 (inert
   without env; retained for kernel-round re-validation) + C6 raise
   (protective) on the STANDING v1.2.24 wheel (no wheel swap).**
   Spec MTP×4 + XGrammar-2 0.2.7 gate-verified on every boot
   (standing requirement).


### Run 7v — P25 bake: C7 rewrite after raw-image discovery; two gate-caught launch failures; v1.2.25-raw bake running (2026-09-29)

1. **Run 7u point 5 REVISED (bake set): opsall fix + C7 ONLY.** No
   C5 — its env routing is unreachable dead code once C7 refuses the
   env. No C6 — its patch anchor requires the C5-patched file, and it
   is redundant once C7 raises at import. No P22B wheel — stays a
   dev-lane kernel-round artifact (asserted by strings fp8_e4m3fn==0
   gate). C7 = import-time RuntimeError in `_xpu_ops.py` when
   `VLLM_XPU_GDN_FP8_NATIVE` in {"1","2"}, message cites the
   P23D/P23F verdicts + kernel-round scope.
2. **DISCOVERY — the P22B switches are NOT on the raw base.** Verified
   directly on v1.2.24-raw (0423c13f8c21): `_xpu_ops.py` (811 lines)
   contains ZERO occurrences of `VLLM_XPU_GDN_FP8_NATIVE`,
   `_GDN_FP8_NATIVE_MODE`, or module-level `import os`. The P22B-lineage
   env handling verified earlier lives ONLY on the v1.2.24 PRODUCTION
   lane-commit image (b13f5654) — the lane venv was P22-patched before
   that commit, so =1 engages silently THERE but is INERT on
   raw-lineage boots. v1225 boots from raw: C7 upgrades the env from
   inert→REFUSED so the knob can never re-engage silently (kernel
   round or accidental re-application) and the verdict travels
   in-code. First C7 draft anchored on the `_GDN_FP8_NATIVE_MODE` line
   and self-aborted (`no module-level 'import os'`) — the abort was
   the discovery trigger; rewritten self-contained (own aliased
   `import os`, appended at module EOF, no anchor, refuses if the env
   name is already referenced — lineage guard).
3. **Two bake launches failed on C7 details; both caught by the bake's
   own gates (gates did their job).** Launch 1: patch self-abort (no
   anchor in raw) → c7_patch_exit/marker/refuse FAIL. Launch 2: guard
   raised but with `NameError: ENV` — the %-format argument referenced
   the PATCH SCRIPT's constant, not a name in the generated guard, so
   the C7 message never printed (refuse_rc gates passed, message gates
   failed). Fix = bind the env value once (`_v125_c7_mode`). Also:
   `pkill -f stage5_bake_v1225.sh` self-matched the plink command line
   and dropped the session before cleanup (exit 128) — use the
   bracket trick (`stage5_bak[e]_v1225`) for self-referencing pkills.
4. **Proof-before-launch pattern added:** before relaunch 3, the full
   patch cycle ran in a throwaway container (lsv-c7t from v1.2.24-raw):
   apply→V125_C7_OK, reapply→V125_C7_ALREADY, --check→APPLIED, =2 →
   RuntimeError with full C7 message, =1 → rc=1 + message, unset →
   IMPORT_OK. All green, then relaunch.
5. **Launch 3 CLEAN through warm-serve gates:** 31 GATE-OK / 0 GATE-FAIL
   at warm round 0 (c7_base_clean=0 proving raw is switch-free,
   c7_marker/refuse_2/refuse_1/unset all OK, opsall_marker+syntax OK,
   serve-config posture inherited OK, barrier default `False False 0`,
   bake_opsall_fired=1 on the warm serve). New v125 gates in
   stage5_bake_v1225.sh: wheel_no_fp8_strings==0, no_c5/c6_marker==0,
   no_c5_in_gmr==0, no_v125_serve_variants==0, c7_base_clean==0,
   functional refusal trio. Bake = v1.2.24-raw + opsall + C7 → commit
   v1.2.25-raw + repro_bootV1225.sh (sed from V1224). Host reboot
   (standing directive) scheduled before the validation round; lane
   torn down for the bake (watchdog stopped, lsv-test removed).

### Run 7w — bake launch 3 COMPLETE: v1.2.25-raw committed (2026-09-29)

1. **Bake finished ALL GREEN: 93 GATE-OK / 0 GATE-FAIL.** Final chain:
   warm rounds (jitwarm seed 412 + warm_ext 519 + --verify 619 +
   xgrammar probe + 4 sampler shapes), zero-JIT round-2 gate (cache
   delta 0 — every first-use was a load), full content gates (v124
   P16=2/P19.5a ×3/mamba fp16 1/0, v123 async+barrier `False False 0`,
   v1222 serve-config parser coder + 85 sizes, v1221 lineage complete,
   pedigree v1218..v1224 + jit stamps v64..v1224, wheel d20260925 +
   wheel_no_fp8_strings=0, C5/C6 absence, no bake scripts left),
   `llm-scaler-exp:v1.2.25-raw` = **4e83528bfd66** (24.7GB) committed
   with `.llm_scaler_exp_v1225_baked` + `.v1225_jit_warmed`;
   `STAGE5_BAKE_V1225_DONE` 08:48:00. repro_bootV1225.sh derived via
   sed from V1224 — diff verified EXACTLY two lines (image tag + boot
   marker): the V1212-lineage async-FORBIDDEN trap cannot recur
   (async gate inverted since V1223, inherited verbatim).
2. **Drivers written + staged (repo + /root/build/), all bash/sh -n
   clean locally and on host:**
   - `validate_v1225_run.sh` — generates sanity_v1225/validate_v1225
     from the on-host v1224 pair (verified present, Sep 28), v125 legs
     appended (opsall/c7/fixline EXTRA3; fail-fast asserts + C7
     functional on the LIVE lane; P23F probe A at section 4.5 PRE-chain
     — fresh-boot semantics, ordering bug caught during writing —
     quality gates in section 7: fresh 0/80 + 0 flips + 30/30; probe B
     post-battery 0/24 + 30/30; JIT cache delta ≤2 vs v1.2.25-raw).
   - `ship_v1225.sh` — preconditions V1225_VALIDATION_COMPLETE + lane
     on v1.2.25-raw; SHIPWARM relaunch; warm rounds seeds 831/833/839
     + sampler×4; lane-commit → v1.2.25; gates via V123_ID/V123RAW_ID;
     prod fresh-boot + sanity; CC battery via :4000 (Bearer sk-dummy,
     python-built JSON); watchdog repoint + `systemctl restart`.
   - `gates_v1225_lane.sh` — v1224 template + v125 section (opsall
     marker + none→all transform + backup present; C7 marker/env-refs
     exactly 2 guard-only/no mode-var; C7 functional refusal trio in
     lsv-gate; wheel_no_fp8_state_strings=0; C5/C6 absent; no serve
     variants) + RESTORED v1223 pedigree/jit-stamp gates (missing from
     gates_v1224_lane.sh — full chain v1218..v1225 now) + v1225 stamps.
3. **Bugs found and fixed while writing the drivers (both recorded as
   lessons):** (a) ship_v1224.sh's prod-boot sed
   `s/v1\.2\.23-raw/v1.2.24/` was a NO-OP on repro_bootV1224.sh (which
   references v1.2.24-raw after the bake sed) — v1224's "fresh-boot on
   the committed image" actually re-booted -raw; v1225 uses
   `s/llm-scaler-exp:v1\.2\.25-raw/llm-scaler-exp:v1.2.25/g` (verified
   by grep in the ship script). Benign for v1224 (prod differs from
   -raw only by the warm lane-commit layer) but recorded. (b) My first
   ship_v1225.sh had `\"content":$(` — ONE missing backslash in the
   t1 JSON body closed the outer -d string early and unbalanced the
   whole script (bash -n caught it; found by isolating the curl block
   and cat -A diff against the proven v1223 pattern). bash -n before
   staging is non-negotiable.
4. **Pre-reboot safety:** lane-watchdog is `enabled` + `Restart=always`
   and its relaunch line pointed at repro_bootV1224_prod.sh — after a
   reboot it would auto-start and boot the WRONG lane image
   mid-validation. Fixed: `systemctl disable --now lane-watchdog` +
   `.paused` flag before reboot; ship_v1225.sh now does
   `systemctl enable` after its restart (final posture enabled+active,
   same as before). litellm-proxy verified `unless-stopped` → returns
   on its own with the correct master-key config.
5. **Host 10.20.3.65 REBOOTED at ~08:59** (standing fresh-hosts
   directive; lane down, watchdog disabled+paused, nothing to
   preserve). Next: fresh boot v1.2.25-raw via repro_bootV1225.sh →
   validate_v1225_run.sh → ship_v1225.sh.

### Run 7x — P25 validation COMPLETE on fresh-boot v1.2.25-raw (2026-09-29)

1. **Fresh-host protocol honored:** host rebooted 08:59 (lane-watchdog
   pre-emptively `disable --now` + `.paused` — it is enabled +
   Restart=always and its relaunch line still pointed at
   repro_bootV1224_prod.sh; left disabled until ship re-points and
   re-enables it; litellm-proxy verified `unless-stopped` → returned on
   its own with correct master-key config). First boot launch failed
   instantly: `usage: repro_bootV1212.sh <MODE>` — MODE is a mandatory
   log tag, not optional; relaunched with `VAL`. Boot clean:
   baked-config verified (gmu0.8 bs64 e4m3 pc-ON async-ON),
   HEALTH_OK ~170s, live_resets=0, lsv-test on v1.2.25-raw.
2. **validate_v1225_run.sh ALL GATES PASS 10:14:08 →
   V1225_VALIDATION_COMPLETE.** Chain evidence:
   - sanity+admission+v125-posture PASS 09:24:06 (v66 admission PASS,
     async engaged, barrier `False False 0`, RPC 60000)
   - v125 deltas: opsall=1 c7=2 fixline=1; C7 functional on the LIVE
     lane (=2 → full RuntimeError message + rc=1; unset → clean import)
   - **P23F quality PERFECT: fresh probe A 0/80 concurrent wrong +
     0 serial-vs-concurrent flips + 30/30 tools clean** (section 4.5,
     pre-chain = genuinely fresh boot), post-battery probe B 0/24 +
     30/30 — the certified fp16-GDN + fp8-KV posture holds on v1225
   - full battery: SOLO_COLD_DONE; 3× 14-phase drill SUSTAIN_COMPLETE_
     NO_WEDGE fence-hits=0; serialized 24 ×3 all SURVIVED ok=24 fail=0;
     bursts a/b/c 36/36 each resets_after=0; parser battery PASS
     (qwen3_coder); serve tracebacks 0; JIT cache delta 1 vs v1.2.25-raw
     (bound ≤2, v1222-raw first-use precedent); v124 lineage p16=2 /
     p195×3 / mamba fp16=1 fp8=0; fp8 dtype resolution intact
     (float8_e4m3fn / float8_e5m2)
   - genspeed: solo 78.6/79.7/79.5 tok/s (sync-parity floor 70);
     **async 4×1024 aggregate 149.80 tok/s** (acceptance ≥100; sync
     era was 72-74) — the v123 async win fully carried through the
     opsall + C7 additions, zero regression.
3. **Ship chain launched** (ship_v1225.sh, background): SHIPWARM
   relaunch → warm rounds 831/833/839 + sampler ×4 → lane-commit →
   llm-scaler-exp:v1.2.25 → gates_v1225_lane.sh (V123_ID/V123RAW_ID)
   → prod fresh-boot on the COMMITTED image (sed actually swaps the
   tag this round — v1224 bug fixed) → CC battery via :4000 →
   watchdog repoint + restart + re-enable.

### Run 7y — v1.2.25 SHIPPED, lane UP, watchdog ARMED (2026-09-29)

1. **Ship gates v2 ALL PASS 10:29:56 — 97 GATE-OK / 0 GATE-FAIL** (full
   log line count; the first run's `tail -80` stdout window had hidden
   the single early failure — grep the LOG, not the tail). The one
   failure was a QUOTING BUG in my new gate, not the image:
   `grep -cF '\"all\" if c == \"none\" else c'` — inside single quotes
   `\"` is a LITERAL backslash, so the -F pattern could never match;
   fixed to plain `'"all" if c == "none" else c'`. The committed image
   (llm-scaler-exp:v1.2.25 = 3f3c91637692, 24.7GB, from the fully-warm
   SHIPWARM lane, shipwarm_new_jit=0 cache=79) was kept — no second
   commit; ship resumed via ship_v1225_resume.sh (gates + step 5) and
   ship_v1225_finish.sh (steps 6-7).
2. **Prod fresh-boot on the COMMITTED image PASS** — the v1224 prod-sed
   no-op is fixed for real: repro_bootV1225_prod.sh line 51 shows
   `llm-scaler-exp:v1.2.25`; sanity + v66 admission PASS + async
   engaged + barrier `False False 0` + RPC 60000 + opsall fix line
   present on the serve log.
3. **CC battery: first in-script attempt http=400 ×3** — body ended
   with a stray `"` after valid JSON ("unexpected content after
   document: char 135"): the v1223-inherited curl tail `)}]}\""`
   appends the escaped quote as payload. ship_v1223.sh STILL carries
   this buggy line in-repo (their green battery ran via
   cc_battery_fix_v1223.sh, so I had copied an untested "proven"
   pattern). Fixed tail `) }]}"` in all v1225 ship scripts + new
   standalone `cc_battery_v1225.sh` building bodies ENTIRELY in python
   (json.dumps → -d @file, body pre-validated) — **re-run ALL GREEN:
   t1/t2/t3 http=200 + thinking block** through :4000 (Bearer sk-dummy).
   Standing lesson re-confirmed: NEVER shell-assemble JSON bodies.
4. **litellm up-probe corrected:** `/health` bare = 401 (master key
   required) and detailed /health round-trips the backend (times out
   at 5s mid-boot); `/health/liveness` = 200 is the right probe. The
   manual 200 (thinking+text) probe confirmed the path before re-run.
5. **Watchdog: repointed to repro_bootV1225_prod.sh (line 75 verified),
   restarted, RE-ENABLED (it was disabled pre-reboot), and ARMED** —
   `2026-09-29 10:54:34 alive armed code=200 cycles=10`. The finish
   script's 220s armed-line window was miscalibrated: the watchdog
   cycles at 60s (not 10s), so the first status line lands ~10 min
   after start — the WARN was timing-only. Lane lsv-test UP on
   llm-scaler-exp:v1.2.25; litellm-proxy up.
6. **Final ship state:** v1.2.25-raw = 4e83528bfd66 (bake, 93/0) →
   v1.2.25 = 3f3c91637692 (lane-commit, gates 97/0) — full chain:
   opsall ROOT FIX (custom_ops 'none'→'all' restoration, native-op NaN
   on spec-off GDN decode fixed, boot line 'v125 root fix') + C7
   fp8-state import-time refusal (P23D/P23F conviction in-code) on the
   v1.2.24-raw base; validation fresh-boot battery ALL GATES PASS incl.
   P23F quality 0/80 + 0 flips + 30/30 and async 4×1024 aggregate
   149.80 tok/s. Spec MTP×4 + XGrammar-2 0.2.7 + e4m3/e5m2 dtype
   resolution verified on every boot gate.
