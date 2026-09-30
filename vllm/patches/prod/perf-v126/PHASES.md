# perf-v126 PHASES — live investigation record (rule: update after every phase)

Round: fp8-state kernel fix — P23D/P23F quality convictions + speed inferiority -> fixed,
certified, shipped as llm-scaler-exp:v1.2.26. Task text + gates in PLAN.md.

---

## P27 — root cause of the fp8-state quality convictions: CONVICTED AT SOURCE LEVEL

Method: read the shipped code paths end-to-end — `_xpu_ops.py` (baked v124 P19.5a unique
bridge + C7), `v1/attention/backends/gdn_attn.py` (metadata builder), wheel sources
`/root/build/vxk/csrc/xpu/gdn_attn/{gdn_attn_interface.cpp, causal_conv1d.hpp,
gated_delta_rule.hpp}`, model layer `gdn_linear_attn.py` + `qwen3_next.py` pool dtypes.

**ROOT CAUSE (the bridge breaks conv-state addressing):**

The SYCL op `gdn_attention` runs TWO kernels per call, driven by the SAME index tensors:

```
gdn_attn_interface.cpp:310  gdn::causal_conv1d(..., conv_state, ..., spec_state_indices_tensor, ...)
gdn_attn_interface.cpp:346  gdn::gated_delta_rule(..., ssm_state, ..., spec_state_indices_tensor, ...)
```

- `causal_conv1d.hpp:232/452`: conv rows addressed `conv_states + states_id * stride_0`
  where `states_id` = entries of the SAME `cache_indices`/state-index tensors; spec path
  loads at `cache_indices[batch, accepted-1]`, writes at `cache_indices[batch, num_spec]`
  (causal_conv1d.hpp:487+).
- Every fp8-SSM posture bridges: ssm pool gathered to a compact fp16 `ssm_run` copy and
  the state-index tensors REMAPPED to compact rows (`_remap`), while `conv_state=
  self.kv_cache[0]` stays the REAL pool with REAL slot ids.
- => the conv kernel reads/writes conv states at COMPACT ROWS of the real conv pool.
  Cross-request conv-state crosstalk whenever the batch's slot ids are not exactly
  0..N-1 in order — i.e. whenever concurrent requests coalesce (prefill or decode).

**Fit against every observation:**
- concurrent-wrong -> serial-right (18/18): single-sequence batches remap to row 0 and
  use row 0 consistently -> self-consistent, correct output.
- post-prefix worse (25-41.7% vs 8.75-13.75%): chunked-prefill continuation loads its
  initial conv window (`has_init_conv_states`, causal_conv1d.hpp:258) from the WRONG row.
- heals after ~3 requests; deterministic-within-arm indices: slot allocation is
  deterministic per schedule; fresh full prefills overwrite conv states.
- BOTH formats, BOTH routes (=1 static/unique bridges, =2 unique bridge on prefill) share
  bridge remapping -> all broken. fp16 pools NEVER bridge (indices real for both kernels)
  -> control 0/104+60/60 clean.
- P23D =2 20% tool salad: conv corruption feeds garbage windows into the GDN stream.
- The v88-int32 "same root family" framing was directionally right (state-slot addressing
  defect) but the actual site is the PYTHON bridge's index remapping, not the kernels.

**Secondary hazards found en route:**
- torch.unique inside the v124 bridge is value-dependent-shape and must never run under
  full-decode-graph capture (P22C5 already knew this for spec; v126 makes static remap
  the rule for ALL capture-eligible decode batches).
- `_uniq`/`_remap` see NULL_BLOCK_ID (-1) padded tails under full graphs: index_select
  wraps to the last pool row and `_remap[-1]` corrupts the LAST REAL SLOT's remap entry
  (benign-by-coincidence today; excluded by narrowing in v126).

**Fix design (P28) — the DUAL bridge:** when the ssm pool is fp8, gather BOTH pools'
rows with the same index set (conv pool is fp16 per `MambaStateDtypeCalculator` — only
the ssm pool follows mamba_ssm_cache_dtype; qwen3_next.py:1937 returns the (conv,ssm)
dtype pair), pass the compact conv copy + compact ssm copy + remapped indices so BOTH
kernels see a consistent address space, scatter BOTH home after. Static-arange remap
(P22C5 pattern) for decode-only/capture-eligible batches; unique-remap only for eager
prefill/mixed batches. C7 reworked from refusal to the certified enablement.

Runtime verification: op-level crosstalk repro + P23F probe re-run on a test lane (P28).

**P27 RUNTIME CONVICTION (op-level repro, throwaway container on v1.2.25-raw,
p27_conv_repro.py):** 2-seq prefill, real slots [5, 9], fp16 pools (isolates
index semantics from fp8 math):
- Variant A (real indices [5,9] against real pools): conv rows written = [5, 9].
- Variant B (bridge sim: ssm gathered to rows [0,1], indices remapped [0,1],
  conv pool left REAL — exactly what the v124/v125 bridges do): conv rows
  written = **[0, 1]** — compact rows of the REAL conv pool, slots 5/9 stale.
- P27_RUNTIME_VERDICT: **CONVICTED** (A_writes_real_slots=True,
  B_writes_compact_rows=True).

P27 CLOSED. The defect is the PYTHON bridge's ssm-only index remapping; both
kernels must be given ONE consistent address space (P28 dual bridge).

---

## P28 — dual bridge implemented + op-level verified (PASS)

Kernel-semantics facts pinned down from the wheel sources BEFORE writing the
fix (they drive the design):
- gated_delta_rule.hpp:527-540: the ssm spec path writes the running state at
  `cache_indices[r, t_local]` per draft step — "this token's dedicated cache
  slot, so that the next forward can pick the right rollback column".
- causal_conv1d.hpp:827-845: the conv spec path CHECKPOINTS the rolling window
  at EVERY column t_local — "column t_local holds the conv state right after
  consuming spec token t_local" (the file-header comment "ONLY the last cache
  slot" is STALE; code is per-column). Control leg confirms: real-index spec
  batch writes ALL n*W rows in both pools.
- => spec index columns are DISTINCT per-rollback slots; the per-VALUE mapping
  of the real pool must be preserved exactly. Static capture-safe remap =
  FLAT: gather all n*W rows, remap[r,c] = r*W+c (all ids distinct; allocator
  guarantee, verified once eagerly pre-capture via
  torch.xpu.is_current_stream_capturing — available on torch 2.11.0+xpu).

patch_v126_dualbridge.py (applies to /opt/venv/.../vllm/_xpu_ops.py,
backup .v126bak, py_compile gate, idempotent):
- Decode-only batches (num_prefills == 0, captured or eager): STATIC FLAT
  remap — spec: _st.reshape(-1) + arange(n*W).reshape(n,W); no-spec:
  ns[:num_decodes] + arange(n). Constant shapes only; torch.unique (value-
  dependent shape) never runs under full-decode-graph capture (the v124
  bridge would have crashed there — e5m2s4 boot death class).
- Eager prefill/mixed: per-VALUE unique remap (v124 logic was addressing-
  correct for ssm), negatives (NULL_BLOCK_ID) filtered, LUT applied to both
  index tensors.
- BOTH pools gathered with the ONE index set (conv must be fp16/fp32 —
  guarded), run op against compact copies, scatter BOTH home (ssm via uint8
  view, conv via fp16 index_copy_).
- C7 -> C7': certified enablement = --mamba-ssm-cache-dtype fp8_e4m3|fp8_e5m2
  directly (no env); legacy VLLM_XPU_GDN_FP8_NATIVE=1/2 stay refused.

p28_dual_check.py (op-level, fp8_e4m3 ssm pool + fp16 conv pool, DK=DV=32,
O(1) inputs so state deltas survive fp8 quantization):
- LEG1 ns-route (prefill, slots [5,9], unique branch): conv rows written
  {5,9} (v124 wrote {0,1}), ssm rows {5,9}, rows 0/1 untouched. PASS.
- LEG2 spec-route (2 reqs, W=5 distinct columns 5..14, accepted [3,2], static
  flat branch): changed-row sets IDENTICAL to the real-index fp16 control for
  BOTH pools; nothing outside 5..14. PASS.
- P28_DUAL_BRIDGE_VERDICT: **PASS** (throwaway container on v1.2.25-raw).

Next: lane-level P23F gate — test lane with --mamba-ssm-cache-dtype fp8_e4m3
(+ e5m2 leg), fresh-boot, concurrent probe: 0/80 fresh wrong, 0 serial
flips, 0/24 post-prefix.

---

## P28 lane-level gate — e4m3 PASSED CLEAN (2026-09-29)

Setup: host rebooted fresh (standing directive; watchdog stopped+disabled for
the test window, prod lane container removed, image v1.2.25 intact).
boot_v126_test.sh = V1225 boot chain on llm-scaler-exp:v1.2.25-raw +
patch_v126_dualbridge.py + sed --mamba-ssm-cache-dtype fp8_e4m3.

**e4m3 arm (fp8_e4m3 SSM pool, fp16 conv pool, spec MTPx4, XGrammar-2,
async, KV fp8_e4m3):**
- Boot: HEALTH_OK ~180s; FULL decode graph capture SURVIVED with the fp8
  pool (v124 postures died at the spec-verify dummy run — static flat remap
  is capture-safe as designed); `v126 DUAL_FP8_BRIDGE engaged:
  ssm=torch.float8_e4m3fn conv=torch.float16` from BOTH workers; resets 0;
  JIT lines 0.
- P23F probe mode A (fresh boot): concurrent math **0/80 wrong** (v125
  e4m3ns baseline 7/80); serial 0/80; **0 concurrent-vs-serial flips**
  (baseline 18/18); tools **30/30 clean, 0 salad** (164s).
- Prefix build: repro_sustain 1 round SUSTAIN_COMPLETE_NO_WEDGE
  (fence-hits 0) + burst_harsh SURVIVED ok=36 fail=0 resets_after=0;
  serve tracebacks 0.
- P23F probe mode B (post-prefix): math **0/24 wrong** (baseline 10/24 =
  41.7%); tools **30/30 clean, 0 salad** (105s).

P23F e4m3 conviction OVERTURNED. e5m2 arm running (mode A already:
0/80, 0 flips, 30/30 — baseline was 11/80; prefix + mode B pending).

**e5m2 arm (fp8_e5m2 SSM pool):** boot HEALTH_OK ~150s, full decode graph
capture survived, DUAL_FP8_BRIDGE engaged on both workers, resets 0.
- Mode A fresh: **0/80** wrong, serial 0/80, **0 flips**, tools **30/30**
  (v125 e5m2ns baseline: 11/80 fresh).
- Prefix: sustain SUSTAIN_COMPLETE_NO_WEDGE (fence-hits 0; round ~18 min —
  e5m2 decodes slower, expected) + burst SURVIVED ok=36 fail=0 resets 0.
- Mode B post-prefix: **0/24** wrong (baseline 6/24 = 25%), tools **30/30**.
- Serve tracebacks 0 across the whole arm.

**P28 GATE: PASSED BOTH FORMATS.** e4m3 and e5m2 each: 0/80 fresh, 0 flips,
0/24 post-prefix, 60/60 tool calls clean, spec MTPx4 + XGrammar-2 active,
wedge-free sustain, burst survival, resets 0. The P23D/P23F quality
convictions are OVERTURNED by the dual bridge (both kernel routes verified:
op-level row-set equality + lane-level zero-wrong). P28 CLOSED.

---

## P29 — remove ALL extra quantization roundtrips + end-to-end fp8 pipeline

Directives: "continue and remove all possbile extra quantization
roundtrips" + "full fp8 implementeation has to work end-to-end fp8 using
fp8 pipleine so all overheads are removed and full benefit of fp8 can
achieve, and full pipeline has to match used fp8 type fp8_e5m2 -> fp8_e5m2
and fp8_e4m3 -> fp8_e4m3".

**Audit (whole esimd_kernels tree, source-level):**
- gdn_conv_fused_seq.h / gdn_conv_fused.h have NO token loops — one state
  load + one store per kernel call = pool residency, nothing to chain.
- The ONLY pool-roundtrip chain: gdn_spec_update_seq (spec kernel) loaded
  prev state from pool slot[t-1] and stored slot[t] EVERY draft token —
  W loads + W requantizing round-trips per verify round (the conv chain
  s0/s1/s2 was already register-chained; conv checkpoints are write-only).
- e5m2 primitives already minimal (3-op bit lift / RNE truncate); e4m3
  LOAD wasted 3 fp16<->f32 conversions + mapped NaN codes 0x7F/0xFF to
  finite 480.0 (c10 decodes NaN).
- Decode-time dual bridge (gather dequant + scatter quant per step) is
  removable overhead: P29C routes fp8 decode through the fp8-native ESIMD
  kernels; the bridge stays for prefill/mixed only (SYCL kernels need
  fp16 pools there — one bridge per forward, inherent).
- Pool dtype keys the whole path (wrapper dispatches on scalar_type):
  e5m2 lane -> e5m2 kernels, e4m3 -> e4m3, no cross-format detour. e4m3
  lane KV already fp8_e4m3; e5m2 lane gets a --kv-cache-dtype fp8_e5m2
  probe leg for the full format match. Weights = e4m3 fp8 checkpoint on
  both lanes (static quantization, not the state path).

**P29a — kernel patch + rebuild (DONE).** p29a_kernel_patch.py (marker
"v126 P29A", backups *.v129prebak):
- utils.h esimd_e4m3_load -> single conversion (magnitude converts once,
  sign ORs into the fp32 bits via u32 bitcast, NaN codes -> qNaN = exact
  c10 parity).
- gdn_conv_fused_seq_spec.h fp8 REGISTER CHAIN: gdn_spec_update_seq takes
  caller-owned fp32 carriers (h0_lo..h1_hi refs + h_chained); token 0 of
  each verify round dequantizes the rollback state from the pool exactly
  as before; tokens 1..W-1 reuse the registers. W-1 loads + ALL requant
  feedback removed; checkpoint stores still quantize once per column.
  fp16 NEVER chains (constexpr kFp8Chain = sizeof(StateT)==1) ->
  bit-identical reload dataflow, zero risk to the fp16 lane.
- Build (build_esimd_v129.sh, esimd-inc, MAX_JOBS=52 KERNELS_MAX_JOBS=52):
  attempt 1 failed — `simd<T,N>(...).bit_cast_view()` needs an lvalue
  (fixed: named qnan_bits). Rebuild clean: .so 95,661,592 B, sha256
  8f222ffe449df2b4d6450edf1432e8f796e7bf820f1b27ebe25072b4416f860d,
  3 esimd_gdn_conv_fused symbols; symbol scope IDENTICAL to the
  production lgrf .so (0 gemm/eagle/moe, 3 gdn — the package's other
  .so files carry those), so the swap is scope-safe.

**P29b window 1 (14:15-14:20, .so swapped into lsv-test, serve down;
lane restored clean after: LANE_RESTORED 14:20:29, health 200, old .so
sha back):**
- L2 fp8-vs-fp16 PASS both formats (e4m3 cos=0.99963 maxdiff=0.00032;
  e5m2 cos=0.99860 maxdiff=0.00059).
- L1 solo-vs-batch FAIL **all three dtypes including fp16** — kernel
  exonerated by construction (fp16 runs P29A-dead code; a real regression
  cannot produce an fp16 mismatch). ROOT CAUSE = the harness, convicted
  against kernel source gdn_conv_fused_seq_spec.h:218-261: init_col =
  max(num_accepted-1, 0) and token 0 seeds BOTH conv (s0/s1/s2) and SSM
  state from pool slot indices[init_col] READ-BEFORE-WRITE; window 1 ran
  the batch call first, then cloned the POST-BATCH pool (and shared the
  mutated conv) for each solo -> token 0 re-read checkpoint slots the
  batch had overwritten. Same root explains L3 det=False (shared mutated
  conv across the two runs). L4 FAIL was a construction error: accepted
  =[2] => init_col=1 => the kernel read slot 56 and the 0x7F/0xFF-filled
  slot 55 was never loaded.
- p29b_check.py REV2: every run starts from PRISTINE pool+conv clones;
  L1 additionally compares final conv rows at the request's slots and
  reports the true failing index; L4 uses accepted=[1] => init_col=0 =>
  indices[0]=55 is on the load path. Rev2 runs as a hard pre-serve gate
  inside p29c_boot.sh.
- Speed A/B on the v129 .so (p22c3_ops.py, K=V=128 H=8 HV=24 NC=512):
  - e5m2 NATIVE SUPERIOR EVERYWHERE: spec ratio 0.932/0.966/0.968/0.690/
    0.825/0.875/0.878 (n=1..64); seq 0.682-0.895. The fp8 chain + cheap
    3-op primitives = genuine bandwidth win.
  - e4m3 remains ALU-bound on the quant primitives: spec 1.218/1.285/
    1.273 (n=1/2/4), 1.022/1.008/1.054/1.062 (n=8..64); 2 seq cells
    >1.10 (P22C3_FAILS=5 total, old <=1.10 gate). e4m3's narrow range
    forces big/tiny merges + denorm magic on store and an exponent
    rebias + NaN merge on load (~6-8 more vector ALU ops/element than
    e5m2's aligned-exponent primitives) — semantics are c10-parity-
    locked (P28 quality rode them), so the ops cannot be dropped without
    changing quality semantics. VERDICT: op-level e4m3 ~=1.0-1.3x fp16 is
    near the pure-ALU floor; the LANE is the arbiter — P29C removes the
    ENTIRE decode-time bridge (gather+scatter per step, the dominant
    fp8-lane overhead), which should more than repay the op delta. If
    the lane still lags fp16, a dedicated store-codegen round follows.

**P29c — integration (RUNNING).** patch_v126_poolpath.py rev2 (markers
"v126 P29C"; TARGET FIXED to layers/mamba/gdn_linear_attn.py — the file
lives under model_executor/layers/mamba/, not models/): E1 flag rename
_gdn_conv_state_fp16_ok -> _gdn_conv_state_ok; E2b spec gate — fp8 SSM
pool rides esimd_gdn_conv_fused_seq_spec with ANY num_spec_decodes (grid
dim0 = num_spec_decodes, one work-group per (seq, hv), read-before-write
seed semantics per the contract above; fp16 keeps nsd==1 bit-for-bit;
mixed nsd>1 + plain-decode batches degrade to the bridge, never corrupt);
E2c one-shot serve-log marker "v126 P29C: fp8 SSM decode NATIVE ESIMD";
E3 state gate: conv pool fp16 required + ssm pool in {fp16, e4m3, e5m2}.
p29c_boot.sh <dtype> [port] [kv] = boot_v126_test.sh chain + poolpath +
v129 .so (sha-gated 8f222ffe...) + P29B rev2 as a hard PRE-SERVE gate.
Lane order: e4m3 -> e5m2 -> e5m2-KV probe -> fp16 same-day baseline;
each lane: quality quick-probe (0-wrong required) + speed matrix
(bench_genspeed.py 4/8-stream + solo).

**P29b window 2 — the nsd>1 spec-kernel bug hunt (p29b2..p29b10; P29B
rev2 gate FAILED for nsd>1 L1 in all dtypes incl. fp16; boot chain
correctly exited 13 pre-serve).** The harness-confound root causes of
window 1 were fixed first (pristine clones per run, valid same-slot
solo references, accepted=[1] for L4) — the failure SURVIVED the clean
harness: real kernel defect, convicted by an escalating probe chain:

- p29b2: batch nondeterministic run-to-run (fp16 flaps rows 5-9);
  solo deterministic. C/D legs self-invalidated (solo refs used
  different slot ids — slot content IS the seed; recorded as a probe
  design lesson).
- p29b3 (clean harness): fp16 flap persists; e4m3/e5m2 run-stable
  in-process; nsd=2 req0+req3 PASS, req1+req2 FAIL; no OOB; conv
  checkpoints BITWISE-correct both runs; damaged pool bytes exist
  (non-pristine, no writer absence); identical twins (same inputs both
  positions) diverge.
- p29b4: Z1 null-seed (idx=-1) differs from batch damage; Z2 pattern
  0.25 visible in solo but batch ignores it; Z3 seed-slot cross-talk
  excluded; Z4 z rows bitwise-identical (divergence confined to the SSM
  segment); Z5 damage = hv>=8, ALL 128 v-rows (full heads).
- p29b5: per-(seq,hv) ran-matrix: out nonzero + pool non-pristine +
  conv solo-equal EVERYWHERE → stores executed; wrong-MATH, not
  absence.
- p29b6: cross-slot/cross-t byte scans: ZERO matches anywhere in the
  512-slot pool or across t → NOVEL values, not an address shift; hv=23
  bitwise CORRECT (ratio 1.0), hv=10/15 corr≈0.53.
- p29b7: M1 (seq×hv) matrices: nsd=4 seq0 clean, seq1 X from hv8,
  seq2 X from hv16, seq3 clean; nsd=2 seq1 ragged from hv8. M2 seq0
  input ×2: seq1 STABLE (no cross-seq data leak). M3 seq0 slot move:
  stable. M4 INVALID (sliced qkvz OOB — discarded, self-caught).
  M5 3-run stability: frozen in-process.
- p29b8 (input nulling, the decisive exculpations): T1 zero-pool,
  T2 zero-qkvz, T3 zero-pool+zero-qkvz — seq1 hv>=8 STILL diverges from
  solo in all three → every in-pool address AND the SLM data content
  are exonerated; remaining vectors: a read landing OUTSIDE the pool
  allocation, or execution-level corruption (scheduling/codegen).
- p29b9 (identical twins): same-slots twins diverge with an UNSTABLE
  per-hv footprint across repeats (in-process flapping — "process
  frozen" was a small-sample artifact); zero-input twins CLEAN (data
  dependence). Same-slots leg tainted by a genuine seed-store race the
  design introduced (slot sharing) — M2 distinct-slots leg INVALID
  (different seed content; self-caught, discarded).
- p29b10 (geometry discriminator): **HV=16 PASSES COMPLETELY** (all
  heads, all tokens, zero-input too); HV=24 fails. Bug confined to the
  double_v code paths (conv_result_hi / hi checkpoint stores / hi SLM
  store / hi chain advance) — which are production-proven at nsd=1, so
  the defect requires double_v AND nsd>1 together. Failure-set shape
  across all configs: wgs {(s,hv): s>=1, hv>=8s} = global gids
  {32..47, 64..71} at HV=24 — not explained by any per-wg source defect.
- Source audit (3 full passes): grid/nd_range exact; early-return guard
  never fires; geometry map (tid->chunk lo/hi, q/k/v writers, readers,
  v_oob remap, z passthrough) all correct; both barriers present
  (post-store pre-load at :378, loop-end at :417); barrier resolves to
  esimd::barrier() (utils.h `using namespace sycl::ext::intel::esimd`)
  = correct SLM-fencing work-group barrier; update helper thread-pure;
  all address math int64-safe. Static analysis exhausted: the code as
  written cannot produce the observed outputs from the observed inputs.
- **P29B-DIAG (in flight): instrumented variant v130** —
  gdn_conv_fused_seq_spec_dbg.h (self-contained copy; production
  kernel byte-untouched) dumps a 20-slot fp32 record per
  (seq, t, hv, tid): global_t/prev/save idx, own conv_result[0]/s2[0],
  SLM-read q/k/v, post-load h0 (t0), A/dt/b/a scalars, q_inv/k_inv,
  kv0, result[0], own conv_result_hi[0], epoch. New torch op
  esimd_gdn_conv_fused_seq_spec_dbg (marker "v126 P29B-DIAG" in
  esimd_kernel_lgrf.sycl / kernel_ops.h / torch_extension_lgrf.cc;
  integrator p29dbg_install.py; build build_esimd_v130.sh ->
  /root/build/v130_so, MAX_JOBS=52 KERNELS_MAX_JOBS=52). Probe plan:
  T3 config (zero pool+qkvz, nsd=2, distinct slots) batch-vs-solo diff
  of the records — the FIRST divergent field names the corrupted stage
  (conv compute vs SLM staging vs h-load vs update math).

**P29b12 — PRE-EXISTENCE PROVEN on the stock production .so (decisive
for blame):** the recorded .v125bak backup was lost in a container
re-create, so the stock .so was extracted fresh from the shipped
v1.2.25-raw image (`docker create tmpso129` + `docker cp`); sha matched
the recorded stock sha `1d9dcf4e…` exactly. p29b12_probe.py ran the
p29b10 geometry with the fp16 pool (the stock .so predates fp8 SSM
state) at nsd=2, 3 reps: **all reps fail identically**
(`ooooooooXXXXXXXXXXXXXXXX`, seq1-t0, hv>=8 band). Verdict: the nsd>1
double_v defect PRE-EXISTS the entire v126/v125/v129 line — an upstream
latent bug in `gdn_conv_fused_seq_spec` never exercised in production
(the model layer gates the ESIMD spec path at nsd==1). NOT a v129
regression; v129 merely exposed it in synthetic probes. v129 was
restored into lsv-test afterwards and verified by sha
(`8f222ffe…`).

**P29B-DIAG build saga (recorded for the process lessons):** build-1
failed on a redefinition storm — my dbg header originally included
BOTH gdn_conv_fused_seq_spec.h AND gdn_conv_fused_seq.h; the latter has
NO include guard and is already included by the former → every helper
double-defined (20 errors). Fix: single include of the spec header
(helpers/SLM constants arrive transitively). Build-2 compiled clean
(nm: 1 dbg symbol; staged sha `3f20069d…`) but ABORTED AT IMPORT:
`Tried to register an operator … multiple times` — the installer's
idempotency guard tested `MARK in txt and tag in txt`, but the tags
("dbg_decl"/"dbg_op"/…) never occur in the inserted text, so each
re-run re-inserted (4x m.def in torch_extension_lgrf.cc, 4x decl in
kernel_ops.h, 2x include). Fixes: (a) p29dbg_dedup.py collapses every
duplicate block keyed on the P29B-DIAG comment markers (keep-first,
idempotent, hard-verifies all 5 occurrence counts == 1); (b)
p29dbg_install.py guard now probes for the inserted symbol itself per
file (`#include "…_dbg.h"` / `at::Tensor …_dbg(` /
`m.def("…_dbg`). One intermediate misfire: first guard fix passed
probe/tag in swapped positions (guard never matched, re-duplicated);
caught from the "patched: <probe-string>" output, re-deduped, fixed
arg order — installer now a true no-op ("already" x4). Build-3
launched 16:0x with exactly-once registration. Lesson reinforced:
idempotency guards must test a string that ACTUALLY EXISTS in the
inserted text, and the guard's own print output must be read
critically ("patched:" with a probe-looking tag is the tell).

**P29b11 — on-device first-divergence capture (v130 .so, dbg op):** the
first REAL divergence is not an input, index or weight — it is the OWN
CONV COMPUTE (s2_0/conv0) at t=0/1 with byte-identical inputs (glb_t
excluded: batch seq1 legitimately runs global tokens NST..2NST-1 vs
solo 0..NST-1). Zero-input leg (pool+qkvz all zeros): conv0 diverges at
t=0 on seq1 only; random-input leg: same first-divergent stage. Both
legs deterministic WITHIN a run (solo-vs-solo and batch-vs-batch record
sets bitwise stable across reps).

**P29b13 — VALUE FORENSICS = ROOT CAUSE (the breakthrough):** post-
processing of the dbg records (T3 config, nsd=2, slots 40..49, solo
45..49, accs [2,4], e4m3):
- [D] conv0 t0: 768 mismatches, ALL hv>=8, and the batch values are
  EXACTLY +0.000000 where solo has small nonzero values (zero-input
  windows return exact zeros by t=2: with zero qkvz the stored window
  collapses to (s1,s2,x)->(0,0,0)).
- [E] the mismatch set splits EXACTLY on dims: corrupted tids
  {0..39, 56..63} = q/k dims [0,2048) + V heads 0..7 = [2048,3072);
  CLEAN tids 40..55 = V heads 8..23 = dims [3072,5120).
- Record fields prev/save/epoch never mismatch — indices and writer
  bookkeeping are correct; only the DATA of the init window is wrong.

**ROOT CAUSE (mechanism, proven by the dim split):** in
gdn_conv_fused_seq_spec the ring convention makes the init row a SAVE
row: ring position init_col is read at work-group start and rewritten
at t==init_col. The writers of that row are SIBLING work-groups of the
same sequence — wg hv==0 tids 0..31 store the q/k dims, wg hv' stores
V dim hv'. At nsd=2/HV=24 the launch has 48 work-groups > ~32 HW
residency slots, so the last 16 wgs (seq1, hv>=8) start only after
early wgs retired; their wg-start init loads then read the init row
AFTER seq1's early wgs rewrote it at t=init_col (their t>=2 windows
are all zeros under zero input). Corrupted dims [0,3072) are exactly
the dims whose writers (hv=0 for q/k, hv 0..7 for V) already ran;
clean dims [3072,5120) are those whose writers (hv 8..23) had not
started. This explains every prior observation: flap = scheduling
jitter; zero-input failure = zero windows; solo pass = 24 wgs all
resident, reads execute before any sibling store; HV=16 pass = 32 wgs
exactly at residency; pre-existence in the stock fp16 .so = same code
path; determinism-within-run = stable schedule per launch shape. The
SSM pool has the same aliasing hazard CLASS for its t==0 read (init
row == save row); t>=1 reloads read rows the SAME wg stored one
iteration earlier (program-ordered, safe given distinct per-column
ids). Sibling hazard noted for the NON-spec gdn_conv_fused_seq —
follow-up audit item, untouched this round (no-degradation rule).

**P29B-FIX — pre-launch ring-row snapshot (root fix applied):**
wrapper (esimd_gdn_conv_fused_seq_spec, .sycl:164) gathers the ring
rows BEFORE launch via at::index_select on the same in-order XPU
stream — conv rows + SSM pool rows, indexed BY RING POSITION (row r ==
seq*NST + col), -1 slots clamp_min(0) (never dereferenced; value
guards stay). The kernel reads (a) the wg-start conv init window and
(b) the t==0 SSM rollback row from the snapshot (never written
in-kernel => race-free by construction); t>=1 SSM reads STAY on the
live pool (own prior store, program ordered — the snapshot would be
stale there and would change fp16 semantics); all checkpoint stores
unchanged. Torch op signature unchanged (snapshot is internal); cost =
one n*NST-row gather (~30 MB at n=8 fp16) ~= 0.1 ms vs 43+ ms steps.
On unraced paths the snapshot bytes equal what the live read returned
=> solo/nsd=1/HV=16/fp16 outputs bit-identical to v129/v130.
Implementation: p29fix_gdn_conv_fused_seq_spec.h (full fixed kernel),
p29fix_gdn_conv_fused_seq_spec_dbg.h (dbg copy regenerated from the
FIXED kernel so instrumentation stays truthful), p29fix_install.py
(whole-function wrapper replacement, span-located by symbol, guarded
by 'p_conv_snap' — a string that exists in the inserted text; one-time
backup to /root/build/p29fix_backup/), build build_esimd_v131.sh ->
v131 sha e050551ea29b729de4ab84af9f17f9b7d8f0ffcf80be302bb5fbc1bf46c7
bea8 (full compiler log preserved p29fix_full_build.log; 0 errors).
Swapped into lsv-test (serve down; v129 backup /root/build/v129_so,
v130 diag backup /root/build/v130_so).

**P29B-FIX VERIFICATION — ALL GREEN (lsv-test, v131 .so):**
- p29b13 rerun: [D] conv0 t0 mismatches 0 (was 768; hv map all '.'),
  [A] solo+batch deterministic, out bands 24/24 'o', [B] seq-crossing
  0/7680, [E] all grids clean.
- p29b11 rerun: ONLY glb_t diverges (expected by construction) at
  every t; no causal field diverges; zero + random legs.
- p29fix_verify.py (production op, bitwise solo-vs-batch incl. pool +
  conv rows): A1/A2 nsd=2 HV=24 (raced geometry) 10 reps x 3 dtypes
  PASS; B1/B2 nsd=4 (96 wgs, every init_col) 3 reps PASS; C1 nsd=1
  PASS; D1/D2 nsd=2 HV=16 PASS; E batch-vs-batch determinism PASS.
- p29b_check.py L1-L4 ALL PASS. L2 re-adjudication CLOSED: e5m2 cos
  0.98292 -> 0.99865 (gate 0.99), e4m3 cos 0.99964; the old value was
  measured on the corrupted nsd>1 kernel, not an e5m2 quantization
  property.
P29B CLOSED: root-caused, fixed, verified. dbg op stays
diagnostic-only (never ships; regenerated from the fixed kernel).

**P29C — lane integration (poolpath + v131 .so, serve-level): FIRST
EXPOSURE RED.** p29c_boot.sh (V1225-lineage chain + dualbridge + poolpath
E1/E2/E3 + v131 .so + p29b_check gate + baked-config checks; bisection
switches NO_POOLPATH / EAGER / ASYNC0 / POOLPATCH added during P29D):
e4m3 lane degenerates from the FIRST request — first token correct
(prefill = bridge, eager), then '!!'/'The!!!' garbage; p23f mode A: MATH
fresh 68/80 wrong, serial re-ask degenerate with flips (P29B race class
ruled out: serial/solo also wrong), TOOLS 0/30 salad. Draft acceptance
98.6% = drafter (MTP shares the corrupt trunk) mirrors the target —
acceptance is NOT a failure signal (green eager lane shows 98.2%/4.93
mean length, same profile). Host rebooted 16:23:53 before the round
(standing directive).

**P29D — serve-level bisection of the e4m3 collapse (all probes p23f
mode A, 10x8 fresh concurrent + serial + 30 tools):**
| leg | posture | verdict |
|-----|---------|---------|
| FAIL | e4m3 + native poolpath + v131 + async+graphs | RED 68/80, 0/30 |
| 0 | fp16 + NO_POOLPATH + v131 + async+graphs | GREEN (128s) |
| A | e4m3 + NO_POOLPATH (bridge) + v131 + async+graphs | GREEN (161s, markers 0) |
| B | e4m3 + native + v131 + EAGER | GREEN (158s, markers 96) |
| C | e4m3 + native + v131 + graphs, async OFF | RED 68/80 (markers 96) |
Convicted: XPU decode-graph capture/replay x native-fp8 spec ride.
Async-scheduling exonerated (C red without it); v131 snapshot build
exonerated under graphs for fp16 (leg 0 green); P28 bridge under graphs
green (leg A) — so the kernel itself under graphs with fp16 pool clones
is a certified-green class.

**P29D kernel exoneration (real-params trajectory, p29d_traj.py in
lsv-test):** ROUNDS=30 chained verify rounds at REAL layer-0 params
(A_log [-5.56,-1.09], dt_bias [-5.72,19.25], conv1d.weight from
layers-0.safetensors; conv bias = zeros, none in ckpt) x scales
0.5/1/2/6: NAT8-vs-BRIDGE8 pool cos == 1.000000, per-round out cos
delta ~5e-4 (both 0.988-0.992 vs fp16 REF = shared fp8 storage noise).
Kernel fp8 chain + fp8 store/load primitives exonerated at real
dynamics incl. saturation. (First run fell back to synthetic params —
real names are model.language_model.layers.0.linear_attn.*, per-layer
shards layers-*.safetensors.)

**P29E — op-level XPU-graph probe (p29e_graph_probe.py, serve down):
ALL GREEN.** Mirrors vLLM's mechanism (capture once, mutate static
inputs in place, replay): P0 machinery control PASS; P1 index_select
capture freshness (plain + out=) FRESH for fp16/e4m3/e5m2; P2 full v131
wrapper (snapshot gather + ESIMD kernel, nsd=8 geometry) under capture:
outputs + z + SSM-pool checkpoint rows + conv checkpoint rows BITWISE ==
eager for fp16/e4m3/e5m2 after full input mutation. The wrapper ops and
kernel are capture/replay-correct in isolation => the serve failure is
in the SERVE INTEGRATION of the native path under graphs (candidate:
vLLM graph padding — E2b is the first time the native kernel runs on a
PADDED multi-seq batch inside a decode graph; fp16 gate kept nsd==1 =
only the unpadded batch-1 graph, P28 bridge handled padding on its own
path).
LEG D (discriminator): POOLPATCH=patch_v126_poolpath_nsd1.py — native
fp8 ride restricted to nsd==1 (fp16-gate semantics), graphs+async ON.
RED => single-seq native-in-graph broken at wiring level; GREEN =>
padded/multi-seq-under-graphs interaction convicted. [running]

**LEG D verdict: GREEN (2026-09-29).** nsd==1-gated poolpath lane,
graphs+async ON, full p23f mode A: MATH wrong 0/80, serial 0/80, flips 0,
TOOL 30/30 clean, 158 s. Same battery on any-nsd posture = 68/80 wrong.
=> native fp8 spec kernel is CORRECT in the unpadded batch-1 graph;
the red surface is exactly nsd>=2 (padded/multi-seq spec decode graphs).

**P29E-2 — code forensics on the two padding vectors (live file reads,
no lane changes):**
- Vector A (spec_token_indx tail): pure-spec path writes
  `spec_token_indx = torch.arange(spec_token_size)` and does NOT
  sanitize the tail slots for padded spec rows -> stale tail values
  index qkvz/ba reads and out/z writes of DUMMY rows only (kernel line
  global_t = token_indx_ptr[state_row + t] drives qkvz/ba reads + out/z
  writes, lines 280/311/435/442/450). Dummy out rows are never read by
  the sampler. BENIGN for real rows.
- Vector B (dummy rows' state walks): dummy rows DO run the full state
  update (idx tail = NULL_BLOCK_ID=0 -> all 5 dummy slots of a padded
  seq walk pool/conv row 0 each round). block_pool.py:176
  `self.null_block = self.free_block_queue.popleft()` over ascending-id
  blocks => null block IS pool row 0, is_null=True, never returns to the
  free queue => NO real sequence can ever own pool row 0. fp16 lanes
  green since v88 under identical dummy walks. BENIGN.

**P29F — serve-exact padding emulation at op level (p29f_probe.py):
real rows BITWISE OK, all 3 dtypes.** Capture nsd=8 (rings 300+, arange
tok, acc=4), replay 3 real + 5 serve-style dummies (idx tail=0, acc
tail=1, tok tail stale, qkvz/ba tail garbage x300). Real out/z/pool/conv
rows bitwise == eager-on-same-tensors for fp16/e4m3/e5m2. Row-0
forensics: capture-exec does NOT write row 0; replay dummies DO write
row 0 (e4m3 row0 NaN 393216/2 per side identical under x300 garbage;
fp16/e5m2 replay-vs-eager differ benignly — 5 dummies racing the same
row, unowned null block). Confirms vectors A+B are not the corruption.

**P29G — graphed multi-round trajectory (p29g_probe.py): ALL GREEN.**
ONE capture per dtype, 30 replay rounds with fresh qkvz/ba per round,
acc cycling [4,2,3,1,5,2,4,3], ring rotation torch.roll(idx, acc, dim=1):
per-round out BITWISE, worst_cos 1.000000, ring pool + conv + FULL pool
bitwise vs eager twin. => frozen graph replaying rotated rings and
compounding state is op-level correct. Op level now EXHAUSTIVELY
exonerated: corruption lives in the SERVE INTEGRATION of the native path
(nsd>=2 graphs only).

**P29H — in-situ serve-level diagnostic (design + 4 failed boots, live).**
Design: instrumented poolpath variant where the layer-0 GDN native fp8
spec call is mirrored by a bridge reference on IDENTICAL live inputs
(fresh fp16 snapshot of touched pool rows + conv rows + clones), metrics
[maxdiff_out, maxdiff_pool, nan_out, nan_pool, qkvz_absmax, nsd]
accumulated in persistent device tensors updated IN-GRAPH every replay;
flush dumps the ring buffer to /root/p29h_diag_<pid>.pt. Boot lessons
(all reproduce on sight, 3 boot cycles):
1. v1: `.item()` in the flush during capture ->
   RuntimeError: wait method cannot be used for an event associated
   with a command graph. Host syncs inside capture are ILLEGAL, not
   just slow. (v1.2.23 async-path precedent: non_blocking D2H + pinned
   mirror is the legal pattern HOST-SIDE, but see 3.)
2. v2: `torch.zeros(..., pin_memory=True)` under the worker's default
   XPU device context -> RuntimeError: Only dense CPU tensors can be
   pinned. pin_memory NEEDS explicit device="cpu".
3. v3: `torch.dot` / `.norm()` inside capture -> same command-graph
   RuntimeError (reduction dispatch does an event wait). `abs().max()`
   reductions are PROVEN capture-legal (ran in PRE before the crash) =>
   all in-graph metrics are max-based; NO dot/norm/sum inside capture.
   Final v4 design: flush reads the device buffer ONLY in the eager
   prefill branch (decode-graph capture dummies never enter it).
4. v4/v5: instrumentation silently DORMANT — V126_P29H never reached
   the spawned TP workers. /proc/PID/environ: serve pid 402 hits=1,
   EngineCore 675 + workers 874/880 hits=0 — env inheritance between
   vLLM v1 process tiers is broken on this build (docker exec -d -e AND
   an export line in serve_user.sh both lost at the EngineCore spawn;
   numa_utils.configure_subprocess is a no-op here, scrub not run down).
   v4 lane still delivered the RED reference UNINSTRUMENTED: MATH 68/80
   wrong, serial 80/80, flips 75 — AND a NEW severity datum: the engine
   DIED mid-battery (EngineCore sample_tokens -> collective_rpc ->
   shm_broadcast acquire_read timeout -> EngineDeadError; workers stuck
   in 'sample_tokens' 603 s > 600 s watchdog; API server 500s then
   shutdown). Corruption -> garbage decode -> hard wedge, not just
   wrong answers.
   v6 fix: diagnostic activation switched from env var to MARKER FILE
   /root/.v126_p29h (os.path.exists in the layer gate; boot script
   touches it) — filesystem is process-tier-proof.

**P29H v6 RESULT — marker file healthy, corruption live, STILL zero
diag dumps.** Boot 19:55:48 HEALTH_OK with marker verified in place
before serve start; battery round 0 already bang-salad (`got='3!!!...'`)
= native fp8 path ACTIVE, so `_gdn_conv_decode` spec branch IS running
in the workers — yet no /root/p29h_diag_*.pt ever appeared. Static
exhaustion: import os at line 5, DIAG_INIT ~453 (marker gate 457, owner
set 465), flush guard 992 inside the prefill branch, PRE guard ~1107;
routing traced — forward_xpu 1224 has no prefill early-branch,
_gdn_core_and_output 1373 no early return, only early return in
_gdn_conv_decode is the `attn_metadata_raw is None` warmup check ~974;
process tree plain (396 serve -> 669 EngineCore -> 868/874 workers).
Something in the chain DIAG_INIT-runs -> owner-matches -> prefill-
branch-entered -> torch.save-succeeds is FALSE, and static analysis
cannot see which.

**P29H v7 (running) — comprehensive telemetry capture points (user
directive: "add comprehensive telemetry capture points to locate issues
from pipelines and etc").** mk_poolpath_p29h.py rebuilt with five
capture points, every one capture-legal:
- T1 ARMED print at DIAG_INIT (owner hex id + pid) — did init run in
  the workers at all?
- T2 CALL# trace at the head of the prefill gate (inside
  _gdn_conv_decode): self/owner ids + match= + npref/ndec/spec for the
  first 4 eager calls + first 12 prefill calls. Graph replays never
  execute Python, so lines = eager routing evidence; match=False
  convicts the owner guard; npref>0 with no FLUSH convicts the flush
  guard; no npref>0 lines = prefill never reaches _gdn_conv_decode.
- T3 SPEC# trace in PRE (first 8 spec-branch executions incl. the
  capture pass): nsd/nst/pool-conv-qkvz dtypes, host values only.
- T4 metrics ring widened 6 -> 8 columns: [md_out, md_pool, md_conv,
  nan_out, nan_pool, nan_conv, qkvz_absmax, nsd] — conv-row divergence
  added (conv cache is fp16 in every lane; if conv diverges but pool
  does not, the convicted stage moves upstream of the SSM pool).
- T5 FLUSH dump payload now self-describing: saves/ctr + ring + meta
  {pid, pool_dtype, conv_dtype, cols}. First divergent ring slot =
  convicted stage; nsd column = convicted graph size.

**P29H v7 RESULT — telemetry LIVE; boot died to a scope bug that is now
a law.** T1 ARMED fired in BOTH TP workers (marker file reaches worker
__init__ — activation mechanism proven). T2 CALL#0 fired with
`npref=64 ndec=0 spec=False match=True`: an EAGER prefill-shaped dummy
batch (the boot profile pass, ~28 s after ARMED, at decode-graph capture
start) DOES route through forward:769 -> forward_xpu:1396 ->
_gdn_core_and_output:1466 -> _gdn_conv_decode and reaches the prefill
gate with the owner matching. Then the boot DIED:
`UnboundLocalError: cannot access local variable 'ssm_state'` at the
FLUSH meta — ssm_state/conv_state are function locals bound only DEEPER
in the spec branch; at the prefill-gate site they do not exist.
=> FUNCTION-SCOPE LAW added to the capture-legality list: gate-site code
may reference only module globals and self attrs; anything else must be
stashed into globals from the spec branch (PRE now stashes dtype
strings; FLUSH meta reads them via .get()). v8 = v7 + this fix only.
Side datum: v7 crash also proves exceptions from _gdn_conv_decode
propagate unswallowed (no try/except around the dispatch).

**P29H v8 RESULT — THE CORRUPTION IS CONVICTED, IN-SITU.** v8 = v7 +
scope fix only; boot HEALTHY 20:39:32, full telemetry live (ARMED both
workers; CALL trace npref=64 eager dummy through forward:769 ->
forward_xpu:1396 -> _gdn_core_and_output:1466 -> _gdn_conv_decode;
SPEC trace shows the capture sweep nsd=64..60 pool=e4m3fn conv=fp16).
Battery: RED as designed (serial bang-salad, TOOL 30/30 engine death,
408 s) with 434 in-graph replays recorded. Ring analysis
(p29h_analyze.py, both TP dumps, 434/434 valid rows):
- nan_pool=1 on EVERY row, from row #0 (nsd=7) — the live e4m3 pool
  reads back NaN/Inf after the native spec call.
- TP1 row #0: md_out=47.01 FINITE, nan_out=0, nan_pool=1 — the fp16
  pre-call snapshot was finite (pool clean going in), inputs finite
  (qkvz_absmax 60-67) => NaN appears DURING the native call. Same-call
  native-vs-fp16-reference divergence 47 on ~60-scale activations =
  overflow class, not quantization noise.
- 364 nsd=1 rows all nan_out=1: serial bang-salad = reads of the
  ALREADY-POISONED pool. Explains LEG D green (solo lane never runs the
  wide graphs that poison) and why corruption appears "any-nsd": one
  nsd>=6 replay poisons rows every later batch reads.
- nsd coverage {1:364, 6:7, 7:63}; md_conv spikes (~60) co-located with
  the wide rows; conv cache itself stays finite (nan_conv=0) — the SSM
  pool write path is the sole NaN source.
=> MECHANISM: native spec kernel's fp8 SSM-pool state writes under REAL
activation magnitudes (absmax ~60, accumulated walk states) overflow
e4m3fn (max 448, overflow -> NaN) — op-level probes were green only
because synthetic inputs (randn*0.5, absmax ~2-3) were 20-100x below
real scale. NEXT (P29I): op-level real-scale sweep to pin the exact
overflow site (per-slot ring-row writes vs final checkpoint) -> root
fix in the kernel (clamp/saturate or scaled/side-buffer checkpoint),
rebuild .so (KERNELS_MAX_JOBS=52), re-run P23F must be GREEN.
Side solves: v6 no-dump mystery = FALSE-NEGATIVE check (host shell
expanded the /root/p29h_diag_*.pt glob before docker exec could see it
— same family as the KNOWN_ISSUES #28 grep -c trap); container-side
quoting required.

**P29I RESULT — op-level real-scale repro: the KERNEL IS INNOCENT; the
e4m3 FORMAT cannot hold the state. Corruption mechanism refined from
"kernel overflow bug" to "format range inadequacy".**
Run-1 (p29i_probe.py): pure-randn sweep, qkvz absmax up to ~198
(>serve's 60-67), 5 compounding rounds each scale, e4m3 pool + fp16
twin on identical starting values — ALL CLEAN (md_out <= 0.006, zero
NaN, ring rows finite). Magnitude alone does NOT convict; run-1 held
ba fixed at randn*0.5 (absmax ~1.7 -> dt=softplus ~1.9 -> increments
~90 << 448).
Run-2 (p29i_probe2.py), all vs fp16 twin, both pool dtypes:
- Sweep A ba scale 0.5->60 (ba_max up to 206, dt_max 206): zero NaN on
  BOTH dtypes — increments alone never crossed 448 under randn
  conjunctions (needs large dt AND beta AND B at the same element).
- Sweep C heavy-tail ba (0.05% outliers at +-U(200,2000), ba_max 1729,
  dt_max 1232): still clean. Same conjunction-rarity reason.
- Sweep B pool START scale 5->400 (state magnitude sweep, ba modest):
  e4m3 at seed absmax ~1280 -> NaN INSTANTLY at r0 (nan_out ~1e5/107k,
  ringnan 35/35) — INCLUDING the fp16 twin, because the XPU e4m3 CAST
  ITSELF produces NaN beyond 448 (no satfinite): the pool cannot even
  be SEEDED with legit-magnitude state. e5m2 holds 1280/2560 fine
  (range 57344 ~ 0.87x fp16) but md_out vs the fp16 twin grows to
  0.93-1.84 on large states = the 2-mantissa-bit quantization error —
  numerically matching the earlier e5m2 P23F tool-salad quality RED
  (format precision, not a bug).
=> VERDICT: the v131 kernel has no reproducible write bug (clean in
every synthetic sweep whose values are representable); the serve
corruption is the FORMAT: GDN SSM state legitimately exceeds e4m3's
+-448 (v8 ring: pre-state finite <=448, NaN appears DURING the call =>
real-activation increments cross 448 in-situ where randn conjunctions
cannot; real conv-output/B magnitudes far exceed randn*0.1-weight
synthetics). e4m3 state storage is numerically impossible without
changing the math (clamp-to-448 = saturated-state systematic
corruption, not a fix). e5m2 state storage is range-safe but
precision-broken (measured 0.9-1.8 out-diff on big states).
=> ROOT-FIX DIRECTION: fp16 SSM pool stays the certified posture (as
v1.2.25 ships); fp8 superiority must come from KV + compute path.
Only conceivable fp8-state path = per-head scaled-space kernel
arithmetic (rewrite state update in scaled coordinates + unscale on
read) — weeks-scale, quality-risky, documented as future plan, NOT a
clamp.
MISSING NUMBER for the verdict doc: the REAL fp16-lane state magnitude
(ground truth for what storage must hold; also decides whether e5m2's
57344 range would even suffice) => P29H v9 leg: running-max scalars
(touched-rows pool absmax pre/post, ba absmax) + flush print, booted on
the fp16 CERTIFIED posture (p29c_boot.sh float16 + P29H=1).
v8 ring dumps preserved to host lce1/p29h_diag_{869,875}_v8.pt before
container teardown.

**P29H v9 LEG — SECOND DEFECT DISCOVERED (pool-dtype-INDEPENDENT): the
poolpath lane WEDGES the engine under concurrent bridge load. The
running-max telemetry itself works; the lane died before the serial
(nsd=1 native) phase could advance it.**
Boot: v9 patch (fp32 device scalars _g_dn_p29h_pmx/_premx/_bmx updated
IN-GRAPH via in-place copy_(torch.maximum(...)); abs().max() = proven
capture-legal), fp16 posture, P29H=1 POOLPATCH v9 + v131 .so, LANE_UP
21:12:46, live_resets baseline 12.
p23f battery (p23f_p29h_v9fp16.log): concurrent phase answers
QUALITY-CLEAN (FLIP lines: 11/11 rows round0 concurrent='33' etc. —
correct multiplications; the 33/80 wrong = FAIL rows from the DEAD
engine, not corruption) for ~40 reqs, then ENGINE WEDGE: +2 XPU resets
(12->14), 'TimeoutError: RPC call to sample_tokens timed out' ->
EngineDeadError 21:19:30 (EngineCore pid 667), workers dead, TOOL 30/30
connection-refused, battery 315s. ctr=0 = NO native instrumented calls
ever ran: concurrent batches are nsd=8 -> on the fp16 lane E2B routes
nsd!=1 to the BRIDGE (uninstrumented), and the serial nsd=1 phase never
executed (lane died in concurrent phase). FLUSH prints observed only at
boot capture (saves 57->73, capture-dummy flushes).
POOLPATH IS BEHAVIOR-NEUTRAL ON FP16 (source-verified, patch_v126_
poolpath.py:85-175): E2B gate returns to bridge when nsd!=1 and pool is
not fp8 — identical to certified pre-patch behavior for the fp16 lane
(bit-for-bit gate semantics); E3 only widens the allowed pool dtypes;
E2C is a print. => poolpatch CANNOT cause an fp16 wedge by code path.
PRIME SUSPECT: the v131 .so under BRIDGE + nsd>1 CONCURRENT load — a
combination NEVER certified (P28 dualbridge was certified on the
PRE-v131 .so; the fp8 poolpath lanes never exercise bridge+nsd>1
because fp8 rides native at any nsd per E2B). Host reboot (standing
directive, +2 resets) before the bisection legs.

**P29H v9b — SERIAL WEDGE on FRESH HOST overturns the concurrency
hypothesis; GRAPH-CAPTURE ROUTING LAW established; suspect narrowed to
v131 .so INSIDE THE BRIDGE (fp16-pool).**
Host rebooted 21:31 (uptime 2 min on return; litellm-proxy auto-returned
via unless-stopped — untouched per directive). Fresh v9 fp16 lane
LANE_UP 21:37:55, live_resets=0. Serial-only magnitude probe
(p29_serial_mag.py, 24 sequential 512-tok requests, NO concurrency):
req00-04 COMPLETE with quality-clean answers (4.6-27.5s), then req05-23
ALL HTTP 500 — engine death at 21:46:11 with the SAME signature
('TimeoutError: RPC call to sample_tokens timed out' -> EngineDeadError,
+2 Engine resets on a fresh host = both from this wedge). ok=5/24.
=> CONCURRENCY IS NOT REQUIRED: the lane wedges under pure SERIAL load
after 5 requests. The "bridge + nsd>1 concurrent" framing is DEAD.
=> GRAPH-CAPTURE ROUTING LAW (from ctr=0 / pmx=0.0 through BOTH boots
incl. this probe era + dump meta frozen at 21:40): on the fp16 lane the
E2B nsd==1->native gate is evaluated ONLY at capture; capture dummies
carry nsd>1, so EVERY decode graph records the BRIDGE branch — serial
nsd=1 replays included. The instrumented native branch NEVER executes
(ctr=0, pmx=premx=bamx=0.0 across boot capture AND real traffic). On
fp8-pool lanes E2B routes native at ANY nsd, so capture records native
— exactly why the v8 ring filled from real replays there. Consequence:
v9 telemetry can never collect fp16-lane decode magnitudes; the
magnitude ground truth must come from an e5m2-pool lane (native at all
nsd, battery-stable) with the e5m2-rounding caveat (~2 sig bits — fine
for the 448/57344 RANGE question).
=> NARROWED SUSPECT: the fp16 lane's ENTIRE decode load (serial and
concurrent alike) executes dualbridge -> esimd_gdn_conv_fused_seq (the
.so) with an fp16 pool. Stable configurations measured: PRE-v131 .so +
bridge + fp16 pool (P28 certification, full batteries) and v131 .so +
native + fp8 pools (P29C e4m3/e5m2 batteries — those lanes never run
bridge-with-fp16-pool decode). Wedged: v131 .so + bridge + fp16 pool
(serial v9b: 5 reqs; concurrent v9: ~40 reqs). Single remaining
variable = the v131 .so in the bridge's fp16 seq-kernel path.
=> BISECTION LEG LAUNCHED: PRODSO=1 switch added to p29c_boot.sh (skips
the v131 docker cp + the v131-specific sha gate + the v131-specific
P29B op gate; production .so from v1.2.25-raw kept; all else identical
incl. poolpath + P29H v9). Verdict rule: battery GREEN => v131 .so
CONVICTED for the fp16 bridge wedge; RED => bridge/other. Artifacts:
p29_serial_mag_v9b.log, p29h_diag_{869,875}_v9b_serial.pt (ctr=0/pm=0
frozen evidence).

**P29H v9c/v9d — BISECTION ROUND: v131 .so ACQUITTED, dualbridge-on-fp16
ACQUITTED; conviction set collapses to {poolpath patch | P29H
instrumentation} on the fp16 lane.**
Matrix (all legs same host boot-cycle, all fp16, full p23f battery or
serial probe):
- LEG C PRODSO=1 (poolpath+p29h, PRODUCTION .so): WEDGED — concurrent
  round05 (~45 reqs), identical signature (sample_tokens RPC timeout ->
  EngineDeadError 21:59:49, +2 ccs resets). => v131 .so ACQUITTED (both
  .so variants wedge identically).
- LEG D NO_POOLPATH=1 (dualbridge + v131 .so, NO poolpath, NO p29h):
  **GREEN** — wrong=0/80, flips=0, tool 30/30 clean, OVERALL 128s, ZERO
  new resets. => the P28 dualbridge decode-only flat branch under graph
  replay WITH FP16 SSM POOLS is battery-certified GREEN (first time this
  cell was ever run — P28 certified the branch on fp8 pools only; the
  earlier suspicion "bridge+fp16 never certified" is now TESTED GREEN).
  Also green with the v131 .so present => v131 .so fully acquitted.
- E2B_OLD read in full: `or attn_metadata.num_spec_decodes != 1` —
  the nsd==1-only native gate PRE-EXISTED poolpath; E2B_NEW fp16
  semantics are textually IDENTICAL to E2B_OLD (nsd!=1 -> return False
  -> dualbridge; E3 same outcome on fp16; E2C print fires on fp8 only).
  => routing provably identical between leg D (green) and the wedged
  legs.
- CONVICTION SET REMAINING: the poolpath patch text and/or the P29H
  insertions (they ride together — mk_poolpath_p29h builds the p29h
  variant FROM the plain poolpath patch; every wedged leg ran the p29h
  VARIANT; the PLAIN poolpath on fp16 was never run before). On fp16
  the P29H executed subset = INIT allocs + CALL/FLUSH prints + FLUSH
  torch.save every 8th eager prefill (PRE/POST provably never execute:
  ctr=0/pmx=0 through capture AND traffic) — all eager, all identical
  in kind to the fp8-lane-stable v8 pattern. Source-level analysis
  CANNOT explain the wedge => empirical split required.
- LEG E LAUNCHED (decisive cell): fp16 + PLAIN poolpath (P29H=0) +
  full p23f. GREEN => P29H-on-fp16 convicted; WEDGE => poolpath-core
  convicted (capture-time semantics to then be root-caused).
- Boot-window anomaly noted: leg C (+2 bcs resets 21:49 during boot
  capture) and leg D (+2 by LANE_UP 22:08) showed boot-time resets
  while leg B showed live_resets=0 — reset noise during startup does
  not predict battery outcome (leg D green despite boot resets).

**P29H v9e — LEG E GREEN: P29H INSTRUMENTATION CONVICTED for the
fp16-lane wedge. Bisection CLOSED.**
LEG E = fp16 + PLAIN poolpath (P29H=0) + v131 .so: FULL p23f battery
GREEN — wrong=0/80, flips=0, tool 30/30 clean, OVERALL 128s, zero new
resets (22:30:08 lane, battery done by 22:36).
FINAL MATRIX (all fp16, identical routing — E2B_OLD carries the
nsd!=1 gate, so poolpath changes nothing on fp16):
  poolpath+p29h (v131 .so)  -> WEDGE concurrent ~40 (v9)
  poolpath+p29h (v131 .so)  -> WEDGE serial 5 (v9b)
  poolpath+p29h (prod .so)  -> WEDGE concurrent ~45 (PRODSO)
  no poolpath (v131 .so)    -> GREEN 0/80 + 30/30 (leg D)
  plain poolpath (v131 .so) -> GREEN 0/80 + 30/30 (leg E)
=> The P29H diagnostic overlay (mk_poolpath_p29h.py build on top of the
poolpath patch) is the wedge agent on fp16 lanes; poolpath itself,
the dualbridge (incl. its never-before-tested fp16-pool decode-under-
graphs cell), and the v131 .so are ALL battery-green on fp16.
PARADOX FOR ROOT-CAUSE: on fp16 the P29H executed subset is a STRICT
SUBSET of what P29H executes on fp8-pool lanes (which ran stable):
INIT allocs + CALL prints + FLUSH every-8th-prefill torch.save; the
PRE/POST spec-branch code provably never executes on fp16 (ctr=0,
pmx=0.0 through capture AND traffic on every wedged boot). A subset of
a stable set wedging = the mechanism is NOT straightforward code-path
execution (candidates: capture-time interaction of the inserted
text/global state with graph recording on the bridge path; the
owner-gate owner!=self layers taking subtly different gate outcomes;
async-scheduling interplay of the flush's D2H under concurrent
prefill; or an allocation-timing effect of the lazily-allocated
diagnostic buffers). Dead lanes' serve logs were lost to successive
container teardowns before full-traceback capture — hang-site evidence
limited to the EngineCore sample_tokens RPC-timeout signature.
=> SHIP DECISION (stands regardless of the unresolved micro-mechanism):
P29H/diagnostics NEVER SHIP — production posture excludes the entire
diagnostic overlay. The v126 ship stack on fp16 = certified v1.2.25 +
plain poolpath (green, leg E) or even bare dualbridge (green, leg D);
given fp8 SSM pools are format-impossible (P29I), poolpath's fp8
features are moot for production and the MINIMAL certified posture
(leg D class: standing stack + dualbridge only, matching what P28
certified) is the ship candidate. P29H wedge root-cause = recorded as
KNOWN_ISSUES diagnostic debt (diagnostic-lane-only impact).
NEXT: e5m2+P29H magnitude leg (fp8 pool -> native at any nsd ->
scalars advance from request 1; robust to the wedge since even pre-
death FLUSH prints carry the running maxes) to bank the REAL state
magnitudes for the P29I format-impossibility verdict.

## P29H v9f — e5m2 magnitude leg: "records nothing" ROOT-CAUSED
(snapshot semantics — NO telemetry defect) + WEDGE EXTENSION to e5m2
Evidence (lane: p29c_boot fp8_e5m2 + P29H v9 + v131 so, LANE_UP
22:42:37; serial probe 24x512 ran ~22:44-22:52, answers garbage
'!!!!' = the expected P23F e5m2 precision RED, re-confirmed live):
- TRACE LINES: 'P29H ARMED' x1/worker (owner=first instance,
  0x741255924da0/0x70269901cb90, pids 869/875); CALL trace shows the
  method is entered by ~12 distinct layer instances per worker per
  batch — owner matches on exactly ONE (CALL#0 match=True, all others
  False — by design, one reference layer); SPEC#0..7 x1/worker fired
  (nsd=64 nst=5 pool=e5m2 conv=fp16 qkvz=fp16 — the owner's native-
  branch executions at the capture ladder, PROOF PRE ran eagerly);
  NATIVE marker x96 (all 96 GDN layer instances took native >= once);
  FLUSH x18, saves 1->65 ALL DURING the capture ladder (FLUSH lines
  interleave with the tqdm; last dump mtime 22:45:08 = capture end),
  then saves FROZE for the whole probe.
- THE PUZZLE + RESOLUTION: dumps read saves=65 ctr=0, buf all-zero,
  pmx/premx/bamx=0.0 — looks like the telemetry never recorded, but
  PRE provably ran and no exception can have fired (only one 'except'
  exists in the whole patched file and it is the __init__ autotune
  warmup guard; an uncaught exception would have crashed the engine,
  yet 24 answers were served). ROOT CAUSE = SNAPSHOT SEMANTICS: the
  FLUSH dumps are CAPTURE-TIME snapshots. ctr/buf/pmX are updated by
  device ops RECORDED at capture which only EXECUTE on replay — and
  the readout window (FLUSH at the eager prefill gate INSIDE
  _gdn_conv_decode) can never open during serial traffic: pure-prefill
  steps route through the prefill path and never enter the decode
  gate, decode steps replay graphs without running Python. saves
  1->65 came exclusively from capture-ladder mixed dummies. The
  post-replay values (thousands of accumulated maxes) died with the
  worker — undumped. The v9e plan assumption "pre-death FLUSH prints
  carry the running maxes" was FALSIFIED. buf's nsd-column argument
  sealed it: an eager POST tail writes row0=[0..0, nsd=64.0] != 0;
  buf row0 == 0 => POST's tail never even once executed EAGERLY —
  exactly right, because PRE/POST sit in the NATIVE spec branch which
  under XPU capture records without executing; the ONLY eager native
  executions are the capture-ladder warmups whose dummy pool rows are
  zeros (hence premx=bamx=0.0 is the CORRECT capture-time value).
- WEDGE EXTENSION: the e5m2+P29H lane WEDGED at 22:52:14 with the
  IDENTICAL signature (sample_tokens collective_rpc -> shm_broadcast
  TimeoutError -> EngineDeadError -> 500-cascade) as every fp16+P29H
  leg (v9, v9b, PRODSO). P29H is now convicted on BOTH pool dtypes.
  On e5m2 the P29H footprint additionally includes the RECORDED
  reference-kernel + ring ops replaying every decode step; on fp16
  the traffic-time footprint is provably near-inert (Python int
  increments at the FLUSH gate) — the strict-subset paradox stands
  for fp16, now with an even sharper edge (diagnostic debt,
  KNOWN_ISSUES; ship-unaffected).
=> READOUT LAW (new, permanent): never read device state from the
forward path — host syncs are async-scheduling-hostile AND gate sites
are unreachable on serial lanes. Readout belongs to a TIMER THREAD.

## P29M v10 — magnitude-minimal leg (supersedes P29H for magnitudes)
mk_poolpath_p29m.py (generator; patch_v126_poolpath_p29m.py): strips
P29H to the magnitude essentials — PRE: premx=max|pool[spec rows]|,
bmx=max|ba| (fp8 pool cast to fp32 BEFORE abs(); abs() on float8 is
not universally supported); POST: pmx=max|pool rows| post-kernel; NO
ring, NO reference kernel, NO FLUSH (no forward-path host syncs at
all; replay overhead = 3 index_select/max ops on the owner layer).
DIAG_INIT starts a DAEMON TIMER THREAD that every 20 s dumps
{t,pid,pmx,premx,bamx} to /root/p29m_mag_<pid>.pt + prints a TICK
line — wedge-proof by construction (dumps land before any death).
Activation reuses the /root/.v126_p29h marker (p29c_boot.sh P29H=1).
Boot: P29H=1 POOLPATCH=/root/build/patch_v126_poolpath_p29m.py
p29c_boot.sh fp8_e5m2 8000 fp8_e5m2; then p29_serial_mag.py; harvest
p29m_mag_*.pt + TICK lines. e5m2-rounding caveat (~2 sig bits)
acceptable for the RANGE question (|state| vs e4m3's 448 / e5m2's
57344) that the P29I format verdict documents.

## P29M RESULT — magnitude ground truth BANKED: e5m2 SSM pool = NaN
from the FIRST traffic tick (09-29 23:07-23:20)
Lane: P29M v10 (6 edits applied, py_compile OK), LANE_UP 23:09:58,
P29B PASS, live_resets=12 at boot AND 12 after the probe (zero new —
NaN is a numeric event, not a hardware event). Serial probe
24/24 requests HTTP 200 (answers garbage — expected e5m2 RED).
TELEMETRY WORKED EXACTLY AS DESIGNED: 82 TICK lines; all-zero during
capture (correct — recorded ops don't execute until replay, dummies
are zeros); first traffic tick after capture END ALREADY READS
pmx=nan premx=nan with bamx=15.57 (finite). NaN onset bracketed to
the first ~20 s of traffic — premx=nan means the pool contained NaN
BEFORE the owner layer's first spec gather, i.e. NaN entered via the
first prefill-written states and/or the first spec update (prefill
bridge scatter casts fp16 chunk_gated_delta_rule states to e5m2; the
v131 fp8 register chain computes the recurrence in e5m2 — either
overflow-to-inf (e5m2 has inf/NaN encodings) or precision-collapse
then normalize/divide -> NaN; recurrence then NEVER clears it).
FINAL NUMBERS for the P29I format verdict:
  - |ba| (gate inputs, fp16): runmax 16.0-20.2 — normal range, e5m2-
    representable, NOT the problem.
  - SSM state on e5m2: NaN within the first traffic tick — the GDN
    state recurrence is NOT representable in e5m2 on this stack.
    (= the live mechanism behind the '!!!!' garbage answers: NaN pool
    -> NaN core_attn_out -> junk sampling, from request 1.)
  - e4m3 side (P29I kernel evidence, prior): legit SSM states >=1280
    vs XPU e4m3 cast NaN beyond 448, no satfinite; pool cannot even
    be seeded at absmax 1280.
=> BOTH fp8 dtypes are FORMAT-IMPOSSIBLE for GDN SSM state. "Full
fp8 pipeline e5m2->e5m2 / e4m3->e4m3" for the SSM component is
mathematically impossible on this hardware/stack; the certified fp16
GDN pool + fp8_e4m3 KV (v1.2.24/25 posture) is the ENDGAME. Only
conceivable future = per-head scaled-space kernel arithmetic
(weeks-scale; documented as future plan, not attempted).
BONUS WEDGE DATA: the P29M lane SURVIVED the full probe (health 200
after 24/24; thread still ticking) where every P29H lane died — P29M
= P29H minus ring/reference-kernel/FLUSH. The convicted wedge surface
narrows further to that machinery (diagnostic-only; P29H still never
ships; fp16 strict-subset paradox remains diagnostic debt).

## P29S — ship-matrix adjudication: v131-.so TAINT + cadence ground truth
(2026-09-30 00:0x)

The 09-29 23:35-23:55 "ship-posture" speed matrix (solo 26.83/30.95/31.03,
4st 111.45/139.71/117.42, 8st 244.37/260.48/258.57) vs Run-7n cert16s4
(181.81/244.88 agg, 75.06/74.89 solo) initially read as a 2x solo/4st
regression. ADJUDICATION:

1. METHODOLOGY MISMATCH (real but secondary): bench_genspeed.py used
   short-task prompts that EOS at ~276<512 (TTFT overweighting) and did
   NOT set enable_thinking=False; Run-7n (p22c5_agg2.sh) used
   "Count from 1 to 1000.", thinkOFF, agg 4x256 x2 + solo x200 x2.
2. DEcisive probe p29s_cadence.py (streaming, per-token timestamps):
   gap p50=131ms, p90=132ms, max=135ms — DEAD FLAT, zero stalls,
   TTFT 0.20-2.56s (small). Solo 31 tok/s with a contended stream
   measured at 63 tok/s (impossible for real resource contention).
   => NOT queueing/EOS: the decode STEP itself = 131ms vs certified
   ~43-48ms. 3x step-time.
3. ROOT CAUSE: the "ship-posture" lane was booted WITHOUT PRODSO=1 —
   live .so sha e050551e... = the V131 DIAGNOSTIC .so (P29A fp8 register
   chain + P29B-FIX ring-row snapshot). The matrix convicted a
   diagnostic kernel, not the ship stack. Preserved v131-lane spec
   metrics (shipmat1_v131so_serve_full.log): acceptance length 3.21-3.28,
   per-position 0.79/0.62/0.48/0.37 — HEALTHY => slowdown is pure
   per-step kernel cost (ring-row snapshot ~+7ms x ~12 GDN layer calls
   ≈ the +88ms/step), not speculation collapse.
4. VERDICT: matrix INVALID for ship adjudication. Lane relaunched as
   SHIPMAT2 = PRODSO=1 NO_POOLPATH=1 float16 (production .so, dualbridge
   only) and the matrix re-run with bench_run7n.py (exact Run-7n
   methodology replica: agg 4x256 x2 + solo 200 x2 + optional 8x256 x2).
   Lesson recorded: EVERY ship-posture measurement must assert the
   production .so sha (PRODSO legs) before numbers are banked.

## P29S-2 — SHIPMAT2 matrix: fp8-KV posture SUPERIOR to fp16 baseline
(2026-09-30 00:1x)

SHIPMAT2 lane (PRODSO=1 production .so sha 1d9dcf4e, NO_POOLPATH=1
dualbridge-only, fp16 SSM + e4m3 KV + async + MTPx4, LANE_UP 23:52:06,
resets 0):
  cadence: gap p50=50ms (vs 131ms on v131 .so) — certified range restored;
  production .so acquitted of the entire 3x step cost (v131 ring-row
  snapshot convicted by difference).
  bench_run7n.py (exact Run-7n methodology):
    agg4x256#1  236.80 tok/s (baseline 181.81, +30.2%)
    agg4x256#2  253.74 tok/s (baseline 244.88, +3.6%)
    solo200     75.58 / 75.69 tok/s (baseline 75.06 / 74.89, +0.7/+1.1%)
    agg8x256    423.11 / 422.93 tok/s, per-stream 52.9 x8 (new axis,
                high-water mark for aggregate on this hardware)
VERDICT: the v125 fp8 deficit (n2e4m3s4 solo -8.9%) is INVERTED — the
ship posture (fp16 GDN pool + fp8_e4m3 KV + dualbridge + production .so
on the async/MTPx4 base) is >= the fp16 control on EVERY measured axis,
above on all of them. Round governing gate "fp8 superior of baselane on
any possible cases": SPEED leg now PASS (capacity +3.50% banked at
v1.2.25; correctness legs = P30 battery next on this same lane).

## P29T — P30 battery ABORT_P195_OPS: dualbridge consumed the v124 P19.5a
lineage site (2026-09-30 00:5x)

The first P30 battery run (validate_v1226_run.sh, launched 23:57:07 on
SHIPMAT2) aborted at the v124 lineage asserts:
  v124 deltas: p16=2 p195_cache=1 p195_mamba=1 p195_ops=0 mamba_fp16=1
  mamba_fp8=0 -> ABORT_P195_OPS
ROOT CAUSE (patch defect, lane healthy): patch_v126_dualbridge.py REPLACES
the v124 P19.5a fp8 ssm-bridge block in _xpu_ops.py (the dual bridge is its
successor — same gather/run/scatter shape for the ssm pool, extended to the
conv pool), and it both dropped the P19.5a marker comment AND asserted
`"v124 P19.5a" not in chk` — actively consuming the lineage tag that the
validation battery (P195OPS gate) and ship gates (v124_p195_sycl_bridge)
assert. Everything before the abort was GREEN: sanity+admission+v125
posture PASS 00:01:29, opsall/c7/fixline deltas =1/1/1, dualbridge
marker=1, C7 functional refusal PASS, P23F probe A fully clean (fresh math
0/80, serial 0/80 flips=0, tools 30/30), sustain drill + serialized legs
SURVIVED (6 SURVIVED markers by abort time), tracebacks/resets clean.
FIX (three-part, in patch_v126_dualbridge.py):
  1. NEW1 (the inserted dual-bridge comment) now opens with a lineage note
     carrying the v124 P19.5a tag.
  2. The "v124 block must be gone" assert was retargeted to the old
     comment's unique phrasing ("P19.5a: fp8 SSM cache bridge") — the old
     BLOCK must still be gone, the lineage TAG must now be PRESENT
     (new assert count >= 1).
  3. The ALREADY_APPLIED path self-heals: on a pre-2026-09-30 dualbridged
     file lacking the marker, it inserts the same lineage note above the
     bridge header + py_compile (idempotent).
VERIFIED: heal path run on the live SHIPMAT2 lane -> "ALREADY_APPLIED +
v124 P19.5a lineage restored"; P19.5a=2, DUAL BRIDGE marker=1, unset
import IMPORT_OK. Fresh-apply path picks the fix up automatically on the
next lane boot (p29c_boot.sh docker-cps the patch from /root/build).
Lane relaunched fresh (shipmat2b, same PRODSO=1 NO_POOLPATH=1 float16
posture) so the full battery re-runs end-to-end on a genuinely fresh boot
(P23F probe A freshness requirement).

## P30 — full validation battery on the ship posture: ALL GATES PASS
(2026-09-30 01:08-02:03)

Run 2 (fresh boot shipmat2b, PRODSO=1 NO_POOLPATH=1 float16, prod .so
1d9dcf4e, dualbridge WITH the P29T lineage fix) — COMPLETE, every gate
green (run 1 aborted at ABORT_P195_OPS, root-fixed in P29T):
  sanity+admission+v125-posture PASS 01:13:09
  code deltas: opsall=1 c7(v126)=1 fixline=1 dualbridge=1
  C7 functional: =2 refuse+raise rc=1, unset import clean
  P23F probe A (fresh boot): MATH 0/80 wrong, SERIAL 0/80 flips=0,
    TOOL 30/30 clean
  sustain drill: SUSTAIN_COMPLETE_NO_WEDGE (3 rounds)
  serialized 24 x3: SURVIVED ok=24 fail=0 (all three legs)
  burst_harsh x3: 36/36 each, resets_after=0 (a, b, c)
  parser battery (qwen3_coder, engine-direct): BATTERY PASS
  JIT recheck: lines=8 (first-use loads), triton cache 80 vs raw 79,
    delta=1 (bound 2, v1222-raw precedent) PASS
  v124 deltas: p16=2 p195_cache=1 p195_mamba=1 p195_ops=2 (lineage fixed)
    mamba_fp16=1 mamba_fp8=0
  P23F probe B (post-battery): 0/24 wrong, tools 30/30
  tracebacks 0, engine resets 0
  solo genspeed 78.80/79.80/79.70 (async floor 70) PASS
  async 4x1024 aggregate=167.67 tok/s (acceptance >=100; sync was 72-74)
    — new high-water aggregate for this probe
  runtime fp8 resolution: float8_e4m3fn float8_e5m2 (both formats
    selectable end-to-end)
  === VALIDATE_V1225_RUN ALL GATES PASS 02:03:00 === (caps literal is the
  v1225 rename carryover; this is the v1226 battery log)
ROUND GATE STATUS: speed leg PASS (P29S-2), capacity +3.50% banked
(v1.2.25), correctness legs ALL PASS (this battery) — the governing
"fp8 superior of baselane" gate is now fully green on the exact posture
being baked. Next: P31 bake v1.2.26-raw.

## P31 — bake v1.2.26-raw
(2026-09-30 02:05-)

Run 1 ABORTED 02:17:03 on exactly one gate:
  GATE-FAIL dualbridge_patch_exit expected=1 got=0
ROOT CAUSE (output-contract defect, NOT a patch-logic defect): the bake
gate greps the patch's `tail -1` for caps status tokens
(V126_DUALBRIDGE_OK / V126_DUALBRIDGE_ALREADY) — mk_stage5_bake_v1226.py
modeled the gate on the v125 opsall/C7 gates whose patches print such
tokens, but patch_v126_dualbridge.py printed only lowercase human text
("APPLIED + py_compile OK" / "ALREADY_APPLIED") as its final line. The
patch itself applied correctly (dual_base_clean=0, marker=1, C7 marker=1,
all downstream gates incl. warm rounds and the full v1218..v1226 lineage
chain GATE-OK; the single GATE-FAIL was the only one in the whole log).
The abort fired BEFORE docker commit — no v1.2.26-raw image, no
repro_bootV1226.sh; host verified clean (no lsv-bake/lsv-test, :8000
free, only v1.2.25 / v1.2.25-raw images present).
FIX (v125 convention enforced): patch_v126_dualbridge.py now prints the
machine-greppable caps marker as its FINAL line on every success path —
V126_DUALBRIDGE_OK (fresh apply), V126_DUALBRIDGE_ALREADY (both ALREADY
branches, plain + lineage-heal). Error paths (PARTIAL_STATE rc=1, anchor
ABORT rc=2/3) keep lowercase text so tail -1 correctly fails the gate.
LESSON (now standing law): every patch consumed by a gate script MUST
emit its caps status marker as the final output line — inventing gate
literals the patch never prints is a generator defect the bake only
catches at apply time; post-generation audits must check BOTH sides of
every output-consuming gate (script literal AND patch print).
Run 2 relaunched 02:4x with the fixed patch: dualbridge_patch_exit=1
GATE-OK, zero GATE-FAIL through the warm rounds.

Run 2 COMPLETE — === BAKE DONE (image committed, boot script ready)
02:41:34 ===: 101 GATE-OK / 0 GATE-FAIL, no ABORT lines.
  image: llm-scaler-exp:v1.2.26-raw = e805da1449b0
    (sha256 e805da1449b074ce0a891b4f6dafd19fd095f6e18a935c38d121f06b601a1510)
  boot script built: /root/build/repro_bootV1226.sh (7285 B, async-REQUIRED
    lineage, .llm_scaler_exp_v1226_baked marker gate + V1226 DUALBRIDGE
    marker gate DBN baked in)
  warm rounds: JITWARM_OK; warm_ext wall=26.6s; verify replay wall=11.9s
    rc=0; xgrammar t2 200 + PARSED_JSON; sampler default/top_p/top_k/all
    all HTTP=200
  lineage: pedigree markers v1218 v1220 v1221 v1222 v1223 v1224 v1225
    all present; jit warm stamps v64 + v1220..v1226 all present;
    xgrammar=0.2.7; serve_spec_mtp4=1
  baked_serve_config (gmu0.8 bs64 e4m3 pc-ON spec-ON mnbt8192 async-ON
    mamba-fp16 no-template) — the certified v1225 posture + dualbridge
  no_bake_scripts_left=0 (bake artifacts cleaned inside the image)
Next: host reboot (standing directive) -> fresh boot repro_bootV1226.sh
-> fresh-boot sanity/admission/posture + validate_v1226_run.sh re-run on
v1.2.26-raw -> ship_v1226.sh chain.

### P31b — fresh-host raw validation: ALL GATES PASS
(2026-09-30 02:53-03:57)

Host rebooted 02:45 (up 5 min at 02:53, litellm-proxy auto-returned as
designed). Fresh boot repro_bootV1226.sh RAWVAL from v1.2.26-raw:
  BOOT_RAWVAL baked config verified (gmu0.8 bs64 e4m3 pc-ON async-ON)
  BOOT_RAWVAL serve started 02:58:36, HEALTH_OK after ~170s,
  live_resets=0; dualbridge boot gate passed (script exits 9 if the
  DUAL BRIDGE marker is missing on the booted image)
  (first RAWVAL attempt printed the usage line — the boot script takes
  a free-form MODE label; relaunched with MODE=RAWVAL)
Battery validate_v1226_run.sh 03:04->03:57 — ALL GATES PASS 03:57:43:
  sanity+admission+v125-posture PASS 03:08:26
  deltas: opsall=1 c7=1 fixline=1 dualbridge=1
  C7 functional PASS (refuse+raise on =2, clean import unset)
  P23F fresh: MATH 0/80 wrong, SERIAL 0/80 flips=0, TOOL 30/30 clean
  sustain 3 rounds + serialized 24 x3 + burst a/b/c all passed (phase
    asserts inside the battery; artifacts burst_v1226_a/b/c.log
    overwritten 03:48/03:49/03:56 this run; any phase failure aborts —
    run 1 proved the abort path fires)
  jit_recheck lines=8, triton cache 80 vs raw 79, delta=1 (bound 2)
  v124 deltas: p16=2 p195_cache=1 p195_mamba=1 p195_ops=2
    mamba_fp16=1 mamba_fp8=0
  P23F post-battery: 0/24 wrong + 30/30 tools
  solo genspeed 78.50/79.72/79.65 (floor 70)
  async 4x1024 aggregate=147.66 tok/s (acceptance >=100)
  tracebacks 0; the only "reset" strings in serve_full.log are 4 v29
    boot-time allocator-reset INFO lines (pre-traffic, benign)
  runtime fp8 resolution: torch.float8_e4m3fn torch.float8_e5m2
  SANITY finish=stop content head READY-v1226
  === VALIDATE_V1225_RUN ALL GATES PASS 03:57:43 === (caps literal is
  the v1225 rename carryover; this is the v1226 battery on v1.2.26-raw)
v1.2.26-raw is battery-certified on a fresh host. Next: ship_v1226.sh
(SHIPWARM relaunch -> warm rounds -> lane-commit v1.2.26 ->
gates_v1226_lane.sh -> prod sed + fresh-boot verify + CC battery ->
watchdog repoint + re-arm).

### P31c — ship run 1: gates FAIL on v124_p195_sycl_bridge (==1 vs >=1)
(2026-09-30 04:06-04:15)

Ship run 1: SHIPWARM boot 04:08 + warm rounds + lane-commit OK
(llm-scaler-exp:v1.2.26 = 29257c1e5034, 24.7GB) — then section-4 gates
aborted: exactly ONE failing gate in the whole battery:
  GATE-FAIL v124_p195_sycl_bridge expected=1 got=2
ROOT CAUSE (gate-semantics mismatch, NOT an image defect): the inherited
v1225 ship gate greps _xpu_ops.py for 'v124 P19.5a' and demands EXACTLY 1
— true on v1225, where the site held the single v124 marker line. The v126
dualbridge replaces that block and carries the tag in its P29T lineage
comment, where the phrase appears TWICE ("supersedes the v124 P19.5a fp8
SSM cache" + "The v124 P19.5a marker is retained") -> got=2. The battery's
P195OPS assert was already fixed to >=1 in P29T (validate_v1226_run.sh:
[ "$P195O" -ge 1 ]; p195_ops=2 PASSed both battery runs) — the ship gate
is the same assertion one link later and missed the same update. Every
other gate was GATE-OK incl. all v126 deltas (dualbridge marker=1,
v125-C7 gone, prod .so exact sha, P29 absence, C7 functional 5/5,
wheel, image ids, full v1218..v1226 lineage, cache floor 79>=79).
FIX: mk_gates_v1226_lane.py transform 3b — v124_p195_sycl_bridge gate
count becomes >=1 semantics (awk '($1>=1)?1:0'), matching the battery;
uniqueness of the v126 surface stays enforced by v126_dualbridge_marker
and v126_c7_marker (both ==1). The patch and the baked image are
UNCHANGED (repo patch == baked bytes; no re-bake needed — this is a
gate-side semantic fix only). gates_v1226_lane.sh regenerated on host,
bash -n OK, line 142 verified.
LESSON: when a marker's SITE moves (v124 block -> v126 lineage comment),
EVERY consumer of that marker across the chain (battery, ship gates,
patch asserts, boot gates) must be re-audited in the same edit — the
P29T fix updated 3 of 4 consumers and the 4th surfaced only at ship time.
Ship run 2 launched 04:2x (watchdog verified inactive + .paused marker
present — no relaunch window risk; SHIPWARM does docker rm -f lsv-test
itself).

### P31d — ship run 2: same gate, new failure mode (awk INSIDE vs OUTSIDE sh -c)
(2026-09-30 04:25-04:34)

Run 2 re-committed v1.2.26 = 872c1b5f8b8c and failed the SAME gate with a
DIFFERENT signature: `GATE-FAIL v124_p195_sycl_bridge expected=1 got=`
(got EMPTY, not 2).
ROOT CAUSE (quoting-layer defect in my P31c fix): the >=1 transform piped
awk OUTSIDE the `docker exec ... sh -c "..."` quotes and kept the `\$1`
escaping — that form belongs to the INNER sh -c context (where the other
v126 gates correctly live, e.g. v126_dualbridge_marker which PASSED).
Outside sh -c, the generated file literally contains awk '{print (\$1>=1)?1:0}'
-> awk receives backslash-dollar -> syntax error -> empty stdout -> got=.
FIX: bare `$1` in the outer awk (single-quoted in the file, so the gates
bash never expands it). Regenerated + verified STANDALONE before ship
run 3 (post-generation test discipline): line 142 shows bare $1;
`echo 2 | awk '($1>=1)?1:0'` -> 1 and `echo 0 | ...` -> 0; committed-image
grep count via `docker run --rm --entrypoint grep v1.2.26` = 2 -> gate
will read 1. 
LESSON (quoting-layer law): the \$1 escaping is a property of the sh -c
INNER context, not of "gates awk" generally — when moving a pipeline
fragment across a quoting boundary in a generator, re-derive the escaping
from scratch and test the exact file line standalone (echo-pipe) before
any consumer run.

### P31e — ship run 3: SHIP_V1226 ALL GREEN (v1.2.26 SHIPPED)
(2026-09-30 04:36-05:02)

Run 3 complete:
  === SHIP_V1226 start 04:36:20 ===
  SHIPWARM boot + warm rounds + lane-commit OK ->
    llm-scaler-exp:v1.2.26 = 92cf94b232e1 (24.7GB)
    (run-1 commit 29257c1e5034 and run-2 commit 872c1b5f8b8c are now
    dangling untagged — same content lineage, superseded by run 3)
  === V1226 SHIP GATES: ALL PASS 04:45:06 ===
    (v124_p195_sycl_bridge now reads count=2 -> awk >=1 -> 1: GATE-OK;
    every other gate identical to run 1/2 = GATE-OK)
  --- 5. fresh-boot verification ---
  prod fresh-boot sanity + admission + v123 posture + v125 opsall +
    v126 dualbridge PASS (dualbridge marker on the RUNNING prod
    container re-verified post-ship: count=1)
  --- 6. CC battery through litellm :4000 ---
  litellm_liveness=200; cc_t1/t2/t3 http=200 blocks=text;
    cc_thinking=thinking — GREEN
  --- 7. watchdog repoint + re-arm ---
  lane_watchdog.sh.pre_v1226 backup; relaunch line repointed to
    repro_bootV1226_prod.sh (count=1 verified); pause marker removed;
    service active
  "SHIP WARN: no fresh armed-line yet" — BENIGN, root-caused: the
  watchdog loop is sleep 60 per cycle and logs 'alive armed' every 10
  CYCLES = 10 MINUTES (plus a 420s post-launch check grace), so the
  ship's early check (04:56) simply preceded the first mark; confirmed
  at 05:02:27: "alive armed code=200 cycles=10". The ship comment
  "10-cycle = 100s cadence" carries a stale cadence assumption from an
  older watchdog version.
  === SHIP_V1226 ALL GREEN 04:56:14 ===
FINAL POSTURE (production):
  image llm-scaler-exp:v1.2.26 = 92cf94b232e1, lane lsv-test UP,
  health=200, watchdog ARMED on repro_bootV1226_prod.sh
  image lineage: v1.2.26 92cf94b232e1 <- -raw e805da1449b0 (bake from
  v1.2.25-raw 4e83528bfd66); v1.2.25 3f3c91637692 retained
Round COMPLETE: fp8-superiority gate closed by measurement (P29S-2
speed + capacity +3.50% v1.2.25 + correctness P30/P31b), dualbridge
shipped, spec MTPx4 + XGrammar-2 0.2.7 intact everywhere. Remaining:
docs (round writeup, README v1.2.26 section, KNOWN_ISSUES #30), memory
update, git commit.






