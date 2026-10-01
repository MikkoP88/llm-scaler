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







## P32 — multi-agent hard-stall deep-dive + v127 program start (2026-09-30)

Task: "do deep analyze there is now running multiple agents and has major
issue multiple sessions has hard stalling, this has to be solved, using any
possible solutions" + start the weeks-scale scaled-space fp8 SSM kernel
project + "Start implementing improvements but currently do not stop running
vllm instance."

Live triage at 12:00 UTC (lane NOT touched — read-only):

- Lane up: lsv-test (Up 3h) health=200, dmesg Engine resets 0 (current
  boot), watchdog alive cycles=180, load 3.58. litellm-proxy up 3h,
  error/timeout lines 0.
- Engine counters (cumulative): 247 requests, avg prompt 48,214 tok
  (53% 5-50k, 47% 50-100k); prefix hit 91.4% (10.67M/11.67M); local_compute
  1.003M tok = ~10 full 100k re-prefills; TTFT p50 ~4s p90 ~15s, 8 reqs
  40-160s + 2 reqs 160-640s; TPOT avg 62.8ms, NO request >0.75s/token;
  KV 16.4%, waiting 0, waiting_by_reason capacity/deferred 0.
- VERDICT: decode flows; the "hard stall" = TTFT tail from full-prefix
  re-prefills under the multi-agent long-context regime (48k avg prompts,
  ~493k-token KV pool fits only ~4-5 resident 100k sessions).
- ROOT CAUSE A (config drift, the trigger): serve at 08:56 launched
  MANUALLY (.bash_history 1072-1080, pts/1) — model mount
  /models/swift-qwen3.8-27b (52GB BF16, no quantization_config) +
  --quantization fp8 (on-the-fly) + --chat-template
  chat_template_qwen38_high.jinja + serve stdout NOT captured
  (serve_full.log untouched since 04:44 = telemetry blind). Four
  deviations from the certified v1.2.26 posture.
- ROOT CAUSE B (load regime beyond certified envelope): every certified
  battery uses <=1024-token contexts; v63/v64 tuned on that regime.
- EXONERATED: GPU wedge (resets 0; f15b_dmesg reset lines = historical
  baked content), 09:10 f15b capture = BOOT-PHASE false-fire (py-spy #1
  stack _init_executor, #2 idle run_busy_loop), preemption/retraction 0,
  spec acceptance 3.19 healthy.

v127 program (perf-v127/): STALL_ROOT_CAUSE.md + FIX_AND_TEST_PLAN.md
(4 layers) + cc_fleet_replay.py (new long-context replay gate: 6 streams,
20k base ctx, 8 turns, TTFT/TPOT/re-prefill counters; PASS = p99<20s,
max<45s) + boot_v1227_restore.sh (window restore script: certified chain +
knob legs V1227_MNBT/GMU/MAXSEQS/V63_BUDGET + drift assertions + caps
BOOT_V1227_RESTORED) + watchdog_v2_drift_check.sh (WARN-only drift guard:
image/model/--quantization/--chat-template/async/KV/spec/log-freshness)
+ scaled-space kernel project START:
  M0 calibrate_offline.py (per-feature runmax -> scale=0.98*448/runmax,
  e4m3 representability report; demo prior from P29M constants)
  M1 patch_scaled_space_v127.py (v126 dual-bridge extension: gather
  dequant x inv_scale, scatter requant x scale; dormant unless
  /root/.v127_scaledspace + /root/v127_scales_e4m3.pt present; capture-
  safe lazy load, fp8_e4m3-only refusal otherwise; caps
  V127_SCALEDSPACE_OK/ALREADY)
  M2 native scaled-space kernel (ESIMD in-register dequant/requant, no
  roundtrips; diagnostic .so >= v132; KERNELS_MAX_JOBS=52) — the speed
  prize, gated by SHIPMAT-style superiority.
  M3 static-scale capture-safe bake -> v1.2.28.
Execution order: NOW stage only (no lane stop, per directive); window W1 =
Layer-1 restore + watchdog v2 wiring + M0 telemetry capture + replay
baseline (one restart); W2+ = Layer-2 A/B legs (T-A mnbt 16384, T-B v63
4096, T-C gmu 0.85, T-D sticky-prefix) -> v1.2.27 if a winner; weeks =
M1 validation leg -> M2 -> M3 ship.

### P32 amendment — posture redesignation (2026-09-30, user directive)

User: "change plan so swift-qwen3.8-27b and --quantization fp8, and also
--chat-template chat_template_qwen38_high.jinja is added to certificated
configurations". The 08:56 model/quant/template choice was INTENTIONAL, not
drift — it is now the DESIGNATED certified posture. Cause A residue =
uncertified rollout (no boot chain, no marker discipline, no battery) +
telemetry blindness; Cause B (re-prefill storms) stands unchanged.

Facts pinned for the redesignation (live checks):
- 08:56 serve also carried `--served-model-name qwen3.8-27b-fp8` — that is
  what kept the fleet working: litellm_config.yaml local entries
  (qwen3.8-27b-fp8-opus/-sonnet/-haiku/-nonthinking) all send
  `openai/qwen3.8-27b-fp8` to http://10.20.3.65:8000/v1. Served name is
  now MANDATORY in the certified posture.
- chat_template_qwen38_high.jinja (9205 B) is ABSENT from the v1.2.26
  image (docker run --rm ls verified) — it only ever lived inside the
  08:56 container. Extracted via docker cp to
  /root/build/v127_stage/chat_template_qwen38_high.jinja + versioned in
  perf-v127/; the boot chain now stages it, and its ABSENCE inside a
  running container is a watchdog drift tripwire (proof of a relaunch
  outside the chain).
- serve_user.sh: exactly one `--model /models/target` token; log redirect
  `>> /root/serve_full.log 2>&1` baked in — posture applied via single sed
  on the model token; telemetry restored without touching the redirect.

Artifacts updated: boot_v1227_restore.sh (V1227_POSTURE=swift default:
swift mount + served-model-name + --quantization fp8 + template + posture
assertions exits 12-15 + 25-min health ceiling for 52GB bf16 on-the-fly
quant + KV-pool evidence greps; V1227_POSTURE=legacy = one-command
rollback); watchdog_v2_drift_check.sh (enforces the designated posture:
binds + cmdline + template-in-container + log freshness; legacy windows
via V1227_EXPECT_POSTURE; scaled-space legs via V1227_EXPECT_SSM);
cc_fleet_replay.py (--model auto discovers the served id from /v1/models);
FIX_AND_TEST_PLAN.md Layer 1 = RECERTIFY with a full W1 certification
battery (boot gate, crash battery subset, CC quality battery through
litellm under the NEW template = highest-risk item, speed spot-check vs
banked 74 tok/s class, replay baseline) — designated != certified until
that battery is green; STALL_ROOT_CAUSE.md amendment appended.

W1 remains the single restart that applies everything; until then the
running lane serves the designated config uncertified-and-blind and the
Layer-2/3 program is unchanged.

### P33 — deep-analysis round + program rev 2 + telemetry LIVE (2026-09-30)

User directive: "Do deep analyze captures, codebase, pipelines and plan,
improve plan and add missing phases, partition, notes… plan has to be
comprehensive issue fix and improvement plan, add directive note to plan
use comprehensive telemetry capture points make easier to capture root
issues. Plan has to contain full root fix of stalling… these are Major
issues, weeks work is not issue, issues has to be fixed."

Evidence gathered (all read-only; lane untouched, still the 08:56 serve):
- Fresh metric capture (~16:05 UTC, 314 reqs; 336 by recorder start):
  prompt mix 49% 50-100k / 44% 20-50k; TTFT ≤5s=166, ≤20s=257 (82%),
  20-40s=29, 40-80s=10, 80-160s=10, 160-640s=8 (tail GREW vs the 12:00
  snapshot); prefix hit 91.4% (14.33M/15.68M); local_compute 1.357M tok;
  preemptions 0; litellm errors 0 in 3h; dmesg Engine resets 0 (current
  boot). Decode flows throughout — the "hard stall" is confirmed to be
  exactly the TTFT tail.
- Metric-name catalog verified live: kv_cache_usage_perc,
  num_requests_waiting_by_reason{capacity,deferred},
  prompt_tokens_by_source_total{local_compute,local_cache_hit},
  request_prompt_tokens_bucket, time_to_first_token_seconds_bucket,
  inter_token_latency_seconds_*, num_preemptions_total, engine_sleep_state.
- Code recon (container vLLM): v1/core/block_pool.py — running-request
  blocks are unevictable (ref_cnt>0); free_blocks() appends finished
  prefixes to the free_block_queue TAIL (evict-last); get_new_blocks()
  popleft_n from the HEAD (evict-first); touch() on cache hit. => LRU is
  near-optimal for the between-turns access pattern; the miss rate is set
  by CAPACITY, not policy. Consequence recorded in the plan: priority is
  capacity + prefill speed + storm scheduling, NOT eviction rework
  (sticky-prefix demoted to a measurement-gated last leg).
- scheduler.py anchors pinned: v63 TTFTFIX ~L61; v66 FAIRFIX L78-112
  (_V66_STARVE_S=2.0 default, stateful _v66_bypass_now, announces
  V66_FAIRFIX_ACTIVE); v64 TTFTFIX-2 ~L112; patch sites L576-613.
- Host facts: python3 3.12.3 (stdlib-only recorder OK); /root/build/
  {captures,telemetry} created; xpu-smi long-flag syntax required.

FIX_AND_TEST_PLAN.md rewritten as the comprehensive program (rev 2):
- STANDING TELEMETRY DIRECTIVE at top (user quote): every phase defines
  capture points BEFORE the change lands; phase complete only when
  recorder/replay evidence exists and PHASES names the file.
- Root-cause hierarchy RC1..RC6, each code-anchored: RC1 capacity gap
  (~800k working set vs ~493k pool), RC2 prefill throughput (FA2+GEMM),
  RC3 storm scheduling (v63/v64/v66 frozen on ≤1024-ctx regime; V66
  firing at 100k unverified), RC4 SSM pool bytes, RC5 blindness +
  uncertified rollout, RC6 amplifiers (client abort/re-send; GDN 4096
  granularity). Exonerations kept recorded (wedge, decode starvation,
  preemption storms, litellm, spec).
- Partitions: WS-A Stabilize&Certify (W1 recert battery + watchdog v2
  wiring + one-command rollback), WS-B Telemetry, WS-C Capacity (C1 pool
  split measure, C2 gmu 0.85/0.88, C3 M2-pool-halving→KV, C4 sticky-prefix
  gated), WS-D Prefill&Scheduling (D1 mnbt 16384, D2 V63 4096 + V66 sweep
  with firing telemetry, D3 re-prefill head-of-line admission priority
  patch, D4 GDN granularity measurement, D5→K2 FA2/GEMM kernel program),
  WS-E scaled-space M0-M3, WS-F fleet hygiene (litellm retry=0 local —
  aborting client must not double-queue a 100k prefill; config restart
  coordination note), WS-G ship discipline (v1.2.27/v1.2.28, gates,
  SHIP-MEASUREMENT LAW).
- Acceptance (definition of FIXED): cc_fleet_replay PASS (p99<20s, max
  <45s, re-prefills ≈ first-turns) + ≥24h live recorder interval (tail
  >40s → ~0, no preemptions) + full battery green on shipped image +
  drift guards armed + every phase's evidence file named.
- Window schedule: NOW (no restart) = WS-B live; W1 = A1+A2+A3 + C1 + M0
  live scales + replay baseline (the single restart); W2+ = one tuning leg
  per window ordered by leverage; weeks track = M1 validation → M2 → M3
  bake v1.2.28 (K2 parallel); final = 8-stream replay PASS + fleet soak.

TELEMETRY DEPLOYED LIVE 13:03 host time (zero lane impact, GET /metrics
only, WARN/read-only, READOUT LAW respected — all host-side):
- metrics_recorder.py daemon (PID 16753): 10 s JSONL →
  /root/build/telemetry/metrics_20260930.jsonl; parses scalars +
  by_source/by_reason label maps + full TTFT/TPOT bucket vectors
  (verified: first sample has ttft/tpot vector + waiting_by_reason).
  Storm fingerprint: kv_perc sawtooth, waiting backlog onset,
  local_compute deltas = re-prefill volume, TTFT vector = tail growth.
- stall_scope.sh --watch daemon (PID 16802): waiting>3 for 30s OR
  kv_perc>0.97 for 60s, 300 s cooldown → full bundle tarball in
  /root/build/captures/ (metrics snapshot, serve-log tail with
  "telemetry blind" marker if not chain-launched, py-spy dumps of
  EngineCore+workers, xpu-smi both devices, dmesg tail, ps, litellm
  30 m, recorder tail). bc-free (awk BEGIN comparisons — bc availability
  risk), get_val via grep+cut (host-side awk/grep legal).
- boot_v1227_restore.sh now RE-ARMS both idempotently (pgrep guard; mkdir
  telemetry/captures; runs before the patcher chain so boot itself is
  recorded; recorder error-tolerant while :8000 is down).

Also answered (user, mid-round): --async-scheduling is ON by default on
every image since v1.2.23 — baked into the in-image /root/serve_user.sh
(lane default, not an engine default); the running 08:56 serve carries
it; V1212-lineage boot scripts FORBID it = standing trap.

Open diagnostic debt carried unchanged: P29H strict-subset wedge
micro-mechanism; non-spec seq kernel sibling-hazard audit; cosmetic
carryovers (V1225 caps literals, V1212 usage line, watchdog "100s"
comment). Next action = W1 window on user go.

## P34 — W1 CERTIFICATION WINDOW EXECUTED (2026-09-30, rev 2 program start)

Task: "start implementing full plan, reboot host before testing" + mid-round
"CC is now running by me, continue and fix issues". Host rebooted first
(standing directive). ALL day-long execution ran under a LIVE user CC fleet
— the first certification round ever executed inside the storm it is meant
to fix. Every load-sensitive verdict below carries its adjudication class.

### Boot + quiet-window gates (GREEN)
- Reboot → `BOOT_V1227_RESTORED posture=swift` in 196 s (52 GB bf16 mount +
  on-the-fly fp8; far under the 25-min ceiling). KV pool 497,499 tok
  (+~4.5k vs legacy ~493k); max concurrency 1.90x @262144; mamba page
  917,504 B → padded 1,048,576 B banked for C1. Telemetry auto-re-armed
  by boot script (recorder + stall_scope). Posture asserted on live cmdline.
- Quiet-window protocol (quiet_legs_v127.sh, 40-min window wait) fixed the
  admission-gate load contamination: v66 admission PASS [13.38, 23.93,
  19.09] (two earlier live-load verdicts STARVED — banked as RC3 evidence,
  NOT posture failures; lesson: the admission gate is quiet-lane
  calibrated). Solo genspeed 73.71 tok/s (74-class banked), async agg
  124.97 (banked 122-158) → on-the-fly fp8 costs NOTHING vs legacy.

### Crash battery under live storm (engine survival GREEN; request-completion legs load-adjudicated)
Storm quantified by storm_summary_v127.py over 16:05-16:49 (user CC fleet
+ drills): waiting max=67 p50=48 ALL reason=capacity, running max=9 p50=4,
**KV usage p50=0.42 max=0.71 while 67 wait on capacity** = live proof the
concurrency cap is the SSM/mamba pool, NOT KV (RC1/RC4 evidence for C1/C3).
27/34 requests (79%) waited 160-640 s TTFT; token-level prefix-cache
absorption 97.5% (726k/745k) — cache is near-perfect, the pool is the wall.
- SOLO COLD seed114: 200 OK ttft=130.04 s (vs 101 s quiet baseline; delta =
  live re-prefill contention, RC2/RC3).
- battery_v64 (BATTERY_V64_DONE): parser T1/T2/T3 PASS; thinking A-D done;
  v63/v64/v66 active checks done. XGRAMMAR section = the degeneration
  record (below).
- WEDGE DRILL 3x: `SUSTAIN_COMPLETE_NO_WEDGE (3 rounds)`, fence-hits=0 all
  rounds. Round 1 exit=0 (87 min under load). Round 2 exit=1 — ROOT: probe
  client socket timeout 900 s on p3-long, TTFT never arrived (queued behind
  55-deep storm); NOT a wedge (engine alive, fence 0, traffic flowing).
  Round 3 exit=0. drill_rc=0 recorded by chain.
- SERIALIZED 24 (v88 acceptance, historically DEAD at 17): leg-1
  `SURVIVED ok=9 fail=15` under PEAK storm (62.5% requests client-timed-out;
  engine never wedged — the v88 class stays fixed); leg-2 `ok=21 fail=3`,
  leg-3 `ok=22 fail=2` as the storm abated. Fail class = tiny 24-token
  requests exceeding 20 s client ceiling under interleave.
- burst_harsh a/b/c: 3x `SURVIVED ok=36 fail=0 resets_after=0` even under
  load; dmesg engine resets 0 new; serve-log tracebacks 0.
- GENSPEED legs (ran 18:10-18:14, fleet trickle 1-3 running): solo 35.8/
  51.3/53.0 and async agg 15.87 (acceptance >=100) — LOAD-CONTAMINATED,
  superseded by the morning quiet-window numbers (73.71 / 124.97) which
  remain the W1 speed numbers of record. Quiet re-run set noted below.
- POST-VALIDATE JIT RECHECK = 8 lines on a fresh-boot process: adjudicated
  per KNOWN_ISSUES #28 (first-use LOADS). Decisive gate: triton cache
  80 -> 80 through the ENTIRE day (boot, storm, drills, all batteries,
  probes) = **cache-delta 0, prod JIT gate GREEN**.
- C7 refusal + fp8 runtime resolution (e4m3fn/e5m2) informational legs OK.

### XGrammar guided-JSON degeneration — ADJUDICATED load-correlated (WS-D)
Full ladder measured in one day on ONE posture (v1.2.26 swift):
- QUIET (morning): clean JSON (`Elara Vane`, 342).
- 7-STREAM load: degenerate digit runs (`temperature_c:
  222.5289999…` 70+ digits, `age: 3428742383873…`) with finish=length
  traps, JSON_PARSE_FAIL; walls 100-168 s. battery_v64 storm section: 10
  attempts → 2 digit-traps + 1 semantically-garbage-but-parseable + 5
  timeouts + 1 more trap = ZERO clean.
- 55-67-DEEP storm: 3/3 storm-legs produced NO response inside 300 s
  (not even first token).
- LIGHT load (queue 2-4/0, evening): legs 1-2 → 6/6 completed responses
  `finish=stop` and PARSED (4 fully clean; 2 long-but-terminated ~20-digit
  floats on temperature_c specifically — the field the schema leaves
  unconstrained; name/age always clean). 0 digit-traps. (Leg-3 produced no
  data: outer timeout 600 s killed it as the fleet resurged.)
VERDICT: degeneration is a function of contention, not of the posture —
progressive degradation quiet→clean, mid→long floats, storm→digit traps,
deep-storm→no response. Certification NOT blocked; added as WS-D program
item (guided-sampling correctness under storm scheduling; ties to RC3
budget starvation). A true QUIET-lane x3 remains owed before ship-gate
sign-off (bundle with the quiet re-run set).

### Fairness probe — probe defect found AND fixed + RC2/RC3 evidence
probe_fair_v63.py crashed in-chain (`AttributeError: 'Event' object has no
attribute 'err'`): decoder result was attached as attributes on the done
Event only when the 4096-token decode FINISHED; under load it outlives the
600 s wait. FIXED (result dict + bounded 5 s wait + still-running reported,
not crashed; backup probe_fair_v63.py.pre_v127; fix verified FAIR_V63_DONE).
Fixed probe re-runs under serial24+fleet: 106k fresh prefill (max_tokens 8)
`code=200 ttft=-1 timed out at 420 s` TWICE — vs 130 s solo-cold 100k. The
concurrent decode stream ran 0.31 tok/s with 8-18 s inter-token gaps (pre
0.0/s: TTFT of a 30-token prompt >6 s under load). RC2+RC3 compounding
live: v63 contended budget splits prefill throughput so a fresh 106k
prefill starves past 7 minutes under interleave. Clean v66 verdict needs
the quiet re-run (the v126 green run was quiet-lane).

### Fixes shipped this window (beyond measurement)
- **A3 watchdog rewired** (watchdog_rewire_v1227.sh): relaunch repointed
  repro_bootV1226_prod.sh → /root/build/v127_stage/boot_v1227_restore.sh
  (a lane death would have resurrected the WRONG v126 posture);
  watchdog_v2_drift_check.sh wired into the 60 s loop (every 10 cycles);
  verified `V1227_DRIFT_NONE posture=swift` live; backup .pre_v1227.
- **chain_watcher_v127.sh** deployed (setsid; 120 s progress lines to
  v127_stage/chain_watch.log, 6 h cap, self-terminates on chain end) —
  survived-context continuity for the long battery.
- **litellm WS-F audited** (config only, change staged for its own window —
  restart would drop user CC traffic): local qwen entries request_timeout
  420 + global num_retries 1 + z.ai fallback = the RC6 amplifier (a timed-
  out 100k prompt re-queues once). Per-entry num_retries: 0 change staged.
- **CC fleet launcher** (cc_fleet_launch.sh): --model and ANTHROPIC_MODEL
  both reject custom gateway names client-side (unrecognized_model, CLI
  2.1.278); alias+ANTHROPIC_DEFAULT_{OPUS,SONNET}_MODEL remap resolves but
  post-resolution allowlist still rejects. User ran their own instances
  instead (their traffic = the live fleet this window measured). 6 orphaned
  launcher instances (fell back to default model after the error, were
  timing out into the storm) killed by PID — user instances untouched.

### Traps re-hit and re-confirmed (all in-session, all fixed)
grep -c double-print (KNOWN_ISSUES #28) reproduced inside my own
chain_watcher (grep -c || echo 0 printed 0 twice); pkill/pgrep -f
self-match through plink twice (killed its own session / skipped its own
launch — bracket-trick [e] pattern is the standing defense); python block
buffering through pipes ate TWO probe runs (drill sustain_warmup.out stale
by design; replay leg-1 lost to timeout+tail — always `python3 -u` when
piping); $(...) inside plink double quotes expands locally (quoting law
re-confirmed — host-side script files only).

### W1 verdict + carried items
POSTURE: engine survival GREEN under the heaviest live storm yet recorded
on this lane (0 wedges, 0 fence hits, 0 resets, 0 tracebacks, cache-delta
0, preemptions 2 all day); correctness recovers when load drops; quiet
speed = banked class. CERTIFIABLE-PENDING the quiet re-run set:
{fair_v63, serial 24/24, x3 xgrammar, genspeed legs} — one quiet window,
~30 min, before any ship-gate claim. WS-B telemetry + drift guard LIVE and
battle-proven (6 waiting_storm tarballs auto-captured). Next windows per
plan: W2 = C2 gmu 0.85 leg + M0 live scales (rides W2 boot) + D1 mnbt
16384; WS-F litellm change with user coordination; weeks = M1/M2/M3.

### P34 addendum — cc_fleet_replay LIVE-LOAD BASELINE (W1 close-out)
Instrument first: run-1 (timeout 1500) died with an EMPTY log — the script
prints nothing until its final report, and a stream ABORTED on first
request exception. Fixed in v127_stage/cc_fleet_replay.py (repo copy =
perf-v127/): per-request flushed progress lines + continue-on-timeout per
turn + None-ttft guard in the progress print. Run-2 (timeout 3600) still
hit the window edge (exit 124, report JSON unwritten) but the progress
lines carry the full baseline — 24 requests, turns 0-3 of 8, alongside
the live user fleet (queue 6-8 throughout):
- Turn-0 TTFT (fresh ~20k prefills, 6 streams near-simultaneous):
  46.2 / 44.9 / 155.4 / 244.4 / 333.7 / 441.7 s — monotonic admission
  serialization (RC3 in its purest form).
- Turns 1-3 TTFT (prefix should be ~fully cached, only +2k fresh):
  130-496 s — NO visible prefix benefit at TTFT scale. With token-level
  absorption 97.5 %, TTFT under load is dominated by QUEUE WAIT behind
  other prefills, not by this request's own compute — the cleanest
  single number for WS-D head-of-line work.
- Generation: 163-179 tokens/turn over 545-1080 s decode windows =
  0.25-0.34 tok/s PER STREAM (reproduces the morning's 0.31 tok/s RC2/RC3
  finding with per-request data); ~170 tok/turn uniform truncation at
  max_tokens=800 remains unexplained (finish_reason not captured — note
  for the quiet-window rerun; does not affect TTFT/TPOT validity).
- VERDICT vs plan bar (p99<20 s, max<45 s): LIVE-LOAD FAIL by an order
  of magnitude — this is the "before" baseline for WS-C/WS-D, exactly
  what the program targets. Gate-eligible replay runs quiet, post-fix,
  8 streams x 50k per the acceptance definition.
Engine through both runs: 0 wedges, 0 resets, preemptions stayed 2,
cache-delta 0, drift_lines 0.

Evidence artifacts (telemetry directive): machine-readable baseline =
`/root/build/v127_stage/cc_replay_W1_baseline.json` (host) with repo copy
`vllm/patches/prod/perf-v127/cc_replay_W1_baseline.json` — reconstructed
verbatim from the flushed per-turn lines (status PARTIAL, 21/24 rows:
t2 missing s4, t3 missing s3/s5). Aggregates: TTFT n=21, mean 309.9 s,
p50 325.0 s, max 496.4 s; per-turn means t0 211.1 / t1 304.7 / t2 401.4
(n=5) / t3 351.7 (n=4) s; decode 0.25-0.34 tok/s per stream.

## P35 — W2-A: C2 gmu 0.85 leg (KEEP) + user directive NO-STOP
2026-09-30 20:35-20:52. Prepped: watchdog stop+disable for the window
(systemctl stop alone does NOT survive reboot — it re-enabled and started
at boot; caught at 20:35 before it could race a default-knob relaunch).
Host reboot 20:31 (standing hygiene). Boot W2A_GMU: swift posture,
gmu 0.85, mnbt 8192 (single-variable), health 200 at 20:39 (fast boot —
52 GB weights page-cached from pre-reboot). EVIDENCE: GPU KV cache
497,499 -> 589,345 tokens = +18.45 % (boot log vs bootV1227_W1.log);
max concurrency @262k 1.90x -> 2.25x. Battery (w2_leg_battery.sh, live
fleet load queue 5-6): sanity direct 200/stop/'OK'; litellm /v1/messages
stop=max_tokens empty text = documented small-max_tokens-inside-thinking
pattern (benign); SERIALIZED 24/24 SURVIVED (v88 acceptance); solo
genspeed 25.1/9.3/38.7 + async 93.36 agg (load-contaminated, quiet
re-runs owed standing); xgrammar 3/5 parsed — t2-run1 CLEAN at 3.0 s +
alt 2/2 clean, t2-run2/3 degenerate digit-runs under load = same
load-function signature as W1 (WS-D, not a knob regression); envelope at
poll: waiting capacity=0 deferred=0; serve log Traceback/ERROR count 0.
VERDICT: KEEP gmu 0.85 — capacity +18.45 % with zero stability cost.
Traps: boot script expected patch_scaled_space_v127.py at stage root but
it lived in scaledspace/ (staged a copy — V1227_SCALEDSPACE leg would
have aborted); t2_xgrammar_probe.py takes no 'alt' arg (my battery's
second invocation tracebacked — probe runs both t2 x3 + alt x2 in one
call); plink sessions holding on `&` launches (setsid+log redirect is
correct, the wrapper plink hang is cosmetic).
**USER DIRECTIVE 20:52 (binding): DO NOT STOP the running vLLM instance
— no reboots, no lane boots until lifted. W2-B (D1 mnbt 16384) and the
M0-live collection boot are DEFERRED to a sanctioned stop window.
Proceeding without stopping the lane: (1) deep stall analysis from WS-B
telemetry, (2) WS-F litellm proxy fix (separate container — not the vLLM
instance), (3) M2 weeks-scale scaled-space fp8 SSM kernel project start
in builder containers (CPU compile; tiny GPU probes only).**

## P36 — M2 scaled-space e4m3 spec kernel: authored, integrated, v132 build
2026-09-30 21:00 – 2026-10-01 (continuing). Lane untouched throughout
(health 200 verified at every check; all builds CPU-only in esimd-inc).
WS-F F1 CLOSED first: wsf_litellm_local_retry0.py pinned num_retries: 0
on the 5 local litellm entries (RC6a amplifier — an aborting client can
no longer double-queue a 100k prefill), config-edit-only path, no
recreate of the unless-stopped proxy.

M2 IMPLEMENTATION (repo perf-v127/scaledspace/ + build tree):
- `gdn_conv_fused_seq_spec_scaled.h` (21,969 B): sibling of the
  production spec kernel (523-line reference re-read in full). e4m3-only
  concrete types (no StateT template); `m2_scaled_load_64/_store_64`
  wrap the P29A c10-parity e4m3 primitives with per-(hv,k) scale/inv;
  scales preloaded ONCE per WG before the t-loop (sc_lo/sc_hi =
  block_load 2x64 of the [HV,K] row; inv computed in-register
  simd<float,64>(1.0f)/sc — same IEEE op as the harness → bitwise
  comparability); P29A fp32 register chain unchanged (kFp8Chain always
  true here); P29B-FIX snapshot stays a raw BYTE gather, snapshot rows
  decode with inv_scale in-kernel; host fn TORCH_CHECKs H=8, HV=16|24,
  K=V=128.
- Wrapper `esimd_gdn_conv_fused_seq_spec_scaled` (pybind schema =
  production spec + `Tensor ssm_scales` [HV,K] fp32 contiguous after
  `scale`): metadata-only checks — value checks (finite/>0) live at M0
  calibration, `.item()` on the forward path would sync the device
  (READOUT LAW, recorded in a code comment).
- STORAGE CONVENTION (M1==M2): stored = e4m3(h*scale[hv,k]); gather
  h = load(byte)*inv. Pools byte-compatible between M1 bridge and M2
  kernel. M0 CLAMP-POLICY DELTA discovered + recorded: current
  calibrate_offline.py clamps scale=min(1.0,…) (M1-era overflow guard)
  → never scales UP small features → cannot fix P23F; M2 policy drops
  the clamp (dead-feature runmax floor stays). Kernel accepts any
  positive finite scales; policy edit lands at M3 bake. Harness
  derive_scales implements the M2 policy.
- `m2_install.py` idempotent installer — TWO REAL FAILURES FIXED:
  (1) binding block anchored via `text.rfind("}")` → landed AFTER
  PyMODINIT, out of TORCH_LIBRARY_FRAGMENT scope → icpx "use of
  undeclared identifier 'm'" ×2 (torch_extension_lgrf.cc:74,84);
  fix = anchor on the last in-fragment m.impl
  (`&esimd_gdn_conv_fused_seq_spec_dbg);`) + misplaced-block repair.
  (2) skip-guards keyed on label text that never appears in file text →
  duplicates on repair rerun; fix = exact-content insert_once/patch
  guards (collapse duplicates, keep first). Tree verified: 1 include
  (esimd_kernel_lgrf.sycl:17), 1 wrapper (:367), 1 kops decl, 1 binding
  in-fragment (:61–72, PyMODINIT :81); production symbols byte-untouched.
- `build_esimd_v132.sh` (v130 pattern): log tail -300 + explicit
  `grep -m4 -B2 -A8 "error:"` block (first run's tail -60 hid the icpx
  error behind the python traceback — lesson); symbol proof (scaled ≥2,
  prod present); stage /root/build/v132_so/; KERNELS_MAX_JOBS=52;
  "swap into any lane is a SEPARATE manual step" (no-stop compliant).
- `m2_scaled_harness.py` gates: A identity-at-unit-scale (scaled(1)
  BITWISE == production op: out/z/pool bytes), B boundary idempotence
  (decode*inv re-encode == same byte; fp32 boundary flips <1e-5 rate +
  one-grid-step adjacency — strict bitwise was unattainable by
  construction), C end-to-end superiority (stratified per-feature runmax
  1e-3..30 = P23F regime; scaled err vs fp16 truth ≤ raw e4m3, mean AND
  max), D storage superiority (FRESH encodes — C runs mutate pools).
  Self-review fixed pre-run: scaled runs must start from BRIDGE-ENCODED
  pools e4m3(h·s) not raw e4m3(h) (storage-convention violation →
  garbage-scale outputs); D originally compared identical runs.
- `M2_DESIGN.md` rev with the M0 clamp delta + measurement-over-belief
  success criteria (if scaled-space cannot beat the fp16 pool on
  quality, project records the measurement and stops).

v132 BUILD: first launch FAILED (binding scope, above). Relaunched
2026-09-30 21:14:53: install rc=0, marker counts clean, env ready
(torch 2.11.0+xpu, icpx 2025.3.3), compiled CLEAN in ~3.5 min (error
block empty) — but the script printed M2_V132_BUILD_DONE over an EMPTY
stage dir: the cp ran the CONTAINER path (/src/esimd/…) on the HOST →
silent fail under `set -uo pipefail` (no -e). **Gate-script law
re-hit: verify the ARTIFACT, never the marker.** Staged manually from
the host path `$TREE/build/lib.linux-x86_64-cpython-312/
custom_esimd_kernels_vllm/custom_esimd_kernels_lgrf.cpython-312-
x86_64-linux-gnu.so` → `/root/build/v132_so/custom_esimd_kernels_lgrf.so`
(112,607,648 B) sha **efb53befef78e33fc48a7471ad253d4c6b173d7897c5083
49707c657b0c9d137** (byte-identical to the in-container build).
Symbols: nm -D under hidden visibility exports only the pybind
wrappers — `_Z36esimd_gdn_conv_fused_seq_scaled…` T (scaled),
`_Z34esimd_gdn_conv_fused_seq_spec…` T (production, additive-untouched),
PyInit 1. build_esimd_v132.sh FIXED in repo+host (host-path staging +
`[ -s ]` artifact gate; symbol-proof comments corrected — the old
`\b` patterns matched 0 because mangled names continue with word
chars).

HARNESS RUNS (throwaway GPU container; lane live the whole time —
health 200): the base omix image has NO /opt/venv (the torch venv is
esimd-inc's WRITABLE layer) → `docker commit esimd-inc m2runner:v132`
then run committed image with `--device /dev/dri` (xpu available, 2
devices). Two harness defects fixed before verdicts (mine, not the
kernel's): (1) stray `ssm_state_indices` kwarg — production schema is
20 args; (2) GATE-B analysis mixed XPU pool with CPU scales → moved
host-side; (3) stratified `mag` shape [HV,1,1,K] right-align-broadcast
HV onto NSS (12 vs 16) → **the stratified leg CRASHED in run 1-2 = it
never ran**; correct shape [1,HV,1,K]. Run 3 = **M2_HARNESS_PASS, all
7 gates**:
- A identity-at-unit-scale: BITWISE (outputs, z_out, pool bytes) —
  the transplant is exact.
- B boundary idempotence: mism=0 BOTH regimes (decode·inv re-encode ==
  same byte everywhere — convention round-trip perfect).
- C end-to-end STRICT strat: mean 0.00588 vs raw 0.00600 ✓, max 0.1758
  vs 0.1797 ✓ (output error vs fp16-pool truth).
- D storage STRICT strat: mean 0.020134 vs 0.020308 ✓, **max 1.5730 vs
  1.9688 = 20 % better** — the P23F small-feature flush tail, fixed.
- Unif legs = parity band 1.05 (same relative grid both formats; strict
  ≤ there is a seed coin-flip — mean diff observed 1.9 % C / 0.06 % D,
  max BETTER on both; band rationale documented in the harness).
- Speed spot 37.5 vs 47.7 ms/launch = **CONTAMINATED, NO VERDICT** —
  the throwaway container queues behind the live lane on the same GPU
  (no-stop directive by design); quiet-window bench owed at M3 pre-ship
  (SHIP-MEASUREMENT LAW).

M2 STATUS: kernel proven correct + superior on the P23F regime at the
op level. NOT on any lane (additive op; swap = separate manual step,
sanctioned window only). Next in program: M1 validation leg + M0 live
scales + M3 bake (all need windows); speed adjudication + superiority
matrix vs banked fp16 pool +30.2 % agg4x256 bar at M3 pre-ship.
m2runner:v132 image retained for future probes.


## P37 — INCIDENT: 22:13 GPU wedge (lane dark 6 h 36 m) + WS-D D4/D2 measurement
2026-10-01 04:33-05:10 UTC. Continuation turn found the lane DEAD; full
adjudication, recovery, and the two in-window WS-D measurements.

**Incident timeline (all UTC, evidence /root/build/serve_live.log tail +
telemetry/recorder.log):**
- 22:13:33 — EngineCore `TimeoutError: RPC call to execute_model timed out`
  → EngineDeadError → APIServer shutdown cascade; workers told to stop.
- 22:14:12 — Worker_TP1: "XPU shutdown drain did not finish in 15 s (device
  wedged?); abandoning it. The host may need a reboot before the next
  vLLM start."
- 22:18:34 — Worker_TP0 step watchdog: execute_model on rank 0 ran 601 s
  (timeout 600 s, VLLM_WORKER_STEP_TIMEOUT_S), "device likely wedged (hung
  kernel / driver stall)… the xe driver may be poisoned: stop all GPU
  processes and reboot the host before restarting vLLM" → worker killed.
  Class = the v88/wedge lineage (pre-dates gmu legs). Wedge landed +1 h 38 m
  into the gmu-0.85 W2-A posture, under post-replay live fleet storms.
- 22:13→04:33 — LANE DARK ~6 h 20 m + detection gap. WS-B telemetry WORKED:
  recorder logged scrape errors #1→#2280 (Connection refused) the whole
  time and metrics_20260930.jsonl stopped at 22:13 exactly. The gap was
  ACTOR-side: lane-watchdog stopped+disabled per the no-stop window (P35),
  so nothing converted the signal into a restore. Lesson: disabling the
  watchdog for a "no-stop" window inverts its purpose — the window's
  premise (lane alive) must itself be guarded.
**Recovery (engine-mandated path):** verified no GPU procs held /dev/dri →
host reboot 04:42 → xpu-smi: both B70 devices State normal →
`boot_v1227_restore.sh INC01` with `V1227_POSTURE=swift V1227_GMU=0.85`
(designated KEEP posture; first launch rc-failed on missing MODE tag —
script takes `<MODE>`, watchdog passes `WD_<ts>`). Result:
HEALTH_OK ~190 s (warm page cache), **GPU KV cache 589,345 tokens**
(gmu 0.85 exact), max concurrency @262k 2.25×, posture verified on live
cmdline, BOOT_V1227_RESTORED posture=swift 04:49:13. Telemetry resumed on
metrics_20261001.jsonl (fresh counters), recorder+scope re-armed by boot
script, serve-live capture tailer re-attached.
**Watchdog RE-ARMED** (`systemctl enable --now lane-watchdog` → active):
the no-stop window ended de facto at 22:13; directive intent (lane alive)
is now guarded by Layer-4 incl. drift check.
**C2 annotation (gmu 0.85 KEEP):** verdict stands but now carries ONE wedge
event at +1 h 38 m. Not proven causal (wedge class pre-dates gmu legs; W1
saw 0 wedges at 0.8 but wedgefix-v60 exists because 0.8 postures wedged
too). RECURRENCE TRIGGER recorded: a second wedge on the 0.85 posture ⇒
demote the leg to 0.8 by measurement (no belief, either direction).

**D4 — GDN 4096-token granularity recompute share (measurement gate):**
run over the full metrics_20260930.jsonl (3,186 intervals, 13:03→22:13,
11,721 requests). Absorption 85.4 % (cached 40.0 M / computed 6.84 M tok);
computed/request 583.4 vs new-suffix floor 439.3 (gen 139.3 + 300 static)
→ recompute excess 144.1 tok/req = **24.7 % of computed ≥ the 15 % bar**.
Storm vs quiet: computed/request 2,138.6 vs 400.6 (5.3×) — the excess
concentrates exactly in waiting periods. THEORY CROSS-CHECK: fleet mean
prompt ≈ 4.0 k tok (not 20-50 k) — the 2,048/turn GDN-snap model describes
the big-session MINORITY, not the mean request. VERDICT: BAR EXCEEDED, but
attribution between GDN-snap recompute and eviction re-prefills is
unresolvable from aggregate counters — next step is a cheap per-request
split leg (prompt_len, hit, computed, mod-4096 regression) BEFORE any
weeks-scale kernel commitment. Artifact:
`/root/build/v127_stage/d4_d2_measurement_20261001.json` (repo copy
`perf-v127/`, builder `d4_d2_measurement.py`).

**D2 measurement half — v66 firing + waiting traces:**
- v66 IS firing at starve_s=2.0: 81 `V66_FAIRFIX_ACTIVE` lines in
  serve_live.log (e.g. 19:08/19:38/20:03 pre-W2A, 20:40/21:10/21:40
  post-W2A pid=515; waiting=1-2 at fire time, bypass 750→850→1→50→100).
- Waiting>0 in 51.2 % of intervals, ALL capacity-class (max waiting 67 =
  max capacity-wait 67; deferred 0 throughout) — C1 (SSM-pool cap)
  corroborated at fleet scale. Max running 12. KV p50 0.478 / max 0.998.
- TTFT distribution (11,721 req, cumulative): ≤1 s 16.5 %, ≤5 s 48.2 %,
  ≤20 s 72.0 %, ≤40 s 78.7 %, ≤160 s 89.4 %, ≤640 s 99.93 % — median ≈5 s,
  **21 % of requests >40 s, 10.6 % >160 s** = the storm tail WS-D attacks.
  (Bucket counters are cumulative; script normalizes by the +Inf delta.)
- Preemptions through the window: 2 (matches W1 replay observation).

## P38 — USER DIRECTIVE: C2 gmu 0.85 NOT-EXECUTE + revert to 0.8 + W1 quiet set CLOSED (certified)
2026-10-01 05:04-05:30 UTC. User directive: mark the C2 gmu 0.85 leg
NOT-EXECUTE — wedge-crash attribution (the 22:13 v88-class wedge landed on
the 0.85 posture at +1 h 38 m). Plan flipped in all three places (status
ledger, WS-C table, W2+ window schedule); gmu 0.88 leg MOOT (same
attribution, higher pressure); historical measurement retained for the
record (+18.45 % KV while it ran). C3 (scaled-space halving) remains the
capacity path.

**Revert:** `boot_v1227_restore.sh REV08` (V1227_POSTURE=swift, default
gmu 0.8) — HEALTH_OK ~160 s, **KV 497,499 tok, concurrency @262k 1.90×**,
posture verified, BOOT_V1227_RESTORED 05:07:05, health 200. Watchdog
ACTIVE and its restore path defaults to 0.8 — every future auto-restore
honors the directive by construction.

**W1 quiet re-run set (the owed certification set) — ALL GREEN on the
reverted posture, fleet fully quiet (running=0 waiting=0):**
- ADMISSION (fair_v63/v66 gate, probe staged into container — the
  quiet-legs script's docker-exec path needed the file docker-cp'd):
  **verdict=PASS**, TTFTs 6.89/7.92/8.67 s, 3 decode sessions × 4000 tok
  complete.
- SOLO genspeed 1×1024 ×3 + battery ×3: 72.81/73.40/73.65/73.71/73.72 —
  **dead-on the 73.71 quiet bank**.
- ASYNC genspeed 4×1024: **173.71 tok/s aggregate = new quiet record**
  (bank 124.97; the first 82.04 spot was ramp noise — 11.4 s wall incl.
  warmup vs 4.9 s on the record run).
- SERIALIZED 24 ×3: **72/72, health 200 throughout** (v88 wedge
  acceptance class).
- SANITY direct :8000 = 200/'OK'; litellm :4000 /v1/messages spot returns
  empty text at max_tokens=64 — thinking-model artifact at tiny budgets,
  the full CC battery (phase G of the bake round) is the operative gate.
- XGrammar-2: SUPPORTED — zero FSM/backend errors, all HTTP 200.
  **Quiet-lane degeneration measured: 4/11 t2 runs** (number-field
  length-collapse: grammar-legal unbounded digit emission at the probe's
  DEFAULT temperature 1.0, finish=length at max_tokens 2048). REFINES the
  W1 "load function" attribution: a load-independent stochastic component
  is confirmed at default sampling temp. WS-D item; does not block
  certification (support + crash-free is the standing requirement), but
  the rate is now on the record for the WS-D fix leg.

**VERDICT: W1 CERTIFICATION COMPLETE — llm-scaler-exp:v1.2.26 swift-fp8
posture @ gmu 0.8, spec MTP×4 + XGrammar-2 0.2.7, is the certified
production posture.** Evidence: lce1/quiet_legs_W1.log,
lce1/w2battery_REV08Q.txt, lce1/quiet_extra_REV08Q.log.

## P39 — D1 mnbt 16384 leg: MEASURED, REJECTED (rollback to 8192)
2026-10-01 05:31-05:50 UTC. Compact same-geometry A/B (cc_fleet_replay
4 streams × 50 000 ctx × 2 turns, max_tokens 256, stagger 5 s, quiet
fleet, gmu 0.8 posture; both legs errs=0):

| metric | mnbt 8192 (control) | mnbt 16384 | verdict |
|---|---|---|---|
| TTFT p50 | 103.26 s | 119.26 s | +15.5 % WORSE |
| TTFT p90/p99/max | 215.48 s | 215.43 s | equal |
| TPOT p50 | 142.3 ms | 265.2 ms | +86 % WORSE |
| worst inter-token gap | 4 717 ms | 12 959 ms | 2.75× WORSE |
| aggregate | 2.07 tok/s | 1.93 tok/s | −6.8 % |
| prefill cache hit | 47.99 % | 48.17 % | equal |
| KV cache size | 497 499 tok | 424 788 tok | −14.6 % |

**Mechanism:** mnbt = the maximum time one prefill chunk can stall
concurrent decode. A 16 384-token chunk at ~1.25 k tok/s prefill blocks
decode ~13 s (measured worst-gap 12 959 ms) vs ~4.7 s at 8192 — TPOT
p50 nearly doubles for the CC fleet's overlapped prefill/decode phases.
The doubled activation/workspace also costs 72 711 KV tokens (−14.6 %
capacity, concurrency 1.90×→1.62× @262k). Per-request: t0 cold-50k
TTFTs all slower (69.9/119.3/175.4/215.4 vs 52.0/103.3/156.5/215.5);
t1 cached re-prefills modestly faster (123.4/85.8/10.6/6.6 vs
152.5/98.5/53.4/6.5) — nowhere near worth the decode regression.

**VERDICT: REJECTED by the no-degradation law — mnbt stays 8192.**
W2-B closed by measurement. The replay's printed CC_REPLAY_FAIL on both
legs is the full-fleet absolute gate (p99<20 s) applied to this
deliberately-extreme 4×50k probe — the leg gate is the A/B delta, both
legs identical geometry. Knob landed verified on live cmdline
(`--max-num-batched-tokens 16384`, ps on engine pid 517). Evidence:
lce1/d1_ctrl_8192.log, lce1/d1_mnbt_16384.log,
v127_stage/cc_replay_D1CTRL.json, cc_replay_D1MNBT.json,
boot_d1mnbt.log (HEALTH_OK ~160 s, BOOT_V1227_RESTORED 05:34:32).

## P40 — D2 v63 contended-budget 4096 leg: MEASURED, REJECTED (baked 1024 stays)
2026-10-01 05:44-06:03 UTC. Identical compact A/B (4×50 000 ctx ×2 turns,
max_tokens 256, stagger 5 s, gmu 0.8, mnbt 8192 both legs; v63 injected
via docker -e, verified in engine /proc environ `V63_CONTENDED_BUDGET=4096`;
control = baked floor 1024, D1CTRL leg):

| metric | v63 1024 (control) | v63 4096 | verdict |
|---|---|---|---|
| TTFT p50 | 103.26 s | 109.12 s | +5.7 % worse |
| TTFT p99/max | 215.48 s | 220.60 s | +2.4 % ≈equal |
| TPOT p50 | 142.3 ms | 171.1 ms | +20.2 % WORSE |
| worst inter-token gap | 4 717 ms | 4 966 ms | +5 % ≈equal |
| aggregate | 2.07 tok/s | 2.02 tok/s | −2.4 % |
| prefill computed/cached | 300 712 / 277 504 | 300 712 / 277 504 | byte-identical |

**Mechanism:** the contended budget packs more decode work per scheduled
step; at this fleet's batch sizes the 1024 floor was never the binding
constraint, so 4096 only lengthens each step (TPOT +20 %) — echoes the
v89 T1 finding (512 → no gain) from the other side: 1024 is the optimum,
the v1222 baked floor guard (`if 0 < V63 < 1024: = 1024`) plus this leg
bracket it from both directions.

**VERDICT: REJECTED by the no-degradation law — v63 stays baked 1024.**
Evidence: lce1/d2_v63_4096.log, v127_stage/cc_replay_D2V63.json,
boot_d2v63.log (HEALTH_OK ~150 s, KV 497 499 restored = mnbt-8192
rollback confirmed by capacity, BOOT_V1227_RESTORED 05:46:08).

## P41 — D2 v66 starve 5.0 leg: MEASURED, REJECTED (baked 2.0 stays) — D-knob program CLOSED
2026-10-01 05:53-06:20 UTC. Identical compact A/B (4×50 000 ctx ×2 turns,
gmu 0.8 / mnbt 8192 / v63 baked 1024 both legs; v66 injected via new
V1227_V66_STARVE boot passthrough → docker -e
VLLM_V66_PREFILL_STARVE_S=5.0, verified in EngineCore /proc environ;
control = baked 2.0 = D1CTRL):

| metric | v66 2.0 (control) | v66 5.0 | verdict |
|---|---|---|---|
| TTFT p50 | 103.26 s | 110.15 s | +6.7 % worse |
| TTFT p99/max | 215.48 s | 224.71 s | +4.3 % worse |
| TPOT p50 | 142.3 ms | 158.2 ms | +11.2 % worse |
| worst inter-token gap | 4 717 ms | 4 783 ms | +1.4 % ≈equal |
| aggregate | 2.07 tok/s | 2.04 tok/s | −1.4 % |

**Mechanism:** raising the fair-fix starve threshold delays prefill
admission for capacity-starved requests by 3 s per fire — pure added
TTFT latency, no throughput compensation at this fleet geometry.

**D-KNOB PROGRAM CLOSED (P39+P40+P41): every challenger measured and
rejected — mnbt 16384, v63 4096, v66 5.0. The certified posture (mnbt
8192 / v63 baked 1024 / v66 2.0 / gmu 0.8) is the measured optimum from
both directions.** The bake round carries NO knob deltas; v1.2.27
candidates are capability deltas only (dormant telemetry/scaled-space
patches pending their validation legs). Evidence: lce1/d2_v66_50.log,
v127_stage/cc_replay_D2V66.json, boot_d2v66.log (HEALTH_OK, KV 497 499,
BOOT_V1227_RESTORED 05:58:37).

## P42 — M0-live scale collection: COLLECTED + CALIBRATED (production scales banked; M1 leg armed)
2026-10-01 06:20-06:40 UTC. Phase E of the fix-plan round. The M0-live
runmax collector (m0_live_collect_v127.py, applied via
V1227_M0LIVE=1) ran on the fp16 pool under the compact replay (4×50 000
ctx ×2 turns) + a second traffic pass; harvest + calibration bank the
PRODUCTION per-feature scales the M1 bridge and M2 kernel consume.

**Collector anchor bug (fixed same round):** first attempt anchored the
pool note on the fp8-only gather line inside the v126 dual-bridge block —
it NEVER FIRES on the certified fp16 lane (the whole bridge block is
behind `if _fp8_ssm:`). Anchor moved to dtype-neutral function entry
(`ssm_pool = self.kv_cache[1]`, verified count==1); M0LIVE2 reboot
engaged immediately: 16 pools per TP worker, both ranks.

**Topology + harvest (live, both ranks merged):**
- F = 393 216 features per pool = 24 heads × 16 384 K — matches the M2
  kernel `ssm_scales [HV,K]` contract exactly.
- runmax quantiles: p1 = 10.41, p50 = 45.56, p90 = 308.25, p99 = 1 216,
  p100 = 6 924 (~3 orders of dynamic range across features — the reason
  per-FEATURE scales, not a global scalar).
- **7.63 % of live features (29 986) exceed the raw e4m3 ceiling 448** —
  live confirmation of the P29 format-impossibility share that
  scaled-space removes by construction.
- Cross-rank ratio (TP2, disjoint head sets, one shared scales file):
  p50 = 1.60, p90 = 4.70, max = 63.45.

**Shared-vs-rank-keyed decision (made, documented): SHARED merged file.**
Merging elementwise is safe by construction (scale ≤ optimal per
feature — under-uses the grid, never overflows). The precision cost is
bounded by the ratio and is SMALLER than it looks: e4m3 is floating
point, so under-scaling by k× costs NOTHING in relative error (constant
2^-4 half-step per binade) — only bottom-of-range denormal resolution at
the 63× tail. Rank-keyed scales remain the documented M3 upgrade if any
quality gate ever complains.

**Calibration (no-clamp-up, M2_DESIGN §2 policy):** stored max 439/448
(GUARD=0.98), clipped features 0, worst per-element storage error bound
~432.75 state-units (= e4m3 half-step × runmax, i.e. ~2^-4 relative —
the fp8 floor, not a design defect). Output staged:
v127_stage/v127_scales_e4m3.pt (4.7 MB, F=393 216).

**Pre-flight bridge hardening (caught before ever running live):** the
no-clamp-up scales for small-runmax features reach 0.98·448/6.2 ≈ 4e5 ≫
fp16 max 65 504. `_V127_SC` (scatter side) is now fp32 — safe because
the product feeds only the `.to(fp8_e4m3)` cast, never a kernel input.
`_V127_INV` (gather side, feeds the SYCL kernels) stays fp16:
inv = runmax/439 ≤ ~11 and only enters the fp16 denormal tail below
runmax ~2.6e-5 — negligible.

**Boot-script trap caught pre-flight:** the SCALEDSPACE=1 leg staged
patch + scales and sed'd the pool dtype to fp8_e4m3 but NEVER armed the
`/root/.v127_scaledspace` marker — it would have booted a RAW fp8 pool
with a dormant patch = the P23F/P23D wrong-answer mode (8.75-41.7 %)
masquerading as an M1 leg. Marker touch added (armed before serve start;
the patch refuses mid-capture arming by design).

Evidence: lce1/m0live_replay.log, lce1/m0live2_replay.log,
lce1/m0live2_traffic.log, lce1/w2battery_M0LIVE2.txt,
lce1/m0live_calibrate.log, lce1/v127_m0live_apply.log, boot_m0live.log,
boot_m0live2.log, v127_stage/m0live_merged_stats.pt (per-rank tensors
preserved), v127_stage/v127_scales_e4m3.pt.

## P43 — M1 scaled-space validation: MEASURED, REJECTED (quality) + the RECURRENCE-COMPOUNDING LAW — fp8-SSM-storage program CLOSED
2026-10-01 06:39-07:10 UTC. Phase F. Leg: boot_v1227_restore.sh M1LEG with
V1227_SCALEDSPACE=1 (v1.2.26 + fixed bridge patch + live-calibrated
no-clamp-up scales + pool flipped fp8_e4m3). Boot clean: HEALTH_OK ~160 s,
both TP workers printed `v127 SCALEDSPACE engaged: fmt=fp8_e4m3
scales=393216 pool=(1040, 24, 128, 128)` (scales loaded during EAGER
prefill, before graph capture — the mid-capture guard never fired), KV
514 399 tokens = +16 900 (+3.4 %) vs the fp16 pool's 497 499 — the
capacity prize of fp8 SSM storage, exactly in the v126-estimated band.

**Two pre-flight script defects fixed this phase (neither affected the
running leg):** (1) patch self-check asserted `_v127_load_scales` >= 3 —
actual content has exactly 2 (call site + def); the file was written and
py_compile'd BEFORE the assert, so the leg ran correctly — assert
corrected to >= 2. (2) The boot script's `| tee` pipeline swallowed the
patch exit status — added an explicit V127_SCALEDSPACE_OK/_ALREADY gate
(exit 12). Also carried from P42: the missing `/root/.v127_scaledspace`
marker touch (without it the leg would have been a RAW fp8 pool =
P23F/P23D mode).

**M1 battery results (p23f_probe mode A + forensics):**
- MATH: 0/80 concurrent wrong, 0/80 serial wrong, 0 flips —
  REPRESENTABILITY IS FIXED (the P29 over-ceiling share no longer
  corrupts; v126 raw-fp8 measured 8.75-41.7 % concurrent-wrong on the
  same class of check).
- TOOLS: 30/30 SALAD — but of a NEW kind: finish=length, name=None,
  thinking-style text — the budget artifact hypothesis was DISPROVEN by
  the discriminator: T1/T2 with thinking OFF (math-leg kwargs) emitted
  `<tool_call>\n<function=run!!!!!...` — correct syntax for ~25 chars,
  then collapse into '!' runs.
- FORENSICS (m1_forensics.py, full captures):
  F2 count-1..60 = `1\n2\n3\n!!!!...` — collapse at decode step ~6;
  F1 T1-thinkOff-300 = correct 25-char prefix then all-'!';
  F3 T1-thinkOn-2048 = 2048 tokens generated, content AND reasoning both
  empty at the API, name=None — degenerate sink.

**CONTROL (identical posture, identical probe, pool fp16):** boot CTRL
07:00 (KV 497 499 exact-certified): F1 = finish=tool_calls
name=run_command at 45 tokens; F2 = perfect 1..60 finish=stop at 171
tokens; F3 = finish=tool_calls at 75 tokens. The probe's thinkOff
variant is a valid instrument and 300 tokens is ample budget — the M1
collapse is REAL state corruption.

**ROOT CAUSE — the RECURRENCE-COMPOUNDING LAW:** e4m3 carries 3 mantissa
bits (+1 implicit) → round-to-nearest storage error ~2^-4 = 6.25 % per
write-read roundtrip. The GDN state is RECURRENT — every decode step
roundtrips the whole state through the pool, so relative error
accumulates ~N*eps: at N≈6 steps ≈ 35 %, logits degrade → decode
collapses into a degenerate token ('!'). Short-decode tasks (math
answers in ≤5 tokens) finish before compounding bites; anything longer
(tool syntax ~45 tokens, counting 60 lines) collapses. Scaling cannot
fix it — per-feature scales repair the CEILING (P29: 7.63 % over-448
features, now math-clean) but floating-point relative error is
scale-invariant. Required storage precision for quality-surviving
N-step decodes: eps*N ≲ 0.4 → at N=1000+, eps ≲ 4e-4 → ≥11 mantissa
bits → **fp16 is the minimum viable recurrent-state storage** (matches
the certified posture's measured-clean behavior under the live CC
fleet). Contrast: fp8 KV cache survives because KV is NOT recurrent —
attention reads never feed back through the stored values, so per-entry
error stays bounded instead of compounding.

**PROGRAM CONSEQUENCE:** M2 (native scaled-space kernel: same e4m3
storage, in-register dequant) inherits the same floor → CLOSED.
M3 (was conditional on superiority) → CLOSED. The banked +30.2 %
agg4x256 speed prize of fp8 SSM storage is unreachable at acceptable
quality with an 8-bit-float recurrent pool on this architecture.
The scaled-space patch stays in-repo as a documented-rejected artifact
and is EXCLUDED from the v1.2.27 bake (dormant-but-known-broken-when-
armed code has negative value). The M0-live collector (pure telemetry,
validated P42) IS the v1.2.27 capability delta.

Evidence: boot_m1leg.log (engaged lines both ranks, KV 514 399),
boot_ctrl_fp16.log (KV 497 499), lce1/m1_p23f_A.log (MATH 0/80+0/80,
TOOL 30/30 salad), lce1/m1_tool_discriminator.py output in transcript +
lce1/m1_forensics.log vs lce1/ctrl_forensics.log (side-by-side F1/F2/F3).

## P44 — Phase G full testing suite on the certified v1.2.26 lane: ALL GREEN (07:08-08:31 + CC 08:31)

Posture under test: llm-scaler-exp:v1.2.26 (production), fp16 SSM pool +
fp8_e4m3 KV, gmu 0.8, all knobs baked defaults, async + barrier 0, spec
MTP x4, XGrammar-2 0.2.7, parser qwen3_coder. Driver
validate_v1226_run.sh (launched 07:08, setsid-detached; evidence lands in
the internal _v1226_*.txt files, NOT the tee'd log — standing quirk; the
stale-file trap was caught: sustain/p23fB first read as "done" were
YESTERDAY's ship-time leftovers, distinguished by mtime).

- SANITY + ADMISSION: PASS (admission c0 ttft 13.30s, c2 8.47s; V123
  posture async-engaged, barrier "False False 0", RPC 60000; V125 code
  deltas opsall=1 c7>=1 fixline=1; v124 runtime fp8 resolution OK;
  C7-on-lane refuse_rc=1 with the v126 message, unset-import clean).
- SOLO COLD seed114 34k: code=200, ttft 96.29s (parity with lineage).
- WEDGE DRILLS 3x14-phase: SUSTAIN_COMPLETE_NO_WEDGE, fence-hits 0.
- FAIRNESS V66/V63 probe under 106k prefill: big 200, decode 961 tok
  clean, gaps during steady 0.06s (max 3.63 at window edge), post 0.
- SERIAL24 x3: 24/24 + 24/24 + leg3 24/24 (ok=24 fail=0 each).
- BURST_HARSH a/b: 36/36 each, resets_after=0; dmesg new engine resets 0.
- PARSER battery (engine-direct, qwen3_coder): T1 flat-tool PASS
  (run_command + args), T2 nested-tool PASS (list n=1), T3 plain PASS.
- GENSPEED: solo 1x1024 72.48/73.55/73.55 tok/s (sync-parity floor 70 ✓);
  async 4x1024 aggregate 181.80 tok/s (gate >=100; banked posture number
  149.80 came from the longer-window ship probe — this battery's r0 window
  is shorter; both far above gate).
- P23F QUALITY A/B: [A fresh] MATH 0/80 wrong, serial 0/80, flips 0, TOOL
  30/30 clean (73s); [B post-battery prefix] MATH 0/24, TOOL 30/30 clean
  (47s) — no degradation through the whole battery.
- JIT/CACHE: recheck lines 8 (= first-use loads, monitor semantics per
  KNOWN_ISSUES #28), cache 80 vs raw 79 -> delta 1 (bound 2; the strict
  delta-0 law applies to committed production images via the lane gates,
  this is the battery's own bound on a live lane with novel probe shapes).
- SERVE LOG: 0 tracebacks after boot.
- CC BATTERY through litellm :4000 (cc_battery_v1225.sh, after the speed
  gates — sequential law): litellm 200; t1/t2/t3 all http=200; thinking
  leg returns a thinking block; CC_BATTERY_V1225: ALL GREEN.

Verdict: the v1.2.26 production posture reproduces its full certified
battery clean — Phase G closed. Phase H (v1.2.27 bake: capability deltas
ONLY = dormant m0-live collector; scaled-space excluded per P43) cleared
to proceed. Evidence: /root/build/_v1226_sanity.txt, _v1226_validate.txt,
_v1226_sustain.txt, _v1226_p23fA.txt, _v1226_p23fB.txt,
lce1/g_v1226validate.log (=== VALIDATE_V1225_RUN ALL GATES PASS
08:08:20 === + V1225_VALIDATION_COMPLETE), cc_battery transcript in
session log.

## P45 — Phase H: v1.2.27 BAKE + PROMOTION VALIDATION + SHIP (2026-10-01)

Task: bake llm-scaler-exp:v1.2.27 with the round's capability deltas ONLY
(dormant M0-live collector per P42; scaled-space EXCLUDED by the P43
recurrence-compounding law), full promotion validation, ship. All
execution direct (no subagents, standing directive).

### H1 — bake (stage5_bake_v1227.sh, generated by mk_stage5_bake_v1227.py)
- Fresh lsv-bake from llm-scaler-exp:v1.2.26-raw; ONLY delta applied:
  m0_live_collect_v127.py (DORMANT — no /root/.v127_m0live marker, no
  scales/dumps in image). All inherited lineage asserted: dualbridge+C7
  markers, prod .so sha 1d9dcf4e..., wheel d20260925, v1218..v1226
  pedigree + jit stamps, baked serve config (gmu0.8 bs64 e4m3 pc-ON
  spec-ON mnbt8192 async-ON mamba-fp16 no-template).
- New bake gates: m0live presence (marker=2), scaled-space code/marker/
  scales/dumps ABSENCE, no bak_v127 files, bake_m0live_dormant=0 under
  warm traffic. Boot diff = exactly 3 expected deltas.
- RESULT: v1.2.27-raw = 810b98f26d7f (24.8GB), 112 GATE-OK / 0 GATE-FAIL
  (08:47:09). Repro boot script repro_bootV1227.sh (marker gate
  .llm_scaler_exp_v1227_baked + M0LIVE-COLLECTOR grep) generated+staged.

### H2 — promotion validation on fresh boot (validate_v1227_run.sh)
- Fresh boot from v1.2.27-raw (post host-reboot hygiene), HEALTH_OK
  ~170s, baked config verified.
- NEW v127 gates green: m0live collector marker=2 dormant=1
  scaledspace_absent=0.
- Full battery: sanity+admission PASS (ttft 13.30/8.47s), solo cold
  96.29s, drills 3/3 SUSTAIN_COMPLETE_NO_WEDGE fence-hits 0, serial24 x3
  24/24, bursts 36/36 resets 0, parser T1/T2/T3 PASS, genspeed solo
  78.77/79.90/79.79 (floor 70), async 4x1024 150.38 agg (gate >=100),
  P23F A fresh 0/80 wrong + 0 flips + 30/30 tool-clean (74s), B
  post-battery 0/24 + 30/30 (47s), cache delta 1 (bound 2), tracebacks 0.
- RESULT: === VALIDATE_V1225_RUN ALL GATES PASS 09:55:53 === +
  V1225_VALIDATION_COMPLETE x2 (caps literal = lineage rename carryover;
  the ship precondition greps it by design).

### H3 — ship attempt 1: GATES-SCRIPT GENERATOR BUG (root-caused + fixed)
- ship_v1227.sh ran clean through preconditions, SHIPWARM relaunch, warm
  rounds (new_jit=0, cache 79), lane-commit (v1.2.27 = c92b8701f7f9) —
  then ABORTED at gates: "GATE-FAIL v126_no_p29h_markers expected=0
  got=0".
- ROOT CAUSE: mk_gates_v1227_lane.py's step-4 insertion anchor matched a
  line PREFIX (ended at ...print s+0}'" without the line's closing `)"`),
  so the inserted v127 block SPLIT the v126 command mid-line and stranded
  the orphan `)"` on the v127_no_bak_files line. Bash parsed a mega
  command substitution that swallowed the whole v127 block; ck() received
  "0" + swallowed newlines as its actual-value argument -> string compare
  failed on INVISIBLE bytes (log shows identical expected/got).
  Ship-side evidence was tail -80 only — the GATE-FAIL sat above the
  window (grep the full lce1/v1227_gates_lane.log, not the _run.out).
  All 7 v127 gates themselves ran GREEN; committed image untouched
  (host-side script bug only).
- FIX: fix_gates_v1227.py restored the v126 line to its certified full
  form + removed the orphan paren; repo generator anchor hotfixed to
  match through EOL. Verified: direct gates run against the committed
  image -> === V1227 SHIP GATES: ALL PASS 10:14:33 ===.
- LESSON (generator discipline): a replace-anchor must match through the
  END OF LINE, never a mid-line prefix — a prefix match silently
  re-parenthesizes the script; and ck()-style failures with
  expected==got displayed mean invisible bytes (multi-line swallowed
  substitution), always inspect the full log not the tail window.

### H4 — ship attempt 2: ALL GREEN
- Full chain re-run (certified sequence, fixed gates): SHIPWARM relaunch,
  warm rounds new_jit=0 cache 79, lane-commit with dormant-refusal gates
  (no ARMED m0live / no scaled-space marker — refuse to commit),
  llm-scaler-exp:v1.2.27 = a7a950148fef (24.8GB), === V1227 SHIP GATES:
  ALL PASS 10:23:39 ===, fresh prod-boot verification incl. NEW gate
  "v127 m0live collector (dormant) PASS", CC battery t1/t2/t3 http=200 +
  cc_thinking=thinking, watchdog repoint + restart + enable.
- RESULT: === SHIP_V1227 ALL GREEN 10:35:24 ===. SHIP-MEASUREMENT LAW:
  prod lgrf .so sha 1d9dcf4e... asserted by the in-chain gates.

### H5 — post-ship watchdog hardening (3 findings, all fixed same hour)
1. Ship WARN "no fresh armed-line yet" was window timing only:
   10:41:36 alive armed code=200 cycles=10 — watchdog CONFIRMED ARMED,
   pause marker removed, service enabled.
2. The ship's watchdog-repoint sed was a NO-OP: lane_watchdog.sh was
   rewired (2026-09-30) to boot_v1227_restore.sh, whose V1227_IMAGE
   default still said v1.2.26 — a wedge would have restored the PREVIOUS
   image. Fixed: default bumped to llm-scaler-exp:v1.2.27 (file-level
   edit; invoked fresh per WD cycle, no daemon restart needed). v1.2.27
   carries BOTH lineage markers (v1226+v1227) so the restore marker
   checks pass.
3. watchdog_v2_drift_check.sh cried 7 DRIFT lines on the CERTIFIED lane:
   its expectations froze at the pre-certification DESIGNATION (external
   swift mount, --quantization fp8 + --chat-template flags, image pinned
   v1.2.26) while the W1 certification + bakes FOLDED the posture into
   the checkpoint (pre-quantized fp8 swift export at /models/target,
   bind source qwen3.8-27b-fp8; template baked into tokenizer_config.json
   — ship gates assert the flags' ABSENCE). Guard is WARN-only by design
   (exit 0 always) so no action risk — but a monitor that cries drift on
   the certified posture trains DRIFT-blindness, the exact failure mode
   (2026-09-30 Cause A) it exists to prevent. REBASED onto the certified
   surface: EXPECT_IMAGE=v1.2.27, ckpt mount/model/served-name checks,
  runtime --quantization/--chat-template overrides treated as DRIFT
   themselves, template checked host-side in tokenizer_config.json
   (V1227_CKPT_DIR overridable), legacy branch retained for rollback
   windows. Verified: manual run -> V1227_DRIFT_NONE
   posture=certified-swift-fp8 10:45:18; wired every 10th WD cycle.

### Standing posture after P45
- PRODUCTION: llm-scaler-exp:v1.2.27 = a7a950148fef (lane lsv-test UP on
  it; -raw = 810b98f26d7f retained). v1.2.27 = v1.2.26 posture + dormant
  M0-live collector capability (arm with /root/.v127_m0live) + v1227
  pedigree. Zero knob deltas; scaled-space absent by gate.
- Watchdog: armed (100s cadence), restore = boot_v1227_restore.sh ->
  v1.2.27 + telemetry arming; drift guard certifies the lane.
- Round state: W1 certified (P34), D-knob program CLOSED (D1/D2
  rejected), M1 rejected + recurrence-compounding law (P43), M2/M3/WS-E
  CLOSED with reopen conditions A (int8-scaled pool) + B (fp8 frozen-
  state spillover) documented in FIX_AND_TEST_PLAN.md, Phase G battery
  green (P44), Phase H shipped (P45). v1.2.28 CANCELLED.
- Evidence: lce1/v1227_ship.log, v1227_gates_lane.log,
  v1227_validate_run.log, boot_v1227_shipwarm/prod.out; artifacts
  stage5_bake_v1227.sh, gates_v1227_lane.sh (fixed), validate_v1227_run.sh,
  ship_v1227.sh, repro_bootV1227.sh + mk_* generators, fix_gates_v1227.py,
  watchdog_v2_drift_check.sh (rebased) — all mirrored in repo
  vllm/patches/prod/perf-v127/.
