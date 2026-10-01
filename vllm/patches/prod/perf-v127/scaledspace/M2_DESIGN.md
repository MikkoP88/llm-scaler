# M2 — Scaled-Space e4m3 SSM Kernel (v127 weeks-track, design rev 1)

Status: **BUILT + HARNESS-PASSED 2026-10-01** (PHASES P36). v132 .so =
`/root/build/v132_so/custom_esimd_kernels_lgrf.so` sha `efb53bef…`
(112,607,648 B; production symbols additive-untouched). Harness
verdict `M2_HARNESS_PASS`, all 7 gates: A identity-at-unit-scale
BITWISE; B boundary idempotence mism=0 both regimes; C end-to-end
STRICT strat mean 0.00588 vs raw 0.00600, max 0.1758 vs 0.1797; D
storage STRICT strat mean 0.020134 vs 0.020308, **max 1.5730 vs
1.9688 = 20 % better** (the P23F flush tail). Uniform legs = parity
band 1.05 (same relative grid; rationale in the harness docstring).
Speed spot contaminated (throwaway container queues behind the live
lane) — quiet-window bench owed at M3 pre-ship. NOT on any lane: the
swap is a separate manual step inside a sanctioned window.

Started 2026-10-01 (user directive: "start the weeks-scale
scaled-space fp8 SSM kernel project… do not stop the running vLLM
instance"). Lane untouched throughout — all builds run in the
`esimd-inc` builder container (CPU-only), the .so is staged to
`/root/build/v132_so/` and NEVER installed on a lane without a
sanctioned window + full battery.

## 1. Problem

Raw e4m3 SSM pools are FORMAT-BROKEN for GDN state (P23D/P23F: 8.75-41.7%
concurrent wrong answers, 20% tool salad) — but not because 8 bits are
too few. GDN state features span a dynamic range of ~4-5 orders of
magnitude across `(hv, k)`; e4m3's fixed grid (max 448, min normal
2^-6, denormal floor 2^-9) crushes small-magnitude features into the
denormal tail or zero, while large features quantize fine. Full-fp8 SSM
*arithmetic* is FORMAT-IMPOSSIBLE (proven by telemetry, v1.2.26 round);
the remaining path to an fp8 SSM pool is **scaled storage with fp32
arithmetic**: keep math in registers exactly as the production fp8 lane
does, but map each feature's live range onto the full e4m3 grid.

## 2. Storage convention (M1 == M2)

```
scatter:  stored_byte[hv, v, k] = e4m3( h * scale[hv, k] )
gather:   h (fp32 register)      = e4m3_load( stored_byte ) * (1 / scale[hv, k])
```

- `scale[hv, k] = 0.98 * 448 / runmax[hv, k]`, `runmax` from **live
  telemetry** (M0 `calibrate_offline.py` from P29M collector dumps —
  production scales MUST come from live magnitudes, never synthetic).
  **Calibration-policy delta vs M0** (recorded, resolved at M3): the
  current `calibrate_offline.py` clamps `scale = min(1.0, …)` — an
  M1-bridge-era overflow guard that never scales UP small features and
  therefore cannot fix P23F. The M2 policy drops the `max=1.0` clamp
  (up-scaling small runmax onto the full grid is the whole point);
  dead-feature pinning (runmax floor) stays. The kernel accepts any
  positive finite scales — this is purely an M0 policy edit at M3 bake
  time. The harness's `derive_scales` implements the M2 policy.
- Scale granularity: per `(hv, k)` — every state row of a work-group
  shares the same K-lane layout, so the lo/hi 64-lane scale slices are
  work-group constants loaded ONCE per launch (512 B per WG; the divide
  for `1/scale` happens in-register with the same IEEE op the harness
  uses, keeping kernel and reference comparable).
- Pools written by the M1 python bridge and by the M2 kernel are
  byte-compatible: both store `e4m3(h*scale)`.
- Relative error of in-range values is UNCHANGED vs raw e4m3 (same grid,
  scaled): worst ~2^-4. The win is coverage: every feature sits in-range
  by construction. P23F's small-feature flush is eliminated.

## 3. Numerics discipline

- Arithmetic stays fp32 in registers — identical dataflow to the
  production fp8 lane (v126 P29A register chain preserved: h* carriers
  persist across draft tokens; the scale applies exactly once per pool
  boundary crossing — first load and each checkpoint save — never
  compounding).
- Conversions reuse the exact c10-parity `esimd_e4m3_load/store`
  primitives (P29A) — no flush-to-zero simplifications.
- Rollback safety: the wrapper-side ring snapshot (P29B-FIX) stays a raw
  BYTE gather; snapshot rows are stored-space and decode with
  `inv_scale` in-kernel. No wrapper change beyond passing scales.
- READOUT LAW respected: the wrapper validates only tensor metadata;
  scale finiteness/positivity is guaranteed at calibration time (M0),
  never by a per-launch `.item()` device sync.

## 4. Implementation (this repo dir → build tree)

| File | Role |
|---|---|
| `gdn_conv_fused_seq_spec_scaled.h` | New kernel: sibling of the production spec kernel, e4m3-only, `m2_scaled_load_64` / `m2_scaled_store_64` primitives, scale preload once per WG. Line-identical to production except the pool I/O boundary. |
| `m2_install.py` | Idempotent installer (`v127 M2` markers): copies the header, adds include + wrapper `esimd_gdn_conv_fused_seq_spec_scaled` to `esimd_kernel_lgrf.sycl` (mirrors the P29B-FIX snapshot logic), declaration to `include/kernel_ops.h`, `m.def/m.impl` to `torch_extension_lgrf.cc`. Production symbols byte-untouched. |
| `build_esimd_v132.sh` | v130-pattern build: `esimd-inc` container, `setup_gdn_only.py`, `MAX_JOBS=52 KERNELS_MAX_JOBS=52` (standing directive), symbol proof, stage to `/root/build/v132_so/`. Never touches `lsv-test`. |
| `m2_scaled_harness.py` | GPU validation harness (throwaway `--device` container): gates A-D below + speed spot. |

Op signature (pybind): production spec schema + `Tensor ssm_scales`
([HV, K] fp32 contiguous) after `scale`. Dispatch is explicit — the
caller (vLLM python side, M3 bake) selects the scaled op; the default
path and the raw-e4m3/fp16 dispatch are untouched.

## 5. Validation gates (harness)

- **A identity-at-unit-scale**: `scaled(scales=1)` BITWISE == production
  spec op on an e4m3 pool (outputs, z_out, final pool bytes). Proves the
  transplant is exact with the scale neutralized.
- **B boundary idempotence**: every pool byte after a scaled run decodes
  (*inv) and re-encodes (*scale, RNE) to the same byte (fp32 boundary
  flips < 1e-5, one-grid-step adjacency only). Proves the convention
  end-to-end.
- **C end-to-end superiority**: on magnitude-stratified states (per-
  feature runmax ~1e-3..30, the P23F regime), output error vs fp16-pool
  truth ≤ raw-e4m3's error (mean AND max).
- **D storage superiority**: bridge-encoded storage decode error ≤ raw
  e4m3's, stratified (the P23F fix measured at the storage layer).

Later (M3 pre-ship, sanctioned windows): superiority matrix vs the
v1.2.26 banked numbers (fp16 baseline banked +30.2% agg4x256 is the
bar), P23-style concurrent-answer battery, crash battery, spec MTP×4 +
XGrammar-2 gates.

## 6. Rollout / rollback plan

1. **Now (no-stop window)**: build v132 .so (CPU-only), harness in a
   throwaway GPU container (tiny probes; lane untouched).
2. **M3 bake (sanctioned stop window)**: pool-format conversion at boot
   (M1 `patch_scaled_space_v127.py` arms the format; scales from M0 live
   telemetry `v127_scales_e4m3.pt`), python dispatch selects the scaled
   op, bake `v1.2.28`. Pre-bake: M0-live scale collection boot (P29M
   patch window) + W2 legs per the one-leg-per-window order.
3. **Rollback**: the scaled op is additive; a lane with the v132 .so can
   run the unscaled path by simply not arming
   `V1227_SCALEDSPACE`/scales. Pool-format rollback = fresh boot without
   the M1 arm (existing pools of the other format are simply not read).

## 7. Success criteria

- Harness gates A-D all PASS on the v132 .so.
- Superiority: scaled-pool lane ≥ fp16-pool lane on quality battery
  (P23 concurrent answers) at the SSM-pool capacity dividend (half the
  bytes → freed memory → KV/attack on the C1 SSM-pool concurrency cap).
  Only then does M3 bake proceed. No degradation anywhere (standing
  law); if scaled-space cannot beat the fp16 GDN pool on quality, the
  project records the measurement and stops — measurement over belief.
