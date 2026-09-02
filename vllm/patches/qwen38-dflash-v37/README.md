# v37 — max-performance image: all keeper improvements + barrier-off default

`llm-scaler-vllm-adv:v37` = v36 recipe + `v33_scalefold.py` (keeper that
v36 missed) + `v37_barrier_default.py` (spec draft-barrier DEFAULT OFF).
Base remains certified v31.1 (#11 fix posture).

## What v37 adds over v36

1. **v33_scalefold.py** — S/P-tile fp8 per-tensor KV scale folding
   (16x fewer f32 scale ops/tile; bit-identical; fp8+spec lanes only;
   v33: 65k 95.1→94.7s, "strictly less work per tile", keeper).
2. **v37_barrier_default.py** — `VLLM_XPU_SPEC_DRAFT_BARRIER` default
   flipped ON→OFF in `gpu_model_runner.py`. The barrier (v2x-era oneCCL
   wedge mitigation) is obsolete under the v31.1 posture; v33 measured
   barrier-off strictly better with unchanged hashes: fp8+spec 2k
   +13.7% (21.16→24.05), 4bit+spec 2k +28.5% (19.65→25.25), 65k +3-4%,
   no wedge in sustained testing. `=1` explicitly restores the drain
   for diagnosis. Spec lanes no longer need the env in the boot recipe.

Excluded (documented, not "not really failed"): `v34_gdn_cache.py`
(REJECTED by A/B gate, 65k warm −45%); `v33_bm16.py` (measured
neutral); MNBT/chunked-prefill 16384 (null); pin-splits + e5m2 (null
as fixes; baked env-gated default-off, inert).

## Build & boot

Build context `/root/build/v37ctx/` (Dockerfile + 9 scripts, md5-gated
against committed repo copies). `/root/.v37_baked` marker;
`bootp.sh` baked-detection generalized to `/root/.v*_baked`
(container-side `sh -c` glob — host-side glob expansion silently
breaks detection; learned the hard way, fail-loud idempotency guards
caught the double-apply with zero corruption).

```bash
setsid nohup bash /root/build/bootp.sh nospec '' <log> '' \
  llm-scaler-vllm-adv:v37 > /root/<log>_boot.log 2>&1 < /dev/null &
```

## Verification (2026-09-02)

- Marker checks passed in-build (all 9 patches + inverted #12 clamp +
  v31.1 gate + barrier "0" default).
- **Prod lane (4bit TQ nospec)** — f8ref hashes bit-exact vs v31.1 refs
  (`0ce080630035/f167d905a10b/87d640ad2ed6`); coh Paris −0.451 ×3;
  2k warm 33.57 tok/s; 65k warm 22.32 tok/s (parity: v31.1 ≈ 22.28,
  v36 ≈ 22.3); conc16 agg 45.4/107.0 with per-stream 11.53 ≈ parity
  (agg delta is EOS-timing variance, #18-class, not decode speed);
  128k cold 121.9 s / warm 51.2 s (17.58 tok/s); 256k cold 264.1 s /
  warm 76.7 s (11.73 tok/s) — ≥ v31.1 everywhere (A/B v36 numbers:
  128k 121.1/51.2, 256k 249.6–264.8/75.5–76.9).
- **k3 spec lane, NO barrier env** (proves the baked default-off;
  boot extraenv empty, `VLLM_XPU_SPEC_DRAFT_BARRIER` absent from the
  container env, tree reads `..., "0") == "1"`): coh Paris −0.452;
  65k warm **29.35 tok/s** vs 29.25 reference (barrier-on state) —
  +0.3%, i.e. the v33 keeper numbers now come for free with no env.
  Spec boots no longer need `VLLM_XPU_SPEC_DRAFT_BARRIER=0`.
- No #11 wedge signature anywhere in the suite (v31.1 posture holds).
