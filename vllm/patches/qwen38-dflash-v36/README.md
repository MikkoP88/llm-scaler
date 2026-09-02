# v36 — all keeper improvements BAKED into the image

`llm-scaler-vllm-adv:v36` = certified v31.1 base (#11 fix posture:
whole-step XPU capture, no inductor compile) + all keeper patches baked
at build time. This replaces the boot-time patch pipeline for prod
(bootp.sh keeps applying patches to v31.1 boots for rollback).

## Baked (md5-gated against committed repo copies at build time)

| Patch | Effect | Lane |
|---|---|---|
| `v32_mq_regpatch.py` | MQ stage1 per-row 2D score+value accumulation | active, prod |
| `v32_align_async_v2.py` | async accepted-count D2H + deferred postprocess | active, spec; nospec-inert |
| `v35_k4_unclamp.py` | k>3 user-selectable (#12 docs-only warning) | no-op nospec/k<=3 |
| `v33_mq3d_triton.py` | fp8-KV spec-verify Triton MQ3D path (#15 fix) | fp8 lanes only |
| `v34_shim_opt.py` | per-call overhead cache on the v33 shim | fp8 lanes only |
| `v34_e5m2_guard.py` | e5m2+fp8-ckpt probe opt-in (#19) | env-gated, inert |
| `v34_f8bi_pin.py` | FA2 split pin (#18 candidate) | env-gated, inert |

EXCLUDED: `v34_gdn_cache.py` — falsified at approach level (v34
addendum; host-only, never committed).

Build: `Dockerfile` in this dir; build context `/root/build/v36ctx/` on
the host (Dockerfile + the 7 scripts, md5-verified against the repo
before build). `/root/.v36_baked` marks the image; `bootp.sh` detects
it and skips boot patching + the restart cycle (single-cycle boot,
~4.5 min vs ~7 min).

## Verification (2026-09-02, all gates)

- **File-state equivalence**: the 4 files patched by v32+v35 are
  md5-IDENTICAL between the baked image and the verified boot-patched
  prodv38 container (xpu.py, triton_turboquant_decode.py,
  gpu_model_runner.py, mamba_utils.py).
- **Prod battery (prodv36i)**: f8ref hashes EXACT refs
  `0ce080630035/f167d905a10b/87d640ad2ed6`; coh P1/P2 distinct=1,
  Paris -0.451 x3; 2k warm 33.56; 65k warm 22.30; conc16 47.3/113.8
  (parity with v31.1 refs 47.2/114.5).
- **128k/256k long-context A/B vs v31.1 (same seeded prompts)**:
  128k cold 121.1s/7.43 vs 143.7s/6.26; 128k warm 51.2s/17.59 vs
  51.2s/17.58 (IDENTICAL wall); 256k cold 249.6s/3.61 vs 264.8s/3.40;
  256k warm 75.5s/11.91 vs 76.9s/11.70. v36 >= v31.1 on every metric —
  zero degradation.

## Boot

```bash
setsid nohup bash /root/build/bootp.sh nospec '' <log> '' \
  llm-scaler-vllm-adv:v36 > /root/<log>_boot.log 2>&1 < /dev/null &
```

Rollback = drop the image argument (v31.1 default, boot-time patches).
v31.1 image RETAINED on the host as the certified base + fallback.
