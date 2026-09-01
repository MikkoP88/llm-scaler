# qwen38-dflash v31/v31.1 — graphs x speculative decode: #11 ROOT-CAUSED and
# GATED, #12 clamped

Image lineage: `llm-scaler-vllm-adv:v30` (other editor's spec+TP>1 fail-safe
gate) → `:v31` (k-clamp + split knobs) → **`:v31.1` (the fix posture:
whole-step XPU graph capture + inductor compile disabled, as the DEFAULT gate
for spec+TP>1)**.

## v31.1 — the fix (gate_v311.patch + Dockerfile.v31_1)

The 2026-09-01 GPU-window discriminator matrix (below) convicted the #11
wedge to the **inductor-compiled piecewise path x speculative decode** — NOT
XPU graph capture itself, NOT any custom kernel, NOT oneCCL. v31.1 replaces
the v30 gate's fully-eager posture with the convicted-clean configuration:

- OLD (v30): spec+TP>1 → `VLLM_XPU_ENABLE_XPU_GRAPH=0`,
  `TORCH_COMPILE_DISABLE=1`, `CompilationMode.NONE`, `CUDAGraphMode.NONE`,
  `enforce_eager=True` → 9.9 tok/s canonical, clean.
- NEW (v31.1): spec+TP>1 → **only** `TORCH_COMPILE_DISABLE=1`. XPU graphs
  stay on; the `CUDAGraphWrapper` captures the whole decode step with eager
  kernels (dynamo never engages, splitting_ops inert). Validated clean
  through the full 65k battery at **17.7 tok/s canonical (+79%)**, coh
  bit-stable (Paris −0.451 == eager reference), boot ~100 s faster.
- `VLLM_XPU_ALLOW_UNSAFE_SPEC_TP_GRAPH=1` still restores the wedging
  compiled configuration for on-demand reproduction of #11.

The v31 k-clamp (patch A) is retained and load-bearing: #12 reproduces under
capture-without-compile (capture-level defect, see below).

## Patch A (v31) — #12 exposure clamp: k=4 → 3 under any active capture

Piecewise/whole-step capture corrupts temp-0 numerics exactly at
`num_speculative_tokens=4` (k<=3 bit-stable == eager reference, KNOWN_ISSUES
#12). Whenever `cudagraph_mode != NONE` with a speculative config present —
TP=1, or TP>1 gated (v31.1) or bypassed — k is clamped to 3 with a warning.
`VLLM_XPU_ALLOW_K4_CAPTURE=1` keeps k=4 for reproduction/diagnosis.

## Patch B (v31) — GDN split arm: REDUNDANT in hindsight, kept as knobs

`VLLM_XPU_SPEC_EAGER_GDN` (default 1) appends
`vllm::gdn_attention_core(_xpu)/vllm::gdn_attention` to `splitting_ops`;
`VLLM_XPU_SPEC_EXTRA_SPLITS=csv` overrides the list. Post-hoc reading of
`vllm/config/compilation.py` shows both GDN ops are ALREADY in the default
`_attention_ops` split list — patch B changed nothing (duplicate-suppressed)
and arm 1 was effectively "v30-bypass + k-clamp". The knobs remain useful:
arms 1b/1c used `VLLM_XPU_SPEC_EXTRA_SPLITS` for build-free kernel-family
exclusion. NOTE: over-splitting (all esimd ops at once) requires
`VLLM_DISABLE_COMPILE_CACHE=1` or the compile-cache artifact fails
serialization.

## Discriminator matrix (2026-09-01, 2x Arc Pro B70, qwen3.8-27b fp8, TP=2,
## MTP k4→3, oneCCL 2021.15, standard provocation battery)

| configuration | 65k fox | 65k long_exp | canonical | verdict |
|---|---|---|---|---|
| compile + capture (v30 bypass / v31 arm 1) | WEDGE @1 chunk | WEDGE @1 chunk | 10/10 CLEAN @17.4 | wedges |
| + GDN splits (arm 1 = redundant) | WEDGE @1 | WEDGE @1 | 10/10 @17.4 | wedges |
| + `moe_ops::moe_forward_full_fp8_block` split (arm 1b; the ESIMD MoE decode variant missing from `_esimd_moe_splits`) | WEDGE @1 | WEDGE @1 | 10/10 @17.3 | wedges |
| + ALL custom ops split (arm 1c2; every `custom_esimd_kernels_vllm::*` gemm/gemv/norm/qkv_rope + `_xpu_C::fp8_gemm*`) | WEDGE @1 | WEDGE @1 | (killed early) | wedges |
| compile, no capture (other editor's v30_safe_p2) | — | — | WEDGE @563 chunks | wedges (slow onset) |
| **capture, NO compile (arm D: `TORCH_COMPILE_DISABLE=1` + graphs)** | **6/6 CLEAN** | **3/3 CLEAN** | **10/10 CLEAN @17.7** | **CLEAN** |
| fully eager (v30 gate, reference) | 6/6 CLEAN | 3/3 CLEAN | 10/10 @9.9 | CLEAN |

Conclusions:

1. **All vllm/esimd/`_xpu_C` custom kernels are exonerated** — the wedge
   survives splitting every one of them out of the captured pieces.
2. **XPU graph capture is exonerated** — whole-step capture with eager
   kernels is clean at 65k, and faster than the compiled posture.
3. **The wedge is the inductor-compiled piecewise path x spec decode.**
   Capture accelerates onset ~500x (chunk 1 vs chunk 563). Flight-recorder
   evidence fits: every stall gap ends with `AR end` then silence — the eager
   collective retires, the following compiled/replayed region spins.
4. **#12 (k4 corruption) is capture-level and compile-INDEPENDENT**: k=4
   under capture-no-compile still corrupts (coh P1 distinct=3, drifting
   logprobs), with no wedge. k-clamp stays mandatory on any capture posture.

Certification of v31.1 (default posture, no bypass env): gate markers
(inductor-disabled ×4, k4→3 ×1, no eager fallback), coh bit-stable, full
battery fox 6/6 + long_exp 3/3 + canonical 10/10 ALL CLEAN — see
KNOWN_ISSUES #11 v31 update for the exact numbers.

## Remaining debug target (optional, for a true upstream fix)

The inductor codegen for the decode pieces under spec shapes (vllm
`compilation/` backends + torch inductor `combo_kernels`) — reproducible on
demand with the bypass env; wedge signature w122359-class. The kernels lane
(`/root/build/vxk`) is NOT implicated by #11; it remains the suspect surface
for #12 only.

## Files

- `spec_fixes.patch` + `Dockerfile` — v31 image (patch A + patch B/knobs)
- `gate_v311.patch` + `Dockerfile.v31_1` — v31.1 image (the fix posture)
- `gpu_window_runbook.sh` — arm-1 boot/probe script (staged to host
  `/root/build/v31/`); arms 1b/1c/D/D-k4/cert were run with the same
  `serve_boot_var.sh` pattern varying only EXTRAENV and image tag.
- `.tmp-tq/v31_semantic_test.py` (repo scratch) — 6-case CPU semantic test
  of `check_and_update_config` against the v31 patch.
