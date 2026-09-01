# qwen38-dflash v31/v31.1 — graphs x speculative decode: #11 ROOT-CAUSED and
# GATED, #12 clamped — and the perf verdict: spec is a NET LOSS, prod runs
# v31.1 nospec

Image lineage: `llm-scaler-vllm-adv:v30` (other editor's spec+TP>1 fail-safe
gate) → `:v31` (k-clamp + split knobs) → **`:v31.1` (the fix posture:
whole-step XPU graph capture + inductor compile disabled, as the DEFAULT gate
for spec+TP>1)**.

## PROD POSTURE (2026-09-01 decision): v31.1 image, NOSPEC serve config

A controlled before/after sweep (`ctxbench.py`, same client/prompts, greedy,
B70x2 TP=2) showed MTP k3 is a **2-4x throughput regression vs nospec-graphs
at every context length and concurrency tested**:

| posture | conc=1 @2k | conc=4 @2k agg | conc=16 @2k agg | conc=16 @32k agg | conc=1 @65k |
|---|---|---|---|---|---|
| v27 nospec (graphs+compile) | 33.6 | 29.5 | 137.0 | 53.9 | 23.5 |
| v31.1 spec k3 (graphs, no compile) | 16.5 | 15.5 | 35.3 | 13.6 | 7.5 |

The "17.7 vs 9.9 tok/s" that motivated the promotion was an artifact: the
9.9 reference was the v30 gate's **fully eager** posture, not v27-nospec.
Spec decode on this stack pays ~7 row-forwards (draft k3 + verify k+1) per
step for ~1-2 accepted tokens at bs=1 (decode is row-serial at small batch:
conc=4 aggregate is BELOW single-stream), so it never wins at low concurrency,
and at conc=16 nospec is still 4x ahead. Decision: **prod = v31.1 image with
NO speculative config** (v27-class perf, plus every v31/v31.1 safety gate
available for opt-in MTP: #11 gate + k-clamp fire automatically if spec is
ever requested). MTP stays available for reproductions/diagnosis via the
normal `--speculative-config` (auto-clamped k4->3, compile disabled by gate).

Warm parity of the promoted posture (v31.1-nospec vs v27, same client;
first-request-after-boot is compile-polluted at any new prefill shape —
measure warm): conc=1 @2k 33.56 vs 33.57; conc=1 @32k 27.81 vs 27.76;
conc=1 @65k 23.52 vs 23.53; conc=16 @2k steady ~353 tok/s both. Coh
bit-stable Paris -0.451 == eager reference. Gate markers inert (0 inductor
warnings, 0 clamps) — engine posture identical to v27, as designed.

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

Native-stack evidence at the live wedge (2026-09-01, `gdb_wedge_evidence.txt`
in this dir; raw dumps on host `/root/build/keeper_gdb_wedge_evidence/`):
both TP workers' main threads parked identically in
`torch.ops...gdn_attention -> chunk_gated_delta_rule_impl_xe2 -> sycl
q.wait() -> urQueueFinish -> ur_queue_immediate_in_order_t::queueFinish ->
libze_intel_gpu` — the EAGER GDN kernel's wait on an IN-ORDER L0 queue,
i.e. something enqueued ahead of it on the same stream never retires; per
the matrix that work is the inductor-compiled region. Repro trigger nuance:
6 concurrent COLD-prefill 65k requests with unique prompt headers (warm
prefix-cache hits and single streams did not re-trigger on an
already-exercised boot). Filed upstream as the #11 livelock issue with this
chain: https://github.com/vllm-project/vllm/issues/54796 (k4 corruption:
https://github.com/vllm-project/vllm/issues/54785; oneCCL #212/#215
cross-posts as triage datapoints).

## Files

- `spec_fixes.patch` + `Dockerfile` — v31 image (patch A + patch B/knobs)
- `gdb_wedge_evidence.txt` — #11 native-stack evidence at the live wedge
  (raw dumps: host `/root/build/keeper_gdb_wedge_evidence/`; capture tooling
  `gdbwedge.sh`/`gdbnow.sh` on `/root/build/`)
- `ctxbench.py` — context/concurrency decode bench used for the perf
  verdict + parity (staged to host `/root/build/ctxbench.py`; unique
  per-request headers defeat prefix-cache sharing; measure WARM — first
  request at a new prefill shape is compile-polluted)
- `gate_v311.patch` + `Dockerfile.v31_1` — v31.1 image (the fix posture)
- `gpu_window_runbook.sh` — arm-1 boot/probe script (staged to host
  `/root/build/v31/`); arms 1b/1c/D/D-k4/cert were run with the same
  `serve_boot_var.sh` pattern varying only EXTRAENV and image tag.
- `.tmp-tq/v31_semantic_test.py` (repo scratch) — 6-case CPU semantic test
  of `check_and_update_config` against the v31 patch.
