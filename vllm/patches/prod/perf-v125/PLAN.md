# v125 ROUND — Kernel-native fp8 GDN state, end-to-end (P19.5b/c + full chain)

User directive (2026-09-28, verbatim intent): *"Start write full end-to-end plan of
implementation, when plan is ready run full implementation and testing suite, goal is
fix all possible issues and bottlenecks to archive a big improvement, bake new
production image llm-scaler-exp:v\* using best values and improvements. Important!
Spec and XGrammar-2 has to have supported and also e5m2 and e4m3 are supported
end-to-end. Do not use subagents."*

## 0. Standing requirements (all phases, no exceptions)

- **Spec MTP ×4 + XGrammar-2 0.2.7 supported on every image, always.**
- **e4m3 AND e5m2 `--mamba-ssm-cache-dtype` supported END-TO-END** (boot → serve →
  generate → tools → graphs): selectable, tested, quality-verdicted. "Supported" ≠
  "default" — the default serve dtype is decided by measurement (P24).
- BARRIER default 0 (never 1), `--async-scheduling` + `VLLM_RPC_TIMEOUT=60000`
  posture, parser qwen3_coder, 85 cudagraph capture sizes.
- Every omix-devel wheel build uses `KERNELS_MAX_JOBS=52` (standing directive).
- Host reboot of 10.20.3.65 before each fresh-host test round; reboot after big changes.
- Live investigation record in PHASES.md after every phase; full write-up + git commit
  at round end. No subagents — all execution direct.
- litellm through :4000 needs `Authorization: Bearer sk-dummy`; container re-creates
  must keep `-e LITELLM_MASTER_KEY=sk-dummy`; never spread litellm config keys.
- Boot scripts: only V1223+/V1224-lineage (async REQUIRED gate); V1212-lineage scripts
  FORBID async = TRAP.

## 1. Baseline and prerequisites

- **P20 (v1.2.24 ship) completes first** — the kernel round iterates ON TOP of it:
  v1.2.24-raw = v123 posture + P16 non-spec gate fix + P19.5a python plumbing (fp8
  accepted E2E via the SYCL bridge; inert on the certified fp16 serve).
  If P20 validation is green → ship v1.2.24, re-arm watchdog, THEN start P22.
  v1.2.23 stays the certified fallback image throughout.
- Certified performance refs (v1.2.23/v1.2.24 parity targets): solo median 74.19
  tok/s, 4×1024 async aggregate 152.25 tok/s, lane jitwarm count 731.

## 2. Key discovered facts (P21 fact-finding, 2026-09-28 — all verified on host)

1. **TWO kernel surfaces, and the ROUTING is the key fact** (verified
   2026-09-28, corrected after deep trace — the first fact-finding pass
   over-credited the ESIMD path):
   - **Surface B — SYCL `torch.ops._xpu_C.gdn_attention`** (package
     `vllm-xpu-kernels 0.1.8.3.dev0+g3cab97a.d20260925`, source
     `/root/build/vxk/csrc/xpu/gdn_attn/`) is **the DOMINANT production
     surface**: `gdn_linear_attn.py`'s own docstring (`:~950-956`) —
     "single-request speculative batches use the rollback-aware sequential
     variant. **Multi-request, mixed, and prefill batches stay on the upstream
     gdn_attention op**". Under the production spec-MTP-×4 fleet (4-8
     concurrent streams → `num_spec_decodes > 1`) EVERY decode step runs
     Surface B, even on fp16 today. `vllm.gdn_attention_core_xpu` is a
     python-registered custom op (`_xpu_ops.py:796-799`) whose impl calls the
     SYCL op — it is NOT a separate kernel. **Already templated**: kernel
     params `StateT* ssm_state` (`gated_delta_rule.hpp:26/:311`), scalar
     static_cast loads/stores (`:120/:247/:409/:539`), all math `float`, and a
     ready-made `DISPATCH_STATE_DTYPE` macro (`:820-837`) branching
     float/bf16/half with a TORCH_CHECK else. Torch 2.11 ships
     `c10::Float8_e4m3fn`/`Float8_e5m2` (header-only float conversions) → the
     B-side change is two dispatch branches + includes; the interface has NO
     ssm dtype TORCH_CHECK (contiguity only).
   - **Surface A — ESIMD fused kernels** (package `custom-esimd-kernels-vllm
     0.1.0`, source tree `/root/llm-scaler/vllm/custom-esimd-kernels-vllm/`
     csrc/xpu/esimd_kernels/ — NOT the older sglang-tree copy, which lacks
     the spec variant) fires ONLY on: non-spec decode batches
     (`esimd_gdn_conv_fused`/`_seq`, num_decodes 1..128) and single-request
     spec batches (`esimd_gdn_conv_fused_seq_spec`, gated
     `num_spec_decodes == 1` + k-heads/tp==8, v-heads/tp==24, dims 128).
     Hard-typed `fp16*` state I/O: `gdn_conv_fused.h` `lsc_load_state_64`/
     `lsc_store_state_64` (`:42/:48`, `fp16* ssm_state_ptr` `:101`);
     `gdn_conv_fused_seq_spec.h` `fp16* ssm_state_ptr` (`:31`), scalar state
     rows (`:66-110`), `block_load<fp16,64>` conv-state. The python gate
     `_gdn_conv_state_fp16_ok` requires BOTH pools fp16 — with an fp8 SSM pool
     these paths turn off and everything lands on Surface B (via the P19.5a
     bridge today, natively after P22-B).
   - **Consequence for the round**: P22-B alone gives fp8 full-engine
     coverage (prefill+mixed+ALL decode incl. multi-spec); the aggregate
     fp8-vs-fp16 A/B measures SYCL state-byte halving DIRECTLY (aggregate is
     SYCL in both formats). P22-A restores the ESIMD fast path for the
     solo-spec-1 / non-spec-decode cases under fp8.
   - **Deferred lever recorded with evidence**: extending the ESIMD spec
     kernel to `num_spec_decodes > 1` would fuse conv+delta-rule for the
     whole multi-stream aggregate regime (today unfused python glue + SYCL
     launch per layer). Potentially the bigger aggregate win than fp8 — but
     a larger ESIMD effort (rollback-aware batched chain). Re-evaluate after
     P23 measurements; candidate centerpiece of a follow-up round.
2. **Torch 2.11 headers ship `c10::Float8_e4m3fn` / `c10::Float8_e5m2`** with float
   conversion operators → the intended implementation is `StateT` instantiation +
   dispatch branches, NOT new math. (`torch==2.11.0+xpu` in the builder venv.)
   All four state touch sites are scalar-indexed static_casts (decode load
   `:120`, decode store `:247`, chunk-init `:409`, chunk-final `:539`) —
   fp8-safe with zero math changes. The launcher macro passes
   `reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr())` — no typed
   accessor, any StateT compiles.
3. **Build discipline exists**: persistent warm builder `vxk-inc`
   (intel/omix:0.1.0-devel-ubuntu24.04, `/root/build/vxk` → `/src/vllm-xpu-kernels`,
   torch at the exact paths build.ninja recorded). `build_vxk_v78_inc.sh` pattern =
   ninja incremental relink of `_xpu_C.abi3.so` → swap into lsv-test (minutes per
   iteration). Full wheel (version vcs-stamped, cf. v88 wheel d20260925) only for
   the bake, with `KERNELS_MAX_JOBS=52`.
4. **Capture-safety**: with the bridge removed there is no `torch.unique` /
   data-dependent shape in the fp8 path → `cudagraph_mode FULL_DECODE_ONLY` is
   restored for fp8 legs (the stage-A runner had forced NONE).
5. **Stage-A lessons that carry over**:
   - acceptance rate ≠ quality (draft and target collapse TOGETHER through a
     degenerate state) → greedy-text probes are MANDATORY alongside acceptance;
   - e5m2 storage is inherently coarse (2-bit mantissa): kernel-native keeps the
     same quantization math, so e5m2 QUALITY IS EXPECTED TO REMAIN DEGENERATE —
     it ships as supported+tested+documented, not as a default candidate unless
     measurement surprises;
   - e4m3 stage-A numerics were coherent (46.9 % acceptance via bridge; expect
     recovery toward fp16 levels once the target engine reads cleanly);
   - `index_copy_xpu`/`sum_xpu` are NOT implemented for fp8 (stage-A pre-flight) —
     irrelevant post-bridge, relevant only if any fallback path is kept.

## 3. Phase plan

### P22 — Kernel fp8-native implementation (iterate on lsv-test)
**Order: B first (small, kills the bridge), then A (the decode hot-path lever).**

**P22-B — SYCL `gdn_attention` fp8-native** (vxk repo, incremental build):
1. Edits (marker-commented `v125 P22B`): `DISPATCH_STATE_DTYPE` +
   `c10/util/Float8_e4m3fn.h`/`Float8_e5m2.h` includes in `gated_delta_rule.hpp`
   (the interface has NO ssm dtype TORCH_CHECK — contiguity only; the dtype
   gate is solely the dispatch macro's else-branch, verified 2026-09-28).
2. Incremental `_xpu_C.abi3.so` rebuild via the v78-inc pattern → swap into
   lsv-test. Full wheel deferred to P24.
3. Python `_xpu_ops.py` (marker `v125 P22B`): capability switch
   `VLLM_XPU_GDN_FP8_NATIVE` (unset → bridge back-compat; 1 → pass
   `self.kv_cache[1]` STRAIGHT to the op — no gather/remap; decision frozen at
   first call, pre-capture; mismatch aborts boot loudly, never silent fallback).
4. Boot e4m3 + e5m2 legs (graphs still NONE at this sub-phase — surface A not
   done yet, and the ESIMD gate still routes decode here); greedy probes +
   acceptance vs stage-A numbers before any speed claims. Measure: how much of
   the 5× bridge penalty dies with native SYCL (expected: most of it; decode
   still on the slower SYCL path until P22-A).

**P22-A — ESIMD fused kernels fp8-native** (`custom-esimd-kernels-vllm`,
source `/root/llm-scaler/vllm/custom-esimd-kernels-vllm/`; covers
non-spec-decode + solo-spec-1 — NOT the multi-stream aggregate, which is
Surface B even on fp16):
1. Full read of the three headers' state I/O + the package binding layer
   (the python wrappers live in vllm `esimd_utils.py:221-307`, thin
   pass-throughs) before editing.
2. Kernel edits (marker `v125 P22A`): fp8 load/store variants per header —
   scalar-path `gdn_load_*` equivalents (uint8 load → dequant; e5m2↔fp16 =
   8-bit shift, e4m3 = rebias+shift with subnormal/NaN edges) and
   vector-path quantizing stores (`simd<float,64>` → fp8 bytes);
   binding-side dispatch on `ssm_state.scalar_type()` (fp16 path
   byte-identical).
3. Build with `KERNELS_MAX_JOBS=52` in the omix builder: the GDN kernels
   live in the lgrf module (`csrc/xpu/esimd_kernel_lgrf.sycl` kernels +
   `csrc/xpu/torch_extension_lgrf.cc` bindings →
   `custom_esimd_kernels_lgrf...so`; vllm imports the package root
   `custom_esimd_kernels_vllm` as
   `_esimd`, `esimd_utils.py:30`) and the tree ships a fast iteration path
   `setup_gdn_only.py` that builds exactly that module — swap the rebuilt
   `.so` into lsv-test (installed 0.1.0 wheel's lgrf .so is from the Aug-20
   tree; verify provenance by fp16-control parity after swap).
4. Python `gdn_linear_attn.py` gate extension (marker `v125 P22A`):
   `_gdn_conv_state_fp16_ok` → accept fp8 SSM pool when
   `VLLM_XPU_GDN_FP8_NATIVE=1` (conv_state stays fp16 — split the check);
   pass-through unchanged otherwise.
5. Boot legs with `cudagraph_mode FULL_DECODE_ONLY` RESTORED. Gate: graphs
   captured, zero tracebacks, greedy probes + acceptance, THEN speeds. The
   P22-A payoff metric is SOLO spec-1 fp8 approaching solo fp16 (~74); the
   aggregate should be Surface-B-limited in both cases (document if not).

### P23 — Full E2E battery, both formats (fresh host after reboot)
For each of {e4m3, e5m2} (and a fp16 control leg):
- sanity/admission; solo ×3 + 4×1024 ×3; contention 7-8 stream replay; wedge
  drill; burst; serial; fairness V66; parser battery T1/T2/T3; CC battery through
  litellm :4000 (python-built JSON + Bearer sk-dummy); tracebacks/resets = 0;
  graphs-ON evidence; jit cache delta gate.
- **Quality verdict per format** (greedy probes + a 200-token creative sample,
  scored for degeneration/repetition) — recorded next to acceptance.
- Speed table vs fp16 refs (solo 74.19 / agg 152.25) — the P22 win claim must be
  measured here, with graphs on.

### P24 — Full wheel + default decision
1. Full wheel build from `/root/build/vxk` (`KERNELS_MAX_JOBS=52`), vcs-stamped
   version recorded; install into a clean lsv-test container (not the .so-swap
   lane) and re-run the decisive battery legs on the wheel.
2. Default-serve decision by measurement + quality:
   - e4m3 clean quality AND ≥ solo-parity AND step-time win → candidate default;
     ship decision documented with numbers (expected: 10–30 % step-time IF the
     state-read share holds — treat as hypothesis, not promise).
   - otherwise default stays fp16; both formats remain first-class supported.
   - e5m2: supported regardless; default only if quality unexpectedly passes.

### P25 — Bake + ship v1.2.25
- `stage5_bake_v1225.sh` from the v1224 pattern: fresh lsv-bake from
  `llm-scaler-exp:v1.2.24-raw` (or v1.2.24 if shipped), apply: new wheel + P22
  python patches + `VLLM_XPU_GDN_FP8_NATIVE=1` env (baked) + decided default
  serve dtype. Full inherited gate battery + new gates: wheel version, fp8
  resolution, native-path marker, default-serve dtype, BOTH formats' E2E smoke
  (boot probe in the bake warm round if default≠fp16, else explicit leg).
- Host reboot → `repro_bootV1225.sh` fresh boot → `validate_v1225_run.sh` full
  battery (v1224 pattern + v125 gates) → ship lane-commit `llm-scaler-exp:v1.2.25`
  → `gates_v1225_lane.sh` → CC battery → fresh-boot prod sanity → watchdog
  repoint to `repro_bootV1225_prod.sh` + re-arm (rm pause marker).
- Spec MTP ×4 + XGrammar-2 0.2.7 gates on every image as always.

### P26 — Post-ship bottleneck re-measure + round write-up
- Re-run the P17 EU telemetry under the SAME 4×1024 load: does stall-51 %/read-55 %
  move? Record the honest delta (including "no change" if that's the result).
- parse_output host share re-check (async posture, post-kernel).
- The 36.5 % GPU-idle share is NOT addressable by this round (scheduler/demand
  domain; V64 K=3 already measured no-gain; raising fleet concurrency is a
  user policy decision) — documented, not silently dropped.
- PHASES.md P22–P26 records, COMPLETE_ROUND_WRITEUP update, git commit
  (Co-Authored-By: Claude Code <noreply@anthropic.com>).

## 4. Risk register

| Risk | Mitigation |
|---|---|
| `c10` fp8 types don't slot into every `StateT` touch site (store form, vectorized loads) | P22 starts from a full source read; worst case a thin proxy struct with `operator float()`/assignment |
| fp8 conversion edge cases (subnormal/NaN/inf) corrupt state | greedy probes + acceptance measured against stage-A BEFORE speeds; rollback = bridge (kept behind flag) |
| e5m2 stays quality-degenerate | expected & documented; ships supported-not-default |
| Native/capture interaction (flag resolved pre-capture) | capability resolved+cached at first call; boot aborts loudly on mismatch, never silent fallback |
| Wheel/link build breakage | warm incremental builder first (.so swap), full wheel only after legs pass; v1.2.24 fallback image untouched |
| Perf win smaller than hoped (state-read share overestimated) | P26 re-measure documents the honest delta; no claim without numbers |
| Time: kernel build round-trips | incremental path = minutes; battery phases reuse v1224 runners (sed) |

## 5. Honest expected outcome

- The single remaining in-engine lever that raises the per-step ceiling: fp8 SSM
  state halves the dominant per-stream state-read bytes in a 51 %-stall /
  55 %-of-peak-read profile. **Where it shows first: the multi-stream aggregate**
  (Surface B carries that regime in BOTH formats, so the fp8-vs-fp16 aggregate
  A/B is a clean state-byte measurement). Solo fp8 needs P22-A to approach the
  fp16 solo number (without it, solo-spec-1 falls to the slower SYCL path).
  Envelope: **~10–30 % step-time** IF the state-read share holds; graphs
  restored for fp8 legs is a second, smaller win vs the bridge. NOT a 2× and
  NOT a fix for the 36.5 % idle share.
- e4m3 = the only realistic default candidate; e5m2 = supported E2E with a
  documented quality caveat. All other posture values unchanged (already at
  measured best).
- The deferred ESIMD multi-spec extension (`num_spec_decodes > 1`) is the
  recorded next-biggest aggregate lever — follow-up round candidate, evidence
  in §2 item 1.
