# vllm/patches — status index

Everything under this directory is a change to the installed vLLM tree
(`vllm-0.21.1.dev0+gad7125a43.d20260826.xpu`, B70/XPU, TP=2) made by this
project. Directories are grouped by **status**, not chronology; names state
the **purpose** (what it does / what it targets), not the model codename the
work started under (`qwen38-dflash`).

## Layout

| dir        | meaning |
|------------|---------|
| `prod/`        | KEEPERS — verified improvements, in the production lineage (baked into images since v36) |
| `failed/`      | attempts that DID NOT fix / were falsified — kept as documented negative results |
| `diagnostics/` | instrumentation / forensics / unbuilt recipes — not serving improvements |
| `legacy/`      | superseded era artifacts (full-file overlays, transitional recipes, superseded bakes) |
| `harmonized/`  | the era-1/era-2 tree state re-expressed as explicit per-file patches (see its README) |

Top-level files are **upstream-owned** (they come from intel/llm-scaler and
evolve with it — do not rename): `vllm_for_multi_arc.patch`,
`vllm_xpu_kernels.patch`, `ai-dynamo-xpu/`, `0001-oneccl-align-global-V0.1.1.patch`,
`miner-u.patch`, `oneapi-samples-enable-correctness-check.patch`.
`harm-vaudit.py` is ours: audits an installed tree against the wheel RECORD
manifest (finds every post-install edit + marker coverage).

## Three patch eras (why styles differ)

1. **Era 1 — full-file overlay (v1–v21):** whole modified vllm files were
   copied into images. No patch scripts; the image IS the record.
   `harmonized/` now captures this state as explicit diffs.
2. **Era 2 — transitional (v22–v31):** overlays + `.patch` files + the first
   custom `vllm-xpu-kernels` wheel (v26 GDN fix). Keepers from this era
   (v26 wheel, v31.1 whole-step capture) live in `prod/`; the rest in
   `legacy/` or `failed/`.
3. **Era 3 — surgical scripts (v32–v38):** fail-loud anchor-replace `.py`
   patchers, idempotent, with `llm-scaler vN` tree markers, grep gates and
   md5 checks. **Baked into images since v36** (`/root/.vN_baked` marker
   makes `bootp.sh` skip boot-time patching).

## prod/ — keepers (in production lineage)

| dir | v | what it does |
|-----|---|--------------|
| `gdn-spec-oob-fix-v26/` | v26 | GDN spec-kernel ragged-batch OOB (the #11 wedge root cause) fixed **in the GPU kernels** — rebuilt `vllm-xpu-kernels` wheel (`gdn_spec_fix.patch`, `build_vxk_wheel.sh`) |
| `whole-step-capture-11fix-v31/` | v31/v31.1 | #11 fix posture: CUDA-graph whole-step capture, compile OFF. Base of the v37 bake |
| `attn-mq-2d-and-spec-async-v32/` | v32 | `v32_mq_regpatch.py` MQ verify per-row 2D accumulation (fix); `v32_align_async_v2.py` async accepted-counts (spec lanes) |
| `attn-mq3d-varlen-and-scalefold-v33/` | v33 | `v33_mq3d_triton.py` fp8-KV spec-verify Triton MQ3D path + `v33_scalefold.py` S/P-tile fp8 scale folding (bit-identical). `v33_bm16.py` measured neutral — not baked |
| `attn-shim-cache-and-guards-v34/` | v34 | `v34_shim_opt.py` per-call descale cache (keeper); `v34_e5m2_guard.py` + `v34_f8bi_pin.py` env-gated (inert by default). `v34_gdn_cache.py` REJECTED by A/B gate (−45% @65k) — never committed |
| `spec-k4-unclamp-v35/` | v35 | `v35_k4_unclamp.py` removes the #12 k>3 clamp → k selectable in boot JSON (docs-only corruption warning, KNOWN_ISSUES #12) |
| `image-bake-keepers-v37/` | v37 | bake recipe: ALL keepers from v32–v37 baked into `llm-scaler-vllm-adv:v37` (base of current prod) |
| `attn-esimd-fp8-reroute-v38/` | v38 | `v38_esimd_reroute.py` — **#18 fix**: fp8-KV decode OFF the ESIMD fast path (run-to-run nondeterministic kernel) → vxk FA2 (deterministic AND faster). Latest keeper |
| `image-prod-v1/` | v1 | **first `llm-scaler-prod:vN` bake** — exactly the certified v38 tree (v37 keepers + v38 reroute) under the new naming; 19/19 md5-identical to `adv:v38`. Validated against 5 images (prod:v1, v37, v31.1, v19, v14) — full matrix in its README. Current prod |

## failed/ — documented negative results

| dir | v | outcome |
|-----|---|---------|
| `spec-prefill-peer-barrier-v23/` | v23 | **DISPROVEN** — #11 wedge; slow rank blocks before the barrier |
| `spec-prefill-peer-rendezvous-v24/` | v24 | **DISPROVEN** — wedge still reproduces |
| `spec-every-step-drain-v25/` | v25 | **DID NOT FIX** — every-step pre-drafter drain + collective shrink |
| `exact-kv-splits-v38/` | v38 | hypothesis **FALSIFIED** — `v38_exact_kv_splits.py`: split machinery inactive at short ctx (wkb=2 < 16 → num_splits=1); pin changed nothing. Tooling/evidence only |
| `tq-nibble-unpack-v39/` | v39 | **REJECTED ON PERF** — `v39_tq_nibble.py` (single-load nibble unpack via `tl.interleave`): bit-exact PROVEN (rig 25/25, probe `0ce080630035` ×10, dt_loop hashes 8/8) but decode regresses −9.5/−34/−53% at 2k/16k/65k; `tl.interleave` lowers worse on XPU than the L1-cached second byte load. `v39b_qrot.py` designed, never applied. Do not bake |

## diagnostics/ — instrumentation & recipes (not serving improvements)

| dir | v | what |
|-----|---|------|
| `wedge-endgame-v27/` | v27 | #11 endgame: drafter-comm isolation (neutral), oneCCL reduce kernel convicted ≤32k. Note: `adv:v27` image served prod for a long time |
| `oneccl-1717-upgrade-v27c/` | v27c | oneCCL 2021.17.2 library-overlay image (wedge verdict: version-independent; #12 persists; image recipe keeper) |
| `cudagraph-debug-v28dbg/` | v28dbg | in-engine flight recorder — NOT a serving image |
| `oneccl-wedge-forensics-v29/` | v29 | live wedge capture, named mechanism (oneCCL SYCL-kernel collective spin), exonerations, upstream ticket drafts |
| `oneccl-wait-timeout/` | — | UNBUILT source-build recipe: convert #11 livelock into recoverable error at `ccl_executor::wait()` |
| `kv-dtype-loop-study-v39/` | v39 | KV-dtype × perf/loop matrix (auto/fp8_e4m3/4bit/k8v4) + thinking-trap differential @4096/@8192 + decode env-knob sweep. Verdicts: TQ wins deep prefill; decode gap is triton-vs-ESIMD architectural (env knobs exhausted); "fp8 thinking loops" = model behavior, dtype-independent (fp16 baseline traps identically). See KNOWN_ISSUES #20 |

## legacy/ — superseded eras

| dir | v | note |
|-----|---|------|
| `dflash-overlay-v1-12/` | v1–v12 | original era-1 overlay work (base `qwen36-b70-vllm:b3-maxperf-final-v7`, dspark drafter). Historical record |
| `reliability-memory-overlay-v17/` | v17 | era-1 reliability/memory overlay |
| `reliability-memory-overlay-v18/` | v18 | era-1 continuation |
| `tq-drafter-mq-verify-v19/` | v19 | TQ × spec enablement (`config/vllm.py` #05b orphan edit originates here) |
| `spec-vs-target-bench-corrected-v20/` | v20 | spec vs target, corrected benchmark methodology |
| `spec-kv-dtype-vram-v21/` | v21 | spec KV-dtype / VRAM matrix |
| `mtp-cudagraphs-eager-head-v22/` | v22 | MTP graphs with eager draft head (#05d `sched/utils.py` ignore_eos guard era) |
| `failsafe-spec-tp-graphs-v30/` | v30 | failsafe spec+TP+graphs — superseded by v31.1 posture |
| `bake-all-first-v36/` | v36 | first all-keepers bake — superseded by v37 (= v36 + v33_scalefold + v37 barrier-default-off) |

## Old → new name map

| old (`vllm/patches/…`) | new |
|---|---|
| `qwen38-dflash/` | `legacy/dflash-overlay-v1-12/` |
| `qwen38-dflash-v17/` | `legacy/reliability-memory-overlay-v17/` |
| `qwen38-dflash-v18/` | `legacy/reliability-memory-overlay-v18/` |
| `qwen38-dflash-v19/` | `legacy/tq-drafter-mq-verify-v19/` |
| `qwen38-dflash-v20/` | `legacy/spec-vs-target-bench-corrected-v20/` |
| `qwen38-dflash-v21/` | `legacy/spec-kv-dtype-vram-v21/` |
| `qwen38-dflash-v22/` | `legacy/mtp-cudagraphs-eager-head-v22/` |
| `qwen38-dflash-v23/` | `failed/spec-prefill-peer-barrier-v23/` |
| `qwen38-dflash-v24/` | `failed/spec-prefill-peer-rendezvous-v24/` |
| `qwen38-dflash-v25/` | `failed/spec-every-step-drain-v25/` |
| `qwen38-dflash-v26/` | `prod/gdn-spec-oob-fix-v26/` |
| `qwen38-dflash-v27/` | `diagnostics/wedge-endgame-v27/` |
| `qwen38-dflash-v27-ccl1717/` | `diagnostics/oneccl-1717-upgrade-v27c/` |
| `qwen38-dflash-v28dbg/` | `diagnostics/cudagraph-debug-v28dbg/` |
| `qwen38-dflash-v29/` | `diagnostics/oneccl-wedge-forensics-v29/` |
| `qwen38-dflash-v30/` | `legacy/failsafe-spec-tp-graphs-v30/` |
| `qwen38-dflash-v31/` | `prod/whole-step-capture-11fix-v31/` |
| `qwen38-dflash-v32/` | `prod/attn-mq-2d-and-spec-async-v32/` |
| `qwen38-dflash-v33/` | `prod/attn-mq3d-varlen-and-scalefold-v33/` |
| `qwen38-dflash-v34/` | `prod/attn-shim-cache-and-guards-v34/` |
| `qwen38-dflash-v35/` | `prod/spec-k4-unclamp-v35/` |
| `qwen38-dflash-v36/` | `legacy/bake-all-first-v36/` |
| `qwen38-dflash-v37/` | `prod/image-bake-keepers-v37/` |
| `qwen38-dflash-v38/` | `prod/attn-esimd-fp8-reroute-v38/` |
| (untracked loose file) | `failed/exact-kv-splits-v38/` |

## Image naming scheme (from the v38 harmonization onward)

- **Production:** `llm-scaler-prod:vN` — `v1` BUILT + VALIDATED 2026-09-03
  (= v38 keeper lineage re-built under the new name, keepers only) and
  **prod is running on it**. Validation battery ×5 vs `adv:v37`, `v31.1`,
  `v19`, `v14`: probe/2k/65k/conc16 all inside the certified envelope —
  see `prod/image-prod-v1/README.md` for the full matrix.
- **Experimental:** `llm-scaler-exp:<purpose>` — throwaway arms, never prod.
- **Historical:** `llm-scaler-vllm-adv:vN` (and `dspark`, `qwen36-b70…`
  ancestors) stay untouched as provenance — never rebuilt, never renamed.
- Baked images carry `/root/.vN_baked` (new bakes:
  `/root/.llm-scaler-prod_vN_baked`); `bootp.sh` skips boot-time patching
  when it finds the marker (host-side glob generalized to `/root/.*_baked`).

## Invariants (do not break)

1. **Patch `.py` contents are byte-identical to what was baked/verified.**
   Directory renames must never edit them — bake gates and `md5sum` checks
   reference these exact bytes (e.g. `v38_esimd_reroute.py` md5
   `9f64c2e495977dc1e00fa2ae506bb73d`, LF).
2. Bake Dockerfiles use the patch dir as build context; `COPY` references
   script filenames only (unchanged by the reorg).
3. `failed/` and `diagnostics/` dirs are load-bearing history — negative
   results gate future retries; do not delete.
4. Top-level files are upstream-owned; renames here would conflict with
   intel/llm-scaler merges.
