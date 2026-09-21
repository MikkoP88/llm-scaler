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
| `dflash2-spec-port-v40/` | v40 | DFlash 2 drafter (upstream PR #52816: grouped depthwise convs + candidate-selector beam walk) ported onto the fork's DFlash v1 machinery, benchmarked vs MTP. Lane WORKS: acceptance 0.46–0.51 @k=7 (4.43 tok/step), correctness gate passed (bf16-draft + params_dtype sync + fp32 conv internals were the landmines), par with MTP @2k/conc8. NOT promoted: open acceptance-decay-with-context defect (72% @2k → 9% @74k, suspected wrong sliding-window range over precomputed context KV) gates any bake. See its NOTES.md |
| `dflash2-window-fix-v41/` | v41 | **closes v40 §6**: TurboQuantAttentionImpl silently DROPPED `sliding_window` — the draft's MQ kernel ramped causal attention over the FULL prefix instead of the trained trailing 2048 window (gradual decay, 9.3% @74k). Fix (11 era-3 hooks): store the window; MQ kernel `HAS_WINDOW` per-row lower bound; device-op block-table rebase + q0 shift (graph-safe, exact, inert when sw=None). Ladder 72.3/30.4/9.3% → **77.7/70.6/62.2%**; DFlash2-k7 now beats MTP @2k (412 vs 380) and @65k (189 vs 181), par @16k/conc8. First image CONTAINING DFlash2: `llm-scaler-exp:dflash2-v41` (prod:v1 + v40 + v41, md5-verified, battery-passed). Prod posture unchanged (v1 nospec) |
| `dflash2-emitk-v42/` | v42/c/d | **Path A: k-adjustable DFlash2 emission with async+graphs preserved.** `DFLASH2_EMIT_K=1..7` (default 7 = byte-identical v41). v42c caps the AsyncScheduler placeholder width (async stays ON — the engine skips `update_draft_token_ids` under async, placeholder width IS the scheduled width; sync fallback is a convicted dead end, and async_scheduling is tri-state auto-on). v42d makes graph capture width-aware at THREE independent derivation sites (runner `uniform_decode_query_len`, config capture lattice → bs·(1+k), dispatcher `__init__`). Verified: k=3 207/179/167/111 acc .799, k=5 297/239/161/126 acc .708 (k=7 stock 412/262/189/124). **k=4 CONVICTED corrupt under graphs** (NaN target hidden states → token-0 flood, 100% garbage acceptance; boundary EXACTLY k=4 — k=3/k=5 bit-clean — matching the v29c MTP k4 conviction on a second spec method+graph mode; serve script refuses the combo). Follow-up engagement: `UPSTREAM_COMPARE.md` (vs upstream PR #52816 — MATCH on drafter math, SUPERIOR on k-adjust/async/graphs, no upstream deltas worth adopting; sole post-merge fix #54282 is in the unported probabilistic path) + **image `llm-scaler-exp:dflash2-v42` BUILT + VALIDATED** (6-patcher in-build bake, all 12 tree files md5-identical to the runtime lane, k=7/3/5 battery on the baked image passed with zero degradation; `Dockerfile` + `new/dflash2_proposer.py` knob-bearing copy committed). Prod posture unchanged (v1 nospec) |
| `dflash2-prod-v49/` | v43–v49c | **DFlash2 prod-promotion arc: KV-dtype matrix, real-life perf fixes, long-context validation, prod images `llm-scaler-exp:v1`→`v1.2.1`.** KV matrix: 4bit_nc/k8v4/k3v4_nc ✓, 3bit_nc ✓@MAXLEN≤258048, fp8_e4m3 ✓ (draft inherits), e5m2 ✗ upstream, auto ✗ DFlash2 defect. Perf root cause (v49b): draft `CommonAttentionMetadata` lacked `seq_lens_cpu_upper_bound` → per-layer TQ `seq_lens.tolist()` D2H sync under async scheduling (propose h−d == tforward-d identity); fix = source the runner's pre-null upper bound — TQ fwd 7.5→0.43 ms, single 31.7→36.7, conc 52.3→71.2, conc2 68.2→77.4, numerics clean. v48 drafter-TP1 replication CONVICTED CORRUPT (acc ~0.2) — dropped; both drafters TP2. v49c: draft KV dtype now DEFAULTS to matching `--kv-cache-dtype` (v21c tq4nc auto-policy retired), `VLLM_DFLASH_DRAFT_KV_DTYPE` optional override; prod pins k8v4. Long-context (user-required): singles MTP wins, but concurrent 2×28k/2×56k MTP k=4 STALLS then DIES (wedge → `TimeoutError: RPC sample_tokens`; full-context draft attention = the #11 comm-stomper wedge family) while DFlash2 (O(2048) window draft) completes ALL concurrent runs clean — the honest superiority claim for DFlash2. Full battery on baked `llm-scaler-exp:v1.2` ≥ certified on every mode, coh bit-stable; prod lane UP on `v1.2.1` |
| `prod/wedgefix-v60/` | v60 | **WEDGEFIX — closes the Claude-Code crash loop** (2026-09-21): async scheduling DEFAULT OFF on XPU (`xpu.py`, opt-in research env `VLLM_XPU_ALLOW_ASYNC=1`) + `VLLM_XPU_SPEC_DRAFT_BARRIER` default ON @MIN_CTX 8192 (reverses v37's short-context posture). NOT OOM — the `Killed` was v55.3 fast-clean SIGKILL after a spec-drafter device wedge (KNOWN_ISSUES #11 class; xe ccs/bcs engine resets both tiles), amplified into full-engine 0 tok/s by the silently-ON AsyncScheduler (§24 K "async OFF" label was a serve-flags grep, not runtime — every boot ran async). Controlled repro on v1.2.14: wedge in ~6 min of sustain, v55.3 KILL 11:30:55, +4 engine resets. Image `llm-scaler-exp:v1.2.15` = v1.2.12 crash-free pedigree (f15b/arstage/v55.3/v58p1/STALFIX) + v60, MTP ×4 and XGrammar-2 0.2.7 kept. Evidence `/root/build/lce1/v60_*.log`; see Invariant 0 |

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
- **DFlash2 prod lineage (from the v43–v49c engagement):**
  `llm-scaler-exp:vN[.N[.N]]` — versioned, prod-capable exp images.
  Scheme (user-specified): `v1 → v2` = full new features, `v1 → v1.1` =
  big fixes/improvements, `v1.1 → v1.1.1` = small bug fixes. Lineage:
  `v1` (first DFlash2-capable, `f02fa561e900`) → `v1.1` (v43-era fixes,
  `468a61a7e79f`) → `v1.2` (v49 rebase + v49b perf fix + v48 overlays,
  full battery, `821b576f0642`) → `v1.2.1` (v49c draft-KV dtype policy,
  `4a10b5f6c81b`) → … crashfix/v52f/v58/kill-drill arc … → `v1.2.12`
  (§24 K crash-avoidance set, standing prod) → `v1.2.13` (+STALFIX
  tool-call stream-stall root fix) → `v1.2.14` (+xgrammar 0.2.7 =
  XGrammar-2) → `v1.2.15` (+WEDGEFIX v60: async default OFF, spec draft
  barrier default ON every step — **current standing lane**; see
  `prod/wedgefix-v60/README.md` and Invariant 0). See also
  `diagnostics/dflash2-prod-v49/NOTES.md`.
- **Historical:** `llm-scaler-vllm-adv:vN` (and `dspark`, `qwen36-b70…`
  ancestors) stay untouched as provenance — never rebuilt, never renamed.
- Baked images carry `/root/.vN_baked` (new bakes:
  `/root/.llm-scaler-prod_vN_baked`); `bootp.sh` skips boot-time patching
  when it finds the marker (host-side glob generalized to `/root/.*_baked`).

## Invariants (do not break)

0. **NEVER build or promote an image that boots with
   `Asynchronous scheduling is enabled`.** The fork silently defaults
   async scheduling ON (`config/scheduler.py` `bool | None`; the XPU
   platform only forces it OFF for PP>1), and the AsyncScheduler event
   pipeline is the amplifier of the spec-drafter device-wedge class
   (crashfix-v58 / GSD-12919): under Claude-Code-shaped long-context
   traffic one dead drafter stream freezes the whole engine at 0 tok/s
   until the v55.3 fast-clean SIGKILL fires — xe ccs/bcs engine resets
   on both tiles bracket every occurrence (2026-09-21 evidence:
   `/root/build/lce1/killan_evidence*.log` on ainode01). Safe states
   only: WEDGEFIX-A baked (`xpu.py` forces `async_scheduling = False`
   unless research env `VLLM_XPU_ALLOW_ASYNC=1`) — every image from
   `llm-scaler-exp:v1.2.15` — or an explicit `--no-async-scheduling`
   serve flag. Speculative decoding (MTP ×4) stays fully supported on
   the sync scheduler; XGrammar-2 ships untouched.
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
