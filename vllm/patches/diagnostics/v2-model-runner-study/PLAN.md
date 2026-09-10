# PLAN — Full fork support for `VLLM_USE_V2_MODEL_RUNNER=1`

Status: **PLAN ONLY — not executed.** The AI machine (10.20.3.65) is running the
user's own testing on the prod lane (`lsv-test`, `llm-scaler-exp:v1.2.1`); every
workstream below is gated on explicit user go-ahead. Nothing in this document
has been applied to any container, image, or lane.

Mandate (user, verbatim): V2 must be **fully compatible and improved versus V1
without any degradation**, validated across **all supported `--kv-cache-dtype`**,
**long context**, and **nospec + MTP + DFlash** spec configurations.

All research for this plan was read-only: CPU-only `docker run --rm` inspection
of `llm-scaler-exp:v1.2.1` (never `docker exec` on `lsv-test`), local repo
reads, and public web sources.

---

## 0. Executive summary

The fork's image (`vllm-0.21.1.dev0+gad7125a43.d20260826.xpu`) already ships a
complete V2 runner tree (`vllm/v1/worker/gpu/`, 1,436-line model_runner vs the
fork-patched 7,628-line V1 `gpu_model_runner.py`) with XPU entry
(`XPUModelRunnerV2`, a thin subclass). Three findings shape the whole plan:

1. **The headline blocker is one conservative config gate, not missing
   machinery.** `_validate_v2_model_runner` (config/vllm.py:1915) rejects
   `has_inner_state` (hybrid/mamba = our GDN-hybrid qwen3.8 target) — yet the
   same tree's `init_model_state` (gpu/model_states/__init__.py) auto-routes
   `model_config.is_hybrid` to `MambaHybridModelState`, which explicitly
   supports GDN (`GDNAttentionMetadataBuilder`) and is spec-decode-aware
   (`MambaHybridAttnMetadata.num_accepted_tokens`,
   `num_decode_draft_tokens_cpu`). Upstream's own blog says linear-attention
   was "not yet supported" as of v0.18.0; our snapshot is newer and carries
   the machinery. The gate is almost certainly "pending validation", not
   "cannot work". **Plan: lift the gate per-lane under controlled flags and
   validate, not rewrite.**
2. **V2 spec decode is a fully separate stack.** `EagleSpeculator`
   (gpu/spec_decode/eagle/speculator.py) imports nothing from the V1 proposer
   stack (`vllm/v1/spec_decode/`, except one utils function). The gate allows
   `eagle/eagle3/mtp` and blocks `dflash`. Consequences:
   - **MTP-under-V2 loses the fork's V1-stack fixes** — critically the #11
     comm-stomper fix (`drafter_comm_v48.py` dedicated drafter communicator,
     patched into `LLMBaseProposer.propose`) and the draft-attention
     integrations. Porting the drafter-comm isolation into the V2 speculator
     is a hard prerequisite for MTP-under-V2, or we re-introduce the wedge
     that kills concurrent long context.
   - **DFlash under V2 = a real port** (method gate extension + proposer).
     Upstream DFlash2 is V2-only by design, so the port direction is aligned
     with upstream; the fork's V1 DFlashProposer semantics (k-adjust
     v42/v42d, TQ draft pool, v49b/v49c) must be re-expressed as a V2
     speculator.
3. **The v49b-class fix is architectural in V2.** V2 `attn_utils` builds
   `CommonAttentionMetadata` with `seq_lens_cpu_upper_bound` natively, and V2
   is async-first (GPU-native input prep, outputs on separate stream). The
   exact class of host-side D2H stall we fixed in v49b should not exist. This
   is where V2's "improved, no degradation" headroom plausibly comes from.

Posture: three-phase bring-up (nospec → MTP → DFlash), era-3 style patch
scripts per phase, full validation matrix per phase, prod image only at the
end (`llm-scaler-exp:v2.0` per the user naming scheme: v1→v2 = full new
feature).

---

## 1. Current-state audit (evidence base)

### 1.1 The V2 gates (image, config/vllm.py)

| gate | site | effect on our lanes |
|---|---|---|
| `model_config.has_inner_state` → "hybrid/mamba models" | :1915 `_validate_v2_model_runner` | **BLOCKS ALL LANES** (qwen3.8-27b is GDN-hybrid) |
| spec method not in (eagle, eagle3, mtp) | same | **BLOCKS DFlash** (`dflash`); MTP allowed |
| prefill CP > 1, DBO, routed-experts capture, custom logits processors, KV-sharing fast prefill, EC transfer | same | not used by our boots — inert |
| `mamba_cache_mode == 'align'` assert | :2050 | **inert for us** — boot never sets align (default `none`, dt_dflash2_serve2.sh has no align flag) |
| trigger | :1233 `if envs.VLLM_USE_V2_MODEL_RUNNER: self._validate_v2_model_runner()` | boot script pins `VLLM_USE_V2_MODEL_RUNNER=0` today (dt_dflash2_serve2.sh:59) |

### 1.2 What the V2 tree already has (image)

- `XPUModelRunnerV2` (v1/worker/xpu_model_runner.py:78) — thin subclass of
  `GPUModelRunnerV2`; all real logic in the base. V1's `XPUModelRunner` is
  equally thin ⇒ runner-level fork work lives in base classes either way.
- `gpu/model_states/`: `interface.py`, `default.py`, **`mamba_hybrid.py`**
  (GDN + Mamba2 builders; spec fields), `whisper.py`. Registry auto-routes
  `is_hybrid`.
- `gpu/spec_decode/`: `EagleSpeculator` (eagle/eagle3/mtp), own
  `RejectionSampler`, `DraftTokensHandler`, probabilistic path, own cudagraph
  managers (`Decode/PrefillEagleCudaGraphManager`), `load_eagle_model`.
- `gpu/attn_utils.py`: `build_attn_metadata` constructs CAM with
  `query_start_loc_cpu` + **`seq_lens_cpu_upper_bound`** natively; per
  kv-cache-group metadata; `build_for_cudagraph_capture`.
- Graphs: `CUDAGraphMode` PIECEWISE + FULL only (no FULL_DECODE_ONLY);
  `ModelCudaGraphManager`; `speculator.capture(captured_attn_states)`
  (model_runner.py:640); FULL = explicit replay (:1127). **V2 FULL mode is the
  natural successor of our v31 whole-step-capture posture.**
- Async-first: `use_async_scheduling` read natively (:137); outputs via
  `AsyncOutput` on a separate stream; Triton prep kernels (input_ids,
  positions, query_start_loc, seq_lens on-device); Triton gumbel sampler.
- `initialize_mamba_ssu_backend` (model_executor/layers/mamba/ops/
  ssu_dispatch.py:193) wired at model_runner.py:397.
- Scheduler V2 branch (v1/core/sched/scheduler.py:857): folds resumed reqs
  into scheduled_new_reqs with full `NewRequestData`. **`scheduler_v48.py`
  overlay carries this branch intact** (same :266/:857 sites — it is a
  full-file overlay of this exact file).
- Fork markers already in the V2 tree: **v28dbg flight recorder**
  (`_fr.log` around replay in gpu/cudagraph_utils.py) — the fork has run V2
  graphs on XPU before, in the v28dbg era.
- Worker-level (shared V1/V2): `gpu_worker.py` selects runner via
  `envs.VLLM_USE_V2_MODEL_RUNNER` (:177); fork stability patches there
  (v29 `_xpu_capture_stab`, #03 compile/warmup fail-fast) apply to both.

### 1.3 Fork keepers classified by V2 coupling

**(a) Auto-carry (runner-agnostic — no work):**
- TQ backend selection: platforms/xpu.py:174-177 routes `turboquant_*` → TQ
  attention backend, platform-level.
- `turboquant_attn.py` v49 backend (baked) incl. seq_lens handling fixes.
- Model layers: qwen3_dflash2.py (bf16 boundary-cast port, v40), DFlash2
  weights, `vllm-xpu-kernels` wheel (v26 GDN OOB fix), v32/v33/v34/v35/v38
  attention keepers (backend/model-layer), MTP head files.
- Worker stability (above); mamba_utils fork marks.

**(b) V1-runner-coupled (11 marker sites in gpu_model_runner.py — each needs
an explicit V2 decision):**

| site | era | what | V2 disposition |
|---|---|---|---|
| :53, :4250, :4286, :4735, :5019 | v20 A2 | `spec_seg`/`spec_step` telemetry (tforward/tlogits/propose segments) | **WS4 port** — needed for propose h−d / TQTIME parity measurements |
| :55, :4736 | v25 | every-step pre-drafter drain (#11 mitigation) | superseded posture (v26 wheel + v27 comm + v31 capture); re-evaluate, likely NOT wanted under V2 async-first |
| :61 | v37 | draft-barrier default OFF | check V2 equivalent knob; default-off preferred |
| :822 | v42d | width-aware graph capture / `uniform_decode_query_len` | **WS3 port** — V2 equivalents: `BatchExecutionDescriptor` + `get_uniform_token_count` + capture lattice |
| :1537 | v32 | async accepted-count D2H, deferred postprocess | V2 async-first may cover natively — verify + measure |
| :1831 | v42 | spec pad into LOCAL tensor | V2 `InputBuffers` persistent fixed-size — verify padding semantics |
| :6994 | v21c | TQ slot size derived from the GROUP's own spec | **WS1 verify** under V2 pool manager (TQ draft-pool sizing with k8v4) |

**(c) V1-spec-stack-coupled (bypassed entirely by V2 EagleSpeculator):**
- `drafter_comm_v48.py` (v27 #11 comm-stomper fix: dedicated oneCCL drafter
  PG during propose/dummy_run) — **must be re-expressed inside the V2
  speculator's propose path or MTP-under-V2 re-opens the #11 wedge family**
  (the one that killed MTP k=4 at 2×56k concurrent and wedged ≥32k eras).
- `dflash_v49c.py` DFlashProposer (whole proposer: TQ draft pool, k-adjust,
  v49b cam sourcing, v49c dtype policy) — WS3 port target.
- v42c AsyncScheduler placeholder-width cap (engine skips
  `update_draft_token_ids` under async) — verify the equivalent contract
  under V2's scheduler/speculator split.

### 1.4 Historical evidence (fork's own runs)

- PERF_TUNING.md (nospec TQ, v6/v7-era images, older base): `c5` = V2 alone:
  26.24 shallow / 19.19 deep64 (−9.6% / +10% vs anchor) "keep (component)";
  `d1` = c2+c5 (V2 + `VLLM_XPU_ALLOW_COMM_IN_GRAPH=1`): 32.75 / 19.45 = best
  stock config of its day. **V2 has booted and served nospec TQ on XPU
  before.**
- UPSTREAM_COMPARE.md (v42 era): upstream DFlash2 is V2-only and silently
  degrades to DFlash1 on V1 — i.e. our current V1 DFlash2 port is the
  non-upstream direction; a V2 DFlash port re-aligns with upstream.
- KNOWN_ISSUES.md: no V2 entries — no convicted V2 defects on record.

### 1.5 Upstream context (web, 2026-09)

- MRV2 blog (vllm.ai, 2026-03-24): no API changes; persistent batching
  (fixed-size state table, no `CachedRequestState`); GPU-native input prep
  (Triton); async-first (spec decode + async scheduling + structured outputs
  together; outputs on separate stream; prep kernels consume rejection
  results directly); Triton gumbel-max sampler; `ModelState` ABC isolating
  per-model logic. Claims: +56% throughput (Qwen3-0.6B, 1×GB200), −6.3% TPOT
  (GLM-4.7-FP8, MTP=1, 4×GB200). Not supported as of v0.18.0: linear
  attention (Qwen3.5, Nemotron 3 Super), spec beyond eagle/eagle3/mtp,
  EPLB/DBO, logits processors, LoRA. Planned to become default.
- Migration issue #41286 (open, 2026-04-29): V2 default-on landed for dense
  (#44443), Llama/Mistral dense (#43458), quantized (#44446), pooling
  (#48290+), GraniteMOE (#45461); MTP weight sharing (#42538); remaining
  TODO #47172. No hybrid/mamba checklist row visible — consistent with "in
  flight, conservative gate".
- Implication: our snapshot post-dates v0.18.0 and already contains
  mamba_hybrid model-states; the residual risk is correctness/perf on XPU
  GDN + TQ, not architecture absence.

---

## 2. Gap matrix (capability × V2 status × work)

| # | capability | V1 lane today | V2 status in image | gap | workstream |
|---|---|---|---|---|---|
| G1 | nospec boot, GDN-hybrid target | certified prod | blocked by `has_inner_state` gate; machinery present (`MambaHybridModelState`, GDN builder) | gate lift + validation | WS1 |
| G2 | TQ attention + KV dtypes under V2 | certified (4bit_nc/k8v4/k3v4_nc/3bit_nc/fp8_e4m3) | backend selection platform-level (shared); TQ pool sizing path (v21c analog) unverified under V2 `pool/` manager | verify + maybe patch | WS1 |
| G3 | XPU Triton kernels for V2 prep/sampler | fork has working Triton on XPU (v33 MQ3D) | V2 Triton input-prep + gumbel sampler unexercised on XPU | run + fix fallbacks | WS1 |
| G4 | graphs on XPU under V2 | V1 whole-step capture (v31 posture) | PIECEWISE+FULL, `speculator.capture`; v28dbg marker proves XPU V2 graphs ran | mode mapping + capture-stab verification | WS1/WS2 |
| G5 | MTP spec under V2 | certified k=4/k=1 (V1 stack) | EagleSpeculator supports mtp; **no drafter-comm isolation** (#11 fix V1-only) | port #11 fix + integrate TQ draft attn | WS2 |
| G6 | DFlash under V2 | certified DFlash2 k=7 (V1-only port) | method gate blocks `dflash`; no proposer in V2 spec stack | method gate + DFlashProposer→V2 speculator port (k-adjust, TQ draft pool, dtype policy) | WS3 |
| G7 | k-adjustable spec width under V2 graphs | v42d 3-site width-aware capture | V2 descriptor/lattice sites differ | port v42d semantics | WS3 |
| G8 | async scheduling + spec under V2 | v32 accepted-count D2H, v42c placeholder cap | native async-first; contract differs | verify placeholder/update_draft_token_ids equivalent | WS2/WS3 |
| G9 | spec telemetry (propose h−d, TQTIME) | v20 A2 segments | absent in V2 runner | port (diagnostics-grade) | WS4 |
| G10 | scheduler overlay compat | scheduler_v48 full-file overlay | V2 branch intact in overlay; resumed-fold semantics untested with our edits | diff-vs-stock + test | WS2 |
| G11 | image/bake | v1.2.1 (overlays baked) | n/a | new bake at completion | WS5 |

---

## 3. Workstreams

> **WS0 — coordination guardrails (always active).** No boots, restarts,
> execs, benches, or py-spy against `lsv-test` while the user's testing runs.
> All V2 experiments in throwaway containers (`llm-scaler-exp:v2study-*`
> tags, never `prod`, never the prod lane port). `DFLASH2_TP1=0` always.
> Every experiment boot uses a dedicated port + its own container name, and
> is torn down same-session. Era-3 patch scripts (`v5x_*.py`, fail-loud
> anchors, idempotent, tree markers, md5 gates) for everything.

### WS1 — Nospec V2 enablement (the foundation)
1. **Gate-lift patch** (`v50_v2gate.py`): allow `has_inner_state` when env
   `VLLM_XPU_V2_HYBRID=1` is set (fail-loud default-off; do NOT remove the
   gate for others). Keep all other gate items intact.
2. Boot matrix (throwaway container, nospec, TP=2):
   a. `--kv-cache-dtype turboquant_k8v4` @ MAXLEN 262144 (prod cell analog)
   b. each supported dtype from §4 (one boot each)
   c. graphs: PIECEWISE, FULL, NONE (XPU graph env interplay:
   `VLLM_XPU_ENABLE_XPU_GRAPH=1` as prod)
3. Verification targets per boot: warmup completes; TQ pool sizing sane
   (compare token capacity vs V1 boot log at same MEMUTIL); GDN state cache
   alloc (`--mamba-ssm-cache-dtype float16` as prod); no CAM construction
   errors; deterministic greedy x3.
4. Fix surface expected: XPU Triton prep/sampler kernel gaps (fallback to
   torch ops if a kernel is XPU-broken — patch, don't rewrite), v21c-analog
   TQ slot-size derivation if V2 pool manager sizes TQ groups wrong,
   capture-stab analog if FULL capture wedges (v29 experience).
5. **Exit gate:** nospec V2 ≥ V1 nospec on bench3 (2k/16k/65k) within −3%
   AND ≥ +3% on at least one mode (else: document why, decide continue/stop).

### WS2 — MTP under V2
1. **#11 comm-stomper port** (`v51_v2draftercomm.py`): dedicated drafter
   ProcessGroup for the speculator's propose/dummy-run window, mirroring
   `drafter_comm_v48.py` semantics inside `EagleSpeculator.propose` (V2
   site). `VLLM_XPU_DRAFTER_PG=0` escape hatch preserved. This is the
   wedge-killer; non-negotiable before ANY MTP V2 stress test.
2. TQ draft-attention integration check: V1 MTP draft rides target KV pool
   (TQ); under V2 confirm the MTP draft layer set still allocates into the
   target-family pool via KV-spec matching (should hold — pool logic is
   runner-agnostic — but verify boot log pool table).
3. CAM parity: verify draft attn gets `seq_lens_cpu_upper_bound` natively
   (expected — attn_utils); no v49b-class tolist() sync in py-spy under load.
4. Graphs: speculator capture paths (Decode/PrefillEagle managers) on XPU;
   start k=1, then k=4. **k=4-under-graphs corruption conviction (v29c, and
   its v42c DFlash twin) must be re-tested under V2 graphs before k=4 is
   allowed** — serve script keeps the refuse-combo guard until proven.
5. Long-context wedge battery early (this is where V1 MTP died): 2×28k,
   2×56k concurrent before celebrating anything.
6. **Exit gate:** MTP k=4 V2 ≥ V1 MTP k=4 on the certified battery, AND
   2×56k concurrent completes (V1 died there — if V2 also dies, that is
   parity, not regression; document; DFlash remains the long-context lane).

### WS3 — DFlash under V2
1. **Method gate extension** (`v52_v2dflashgate.py`): allow `dflash` under
   V2 (scoped: `VLLM_XPU_V2_DFLASH=1`).
2. **DFlashProposer → V2 speculator port** (`v53_v2dflash.py`): re-express
   the V1 proposer inside a V2 `Speculator`-shaped class:
   - propose/dummy_run orchestration on the V2 step path;
   - TQ draft pool (second pool, KV-spec mismatch by design — see
     NOTES "why DFlash2 draft KV cannot ride target KV") via V2 `pool/`
     manager; v49c dtype policy (default match `--kv-cache-dtype`,
     `VLLM_DFLASH_DRAFT_KV_DTYPE` override) carried verbatim;
   - v49b analog: draft CAM sourced from V2 upper bounds (native expected);
   - k-adjust: `DFLASH2_EMIT_K` with V2 width-aware capture at the V2
     sites (`BatchExecutionDescriptor`, capture lattice bs·(1+k),
     dispatcher) — v42d semantics, V2 mechanics;
   - async placeholder contract: V2 scheduler/speculator equivalent of the
     v42c cap (`update_draft_token_ids` skip under async);
   - bf16 boundary casts, conv fp32 internals, grouped-conv loop — all
     model-layer, auto-carry (verify weights load identical md5).
3. Correctness ladder before ANY perf claim: greedy vs V1 eager reference
   (bit-tolerance), b65 acceptance @2k/@16k within noise of V1 cell
   (87/112, 120/217), acceptance-vs-context curve (the v40 decay defect and
   v41 window fix must still hold: 77.7/70.6/62.2% ladder).
4. **Exit gate:** DFlash2 k=7 V2 ≥ V1 on certified battery; concurrent
   long-context (2×8k/16k/32k) all clean (V1 strength — must not regress);
   k=3/k=5 spot cells; k=4 graphs guard evaluated under V2 like WS2.4.

### WS4 — Telemetry parity (diagnostics-grade)
- Port v20 A2 `spec_seg`/`spec_step` segments (tforward/tlogits/propose) into
  the V2 runner behind `VLLM_SPEC_TIMING` as today. Purpose: propose h−d and
  TQTIME comparability across V1/V2 for the §4 gates, and py-spy-independent
  host-stall detection. Eniv-gated default-off; never baked without the gate.

### WS5 — Bake + promote
- `llm-scaler-exp:v2.0` (user scheme: full new feature) = v1.2.1 lineage +
  WS1-4 keepers, era-3 in-build patchers, md5 gates, full battery on the
  baked image (never the overlay lane). Prod promotion only after the full
  §4 matrix passes on the baked image. Rollback = v1.2.1 lane unchanged
  (never touched during the study).

---

## 4. Validation matrix (the acceptance contract)

Baseline = certified V1 numbers on the same hardware/TP/MAXLEN cell
(v1.2.1: single 41.2/36.8, conc 69.9/83.7 comb, mixed 74.4, bench 37.2,
bench3 482.5/280.6/149.1 tok/s @2k/16k/65k, conc8 198.8 agg acc 0.596, b65
87/112 @2k & 120/217 @16k, coh Paris −0.452 ×3 distinct=1; MTP control
46.1/42.0, 79.4/90.6; DFlash2 longctx singles 23.5/15.9/9.8, concurrent
14.5+14.6 / 10.2+9.5 / 5.7+5.6).

### Axes

**Runner:** V2 (candidate) vs V1 (baseline, certified numbers + same-day
re-run for drift control).
**Spec config:** nospec · MTP k=1 · MTP k=4 · DFlash2 k=7 · DFlash2 k=3
(k=4-under-graphs corruption re-test only; e5m2/auto excluded — upstream
gap / known defect, documented not re-litigated).
**KV dtype (`--kv-cache-dtype`):** turboquant_4bit_nc · turboquant_k8v4 ·
turboquant_k3v4_nc · turboquant_3bit_nc (@MAXLEN ≤ 258048 only) ·
fp8_e4m3. DFlash lanes also exercise the draft-dtype policy: default
match-target AND `VLLM_DFLASH_DRAFT_KV_DTYPE=turboquant_k8v4` pinned cell.
**Context:** 2k/16k/65k (bench3) · long-context singles 8k/16k/32k ·
concurrent 2×8k/16k/32k · 128k + 256k spot checks (single + concurrent) ·
mixed crash-repro battery.
**Battery (per cell):** dt_cargame single/concurrent/concurrent2/bench/mixed
· dt_bench3 · b65 acceptance @2k/@16k · coh_probe ×3 (bit-stability) ·
VRAM/pool-capacity log capture · boot determinism (2 cold boots, identical
pool tables).

### Full matrix phasing (per workstream exit, then final)

- **P1 (WS1 exit):** {nospec} × {5 dtypes} × {2k/16k/65k bench3 + cargame
  single/concurrent + coh ×3} + graphs {PIECEWISE, FULL} on the k8v4 cell.
- **P2 (WS2 exit):** {MTP k=1, k=4} × {5 dtypes} × full battery + longctx
  singles/concurrent + 2×56k wedge battery + mixed crash-repro.
- **P3 (WS3 exit):** {DFlash2 k=7, k=3} × {5 dtypes} × full battery +
  longctx singles/concurrent + acceptance-vs-context ladder @2k/16k/65k.
- **P4 (final, baked image):** cross-product spot audit — every spec config
  on k8v4 (prod cell) full battery; every dtype on each spec config at 2k +
  16k + coh ×3; long-context full set on prod cell; 128k/256k spots.

### Gates (hard, per cell)

1. **Perf no-degradation:** every V2 cell ≥ V1 cell − 3% (tok/s single,
   concurrent combined, bench3 all three depths).
2. **Improvement mandate:** aggregate across P4, V2 must beat V1 by ≥ 3% on
   ≥ 1 mode per spec config (target sources: async-first spec decode, native
   CAM upper bounds — the v49b stall class —, Triton sampler), else the
   engagement is documented as parity-only and promotion is a user decision.
3. **Correctness:** b65 acceptance within ±2 points of V1 cell; greedy
   determinism ×3; coh_probe distinct=1 at Paris −0.452 (V1 bit-stability
   carried); acceptance-vs-context ladder monotone-ish (no v40-style decay
   reappearance).
4. **Stability:** zero wedges/stalls/RPC timeouts in ALL long-context
   concurrent cells; 30-min soak at conc8 on the prod cell with acceptance
   histogram stable; no container exit.
5. **Resources:** VRAM ≤ V1 cell + 5%; TQ pool token capacity ≥ V1 − 2%.
6. **Anti-regressions:** k≤3 MTP bit-stability vs eager (the v29c/k4
   conviction boundary re-checked under V2 graphs — refuse-combo guard stays
   until k=4 is proven or convicted under V2).

---

## 5. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| #11 wedge family re-introduced via V2 MTP (no drafter-comm fix in EagleSpeculator) | high if WS2.1 skipped | lane death @ concurrent long ctx | WS2.1 is a hard prerequisite; longctx wedge battery before any k=4 stress |
| XPU Triton gaps in V2 prep/sampler kernels | medium | boot fail or silent slow path | fallback-to-torch patches; kernel-by-kernel bring-up; py-spy screens |
| GDN V2 metadata wrong (hybrid gate exists for a reason) | medium | wrong outputs (silent!) | greedy-vs-V1-eager reference diff FIRST on every boot; b65 acceptance; never trust tok/s before correctness |
| TQ pool sizing under V2 pool manager (v21c analog) | medium | pool too small / OOM / capacity loss | boot-log pool table vs V1 at same MEMUTIL; explicit capacity gate |
| k4-under-graphs corruption persists under V2 | likely (convicted twice on V1) | garbage @ k=4 | keep refuse-combo guard; re-convict or clear under V2 explicitly |
| V2 FULL capture wedge on XPU (oneCCL in graph) | medium | boot hang | PIECEWISE-first; `VLLM_XPU_ALLOW_COMM_IN_GRAPH` axis from PERF_TUNING d1; v29 capture-stab analog |
| Boot lottery / nondeterministic init (historical) | low-medium | flaky cells | 2 cold boots per cell; container-per-cell isolation |
| DFlash V2 port scope creep (proposer is big) | high | schedule risk | strictly mirror V1 semantics (UPSTREAM_COMPARE: fork already superior on k-adjust/async/graphs — port ours, not upstream's); correctness ladder before perf |
| Upstream drift (snapshot frozen 2026-08-26) | low | none now | pin: this is a fork-internal study, no base upgrade in scope |
| User-lane collision | must-be-zero | prod disruption | WS0 guardrails; all work in throwaway containers/ports |

---

## 6. Decision gates / kill criteria

- **K1 (after WS1):** nospec V2 within −3% of V1 AND boots all 5 dtypes →
  continue to WS2. If a dtype fails structurally → fix or document partial
  support (dtype matrix must state V2 column honestly).
- **K2 (after WS2):** MTP V2 parity + no-wedge → continue. If #11 fix cannot
  be expressed in the V2 speculator → MTP stays V1-only; document as V2
  partial (nospec+DFlash), user decides.
- **K3 (after WS3):** DFlash V2 parity or better → WS5 bake. If the port
  cannot reach V1 acceptance/perf → DFlash stays V1-only; V2 image ships
  nospec+MTP capability and the matrix records it.
- **K4 (final):** full §4 matrix on baked `llm-scaler-exp:v2.0` ≥ gates →
  user promotion decision. Any single hard-gate failure = no promotion,
  rollback posture untouched (v1.2.1 lane never modified).

## 7. Deliverables

- Era-3 patch scripts: `v50_v2gate.py`, `v51_v2draftercomm.py`,
  `v52_v2dflashgate.py`, `v53_v2dflash.py`, `v5x` telemetry port — each with
  anchors, idempotency, tree markers, md5 gates; parked in
  `vllm/patches/diagnostics/v2-model-runner-study/` until promoted.
- Study NOTES.md with the honest V1-vs-V2 matrix (all cells, including
  failures), boot matrices, pool tables, wedge forensics if any.
- Image `llm-scaler-exp:v2.0` (bake recipe + Dockerfile, v37/v1-prod style).
- patches/README.md + KNOWN_ISSUES.md updates (any new defect numbers
  allocated sequentially).

## 8. Open verify-items (first session of implementation)

1. `model_config.is_hybrid` true for qwen3.8-27b-fp8 (expected — GDN layers)
   and `get_model_state_cls` absence on the model (routes to
   MambaHybridModelState).
2. EagleSpeculator MTP branch details: draft-model load path for our MTP
   head, draft KV pool spec (target-family ride), propose-window collective
   sites (where to splice the drafter PG).
3. V2 `InputBuffers`/`BatchExecutionDescriptor` exact widths for spec (k
   derivation sites to patch in WS3).
4. Whether V2 sampler path honors server default sampling identically
   (battery cells use server defaults — must match V1 outputs on greedy;
   non-greedy cells compared statistically).
5. `VLLM_XPU_ALLOW_COMM_IN_GRAPH` interplay with V2 FULL mode (d1 datapoint
   was V1-era).
