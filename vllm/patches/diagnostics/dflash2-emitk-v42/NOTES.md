# DFlash2 v42/v42c/v42d — k-adjustable emission (Path A) + async-preserving cap + width-aware graphs

Lane: `lsv-test` on `llm-scaler-prod:v1` (runtime-patched, era-3 chain), TP=2 B70,
target `/models/target` qwen3.8-27b-fp8, drafter `/models/dflash2` (block_size=8
checkpoint), turboquant_4bit_nc KV, block 512, prefix caching, async scheduling.
Baseline lane = v41 (k=7 stock): 411.6 / 262.0 / 189.1 / 123.6 tok/s
(2k/16k/65k decode, conc8 aggregate).

## Goal

DFlash2 trains its drafter at block_size=8 (num_speculative_tokens=7). Path A =
let the drafter **emit only k of 7 candidates** at runtime (8-row internal
layout untouched) so the target verify batch shrinks from 8 rows/req to k+1 —
~37% target FLOPs saved at k=4 — while async scheduling AND XPU graphs stay on.

## v42 — DFLASH2_EMIT_K (drafter emission cap)

`dflash2_emitk_edit.py`: drafter-side cap; emits first k candidates, internal
8-row forward preserved. Default (env unset) = 7 = byte-identical v41.

## The async wall (why sync was tried and is a dead end)

Under `--async-scheduling` the engine SKIPS `post_step ->
take_draft_token_ids -> scheduler.update_draft_token_ids` (engine/core.py gate
`not self.async_scheduling`). Instead `AsyncScheduler._update_after_schedule`
injects a shared placeholder list into every running request each step:

    self._spec_token_placeholders = [-1] * num_spec_tokens   # width 7

The placeholder WIDTH is what schedule() turns into spec slots (live-convicted:
D3 probe total_num_spec_tokens=7 at k=4; metrics num_draft_tokens/drafts =
exactly 7.0; positions 4-6 zero, never accepted).

`DFLASH2_SYNC=1` (force `--no-async-scheduling`) was tried: widths consumed
correctly but output CORRUPT (padded-batch contract mismatch on the sync spec
path). Dead end.

**Tri-state landmine**: `async_scheduling` is `bool | None`, default None, and
config/vllm.py AUTO-ENABLES async when None (dflash qualifies). Dropping the
CLI flag does NOT disable async; only the explicit `--no-async-scheduling`
(BooleanOptionalAction) reaches the sync path.

## v42c — async placeholder cap (async scheduling PRESERVED)

`dflash2_schedk_edit.py`: cap `_spec_token_placeholders` to k. Self-consistent
end-to-end: `_update_after_schedule` derives cur_num_spec_tokens from the
scheduled width, and the engine's `update_draft_token_ids_in_output`
trims/-1-pads real drafts to the scheduled width. Boot evidence:
"DFlash2 v42c async cap: scheduling 4 of 7 spec slots per step (async
scheduling preserved)"; metrics width exactly k.

## Graph corruption conviction chain

1. k=4 + async-cap + graphs: output `' Paris!!!!...'` / `'\n\n!!!...'`,
   **100% garbage acceptance** (draft==target==token 0).
2. `VLLM_XPU_ENABLE_XPU_GRAPH=0` does NOT disable capture on spec+TP2 lanes
   (fork forces whole-step graphs, KNOWN_ISSUES #11): 48/48 "Capturing" lines
   despite the env. An earlier "graphs not the culprit" retraction stems from
   trusting this control.
3. Real switch = `cudagraph_mode` via `--compilation-config` (serve knob
   `DFLASH2_CGMODE`, default FULL_DECODE_ONLY). **NONE => correct output,
   width honored** — graphs convicted.
4. Eager k=4 (proof, not product): bench3 87.9 / 96.8 / 157.4 / 75.8,
   acceptance 0.729, 2.9 tok/step. 2-5x slower than graphed k=7.

## v42d — width-aware graph capture (THREE derivation sites)

The uniform decode width `1 + num_spec_tokens` is derived independently in
three places; ALL must become k-aware or boot asserts / replay corrupts:

1. **runner** `gpu_model_runner.py:~803` `self.uniform_decode_query_len` —
   drives uniform classification (`max_num_scheduled_tokens == udql`) AND
   capture shapes. (dflash2_udql_edit.py edit 1)
2. **config** `config/vllm.py _set_cudagraph_sizes` — capture lattice must be
   rebuilt as bs*(1+k): the dispatcher asserts
   `padded_size % uniform_decode_query_len == 0`; stock lattice {8,16,...}
   vs width 5 => boot AssertionError (cudagraph_dispatcher.py:144).
   (dflash2_udql_edit.py edit 2; lattice = [1,2,4]+8s scaled by 1+k;
   e.g. k=4 -> 15 sizes max 480, k=3 -> 19 sizes max 512)
3. **dispatcher** `cudagraph_dispatcher.py.__init__` — derives its own width
   from config; `initialize_cudagraph_keys` receives the runner width only as
   a PARAMETER (size filter) while `_create_padded_batch_descriptor` asserts
   against the INSTANCE attribute. Left at 8 => same boot assert against the
   width-5 lattice. (dflash2_disp_edit.py)

Audited and clean (no edit needed):
- `gpu_worker.execute_dummy_batch`: `getattr(model_runner, udql, 1)` — reads
  the patched runner attr.
- `config/compilation.py resolve_cudagraph_mode_and_sizes /
  adjust_cudagraph_sizes_for_spec_decode`: called by the runner with the
  patched width; multiple_of = max(w, tp) = w at tp=2 — no-op on an already
  w-multiple lattice.
- proposer dispatcher (llm_base_proposer): PIECEWISE-only keys; the assert is
  uniform_decode&&FULL-only — unreachable. Piecewise padding 8->next-lattice
  entry is contract-safe.
- Buffer sizing: max_num_tokens comes from max_num_batched_tokens, not udql;
  drafter 8-row internal layout is num_spec_tokens-driven (v40 port).

NOTE: patchers run in phase-1 helper containers WITHOUT the env — they write
the env-gated code unconditionally; the gates evaluate at SERVE time (container
has DFLASH2_EMIT_K via -e). Default (env unset): byte-identical stock.

## The k=4 conviction (fork defect, NOT a port bug)

With all three sites width-aware, k=4 boots clean, captures 11 width-5 graphs,
schedules width 4.0 — and output is STILL token-0 garbage. D-probe
(dflash2_dbg_edit.py, debug-only, NOT in the final chain):

    DFLASH_DBG pack_aux n_layers=5 | shape=(5,5120) mean=nan ... PACKED mean=nan

Target hidden states NaN from the first decode step => both drafter and target
argmax NaN = token 0 => draft==target every position => 100% "acceptance".
Draft-side inputs are sane (masks/positions/slot maps correct).

**Boundary is EXACTLY k=4** (verify width 5):
- k=3 (width 4): CORRECT — refs p1/p2/p5/p6 exact, stable x2 (p3 bistable #18)
- k=5 (width 6): CORRECT — refs p1/p2/p4/p5; p6 stable knife-edge variant
  (ebcc8258959448f3, coherent one-word divergence, deterministic in-lane)
- k=7 (width 8): stock v41, correct.

This reproduces, on a SECOND spec method (DFlash2 whole-step FULL graphs) and
a second graph mode, the v29c MTP conviction (piece-capture x k4 corrupts
temp-0 numerics; k<=3 bit-stable == eager ref). Shared fork-level defect
triggered at verify width 5. Root cause not chased further this era (graph
buffer/kernel internals); k=4 is refused by the serve script guard.

## Measurements (dt_probe2 / dt_b65 / dt_bench3)

| lane | 2k | 16k | 65k | conc8 | acceptance |
|---|---|---|---|---|---|
| DFlash2 k=3 (v42d graphs) | 207.2 | 179.1 | 167.5 | 111.4 | 0.799 |
| DFlash2 k=4 EAGER (only correct mode) | 87.9 | 96.8 | 157.4 | 75.8 | 0.729 |
| DFlash2 k=5 (v42d graphs) | 297.4 | 238.5 | 161.3 | 125.7 | 0.708 |
| DFlash2 k=7 (v41 stock) | 411.6 | 262.0 | 189.1 | 123.6 | ~0.7 |
| MTP k=4 control (v41 era) | 354.1 | 431.1 | — | 142.4 | 0.615 |

- k knob works end-to-end: k in {3,5} verified correct+measured, k=7 stock,
  k=1/2/6 in range but untested, **k=4 forbidden with graphs** (serve-script
  guard: FATAL + exit 2 unless DFLASH2_CGMODE=NONE).
- k=3 recovers 2.4x over eager-k4 @2k; conc8 125.7 @k=5 ~= k=7's 123.6.
- Truncation trades tokens/step for cheaper steps; nothing beats k=7 @2k on
  this workload — the cap is a compute-saving / tail-acceptance-tuning knob,
  not a throughput win at these context lengths.

## Files

- `dflash2_emitk_edit.py` — v42 drafter emission cap
- `dflash2_schedk_edit.py` — v42c async placeholder width cap
- `dflash2_udql_edit.py` — v42d edits 1+2 (runner udql + config lattice)
- `dflash2_disp_edit.py` — v42d edit 3 (dispatcher width)
- `dflash2_dbg_edit.py` — D1/D2/D3 + pack_aux/attn_meta/draft_inputs probes
  (DEBUG ONLY — removed from the committed boot chain; convictions above)
- `dt_dflash2_serve.sh` — lane serve script; env knobs DFLASH2_EMIT_K (1-7,
  default 7), DFLASH2_SYNC (default 0 = async), DFLASH2_XPUGRAPH (default 1),
  DFLASH2_CGMODE (default FULL_DECODE_ONLY); k=4+graphs guard
- `dt_dflash2_boot.sh` — final 6-patcher chain (edit -> winfix -> emitk ->
  schedk -> udql -> disp), 12-file inject
- `Dockerfile` — **v42 image bake** (`llm-scaler-exp:dflash2-v42`): prod:v1 +
  all 6 patchers run in-build (the v42 patchers write /w/patched only — the
  explicit cp loop mirrors boot phase 3 so the baked tree == the runtime
  lane tree); grep gates for every era hook + py_compile; markers
  `/root/.dflash2_patched` + `/root/.dflash2_v42_baked`. Boot the baked
  image directly with `DFLASH2_IMG=llm-scaler-exp:dflash2-v42` (serve-script
  knob, default `llm-scaler-prod:v1` = runtime-patch flow)
- `new/dflash2_proposer.py` — the v42 knob-bearing proposer (md5
  `26e34cc691175af518a3ae07f801497d`); the v40-era copy in
  `../dflash2-spec-port-v40/` predates the knob
- `UPSTREAM_COMPARE.md` — deep-research comparison vs upstream PR #52816
  (verdict: MATCH on drafter math, SUPERIOR on k-adjust/async/graphs, no
  upstream deltas worth adopting)

## Upstream comparison + image bake (Sep 4, follow-up engagement)

Deep research vs upstream PR #52816 (merged 2026-08-21) — full analysis in
`UPSTREAM_COMPARE.md`. Summary: conv/selector/top-k/greedy-walk/TP/quant are
formula-identical; ours adds k-adjustable emission (absent upstream), async
scheduling, width-aware whole-step graphs, TQ KV + v41 window fix, and the
bf16-draft/fp16-target numerics the platform requires. The ONLY post-merge
upstream fix (PR #54282) is in the probabilistic gumbel path we deliberately
did not port. **No code changes adopted.** AR baseline measured on the
restored prod nospec lane for the ratio table: 136.6/160.8/130.9/219.3
(2k/16k/65k/conc8) → DFlash2-k7 = 3.01×/1.63×/1.44×/0.56× AR (conc8 spec
loss is fork-generic: MTP is 0.65×).

Image `llm-scaler-exp:dflash2-v42` BUILT + VALIDATED:

- Bake gates all green (`DFLASH2_V42_BAKE_OK`); **all 12 tree files
  md5-IDENTICAL to the runtime-patched lane** (strongest equivalence proof:
  same bytes ⇒ same behavior).
- Runtime chain re-validated first (k=7): probes p1/p2/p5/p6 exact refs;
  415.4/262.2/192.3/123.4 vs baselines 411.6/262.0/189.1/123.6.
- Baked image, k=7: probes exact; bench 416.9/262.9/125.2 (2k/16k/conc8),
  65k 189.0+192.1 on re-runs (one 123.8 outlier = greedy knife-edge
  trajectory, compl 159 vs 182/185 — not tree degradation, md5-identical
  tree).
- Baked image, k=3: lattice gate `uniform width 4: 19 sizes, max 512`,
  udql=4 both workers; probes exact (p1/p2/p5/p6); bench
  229.6/194.5/176.3/117.1 — ABOVE baseline 207.2/179.1/167.5/111.4.
- Baked image, k=5: lattice gate `uniform width 6: 13 sizes, max 480`;
  p6 = documented knife-edge variant `ebcc8258959448f3` (deterministic
  in-lane); one transient p5 flip reverted to ref on re-run (#18 class);
  bench 297.2/237.9/160.6/122.2 acc 0.701 vs baseline
  297.4/238.5/161.3/125.7 acc 0.708 — flat.
- Degradation gate: PASS on every row of every k.

Prod restored `llm-scaler-prod:v1` nospec via bootp.sh; certified
harm_probe10 = 0ce080630035 x10.
