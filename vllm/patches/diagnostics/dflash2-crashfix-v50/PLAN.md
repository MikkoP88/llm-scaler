# Crash-fix + DFlash2 improvement engagement — PLAN (v50 era)

Server `root@10.20.3.65`, 2× B70 TP=2, target `qwen3.8-27b-fp8`, DFlash2 drafter.
Base image `llm-scaler-exp:v1.2.1` (current prod lane). User mandate: deep crash
research + DFlash2 improvements, **zero degradation allowed**, **all crashes
fixed before the final image**, DFlash2 tested k=7 vs MTP k=4 with prompt
`Write a html car game.`, full `--kv-cache-dtype` matrix, concurrent +
long-context + looping-prevention tests on both drafters, loops fixed if found,
then a new prod image. Block sizes {2048, 512, 64} all in scope (user-added).
v49c draft-KV policy (default = match `--kv-cache-dtype`, optional
`VLLM_DFLASH_DRAFT_KV_DTYPE` override) is a wanted improvement → full
validation + issue fixes.

## Crash 1 — `assert len(request.block_hashes) >= num_full_blocks` (block_pool.py:240)

**Cell (reproduced exactly, R2)**: `llm-scaler-exp:v1.2.1`, manual-lane boot
(TP1 default fired: `VLLM_XPU_DFLASH_TP1=1`), tq4nc target, draft KV = tq4nc
(v49c default path), `--block-size 512`, memutil 0.92, MAXLEN 262144,
prefix caching ON, async scheduling ON, dflash k=7, no warmup sidecar.
Pool 270,087 tokens (TP1 replica weights ate ~104k vs certified 373,824).

**Repro (R2, done)**: client `rp_longdecode.py` (tiny 8-tok request then long
streaming car-game requests, max_tokens 8192). Result: tiny OK → long0 OK
(784 toks, stream ended **without usage chunk**) → long1 → fatal assert
~4 s later. Acceptance collapsed 2.1–2.8% (v48 TP1 corruption signature).

**Path**: scheduler.schedule:450 → kv_cache_manager.allocate_slots:414 →
coordinator.cache_blocks:209 → single_type_kv_cache_manager.cache_blocks:292
→ block_pool.cache_full_blocks:240.

**Invariant analysis (why this assert is "impossible")**:
- `num_tokens_to_cache = min(total_computed + new, request.num_tokens)` caps
  caching at real tokens (kv_cache_manager.py:410).
- Hasher maintains `hashes == floor(all_tokens / hash_block_size)` exactly
  (chain hash, appended on every token append; all append sites call
  `update_block_hashes`).
- All group block sizes are divisible by hash_block_size (validated at init).
⟹ `nfb = ntc // group_bs ≤ all // group_bs ≤ all // hbs = hashes`. The assert
can only fire if one premise is false at runtime:
  H1 **hasher granularity ≠ pool hash_block_size** (e.g. hasher built with
     scheduler `lcm` block size 2048 while pool hbs = gcd 512; draft group
     bs=1024 then demands 2× available hashes mid-window).
  H2 **crashing request is a zombie** (stream ended at 784 toks with no usage
     chunk → request possibly still alive/scheduled server-side; its token
     count crossing a block boundary during long1's lifetime).
  H3 **some path clears/trims block_hashes without trimming tokens**.
  H4 a group whose runtime block_size < hash_block_size (topology bug).
**R3 (instrumented block_pool via EXTRA_VOL)** dumps gid/bs/hbs/hashes/all/
computed/placeholders/spec/stack at the failing call + per-group topology log
→ decides H1–H4. Files: `/root/build/r3_block_pool.py` (md5 9536113e…).

**Control (R1, in flight)**: identical but `DFLASH2_TP1=0`. Acceptance healthy
(0.609/0.370/0.174…), long0 >1500 toks and alive past R2's death points.
Note: idle KV usage floor ~8–9.6% (~26–30k tokens) present in BOTH arms →
not TP1-specific; secondary question (structural vs boot leak), investigate
after the assert.

## Crash 2 — fp8_e4m3 boot OOM @ memutil 0.9 (VERDICT: sizing, not a defect)

`--kv-cache-dtype fp8_e4m3 --block-size 2048 -memutil 0.9 MAXLEN 262144` →
needs 8.38 GiB > 8.0 GiB (est. max len 247808). Valid cells: memutil 0.92
(certified), or MAXLEN ≤ 247808, or block 512. fp8 lane runs FlashAttention
backend (not TQ). Action: document certified cells; add a boot-matrix row;
no code change expected.

## Fix list (all gated by repro/instrumentation before writing)

- F1 **TP1 default-off + refuse guard**: `VLLM_XPU_DFLASH_TP1` default 1 → 0
  (code + serve script), plus a loud refusal (or explicit opt-in escape) —
  v48 TP1 replication is convicted corrupt (acc ~0.2) and now implicated in
  the crash cell. Any lane with acceptance < ~20% also starves the
  hash/placeholder regime → the real fix (F2) must be regime-independent.
- F2 **block-hash invariant hardening** (root fix, decided by R3): make
  cache_full_blocks clamp to available hashes (and log loudly) instead of a
  bare assert — engine must never die from caching bookkeeping; AND fix the
  underlying mismatch (hasher granularity, zombie request finish, or hash
  clear path) per R3 verdict.
- F3 **stream-end anomaly**: long0 ended with no usage chunk before the
  crash — verify against R1 whether that is normal-finish variance or a
  symptom (zombie). Fix if zombie.
- F4 **JIT warmup coverage**: extend `dt_warmup.py` to pre-JIT
  dflash/TQ/eagle kernels (crash log showed 5.1 s propose stalls; GuC
  watchdog risk documented in-image).
- F5 **fp8 sizing guard**: log estimated KV requirement vs available at
  boot-fail time is already fine (the error is loud); document cells only.
- F6 **fp8-class KV + spec refusal** (added after B5-series conviction):
  fp8_e4m3 + ANY speculative drafter is a convicted broken lane — dflash k7
  (8-row verify) 0.3–0.9 tok/s + engine-death wedge on stock v1.2.1
  (`sample_tokens` RPC timeout, B5e); MTP k4 (5-row) 0.2–0.5 tok/s (B5d);
  nospec single-row on the same pool is certified healthy 34.4/30.7/22.4
  (B5c). Mechanism = multi-row verify forward on fp8 KV ≈5 s/step
  (upstream kernel class — out of scope to fix). Remedy: refuse the
  combination at boot — (a) serve script fatal guard (fp8* KVDTYPE ×
  DFLASH2_SPEC != none), (b) EngineCore init RuntimeError
  (fp8_e4m3/fp8_e5m2/fp8 × speculative_config), both pointing operators at
  TQ-for-spec or nospec-fp8.

## DFlash2 vs MTP improvement work (k=7 vs k=4, prompt `Write a html car game.`)

Certified baselines (v1.2/1.2.1, must not regress): DFlash2 single 41.2/36.8,
conc 69.9/83.7, bench 37.2, bench3 482.5/280.6/149.1, conc8 198.8 agg
acc 0.596; MTP k=4 46.1/42.0, 79.4, 90.6. Known structural deficits
(measured): tforward +2.6 ms (8-row verify), dforward host +18.5 ms (5-layer
eager drafter: python `_grouped_conv` taps=2 loop, ~11 collectives), yield
2.44 vs 2.55–2.59. Candidate levers (in order of expected value):
1. dforward host cost — batched/grouped conv cleanup (reduce python-loop and
   collectives per propose).
2. k emission tuning per context bucket (v42 machinery: DFLASH2_EMIT_K).
3. Acceptance at depth (v41 window fix landed; re-check @65k).
Superiority claim already established (survival): MTP k=4 stalls/dies at
2×28k/2×56k concurrent; DFlash2 clean at all depths. Raw short-context tok/s
superiority is NOT promised; pursue levers honestly, report real numbers.

## Validation matrix (gates: ≥ certified baselines, zero new defects)

Axes:
- KV dtype: turboquant_4bit_nc, turboquant_k8v4, turboquant_k3v4_nc,
  turboquant_3bit_nc (≤258048), fp8_e4m3 (certified cells), (e5m2/auto:
  documented upstream/known gaps).
- Draft-KV policy: v49c default (match) AND override (k8v4) on tq4nc —
  acceptance + throughput equality check (user-mandated).
- Block size: 2048, 512, 64 (user-mandated) × {tq4nc, fp8_e4m3} minimum.
- Spec: DFlash2 k=7, MTP k=4, nospec control.
- Load: single, concurrent (2 clients), concurrent2, conc8, long-context
  8k/16k/32k single + concurrent, mixed crash-repro (rp_longdecode),
  looping-prevention (repeat-prompt loop probes + long-decode loop scan,
  both drafters; any loop → fix per mandate).
- Stability: coh_probe bit-stability ×3; rp_longdecode ×5 clean.

## Image plan

Fixes F1–F4 → bake → full battery → prod image per user scheme
(small fixes → `llm-scaler-exp:v1.2.2`; if improvement work lands materially
→ `v1.3`). Bake via 6-patcher style recipe (era-3), md5 gates, marker file.

## Crash 3 — certified-lane engine death AFTER loopscan (R6c extras; ROOT-CAUSED via redo salvage 2026-09-06)

Sequence on R6c container (tq4nc + k8v4 ovr, TP1=0, bs512, MAXLEN 262144,
final core): full v50_cell battery ALL-CLEAN → dt_bench3 clean → coh ×3
clean → loopscan cargame-8192 COMPLETED (26,702 chars, NO-LOOP, coherent
road-render tail) → **rp round 1 tiny = HTTP 500** → all subsequent
connections refused; engine log scan = 3 (Traceback/Assertion/clamp class;
content unknown — the log was LOST: Cell A's bootcell `docker rm -f` raced
the manual salvage; docker cp captured Cell A's fresh boot log instead).
First engine death ever observed on a certified (TP1=0) lane. Prior clean
runs of the same battery existed, but the extras sequence (bench3 65k +
conc8 + coh + 8192-decode) had never run on one container.

Suspects (undifferentiated, no log): cumulative-load wedge (v29-era
oneCCL/IPC class), device/host OOM at high cumulative prefix-cache + long
decode, or a new defect. Countermeasures already armed: (1) death-watchdog
(auto docker cp salvage at first :8000 down-transition, one per episode)
covering every remaining cell; (2) v50_rp5redo — exact-sequence replay
(battery → extras) gated after the final cells, to reproduce with evidence.
Mandate: all crashes fixed before prod — if the redo reproduces, forensics
on the salvaged log gates the v1.2.2 promotion; if clean ×1, treat as
load-cumulative edge and document (redo needed the full sequence).

**Event #2 (Cell F rp spot, v1.2.2 certified lane)**: after battery +
bench3, rp long0 ended ZOMBIE-style (1650/8192 toks, usage=None, tail
crawled 0.74 tok/s) → long1 HTTP 500. Same class suspicion: crash-1 shape
(assert removed by F2a; zombie stream-end vehicle survives) on a
post-cumulative-work container. F container also showed 498 DFLASH_STALL
propose(~0.25s) from warmup onward — degraded-boot/IPC signature may be
the enabling condition. Log lost (watchdog was gated inside the redo);
rp client now patched (500 bodies + finish_reason), F2 v2 re-queued with
watchdog armed BEFORE boot + rp ×3.

**Event #3 = the redo CAPTURE (salvage_010017.log, 1885 lines, salvaged
12 s post-death) — ROOT CAUSE**: exact replay of R6c's sequence on the
certified lane (v1.2.1 + final overlays, dflash7, tq4nc, k8v4 ovr, TP1=0,
WARMUP=0, bs512). Timeline: boot 00:43–00:44 clean → battery ALL-CLEAN
replicating R6c (46.32/16.5/9.01; det exact ×3; cargame 512 toks WITH
usage chunk — the battery cargame is EXONERATED as the zombie) → coh ×3
clean (Paris −0.452 ×3 each, 00:54:46/00:54:59/00:55:11) → loopscan
cargame-8192 completed client-side (14,182 chars, NO-LOOP, ~00:55:15+) →
NO further client traffic → **01:00:05 engine death**. Death mechanism:
1,024 SYCL assert hits `ind >= 0 && ind < ind_dim_size_ && "vectorized
gather kernel index out of bounds"` (torch-xpu-ops IndexKernelUtils.h:63,
vectorized gather), first at log line 790 — and it is the FIRST anomaly
in the entire log: everything between boot and it is health GET 200s
(zero DFLASH_STALL, zero warnings) → Worker VllmWorker-0 dies (01:00:05)
→ EngineCore `RuntimeError: cancelled` (shm_broadcast, 01:00:13) →
APIServer 500s → clean container exit. Watchdog worked (salvage 3 s
after the container-log flush; the engine's own dump pickle was lost with
the container, the salvaged stdout log is complete).

**The zombie (death dump, scheduler state)**: exactly ONE running request
`chatcmpl-…-8f028749`, prompt 57 tok (chat-templated cargame = the
loopscan request; the battery cargame is EXONERATED twice over — its
stream ended normally WITH usage at exactly 512 toks, and its raw prompt
is 6 tok, not 57), `num_computed_tokens=22745` /
`num_output_tokens=22688` in `scheduled_cached_reqs` (the NORMAL
running-request decode bucket — scheduler.py `schedule()` chains
running+resumed into CachedRequestData; `new_block_ids=[None]` = no new
KV blocks this step, `scheduled_spec_decode_tokens=[-1]×7` = empty-draft
padding: both benign serializer states), `finished_req_ids=[]`.

**Counter semantics (source-verified, .tmp-tq/vllm-src)**:
`Request.num_output_tokens` = `len(_output_token_ids)` = EMITTED tokens
only (request.py:247; sole writer `_update_request_with_output` →
`append_output_token_ids`); the dump's 22,688 is `CachedRequestData.
num_output_tokens = req.num_output_tokens + req.num_output_placeholders`
(scheduler.py:1060-63) — **async-scheduling optimistic-placeholder
padding** (gpu_model_runner.py:1264 extends worker-side token arrays by
`optimistic_num_accepted` each step; resolution truncates at 1310-14
using the scheduler-sent count). So real emitted ≈ 3.5-4k tok (=
loopscan's client-visible 14,182 chars; ~13 tok/s × ~290 s alone —
matches lane speed at 22k ctx; 22,688 real would need 78+ tok/s,
physically impossible), and **~19k unresolved placeholders** accumulated
because the request's outputs stopped being consumed/resolved after the
usage-less stream end. The OOB surface is exactly the worker-side
alignment: `end_idx = num_prompt_tokens[i] + num_output_tokens[i]`
indexing batch token arrays sized by the REAL history (gpu_model_runner.
py:1310-20) — placeholder divergence overruns it → vectorized gather
index out of bounds.

**Why the request survived past its stream end**: loopscan's client
reads to stream end; the stream ENDED SILENTLY at ~3.5-4k toks with NO
usage chunk and NO APIServer error line (the H2 signature; F's rp long0
= same, sub-fatal). Per async_llm.py:591-95, the engine-side abort fires
ONLY on `CancelledError/GeneratorExit` in `generate()` — a stream that
ends via any other exit path sends NO abort. And `check_stop`'s length
guard legitimately hadn't tripped (real 3.5-4k < max_tokens 8192) — so
between stream-end and 8,192 the zombie is server-side by design;
the placeholder inflation killed the worker before the length guard
could reap it.

**Verdict**: crash 3 = **unreaped (zombie) request under dflash spec
decode + async scheduling on the certified lane**: a request whose
client-visible stream ended SILENTLY and usage-less (no abort sent —
async_llm only aborts on CancelledError/GeneratorExit) kept decoding
alone engine-side; with no consumer left, its async optimistic
placeholders accumulated ~19k unresolved, and the worker-side persistent
batch alignment (indexed by real-emitted + placeholders vs arrays sized
by real history) ran a vectorized gather out of bounds → 1,024 SYCL
asserts → fatal worker death (device assert = process kill, no
recovery). Class recurrence: crash-1 R2/R3 zombies (usage-less stream
ends, TP1 lane — there the fatal assert was the hasher; same vehicle),
event #2/F (sub-fatal: zombie stream-end + tail crawl → request-scoped
500), event #3 (fatal, certified lane). v50 overlays INNOCENT (no
finish/placeholder code touched; F2a resync fired 0×). NOT oneCCL/
cumulative-load (log clean until the assert; no stall signature).
Incidence: 3 events across ~15 cells, always after battery-scale request
volume; never in short cells. Two-layer fix: **F8 runaway guard**
(scheduler-side force-finish on placeholder-overflow or real-count >>
max_tokens — kills the FATAL consequence) + upstream ticket for the
missing engine-side abort on non-cancel stream exits (the usage-less
early end remains a client-visible defect until then).

## Status log (v50)

- R2 (exact cell): CRASH REPRODUCED (fatal assert during long1, 4 s after
  long0's usage-less stream end; acceptance 2.1–2.8%).
- R1 (TP1=0): ALL-CLEAN (tiny + 3×8192, usage chunks present; acceptance
  0.589/0.339/0.161). KV floor ~9.6% in BOTH arms — not TP1-specific.
- R3 (instrumented): **ROOT CAUSE CAPTURED** — gid=14, group bs=1024, pool
  hbs=1024, hasher closure captured **2048**, len(block_hashes)=0,
  num_full_blocks=1, all=1024, status=RUNNING (zombie long0, 8 stuck output
  placeholders). Verdict: **H1** (hasher granularity ≠ pool granularity; the
  dflash2 v48 TP1 draft-spec remap 2048→1024 shrinks the pool gcd AFTER the
  hasher closure captured the resolve value) with **H2** as the vehicle (the
  starving request is the zombie long0 whose stream ended usage-less at ~787
  toks while the engine kept decoding it). H3/H4 ruled out (no hash-clear or
  bypass sites; no group bs < hbs).
- R4 (exact crash cell + v50 fixes via EXTRA_VOL: core hasher resync +
  block_pool clamp): **ALL-CLEAN** — boot shows `v50: request block hasher
  re-synced to final pool hash granularity 2048 -> 1024`; clamp fired 0×
  (resync removes the starvation; clamp = defense-in-depth); tiny + 3×8192
  all ended WITH usage chunks (R2's usage-less zombie stream-end did not
  recur); engine alive; TP1-lane acceptance still degenerate (3.4–9.5%,
  per-position 0.22/0.02/0.00) as expected — corrupt lane survives, F1
  defaults it OFF.
- R5 (certified lane TP1=0 + v50 fixes): in flight — resync silent no-op
  (no remap), full pool 373,824-class (concurrency 1.43× @262144), clamp 0×,
  long streams decoding normally.
- B5/B5b (fp8_e4m3 + dflash k7, v50 overlays): PATHOLOGY — decode 0.3–0.9
  tok/s vs historical 34/31/22; SPECTIMING 20-step medians: step_wall 18.8 s,
  tforward d=4976 ms, propose d=4027 ms, precompute d=1720 ms (TQ lanes:
  tforward 30–60 ms class).
- B5e (fp8_e4m3 + dflash k7, STOCK v1.2.1, no overlays): **ENGINE DEATH on
  first request** — worker wedged in `sample_tokens` RPC (8-token dflash
  verify step, 9 output tokens in) → shm_broadcast TimeoutError →
  EngineDeadError → container exit. Overlays INNOCENT; defect pre-existing.
  (Initial "v49c match policy = prime suspect" hypothesis DISPROVEN by B5d:
  MTP has no v49c draft pool and crawls identically.)
- B5c (fp8 + nospec): pure-target-path discriminator — running.
- B5c DONE: **ALL-CLEAN** — decode 34.39/30.73/22.41 == historical fp8
  record (3-digit match ⟹ historical record was the nospec path; fp8+spec
  likely never healthy). Determinism exact ×3, cargame coherent w/ usage,
  0 errors.
- B5d (fp8 + MTP k4): **CONVICTED CRAWL** — 0.2–0.5 tok/s generation,
  acceptance machinery healthy (acc len 2.5) → spec-on-fp8 is generic,
  drafter-independent. Cell torn down (evidence captured; full battery
  pointless at 0.3 tok/s).
- F6 LANDED: serve-script guard (dry-tested: dflash7/mtp4 refused, none
  passes) + EngineCore RuntimeError (core_v50.py md5
  094d4d5601b51edd2837072dcf934cfc, COMPILE-OK, pushed). R6 validation
  (engine-side refusal + TQ-lane inertness + crash-cell re-run) next.
- R6a DONE: engine-side F6 VALIDATED (no-guard script copy, fp8+dflash7 →
  RuntimeError "llm-scaler v50 F6" at EngineCore init, clean shutdown).
  Found+fixed en route: fork's CacheConfig field is `cache_dtype` (not
  `kv_cache_dtype`) — first F6a build probed the wrong attr and was inert.
- R6b (first attempt) FAILED for a TOOLING reason that yielded gold:
  **serve-script overlay slots are `EXTRA_VOL, EXTRA_VOL2..6 — there is NO
  EXTRA_VOL1`**; this session's B5c/B5d/R6b launches used EXTRA_VOL1..5, so
  core.py mounted as NOTHING (pristine stock core ran; md5-verified
  in-container 74e10065...). Result: accidental clean test of **clamp WITHOUT
  resync** = engine STILL dies, at a *second* site
  (single_type_kv_cache_manager.py:1062 `assert block.block_hash is not
  None`) after the clamp fires (gid=0 bs=2048 hbs=1024 hashes=3 all=6325)
  ⟹ **F2b alone is insufficient; the F2a resync is load-bearing** (matches
  R4 where resync fired + clamp never needed).
  B5c/B5d verdicts unaffected (fp8 pathology is stock-level; those cells
  validly ran stock core).
- R6b2: crash-cell re-run with CORRECT slots (EXTRA_VOL unnumbered for core)
  + resync probe INFO line (logs captured/pool hbs at boot).
- R6b2 DONE: **ALL-CLEAN with the FINAL core** (md5 3581a600, F6a + probe
  line) — probe printed `captured_hbs=2048 pool_hbs=1024 hasher_built=True`,
  resync warning fired 2048->1024, clamp 0, error scan 0 (no Traceback/
  Assertion), determinism exact ×3, cargame coherent WITH usage chunk
  (15.4 tok/s — corrupt-lane speed, engine survives as promised), bench 65k
  decode 5.81 tok/s (TP1-degenerate, expected). F6a correctly INERT on
  tq4nc. Core md5 refs in Dockerfile/NOTES refreshed 6a113164→3581a600 +
  probe grep gate added. R6c (certified prod cell tq4nc + k8v4 override,
  TP1=0, same final core) launched next.
- R6c DONE: **certified prod cell + FINAL core = ALL-CLEAN** — decode
  44.7/15.9/9.13 ≡ earlier cell (43.2/16.4/9.2) within noise ⟹ F6a+probe
  behaviorally inert on TQ. Extended battery on same container: bench3
  ctx2k/16k/65k decode **515.8/298.2/202.0** vs certified 482.5/280.6/149.1
  (meets/exceeds at all depths; 65k +35% = v41 window fix paying off),
  conc8 **283.9** vs 198.8, acceptance **0.602** vs 0.596; coh_probe ×3
  distinct=1 P1+P2, logprobs identical (Paris:-0.452 marker). **No
  degradation.** R6 series closed.
- **v1.2.2 BAKED & VERIFIED** (parallel with cells — build is CPU-only,
  cells pin v1.2.1): all md5 gates + grep gates + py_compile passed;
  in-image marker + 5 md5s exact (core 3581a600, pool 14bf0787,
  dflash2 66d4362, drafter_comm 7a78ba74, warmup 254d1de5).
- **Pipeline automated** (3 gated stages): extras (loopscan 8192 +
  rp_longdecode ×5 on R6c) → queue (L0 dt_loop8 dflash arm; A bs2048;
  B bs64; C mtp4+dt_loop8; D EMIT_K=5; E EMIT_K=4 — D/E on prod cfg for
  direct bench3 A/B vs 515.8/298.2/202.0) → final (F v1.2.2 prod cell
  WARMUP=1 validating F4 phases 4-5 + bench3 + rp spot; G v1.2.2 crash
  cell TP1=1 validating F1 opt-in escape).
- **Improvement lever-1 assessment (parked)**: `_grouped_conv` in
  `qwen3_dflash2.py` = ~10 small host-launched kernels × 10 calls
  (2 convs × prepare+finish × 5 layers) per propose; micro-fusion
  (cached position vector, precomputed tap masks, coeff-mask fold)
  recovers ~1-3 ms of the 18.5 ms host cost; the rest is structural
  eager-launch overhead — graph-capturing the drafter contradicts the
  CGMODE=FULL_DECODE_ONLY design. Not worth the zero-degradation
  validation cost; EMIT_K A/B (zero-code) is the honest lever.
- **Cells D/E — EMIT_K lever CLOSED (improvement lever 2)**: E
  (EMIT_K=4) refused at boot by the pre-existing v42 conviction guard
  (verbatim: "FATAL: explicit DFLASH2_EMIT_K=4 corrupts numerics under
  graphs (v42 conviction).") — the refusal fired correctly in-flow, no
  container left behind. D (EMIT_K=5): temp-0 CORRECT (' Paris.' ×3,
  conc distinct=1) and decode 43.55/22.08/11.25 @2k/16k/65k (≈ k=7
  cell at 2k, below at 16k), BUT cargame emitted 1 token then stalled
  18.2 s (tok_s=0.1, empty head/tail) + in-container error scan = 3 —
  live engine defect under sustained load. Emission truncation unsafe
  at any k<7 tested on this stack; DFLASH2_EMIT_K stays default 7.
  Both improvement levers now closed/parked → v1.3 improvement image
  NOT justified by this axis; v1.2.2 (fixes only) is the deliverable.
- Recipe: overlays + Dockerfile + NOTES.md in this dir → bake v1.2.2 after
  R5.
- **FINAL cells F/G DONE (v1.2.2 in-image)**: G (TP1=1 opt-in) ALL-CLEAN —
  probe `captured_hbs=2048 pool_hbs=1024` + resync warning fired IN-IMAGE
  at boot, engine SURVIVED the corrupt lane, det exact ×3, errs 0, decode
  17.1/6.07/6.66 (TP1-degenerate, expected). F (prod, WARMUP=1)
  functionally clean (det exact, errs 0, cargame TTFT 0.26 s = F4 warmup
  works; conc8 291.0 > 283.9; bench3 2k/16k 498.8/293.3 ≈ R6c) BUT two
  open threads: (a) 65k slow — battery 5.35 vs R6c 9.13, bench3 129.4 vs
  202.0, 498 DFLASH_STALL propose(~0.25s) from warmup window onward +
  8 spec/tq kernels still JIT during battery (F4 = partial coverage);
  (b) rp spot FAILED — long0 zombie end (1650 toks, usage=None, tail
  0.74 tok/s) → long1 HTTP 500 (crash-3-class event #2; log lost, no
  watchdog was running pre-redo). Countermeasures: rp client patched
  (500 bodies + finish_reason); F2 v2 (v1.2.2 + WARMUP=0 + rp ×3 +
  watchdog BEFORE boot) re-queued at chain end after the redo.
- **MTP discriminators H/I DONE + F7 LANDED + mtp1 DISCOVERY**: H (mtp1,
  graphs ON): battery ALL-CLEAN, probes ' Paris.' ×3 CORRECT (v29c k≤3
  claim holds at k=1); decode **44.31/28.11/13.04** @2k/16k/65k — 16k
  +77% / 65k +43% over dflash7 (15.9/9.13), 2k on par; mechanism = 2-row
  verify + no cross-model drafter host cost (itl 42 ms vs ~90). I (mtp4
  EAGER): temp-0 GARBAGE ×4, ALL DIFFERENT — v29c "eager = reference"
  OVERTURNED; mtp4 corrupt in BOTH modes ⟹ defect is in the MTP k≥4
  multi-row path itself. F7 added to dt_dflash2_serve2.sh (repo + server,
  validated exit 1): mtp4/mtp7 → FATAL refuse; mtp1/dflash pass. mtp1
  promotion candidate (v1.3) — full characterization cell (v50_mtp1_loop:
  battery replication + dt_loop8 + FULL extras incl. loopscan + rp ×5)
  queued at chain end after F2. Chain extended: redo → F2 → mtp1_loop.
- **CRASH 3 ROOT-CAUSED (redo, salvage_010017.log)** — full chain in the
  Crash-3 section above. Short form: loopscan req's stream ends SILENTLY
  usage-less → async_llm never aborts (only CancelledError/GeneratorExit)
  → request never reaped, decodes alone → fork async_scheduler placeholder
  accrual (+8/step) never resolved → CachedRequestData.num_output_tokens
  (emitted+placeholders) = 22,688 vs real ~2,836 → worker gather OOB →
  1,024 SYCL asserts → fatal. 4 events / ~17 cells.
- **F2 DONE — bake exonerated, event #4 fatal**: v1.2.2 + WARMUP=0
  healthy (48.89/17.33/9.44, stalls 12 vs F's 498; Cell F slowdown =
  warmup sidecar IPC degradation). Then rp long0 zombie end (796 toks,
  finish=None usage=None) → long1 500 → health 000 — crash-3 class,
  2-of-2 on the loopscan→rp tail, image/overlays/warmup-independent
  (log lost to salvage race; client markers only).
- **F8 LANDED (runaway-request guard)**: async_sched_v50.py overlay
  (EXTRA_VOL6) — dual-leg force-finish in AsyncScheduler.
  _update_after_schedule: (a) num_output_placeholders > 64, (b) real
  count > max_tokens + spec_width + 8 → FINISHED_ABORTED + loud warning;
  boot probe "F8: runaway-request guard armed"; md5 a096c5a8; py_compile
  OK; diff vs in-image = exactly the 2 additions (52 lines). F8a replay
  cell (same crash sequence + EXTRA_VOL6) queued behind mtp1_loop.
  Upstream ticket draft written (upstream_vllm_ticket.md, unposted).
- **mtp1_loop DONE — mtp1 promotion NO-GO**: every quality/speed win
  replicated (45.13/28.39/13.1 ≡ H; coh top-1 -0.452 bit-identical to
  dflash reference; loopscan NO-LOOP w/ usage; 0 battery errs) but (1)
  dt_loop8 3/4 think-trapped (dflash 0/4 — MTP-family entrapment, no
  periodicity) and (2) **rp round-1 long0 → #11-CLASS ENGINE WEDGE**
  @~1.1k/8192 toks: metrics FROZEN 02:04:16 (loop stopped stepping —
  NOT the zombie-alone-decode pattern), Worker_TP0 161%/TP1 100%/
  EngineCore 87% CPU spin, GPUs ~220 W both, 0 SYCL asserts, health
  200 throughout (silent service loss), client stream FROZEN not
  ended (crash-3 trigger absent — engine-side stall under plain long
  decode). Salvage wedge_mtp1_0228.log; chain released by force 02:29:54
  (rp client timeout-less; killed). F8 inapplicable (wedged loop never
  schedules). First #11-family wedge on an MTP lane; dflash7 never
  wedged across 17+ v50 cells → stays the only promotion-grade lane.
  mtp1 remains F7-allowed experimental. F8a unblocked 02:33:33.
- **F8a PASS — guard validated LIVE, 2 firings in one cell**: boot probe
  in-image (`F8: runaway-request guard armed (placeholder limit=64,
  spec width=7)`), battery 45.84/16.82/9.6 + det exact + coh -0.452
  (guard inert on healthy traffic), loopscan NO-LOOP w/ usage. Zombie
  class then recurred DETERMINISTICALLY in rp rounds 1 AND 2 (long0
  output leg stalls at ~500 client-visible / ~1,990 engine-accepted
  toks): guard fired both times at 72 > 64 placeholders (~9 scheduled
  steps), force-finished FINISHED_ABORTED, request reaped in <5 s
  (Running: 0), engine survived both — health 200 throughout, 0 SYCL
  asserts (event #3 had 1,024 fatal), and a fresh completion answered
  correctly while round-1's zombie stream was hung (engine fully
  serviceable). Residual by design: the zombie's own client SSE stream
  never terminates (finish can't traverse the stalled output leg) →
  timeout-less clients hang; serving-layer half = upstream ticket
  Layer 1. Log f8a_final_031420.log. Cell released by force at round 3
  (kill client+extras; class would re-fire every round).
- **v1.2.3 BAKED** (llm-scaler-exp:v1.2.3 = v1.2.2 + async_sched_v50.py;
  Dockerfile gates: md5 a096c5a8 + grep armed-line/env-knob/FINISHED_
  ABORTED + py_compile; content spot-verified: async_scheduler
  a096c5a8, core 3581a600, dt_warmup 254d1de carried through). V123
  certification cell (baked image, prod combo dflash7 + tq4nc/k8v4,
  NO overlays) booted 03:15:14 — gates: both probe lines, battery ≥
  certified, crash-sequence survival.
- **V123 PASS — v1.2.3 CERTIFIED (ALL DONE 03:56:01)**: both probe lines
  in-image (F8 armed; resync probe 2048/2048 silent no-op) + bake
  markers. Battery 55.45/19.11/9.29 vs certified r6c 44.7/15.9/9.13
  (+24/+20/+2%); det exact ×3; battery errs 0; bench3
  516.9/298.7/202.4 + conc8 283.5 + acceptance 0.602; coh×3 ' Paris'
  -0.452 bit-identical ≡ reference (baked image numerically identical);
  loopscan 29,353 NO-LOOP w/ usage. Crash sequence: zombie recurred
  5/5 rp rounds (rounds 1+2 telemetry byte-identical: 72>64,
  real=4034, computed=4171) → guard fired 5×, every request reaped
  FINISHED_ABORTED <5 s → **engine survived all five** (health 200,
  0 SYCL asserts, fresh completion answered post-extras). Residual
  unchanged: zombie client streams hang until client timeout. Log
  v123_final_035601.log. rp clients released per-round by PID (5×).
- **PROD RESTORED on v1.2.3 (03:58–04:05)**: same certified combo
  (dflash7 + tq4nc + k8v4 override) + WARMUP=1 via stock
  dt_dflash2_serve2.sh, no overlays; startup complete 04:01, health
  200, F8 armed + resync no-op in boot log, warmup sidecar present,
  temp-0 probe ' Paris' correct. Leftover cell watchdog killed;
  no other containers. ENGAGEMENT COMPLETE: all 3 crash classes
  root-caused and fixed or contained (crash-1 F2a/F2b baked, crash-2
  sizing-documentation, crash-3 F8 baked + serving-layer ticket
  drafted); spec-lane matrix closed (dflash7 = only promotion-grade
  lane; mtp1 experimental; mtp4 convicted); improvement levers closed
  (EMIT_K, grouped_conv); zero degradation vs certified at every depth.
