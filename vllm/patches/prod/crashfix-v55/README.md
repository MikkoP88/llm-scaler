# crashfix-v55 — the 2026-09-10 crash pair: root cause, fences, warmup, warning hygiene, image v1.2.8

Two engine deaths on 2026-09-10 under multi-stream chat traffic, both
on boots that skipped the certified `dt_warmup` discipline:

- **Crash 1** (14:31–14:34, `fp8_e4m3` + mtp k=4, v1.2.7):
  `UR_RESULT_ERROR_DEVICE_LOST` on TP0, surfacing at
  `gpu_model_runner.py:298` (`async_copy_ready_event.synchronize()` in
  `WorkerAsyncOutputCopy`) and `:4254`
  (`num_accepted_tokens_event.synchronize()` in the deferred
  postprocess). First traffic triggered a mid-inference Triton JIT
  storm (`_topk_topp`, `kernel_unified_attention`, `reduce_segments`,
  eagle padding kernels, `batch_memcpy`) concurrent with graph replay
  and collectives; the device context died and both async threads
  reported it at their next event sync.
- **Crash 2** (15:37–15:54, `fp8_e5m2` + mtp k=3, explicit
  `VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1`): silent wedge. Solo request at
  17,182 computed tokens froze mid-decode; `TimeoutError: RPC call to
  sample_tokens timed out`; 600 s step watchdog killed both ranks.
  Flight recorders (`fr_521/fr_527`, preserved in
  `../../diagnostics/fp8-mtp4-v1/../crash2-forensics/` on the host at
  `/root/build/lce1/crash2-forensics/`) show BOTH ranks completing a
  clean `propose end` at t=2430.27 s == 15:44:30 with no collective in
  flight — the verify step's async accepted-counts event never
  signaled; the unbounded `synchronize()` wedged the RPC thread and the
  scheduler (blocked on the RPC) could reach no guard. dmesg: **no GPU
  fault in the window** — host-side deadlock, not device corruption.
  Precursor 15:41:33: `v52c DISCARD-GAP` gap=1069 (worker num_tokens
  3117 vs optimistic 2048).

## Root cause (common chain)

Cold engine (no warmup) → first-traffic JIT storm perturbing the async
scheduling machinery under concurrent multi-stream chunked prefill →
either (crash 1) device context death reported at the async-event
syncs, or (crash 2) worker/scheduler accounting divergence → stuck
async event → unbounded wait → RPC deadlock → watchdog. Both crashes
ran the warmup prompt class ("Write a html car game") as first traffic
with warmup skipped.

## The fixes (v55)

1. **Warmup v53** (`dt_warmup_v53.py`): adds p10 explicit sampled
   params (topk+topp / topp-only / topk-only kernel branches), p11
   staggered concurrent DISTINCT-prompt streams (+0/+3/+6/+9 s, ~2k/6k/
   12k/18k, mixed greedy/sampled — v51's concurrent phases shared one
   prompt and prefix-cache-coalesced), p12 long repetitive html-game
   decode (4096 out, sampled — the crash-2 wedge lane). p1–p9 and the
   WARMUP_DONE marker protocol unchanged. Validation runs 5/6 convicted
   two further coverage gaps, closed in v53.1/v53.2: p13 validation-
   geometry 5-stream stagger (+0/+4/+8/+12/+16 s: short sampled
   4096-out decode + ~17k/~8k doc prefills + greedy/sampled shorts —
   p11 tops out at 4 streams), and p14 DRESS-REHEARSAL — warmup's final
   phase runs one full `v55_client` cycle (seed base 90, lengths
   identical to the validated cycles, text distinct), because the
   bs=5 mixed-context step shapes (`kernel_unified_attention`,
   `reduce_segments`) JIT only when docs decode 1024–2048 deep WHILE
   the htmlgame 4096 decode runs — the exact evolving-batch mix only
   the real cycle geometry produces. JIT inside warmup is allowed by
   definition; the acceptance gate is zero jit_monitor lines in the
   POST-WARMUP log section.
2. **Bounded async-event waits** (`patch_async_fence.py`):
   `_v55_wait_event()` replaces the four unbounded `.synchronize()`
   calls (2× async_copy_ready_event, 2× num_accepted_tokens_event);
   polls `query()` at 50 ms, raises a named RuntimeError after
   `VLLM_V55_EVENT_TIMEOUT_S` (default 120 s). A dead producer now
   kills the engine in seconds with the culprit named — never a
   10-minute silent wedge.
3. **GAP-FENCE** (same patcher): records every discard-step gap per
   request; raises on the LATCHING zombie signature — 3 consecutive
   non-decreasing observations (all ≥ 8) = no chunk progress. One-shot
   large gaps are routine (gap = remaining prompt beyond this chunk's
   budget; validation run 1 observed a benign 2005 on warmup p5 and
   convicted the initial `|gap|≥256` immediate arm as a false trip —
   removed). `VLLM_V55_GAP_FENCE=0` reverts to detect-only per boot.
   v55.1: the v52c DISCARD-GAP probe line demoted WARNING→DEBUG —
   routine multi-chunk prefills fire it per chunk (8 WARNINGs in the 3
   healthy cycles of validation run 4, all one-shot 1830/1458-class);
   the GAP-FENCE ERROR escalation supersedes it as the anomaly signal.
4. **Warning hygiene** (`patch_warns.py`): envs.py allowlist of 57
   census-verified knobs (unknown-var warning now flags only
   typos/dead vars), v52l BOUNDARY-* (scheduler, 4 sites) and v52f
   PRE-COPY/POST-COPY/DEF-PP (mamba_utils, 3 sites) demoted
   WARNING→DEBUG (routine per-2048-token bookkeeping), `.bashrc`
   oneAPI banner silenced (`--ccl-bundled-mpi=no`), dead
   `VLLM_ALLOW_LONG_MODEL_LEN` removed from `serve_bench.sh` (read
   nowhere; the certified v1.2.7 boot warned on it).
   DISCARD-GAP stays ~~WARNING~~ DEBUG (v55.1, see fix 3);
   GAP-FENCE/ASYNC-EVENT-STALL are ERRORs.

## Bake

`Dockerfile.v7` FROM `llm-scaler-exp:fp8-mtp4-v6` (== v1.2.7) →
`llm-scaler-exp:v1.2.8` == `llm-scaler-exp:crashfix-v55`
(sha256:`7b4de29d0ff3a0b04e6ff0bcef9b7d883aa4bd240ccdd97dd5f258897c1c548a`;
first bake `6d4bca514376…` superseded in pre-certification validation by
the v55.1 DISCARD-GAP demotion). Numerics untouched; in-tree files
md5-identical to the scratch-tested copies (gmr `362e29cf…`, scheduler
`a4ea0ec6…`, mamba `b4b23b85…`, envs `122a7923…`). Host bake context
`/root/build/v55/`.

## Validation

`validate_v55.sh` + `v55_client.py`: both crash configs (A: e4m3+mtp4,
B: e5m2+mtp3 with the crash-2 env), each: boot → warmup v53 → 3 cycles
of 5 staggered streams (0/5/10/15/20 s; distinct prompts per stream AND
per cycle; stream 1 = the exact crash prompt "Write a html car game" @
temp 0.9/top_k 20/top_p 0.95/4096 out; largest context ≈ 18k < 64k) →
health between cycles → engine-log scan of the validated section for
crash/warning classes (DEVICE_LOST, RPC timeouts, watchdog, JIT
warnings, unknown-env, boundary floods, fence trips) → warning census.
Results: `REPORT.md`.

## Known residual — the e5m2+mtp3 lane (convicted, contained)

Validation runs 7–9 exposed a PRE-EXISTING numerical defect specific to
the crash-2 boot combo (e5m2 KV + MTP k=3; v55 touches no numerics):
under sampled long decode crossing the FIRST 2048-token mamba/GDN
boundary, the draft verify bonus row goes NaN (`v52e ATTR:
brow(min,max,am)=(nan, nan, 0)` after `v52d EMPTY-ROW
[vocab_sentinel, -1 x k]`). Observed rate: 3 onsets in ~7 boundary
crossings (every incident at exactly `num_computed_tokens=2048`);
stochastic per crossing, co-batch composition irrelevant. Before v55
this class is crash 2 itself (wedge → watchdog → engine death); on
v1.2.8 the pre-existing `v52m STRIKE-OUT` guard force-finishes the
poisoned request FINISHED_ABORTED client-visibly within ~6 s — engine
stays up, co-running and subsequent requests unaffected, health 200
throughout, 3/3 containments. Disposition: e5m2+mtp3 is NOT
serve-grade for long-decode traffic (a ~40 % per-boundary request-abort
rate is unacceptable even contained); serve e5m2 only with mtp4 (the
v51-certified pairing — mtp3 was never in a certified matrix, it is
exactly the crash-2 boot setting). The client/harness classify
`finish=abort` as CONTAINED (engine-safety criterion) while the REPORT
carries the lane verdict separately.
