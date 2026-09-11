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

## 2026-09-11 addendum — crash-4/5/6 (post-coredump GPU flakiness), the v1.2.8 throughput regression, and v55.2 progressive-wait (v1.2.9)

A "major token generation speed drop llm-scaler-exp:v1.2.7 vs v1.2.8"
report plus three further engine deaths; **two distinct root causes**
convicted the same day.

### A. Morning chaos: post-coredump GPU flakiness (image-independent)

- **crash-3** (03:41, v1.2.7): manual `/bin/bash` cold boot, no warmup —
  classic crash-2-class wedge under first traffic (fr_522/fr_528 tails
  clean `propose end`; preserved in `../../diagnostics/…/crash3-forensics/`
  on the host). Teardown at 03:57 left `xe … Engine reset (ccs)
  guc_id=32` + **Xe device coredump** on GPU0 (da:00.0); further resets
  on both GPUs 03:59:38.
- **crash-4** (04:00, v1.2.8): another manual `/bin/bash` boot, no
  warmup; first real traffic (warmup p8) → HTTP 500 after ~4.5 min;
  serve log lost with the interactive pts (forensics: fr_522/fr_528
  v128 copies + inspect.json in `crash4-forensics/`).
- **crash-5** (04:28, v1.2.8, CERTIFIED recipe via `reg_ab.sh`): p1–p13
  healthy (22.4 tok/s long-decode = certified), then during p14 BOTH
  workers raised `v55 ASYNC-EVENT-STALL: num_accepted_tokens_event
  (deferred postprocess)` at 04:41:36 — TP-symmetric stall, the fence
  fired exactly as designed (culprit named, 120 s, no silent wedge);
  EngineCore then died via `sample_tokens` RPC timeout. The p14 cycle
  ran 400 s for 2794 tok (vs 178 s healthy) — the "slow generation"
  was the wedge window, not steady-state throughput.
- **crash-6** (04:45, v1.2.7, same harness): died at p12 with the
  `sample_tokens` RPC timeout at 04:56:18 — one second after a fresh
  `xe … Engine reset (ccs+bcs)` on GPU0: an actual device reset
  mid-serve.
- **Disposition**: with BOTH images dying on the certified recipe while
  yesterday's identical matrix ran clean for hours, the NODE was
  convicted: GPU0 flaky since the 03:57 coredump. Host reboot 05:10 →
  everything healthy (all §B numbers). Ops rules learned: (1) after a
  coredump-class GPU reset, reboot the node before serving again;
  (2) never launch the server in an interactive shell — crash-4's log
  died with the pts. Use `serve_bench.sh` (logs to `/root/b_<tag>.log`),
  redirect to a file, or run inside tmux.

### B. The real regression: v1.2.8 −14…−44% generation throughput vs v1.2.7

Post-reboot clean A/B (`reg_ab.sh`: back-to-back legs, identical lane
fp8_e4m3+mtp4 @0.9/262144, full warmup v53 + identical benches each):

| warm metric | v1.2.7 | v1.2.8 | delta |
|---|---|---|---|
| ctxscan 2k/16k/32k/65k (tps) | 70.8/53.4/62.1/51.3 | 39.6/38.0/36.5/44.2 | −44/−29/−41/−14% |
| genspeed 4×1024 aggregate (tok/s) | 142–148 | 88–99 | −37% |
| warmup p13 long-decode 4096 (tok/s) | 36.8 | 23.2 | −37% |
| warmup p14 dress rehearsal | 112 s / 7184 tok | 178 s / 7114 tok | +59% wall |

Root cause: v55's `_v55_wait_event` flat `time.sleep(0.05)` poll loop.
The pre-v55 `.synchronize()` blocks efficiently in the driver; the flat
poll quantizes EVERY not-yet-signaled event wait to 50 ms granularity,
and three of the four fence sites are per-engine-step hot paths
(WorkerAsyncOutputCopy ×2, deferred postprocess) → ~+40 ms per decode
step. (Step-time adder math from the A/B: consistent across all four
context lengths.)

### C. The fix — v55.2 progressive wait (`patch_progressive_wait.py`, image v1.2.9)

Progressive cadence, identical stall semantics and identical 120 s
bound (`VLLM_V55_EVENT_TIMEOUT_S` unchanged): 0.5 ms polls while
elapsed < 20 ms (the normal async-completion window), 5 ms < 200 ms,
50 ms < 2 s, 250 ms beyond. Normal-path inflation ≤ 0.5 ms.

- **Scratch test** (live-patched v1.2.8 boot, gmr md5 `51ec1bd2…`):
  ctxscan 61.5/48.6/59.9/49.0, genspeed 134–160, p13 35.1 tok/s,
  p14 5/5 in 124 s — v1.2.7-class restored.
- **Bake**: `Dockerfile.v8` FROM v1.2.8 → `llm-scaler-exp:v1.2.9` ==
  `crashfix-v55.2` (ddc6f15a7061); baked gmr md5 identical to the
  scratch-tested file.
- **Validation ON the baked image** (`validate_v129.sh`): warmup p1–p14
  (p13 36.4 tok/s, p14 5/5 in 112 s = v1.2.7-identical), client cycles
  101/102 both PASS 5/5 (6856 tok / 115 s class), ctxscan
  57.6/49.2/46.0/49.2, genspeed 131.9/141.4/138.8, **0 fence stalls,
  0 post-warmup WARNING lines**, no crash classes. Numerics untouched —
  only the wait helper changed.
- Known residual: `VLLM_ALLOW_LONG_MODEL_LEN=1` is baked as ENV in the
  image lineage → one boot-time unknown-env advisory per boot (read
  nowhere; harmless; drop at the next base refresh).

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
