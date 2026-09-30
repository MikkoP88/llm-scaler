# perf-v126 — fp8-state kernel round: FIX the quality convictions, WIN the speed matrix, ship v1.2.26

Task (verbatim): 'Run full fixing and testing suite "both fp8(e4m3/e5m2) implementations has
to be superior of baselane on any possible cases and work to end-to-end(e4m3/e5m2)", and
baked into new production image llm-scaler-exp:v* using best values and improvements.
Important! Spec and XGrammar-2 has to have supported, all "fresh math wrong, serial wrong
and post-prefix math wrong" (e4m3/e5m2) issues has to fixed. Do not use subagents.'

Standing: no subagents (all execution direct) · spec MTPx4 + XGrammar-2 0.2.7 gate-verified
on every boot · e4m3/e5m2 end-to-end · barrier default 0, never 1 · live PHASES record after
every phase · KERNELS_MAX_JOBS=52 for all wheel builds · host reboot after big changes /
before test rounds · never bake from the running lane (bake from -raw) · V1223+ boot lineage
only · python json.dumps bodies + -d @file · litellm auth `Bearer sk-dummy` via :4000.

## Entry state (from perf-v125, commit bf9342d)

- Production lane: llm-scaler-exp:v1.2.25 = 3f3c91637692 (fp16 GDN pool + spec MTPx4 +
  fp8_e4m3 KV default; opsall root fix; C7 refusal baked).
- CONVICTIONS to overturn (P23D/P23F):
  - e4m3ns (=1 ESIMD): 7/80 fresh math wrong, 10/24 (41.7%) post-prefix wrong.
  - e5m2ns (=1 ESIMD): 11/80 fresh, 6/24 (25%) post-prefix, tool-salad legs.
  - =2 spec-fp8: 20% post-prefix tool-call salad BOTH formats (fp16 control 0/30).
  - Signature: concurrent-wrong -> serial-right (18/18 flips; serial 0/160 wrong) =
    per-request GDN/SSM STATE-SLOT mapping corruption when prefills coalesce. v88 int32
    root family. Silent (tracebacks 0, resets 0).
- Speed matrix (Run 7n): fp8 below fp16 everywhere — spec-4 solo e4m3 -8.9% / e5m2 -12.5%;
  ns solo -1.5..-4.0%; agg -2.6..-13.9%.
- Capacity: pool-sizing peg = equal pages regardless of dtype (+3.50% spec-4 KV tokens,
  +1.16% ns, dtype-symmetric) — savings not yet converted.
- C7: import-time RuntimeError in _xpu_ops.py when VLLM_XPU_GDN_FP8_NATIVE in {1,2}. This
  round removes/reworks it once the paths are actually fixed.

## Phase plan (live-recorded in PHASES.md)

- **P27 — root-cause the state-slot crosstalk.** Read the fp8 state store/load paths
  (ESIMD static bridge =1, SYCL pool =2 + P22B wheel) against the fp16 path; trace slot-id
  computation through prefill coalescing / batch reorder; convict the exact line(s).
- **P28 — root fix.** Patch slot handling; per-slot dtype round-trip test; unit-level
  concurrency repro (two interleaved prefills + decode) proving correct per-request state;
  rework C7 from refusal to (guarded) enablement of the fixed paths.
- **P29 — speed + capacity.** =2 pool-path tuning; e4m3 vectorization window (681 vs
  261 GB/s @ seq n=16); pool-sizing surgery to convert fp8 state savings into KV budget.
  Target: fp8 >= fp16 on every speed leg (any-case superiority via capacity + tuned speed).
- **P30 — full validation battery on the fixed posture, BOTH formats.** P23F rate study
  (>=80 fresh + >=24 post-prefix + serial control + flip detection, 0 wrong required),
  P23D tool-salad A/B (0 salad), Run-7n-style speed matrix (no degradation), spec + parser
  + drills + bursts + serial24 + JIT-cache + tracebacks. Fresh host (reboot first).
- **P31 — bake + ship v1.2.26.** stage5 bake from v1.2.25-raw (93+ gates), reboot,
  fresh-boot validation, ship gates, CC battery, watchdog repoint + re-arm, docs
  (COMPLETE_ROUND_WRITEUP, PHASES, README, KNOWN_ISSUES), git commit.

## Acceptance gates (pre-registered)

1. P23F-style probe on EVERY fp8-state posture: 0/80 fresh, 0/24 post-prefix, 0 flips,
   serial 0 wrong — BOTH e4m3 and e5m2, BOTH =1 and =2 routes.
2. P23D tool salad: 0/30 on both formats (=2 spec), fp16 control 0/30.
3. Speed matrix: no leg below fp16 beyond noise (<=1%); at least one axis superior
   (capacity via pool surgery and/or tuned speed) — else the posture does not ship as
   default; superior-any-case is judged on the shipped composite posture.
4. Full battery green: drills 3/3 no-wedge, serial24 x3, bursts resets 0, parser T1/T2/T3,
   JIT cache delta <=2, tracebacks 0, spec MTPx4 + xgrammar 0.2.7 gates.
5. Bake gates 0-FAIL -> fresh-boot validation -> ship gates 0-FAIL -> CC battery green ->
   watchdog armed on the new image.
