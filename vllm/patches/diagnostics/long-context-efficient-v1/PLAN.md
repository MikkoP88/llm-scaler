# Implementation and qualification plan

2026-09-08, updated 2026-09-09 after the 15-lane measurement campaign. Evidence, constraints, source links and detailed test matrix: [REPORT.md](REPORT.md).

## Completed

- Captured active image identity, installed DFlash source and physical upstream PCIe links.
- Measured correctness-checked two-rank XCCL reductions on the live host.
- Completed exact-token MTP4/TQ4-NC probes at 65536, 131072 and 261888 prompt tokens.
- Implemented version-aware FP32 snapshots and a source-layout-checked DFlash candidate generator.
- Built `llm-scaler-exp:lce1-fp32-cache` with the optimization disabled by default.
- Passed helper contracts and eight small grouped-convolution cases on both CPU and XPU.
- Measured projection microbenchmarks; preserved the failed large-GEMM bit-exact test and diagnostic baseline-repeat comparison.
- **Executed the 15-lane natural-task matrix** (2026-09-08/09): 3 spec lanes × 4 images (prod:v1, v1.2.5, v1.2.6t2, spec-prefill-phase-v1-v125) × 2 KV dtypes (TQ4-NC, FP8 e4m3), lengths 64k/128k/261888, clients 1/2, five repeats, identical seeds, plus the FP32-cache candidate A/B. Zero request errors across all cells; cancel probes all clean. Results and disposition in [REPORT.md](REPORT.md) campaign section.
- **Campaign verdicts**: FP8 e4m3 + nospec is the only lane meeting the 128k falloff gate (−13.1%) with +27/+43/+80% absolute gains — promotion candidate; speculation net-negative at ≥128k everywhere; FP8+spec collapse reproduced identical on all three images (not the scheduler phase race); v52/v53 image lineage perf-identical (±0.2%) with byte-identical KV pools; FP32-cache candidate perf-neutral (±1%); P0 = silent engine death on TQ4-NC nospec over-subscription (co-admission PRE-COPY wedge), logged and preserved.

## Next: qualify the FP8 lane and close the correctness gaps before any kernel work

1. **Quality-validate the fp8_e4m3 + nospec lane** on the preserved suite outputs: rescore generated texts for the natural tasks against TQ4-NC baselines, check natural EOS, think-block structure and near-boundary lengths (261632 prompt + 512 output = exactly 262144). The throughput case is made; the quality case is not.
2. **Fix or fence P0**: reproduce the nospec co-admission wedge once under scoped tracing, then either port the DFlash-style admission deferral to the nospec preemption path or enforce a hard admission cap against measured KV pool budget. Until then, cap concurrent long-context admission in serving configuration.
3. **Diagnose the FP8+spec collapse mechanism** (draft/verify cost over FP8 KV vs acceptance vs placeholder accounting) with per-step timing before any speculative long-context deployment; the phase-race fix is ruled out by the pf8m4 A/B.
4. **Multi-client soak and admission sweep on the FP8 lane**: closed-loop 2–4 clients with admission control at 64–70% pool budget, mixed 2k/262k clients, prefix reuse, cancellation; then production arrival-rate replay.
5. Instrument target attention, draft attention/convolution, collective calls, prefill continuation and scheduler gaps separately (per committed token) only where the above leaves material unexplained cost; retain an uninstrumented control.
6. For the built FP32-cache candidate, decision now rests on cache-hit counts and retained-bytes evidence from engine logs (throughput is proven neutral), plus graph capture/replay and cancellation lifecycle tests.

## Kernel and transport implementation order

1. Implement paged compressed-KV tiling with fused dequantization/online softmax and reuse across speculative queries within the same sequence. Preserve causal masks, quantization transforms, partial pages and recurrent-state rollback. Compare against reference attention before performance tests.
2. Tune split count jointly with tile shape and query grouping by length/client bucket. Keep the existing default control. Forced 64/128 splits have rejected history; do not promote them without new evidence.
3. Replace full-history continuation dequant/concatenation with bounded tiles. Measure this as a TTFT/transient-memory improvement independently from decode.
4. Keep rank-local KV resident. Confirm actual XCCL peer transport using effective library configuration and traces. Measure stable-buffer and allocation-churn collectives separately. Evaluate narrow IPC-cache workarounds only against a reproduced fault signature.
5. Evaluate fixed-depth serving variants before adaptive speculation. A later adaptive implementation must coordinate scheduler token counts, proposal shapes, verifier masks, KV/recurrent rollback and graph selection. Add hysteresis and fallback using committed tokens/time, rather than acceptance alone.
6. Consider fused DFlash greedy edge scoring only if traces show material cost; preserve tie behavior. Existing local top-K/argmax reductions should first be verified active, not duplicated.

## Release gate

Aim for at most 15% decode-throughput loss relative to the original 64k performance at both 128k and near262k, plus an absolute speed improvement. Measure TTFT and p99 latency independently. This target is unproven and may require different hardware or a different trained model if exact full-attention costs dominate after optimization.

Require passing supported-dtype startup/dispatch checks, real-task quality, boundary lengths, mixed clients, prefix reuse, natural EOS, tool parsing, cancellation, graph replay and multi-hour allocation-churn soak. Capture reproducible image/source hashes and configuration. No device faults, corrupted outputs, NaNs, cross-request state contamination or hidden capacity regression may be traded for speed.

Current release decision: the production lane remains on the certified v1.2.5 TQ4-NC image. The campaign promotes **fp8_e4m3 + nospec to release-candidate status** — it is the only lane meeting the 128k falloff gate (−13.1% ≤ 15%) and improves absolute throughput at every length — but promotion is blocked on the quality, soak and P0-fence legs above. Speculative modes stay excluded from ≥128k serving on current evidence.

## UPDATE 2026-09-10: FP8+spec collapse ROOT-CAUSED and FIXED — spec readmitted at all lengths; prod re-based

Diagnostic chain diag1–diag8 (`fp8-mtp4-v1/REPORT.md` is the campaign record; `master_diag4..8.sh` the chains):

- **Root cause (§4.3)**: the entire fp8_e4m3+MTP long-context collapse is the
  v51 `triton_fp8_mq` verify kernel's own cost curve — 0.555 + 0.284·q_len
  µs/KVtok — where flash q=1 charges 0.075. Measured by forcing the kernel
  onto the q=1 shape (dQ-kq1: 0.838 slope, 11× flash). Draft cost ≈ 0; every
  other route measured (stock flash q5: 3.35; tl.dot: 4.33, rejected).
- **Fix (§5)**: position-sliced flash fan-out — decompose the uniform q_len
  fp8 paged verify into q_len flash q=1 calls (shared paged KV/descales,
  device-side per-position causal limits). Knob `VLLM_XPU_FP8_FANOUT`;
  baked default-ON in `llm-scaler-exp:fp8-mtp4-v5`
  (sha256:e536666de558…, = v1.2.5 + SPLITS=16 + fan-out).
- **Gates (§6) — VERDICT PASS**: fp8+mtp4 @c1 = 65.2/52.67/33.21/23.47 tps
  @8k/64k/128k/262k — beats fp8+nospec (29.79/25.91/20.69) by +77/+28/+13.5%
  and tq4nc+mtp4 (25.06/15.31/7.50) by 2.1–3.1×; 2k = 77 tps (no short-ctx
  sacrifice); steps deterministic, cancels clean, no wedge. The 128k
  relative-falloff gate is superseded by direct absolute superiority over the
  previously-best lane at every length.
- **Prod standing since 2026-09-10 12:00:44 UTC**: fp8_e4m3 + mtp4
  @0.9/262144 on `llm-scaler-exp:fp8-mtp4-v6` == `llm-scaler-exp:v1.2.7`
  (sha256:7cf3d51cfc60…; = v5 + baked `VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1`).
  diag11 dual-lane byte-parity cert: dW6-e4m3 13/14 vs dV5-cert (the 1
  diff = documented §8.2 boot-1 near-tie), dW6-e5m2 6/6 vs dX2 with EMPTY
  extraenv (env-passed == env-baked; e5m2 now serve-time selectable via
  `--kv-cache-dtype fp8_e5m2`, no env). Rollback:
  `-e VLLM_XPU_ALLOW_E5M2_FP8_CKPT=0` per boot, `prod_restore_v5.sh`
  (v5), or `prod_restore127.sh` (tq4nc).
- Still open from the legs above: ~~multi-client soak~~ CLOSED (diag10:
  10 cycles × 2k C2 + 64k C2 + 262k C1 on the standing lane, 72 min,
  0 faults, 0 errors, no perf time-trend; cancel re-verified rc=0,
  engine_abort_count=0) and ~~cross-boot determinism~~ CLOSED (3 boots:
  rep1 byte-identical ×3; rep0 byte-identical boots 2+3, boot 1 = one
  documented fp near-tie acceptance flip, correct in every resolution).
  Quality is validated per-cell by the suite's task checks (needle_hit,
  multihop, no repeated 10-grams) across every campaign cell. Remaining:
  the P0 nospec wedge fence is unchanged (nospec remains the rollback
  lane); optional future re-base onto e5m2 KV (+33-39% @262k over e4m3,
  validated; selectable on v6 with `--kv-cache-dtype fp8_e5m2` — the
  guard env is baked).

