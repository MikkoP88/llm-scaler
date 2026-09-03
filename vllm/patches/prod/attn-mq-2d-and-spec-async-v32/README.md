# attn-mq-2d-and-spec-async v32 — spec-decode pipeline decomposed to the metal: MQ
# verify kernel FIXED (regpatch), host chain convicted (overlap-hidden),
# align-drain patch perf-neutral, KV dtype matrix extended to fp8 — spec
# still loses to nospec-graphs, and now we know exactly why

Window 2026-09-01 (fourth GPU window of the day; image lineage v31.1).
Directives: deep analysis of every pipeline stage that makes spec slow
("in reality spec is superior"), fix across ALL dependencies and kernels
(not just vLLM params), prevent CPU from gating tok/s, include the fp8
KV dtype family in testing/improvements, solve k4.

## TL;DR

1. **Spec verify routing SOLVED**: verify (q_len=k+1) routes
   `TurboQuantMetadataBuilder.build` → `is_prefill=(max_query_len>1)` →
   `TurboQuantAttentionImpl.forward` pure-prefill branch →
   `_prefill_attention` continuation-chunk branch (q_len<=128) → the
   multi-query kernel `triton_turboquant_mq_decode_attention` (confirmed
   by triton JIT cache: `_tq_mq_decode_stage1` present, `_tq_decode_stage1`
   absent in a k1 boot). The shared-KV design works at 2k (verify fwd ==
   nospec fwd) but cost +30ms at 65k.
2. **MQ kernel cost ROOT-CAUSED and FIXED** (dependency-level fix,
   `v32_mq_regpatch.py`): three `[Q_BLOCK, BLOCK_KV, BLOCK_D]` fp32 temps
   (FP8 scores, MSE term1, value accumulation) double the register
   footprint vs the tuned single-token kernel → spills → occupancy loss
   that only bites on long scans. Per-row static-loop rewrite keeps temps
   2D. k1: tforward@65k 72.0→**66.1ms** (−8%), drafter same-kernel paths
   −15/−18%, @2k 19.4→19.65, coh PASS (Paris −0.452, distinct=1).
   `VLLM_TQ_MQ_STAGE1_WARPS=4` arm REGRESSED both ctx (77.4 @65k, 18.65
   @2k) — occupancy, not issue-width, was binding.
3. **Remaining verify gap = per-row ALU physics**: nospec 65k scan ≈
   40ms ≈ ~14ms bandwidth + ~26ms compute (MSE unpack + centroid gather
   + score/value math). Verify = bandwidth + n_rows × compute → k1 ≈ 66,
   k3 ≈ ~120 (measured k3 step ≈ 4×compute). Cutting it needs XMX
   (`tl.dot`) score tiles — a numerics decision (tf32), not done.
4. **Host round-trip chain (R2) is what kills spec at short/mid ctx** —
   and it is NOT what py-spy's worker percentages said: the align-mode
   accepted-count drain (57% of worker samples, v32 window's first
   suspect) is a device-wait that async execution already overlaps →
   removing it (async pinned copy + deferred postprocess,
   `v32_align_async_v2.py`, correctness-validated incl. 65k + coh) is
   **perf-neutral both alone and stacked on the kernel fix** (19.78 @2k /
   86.1ms @65k step, identical to kernel-fix-only). The REAL exposed host
   cost at 2k ≈ 54ms/step: drafter propose EAGER python ~7ms (inductor
   off by the v31.1 gate), GDN attn metadata build ~4.4ms (gdn_attn.py
   build per step), block-table commit H2D ~2ms, prep/output/IPC chain
   ~30ms (engine 100% idle in shm_broadcast sched_yield — latency, not
   CPU). Nospec hides all of this via async-scheduling overlap; spec
   CANNOT because acceptance of step N gates scheduling of N+1 (vLLM v1
   architectural serialization).
5. **Verdict** (v31.1 posture, 4bit_nc, B70×2, warm ctxbench conc=1):

   | arm | @2k tok/s | @65k tok/s | coh |
   |---|---|---|---|
   | nospec graphs (prodns3) | **33.56** | **23.61** | −0.451 |
   | spec k1 baseline | 19.42 | 20.7 | PASS |
   | spec k1 + MQ regpatch | 19.65 | 22.2 | PASS −0.452 |
   | spec k1 + regpatch + align | 19.78 | 22.2 | PASS −0.452 |
   | spec k3 + regpatch + align | 15.11 | (n/a) | PASS −0.452 |

   Spec's deficit decomposes as: @2k −41% = host chain (54ms exposed);
   @65k −6% = per-row verify compute. For spec to WIN it needs BOTH the
   worker-resident loop (P2, overlap host with device across steps) and
   XMX verify tiles (P1b). Neither is a parameter.
6. **KV dtype matrix (user directive, COMPLETE)**: TQ lane — 4bit_nc =
   only fused TQ dtype (3bit_nc/k3v4_nc fall back to
   `_tq_full_dequant_kv`, −14% @65k, quality perfect). fp8 family —
   **e4m3 + NOSPEC is the fastest long-ctx lane measured: 33.51 @2k
   (parity), 25.06 steady @65k (+6% over 4bit_nc), coh PASS (−0.446)**
   (raw fp8 attention skips the MSE-unpack ALU that binds the 4-bit
   lane — the #14 B/C decomposition confirmed from both sides; prod
   candidate pending the conc16/capacity battery — KV bytes double).
   e4m3 + SPEC is catastrophic @65k (verify 249.5ms/step = 3.8× TQ;
   −10% @2k; coh PASS). `fp8_e5m2` unbootable (WorkerProc startup
   failure, #05c class). Spec stays off fp8; prod stays 4bit_nc.

## The full step cost map (SPECTIMING 200-step flushes + py-spy, k1/k3)

| component | k1@2k | k3@2k | k1@65k | nature |
|---|---|---|---|---|
| verify forward (device) | 25 | 44.1 | 66.1 (was 72) | B + n·C in ctx |
| logits | 2.2 | 2.3 | 2.2 | device |
| drafter forward | 7.2 | 6.9 | 6.6 | device (eager host path) |
| propose (host py) | 7.2 | 20 | 7.6 | host, ∝ k |
| exposed host chain | ~54 | ~54 | ~0 (hidden) | host |
| step wall | 98.6 | ~134 | 86.1 | |
| tokens/step | 1.918 | 2.02 | 1.918 | k1 accept 91.8% |

Acceptance (k1, 868 samples): 91.8% pos-0 → 1.918 tok/step. k3: 2.02
(82.7/66/53%). Real k1 period @2k = 98.6ms ≠ 80.4 profiled sum → the
~54ms exposed chain was found by difference + py-spy.

## R1 evidence chain (MQ kernel)

- Dispatch: `turboquant_attn.py` — `_TQ_MQ_VERIFY` default ON (env
  `VLLM_TQ_MQ_VERIFY=0` restores v18 synthetic per-row decode),
  `_TQ_MQ_MAX_Q=8`, `_CONTINUATION_DECODE_THRESHOLD=128`; graph capture
  forces the continuation path for multi-token captures (v19b fix, so
  replays read the real KV).
- Launcher comparison: identical grid (1,Hq,32), identical
  BLOCK_KV=4/BLOCK_D=128, only `_TQ_MQ_STAGE1_{WARPS,STAGES}` separate —
  both default 1/1 → schedule identical, kernel body was the difference.
- Kernel body: loads/gathers shared across rows; the three 3D temps were
  the regression. `v32_mq_regpatch.py` = three anchored rewrites to
  per-row `tl.static_range` loops with where-masked row assembly
  (identical reduction axes per row → bit-stable, confirmed by coh).

## R2 evidence chain (host round trip)

- Worker py-spy (patched boot, 2k decode, 3266 samples): 613 @
  gpu_model_runner.py:1489 (drain — later shown overlap-hidden), 148 @
  gdn_attn build, ~228 across drafter propose eager frames
  (all_gather_into_tensor, rotary forward_static, get_top_tokens GEMM),
  68 @ block_table commit copy_to_gpu, 39 @ graph replay (benign).
- Engine py-spy (592 samples): 100% `shm_broadcast.dequeue →
  acquire_read → wait/sched_yield` — pure idle; engine CPU is never the
  bottleneck; the chain is latency.
- Align-drain async patch (`v32_align_async_v2.py`): N-view snapshots at
  deferral + consume at execute_model entry (before `_update_states`
  at :3958 — the N-view survives there), dead-request and moved-state
  guards. Correct at 2k/65k/coh. Perf: +5% @2k alone (noise-level),
  0 stacked → parked, not shipped (prod keeps blocking drain: simpler,
  same speed).

## k4 (#12)

`gated_delta_rule_spec_kernel` (gated_delta_rule.hpp:294-560) EXONERATED
by full code-read: runtime-bounded token loop, init_col clamped to
[0, num_spec_tokens-1], per-token state writeback — no width-4
assumption. Conviction region stays "5-token spec-step numerics"
(verify/rejection assembly under capture). Clamp 4→3 stays. Upstream
vllm#54785 + addendum already posted.

## P1–P6 efficient-pipeline design (what would actually flip spec ahead)

- **P1 (shipped here)** MQ kernel register fix — any-KV-dtype fused
  verify lane now exists and is tuned; 3-bit lane still needs its own
  stage1 kernel (P5).
- **P1b (open, biggest device lever)** XMX score tiles: `tl.dot`-based
  [16-pad(Q_LEN), D]×[D, BLOCK_KV] scores on BMG XMX units — removes the
  per-row compute multiplier; numerics decision (tf32) required by the
  coherence gate.
- **P2 (open, biggest host lever)** worker-resident acceptance loop:
  compute acceptance + next propose ON the worker for m steps before
  returning to the engine (vLLM v1 architectural change; upstream
  discussion material — the acceptance dependency makes async-scheduling
  overlap impossible as designed).
- **P3 (open)** propose metadata on GPU + drafter into a captured graph
  (drafter is EAGER python today because the v31.1 gate disables
  inductor for spec safety): ~7ms/step k1, 20ms k3.
- **P4 (validated, parked)** align-drain async counts — correctness
  pattern preserved in `v32_align_async_v2.py` for whenever the TODO
  upstream ("Remove .cpu() sync to enable fully async for hybrid model")
  is done properly.
- **P5 (open)** fused 3-bit TQ decode kernel (would give nospec AND MQ
  lanes a −25% KV-bytes lane; quality gate already proven perfect).
- **P6 mamba/GDN**: GDN decode is already O(1)/token; align-mode python
  postprocess is overlap-hidden (measured); only GDN metadata build
  (~4.4ms/step, gdn_attn.py build) is a real target — cacheable dict
  work, no kernel change needed.

## Artifacts

- `v32_mq_regpatch.py` — the MQ kernel fix (apply in-container to
  `vllm/v1/attention/ops/triton_turboquant_decode.py`; anchors verified,
  py_compile gate; survives `docker restart`, lost on `docker rm`).
- `v32_align_async_patch.py` (v1, SUPERSEDED — KeyError at 65k; kept for
  the lesson: consume point must precede `_update_states`) and
  `v32_align_async_v2.py` (correct; perf-neutral; parked).
- `longdecode65k.py` — 67,485-token filler + ignore_eos 900-tok driver
  for pure-decode SPECTIMING flushes (host: /root/build).
- Host helpers: `/root/build/bootp.sh` (boot→patch×2→restart→health).

## Posture

Prod stays **v31.1 image, NOSPEC, turboquant_4bit_nc** (33.56/23.61,
coh −0.451). The MQ regpatch is a validated, upstreamable kernel fix
kept as a patch artifact (not baked into the prod image — spec itself is
not shipped). k4 clamp unchanged.
