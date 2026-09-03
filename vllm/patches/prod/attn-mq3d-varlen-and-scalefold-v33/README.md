# attn-mq3d-varlen-and-scalefold v33 — spec fixed on every KV dtype that can boot; the fp8
# verify pathology root-caused and killed (2.56x); our own per-step sync
# barrier found and removed (+13.7% @2k); 4bit+spec now has the flattest
# long-context curve measured — but conc16 exposes the eager-draft ceiling

Window 2026-09-02 (image lineage v31.1). Directives: fix spec on ALL KV
cache dtypes, maximize tok/s on spec AND nospec, flatten the long-context
drop.

## TL;DR

1. **fp8-e4m3 + spec verify SOLVED (#15 closed)**: root cause = verify
   (q_len=k+1, `is_mix_batch=False`) routes to the C++ `_vllm_fa2_C`
   chunk-prefill branch with NO KV splits — grid (1, Hk=2/GPU) = 2 CTAs
   serially scanning 65k fp8 KV. The `num_splits_kv` shim passthrough
   probe was DEAD (branch ignores it; bit-identical, same wall). Fix =
   route small-q paged fp8 verify batches to the in-tree Triton
   `unified_attention` 3D split path (wrapper 3D gate relaxed for
   q<=8; helper + routing in flash_attn.py; `VLLM_XPU_TRITON_MQ3D=0`
   rolls back). Outputs BIT-IDENTICAL to the C++ path (greedy hashes
   stable across 4 boots). Warm-prefix A/B @65k pure decode:
   **C++ 127.2s/900tok (7.08 tok/s) -> Triton 3D 49.6s (18.14)** =
   2.56x. NSEG sweep: 32=124.5s, **64=optimal**, 128=101.2s.
   v32's original "249.5ms/step" alarm was directionally right (warm
   C++ decode = 141ms/token); the mid-window "~113ms correction" was
   a prefill-contamination misread — cold/warm A/B is the truth.
2. **Scale-fold** (`v33_scalefold.py`): mode-1 fp8 per-tensor KV scales
   moved from every-KV-element f32 detour (2x8192 f32 ops/tile) onto
   the S/P tiles (512 elems, 16x fewer; s*(Q.K)==Q.(s*K),
   (P*s)@V==P@(s*V)). BIT-IDENTICAL hashes, 65k 95.1->94.7s (marginal:
   the path is bandwidth/overhead-bound, not ALU-bound). Kept — strictly
   less work per tile.
3. **The per-step sync was OURS**: py-spy on TP0 (40.6% of host samples
   blocked in `torch.xpu.synchronize()` inside `propose_draft_token_ids`)
   turned out to be the v2x-era `VLLM_XPU_SPEC_DRAFT_BARRIER` oneCCL
   wedge mitigation — a full device drain on EVERY spec step, default
   ON, predating the v31.1 "whole-step graph is clean" conclusion.
   `VLLM_XPU_SPEC_DRAFT_BARRIER=0`: fp8+spec 2k 21.16->24.05 (+13.7%);
   4bit+spec 2k 19.65->25.25 (+28.5%); 65k +3-4%. Hashes unchanged
   (timing-only). No wedge in sustained testing (3x warm 65k back-to-back
   + full batteries ~10 min). WATCH ITEM for long prod runs.
4. **Final matrix** (warm prefix cache, k1, barrier off, same scripts):

   | lane | @2k | @8k | @65k warm | conc16@8k agg | greedy |
   |---|---|---|---|---|---|
   | 4bit TQ + spec | 25.25 | 23.25 | **23.73** | 21.92 (!) | stable, coh -0.452 |
   | fp8 e4m3 + spec | 24.05 | — | 18.76 | — | stable |
   | k3v4 + spec | 19.79 | — | 18.65 | — | stable |
   | 3bit + spec | 24.10 | — | 14.98 | — | stable |
   | fp8 e4m3 nospec | 33.38 | — | 23.99 | — | BIMODAL 2/3 |
   | 4bit TQ nospec (prod) | 33.50 | — | 22.28 | **55.67** | stable |

   4bit+spec has the FLATTEST curve measured (2k->65k = -6%, vs -33%
   nospec) and is joint-best @65k single-stream. All TQ dtypes now run
   spec correctly (3bit/k3v4 work, just dominated: 3bit 65k is
   unpack-ALU-bound, 14.98 despite 24% fewer KV bytes).
5. **Spec conc16 collapse — the remaining structural defect**: 4bit+spec
   conc16@8k = 21.92 agg vs nospec 55.67 (2.5x worse; TTFT 80.9s).
   py-spy during conc16: eager MTP draft 52-62% of TP0 host samples,
   TQ prefill python, per-call Triton JIT dispatch (cache-key compute)
   for the MQ kernel — none of it graph-captured, serializing every
   step (the #16 acceptance-gated serialization, now with batch-scaled
   draft work). Fix = P2/P3 (worker-resident acceptance loop / capture
   propose+sample in the decode graph) — upstream-scale work.
6. **e5m2 root cause (was "unbootable mystery")**: upstream guard
   `vllm/model_executor/layers/attention/attention.py:168` raises
   `ValueError("fp8_e5m2 kv-cache is not supported with fp8
   checkpoints.")`. Not an XPU bug. WONTFIX: e4m3 is same 1B/elem with
   better precision for bounded KV values.
7. **fp8-e4m3 nospec greedy BIMODALITY**: 2/3 fixed prompts flip between
   two stable outputs (batch-state-dependent argmax; C++ KV-split
   reduction order varies with resident batch; prompt 2 never flips).
   Not corruption — both modes coherent, reps deterministic within a
   batch state. 4bit lanes are rep-stable on the same prompts. Documented
   as a lane property; would need fixed split counts to "fix".
8. **Warm-cache methodology note**: `--enable-prefix-caching` makes the
   second identical 65k run skip prefill (~45s): always run t65k twice
   and report the warm run as pure-decode. Also: first traffic after a
   fresh boot pays ~10s one-time graph-capture/JIT (lane_bench.sh
   warms up first).

## Where the spec step goes now (post-v33, @65k warm, 4bit)

~53ms/token: graph-replayed verify forward + triton 3D fp8 / TQ MQ
attention + eager MTP draft (python IR-op dispatch per layer op —
rms_norm/rotary/gemm frames) + GDN metadata build (~14% host) + the two
unavoidable per-step event syncs (sampled-ids readback, v32 deferred
counts). Host is no longer sleeping in our barrier; it is busy in the
eager draft python — which is exactly what P3 (draft graph capture)
removes.

## Artifacts

- `v33_mq3d_triton.py` — THE fp8-spec fix (wrapper 3D gate + routing +
  helper). Apply in-container after bootp.sh; lost on `docker rm`.
- `v33_scalefold.py` — S/P-tile fp8 scale folding (bit-identical).
- `v33_bm16.py` — BLOCK_M=16 for q<=8 verify (measured neutral; kept
  for completeness).
- `lane_bench.sh` — warmup + 2k ctxbench + t65k x3 (cold+warm+warm) +
  f8ref hashes, per lane.
- Host helpers: `/root/build/t65k.py` (65k wall timer around
  longdecode65k.py), `/root/build/f8ref.py` (3-prompt greedy hashes),
  `/root/build/bootp.sh` (boot -> v32 patches -> restart -> health).

## Boot recipes (v33)

- fp8+spec k1: `bootp.sh '{"method":"mtp","num_speculative_tokens":1}'
  'VLLM_XPU_SPEC_DRAFT_BARRIER=0' <log> '--kv-cache-dtype fp8_e4m3'`
  then apply v33_mq3d_triton.py + v33_scalefold.py + restart.
- 4bit+spec k1: same minus kv flag (default is turboquant_4bit_nc) and
  minus the v33 patches (they are fp8-path-only).
- nospec: `bootp.sh nospec '' <log> [flags]` (the "booted v22 VAR
  spec=..." echo from serve_user_nospec.sh is cosmetic; verify with
  `ps aux | grep speculative`).

## Posture

Prod stays **v31.1, NOSPEC, turboquant_4bit_nc** (33.50 @2k, 22.28 @65k
warm, 55.67 agg conc16@8k, coh -0.451 band): conc16 throughput decides.
The 4bit+spec k1 lane is a validated OPT-IN for single-stream
long-context workloads (flattest curve, 23.73 @65k, +6.5% over nospec
there, bit-stable across boots, coh PASS) — boot with the barrier off.
Spec-on-fp8 is fixed and usable (18.76 @65k warm) but no longer the
fastest anything. Open levers, in order: P3 draft graph capture (kills
the conc16 collapse AND the 2k deficit), P2 worker-resident loop,
GDN build cache (~5ms/step), P1b XMX tiles.
