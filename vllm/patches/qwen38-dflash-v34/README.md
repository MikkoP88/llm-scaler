# qwen38-dflash v34 — the latency question answered end-to-end: fabric at
# 55µs floor, host chain is the wall; #18/#19 closed with root-cause
# evidence; k3 spec enabled as documented opt-in (best-ever 65k); one
# patch kept (bit-identical), one rejected by the A/B gate

Window 2026-09-02 (image lineage v31.1). Directive: fix latency; test
CCL_ATL_TRANSPORT=ofi / CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 /
bigger batches; minimize GPU<->GPU latency from vLLM code; deep-optimize
the XPU varlen shim; fix #17 (spec conc16), #18 (fp8 bimodality), #19
(e5m2); allow spec k>=2; comprehensive testing incl. 126k/262k.

## TL;DR

1. **ENV arms: all null by construction.** The oneCCL SYCL runtime
   already forces `CCL_ATL_TRANSPORT=ofi` (serve log: "value of
   CCL_ATL_TRANSPORT changed to be ofi (default:mpi)");
   `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` was already in the
   baseline env. Concurrency: agg tok/s @8k saturates ~85 (conc16
   56.4 -> conc32 84.6 -> conc64 86.2); max-num-seqs=64 not binding;
   concurrency helps to ~32 streams, beyond = latency-only.
2. **GPU<->GPU latency measured at the floor**: 2-rank allreduce via
   the real backend (`torch.distributed` "xccl", from the serve log) =
   **55µs/op flat from 256B to 64KiB, 119µs @1MiB**. The fabric is NOT
   the ~30ms "IPC/latency" in the #16 chain — that is the
   engine<->worker host round-trip. No vLLM-side GPU<->GPU optimization
   exists to make; the lever is structural (#16/P3), not transport.
3. **Host-chain profile (k3 @65k, TP0 py-spy)**: draft/propose 71% of
   host (qwen3_5_mtp 64%: eager IR-op dispatch spread across
   rms_norm/rotary/GEMM; qwen3_5 67%: lm_head vocab-parallel allreduce,
   get_top_tokens, sample), GDN build 9.1%, triton 10.2%, jit 6.0%,
   all_reduce 4.9%. No single killer; capture (P3) is the fix. The
   capture blocker is precisely the draft loop's per-position
   `build_per_group_and_layer_attn_metadata(draft_index=...)`
   (fresh tensors per position defeat XPU-graph capture); seam =
   `cudagraph_dispatcher.get_capture_descs`/`_warmup_and_capture`.
   NOT attempted (v29c k4 corruption lesson; upstream-scale surgery).
4. **#18 CLOSED intrinsic** (fp8-nospec greedy bimodality):
   - `VLLM_BATCH_INVARIANT=1` (upstream pin) UNBOOTABLE here —
     reroutes the attention selector (`selector.py:153` mamba/GDN
     batch-invariance gate) -> silent native crash-loop during init.
   - Our `VLLM_XPU_FA2_PIN_SPLITS=N` (v34_f8bi_pin.py): N is a CAP,
     not exact — C++ still picks actual splits <= N -> bimodality
     persists, flipping WITHIN one f8ref invocation (P1/P3 two modes
     each; P2 invariant); cap=64 perf-free (65k 24.04 vs 23.99).
   - splits=1 (the only forcing value) = the #15 serial-scan pathology.
   Needs an exact-splits / batch-invariant XPU attention kernel.
   Prod 4bit lane unaffected (rep-stable).
5. **#19 CLOSED triple-blocked** (e5m2 KV): guard env-bypassed
   (v34_e5m2_guard.py) -> graphs boot dies at decode graph capture
   (silent) -> eager boot healthy but the FIRST request kills the
   worker: `Unrecognized FP8 dtype: fp8_e5m2` — the XPU FA varlen
   dispatch has NO e5m2 kernel at all. Wontfix is now kernel-proven.
6. **Spec k-structure settled**: k1 = hash-lossless vs 4bit-nospec
   reference. k2 AND k3 both diverge on 2/3 prompts (coherent, coh
   Paris -0.452; prompt 2 hash is k-invariant) — #18-class batch-state
   sensitivity of the verify batch, not corruption. **k3 is the
   best-ever single-stream long-ctx lane**: 65k warm 29.25 (+31% vs
   nospec 22.28), 126k 15.00, 258k 11.30, no wedge in 3x65k sustained;
   conc16 13.99 (WORSE than k1 21.92 — draft work scales with k).
   k2 (65k 23.54, conc16 16.98) dominated by k1 everywhere. Posture:
   k1 = default opt-in; k3 = documented opt-in for single-stream
   envelopes <=~100k (advantage inverts by 126k — see crossover).
7. **v34 patch A/B (fp8+spec k1 lane)**:
   - `v34_shim_opt.py` VALIDATED — caches the v33 mq3d shim's per-call
     allocs (descale ones, kv8 dtype views) keyed by cache-tensor
     identity on the impl object. Bit-identical: f8ref hashes equal
     the v33-only phase EXACTLY across a restart; 65k warm 18.59 vs
     18.76, 2k 23.66 vs 24.05 (noise). Kept: strictly less per-call
     work; benefits compound under conc/short-ctx call rates.
   - `v34_gdn_cache.py` REJECTED by the A/B gate — steady-state GDN
     build memo broke ALL THREE hashes (plus within-invocation
     mismatch: metadata fields vary beyond the memo key's coverage)
     AND collapsed 65k warm 18.76 -> 10.24 (-45%): the per-step key
     (`numpy().tobytes()` on staging tensors) forces D2H syncs worth
     ~+44ms/step. Do not resurrect without (a) a sync-free key and
     (b) a completeness proof for the memoized fields.
8. **Long-ctx matrix (warm, tXXk, 900 tok decode)**: nospec prod lane
   and k3 measured at 65k/126k/258k (258k = the max-model-len point,
   262144 - prompt/comp margin; filler calibrated 1.127 tok/word).
   See final table in the window log; curve shape: nospec 22.2 ->
   (126k, 258k measured this window, see below) vs k3 29.3 / 15.0 /
   11.3 — spec flattens the fall until ~126k, converges by 258k.
9. **Warm-cache methodology** (unchanged): report the SECOND identical
   run; first post-boot traffic pays JIT/graph warmup (2k 22.85 ->
   33.49 in this window's own battery).

## Final matrix (warm, this window, same scripts as v33)

| lane | @2k | @65k warm | @126k warm | @258k warm | conc16@8k agg | greedy |
|---|---|---|---|---|---|---|
| 4bit TQ nospec (PROD) | 33.49 | 22.29 | **17.09** | **11.68** | 47.2* / 55.7-56.4 clean | bit-stable = refs |
| 4bit TQ + spec k1 | 25.25 (v33) | 23.73 (v33) | — | — | 21.92 (v33) | lossless vs nospec |
| 4bit TQ + spec k2 | — | 23.54 | — | — | 16.98 | divergent (2/3) |
| 4bit TQ + spec k3 | — | **29.25** | 15.00 | 11.30 | 13.99 | divergent (2/3), coh OK |
| fp8 e4m3 + spec (v33+v34 shim) | 23.66 | 18.59 | — | — | — | bit-stable (A==B) |
| fp8 e4m3 nospec + pin64 | — | 24.04 | — | — | — | BIMODAL (within-inv) |

*conc16 47.2 was measured immediately after 2x258k (KV holding ~516k
tokens = eviction pressure); the clean reference is 55.67 (v33) /
56.38 (v34 phase 0). Warm-prefix conc16 (same prompts re-run) = 114-116
agg (prefill skipped) — do not compare across cache states.

**Crossover finding**: the k3 spec advantage INVERTS with context —
65k: k3 29.25 vs nospec 22.29 (+31%); 126k: 17.09 vs 15.00 (nospec
+14%); 258k: 11.68 vs 11.30 (nospec +3%). Verify cost scales with
k x context while acceptance saturates. k3 opt-in guidance: single-
stream envelopes <=~100k only. 258k = the max-model-len point
(262144 minus prompt/completion margin; filler 1.127 tok/word).

Prod verify (this window): hashes 0ce080630035 / f167d905a10b /
87d640ad2ed6 exact; coh Paris -0.451 x3; 2k/65k = baseline.

## Artifacts

- `v34_shim_opt.py` — THE keeper: mq3d shim per-call alloc removal
  (bit-identical). Apply in-container AFTER v33_mq3d_triton.py (it
  anchors on the v33 helper) + restart. Lost on `docker rm`.
- `v34_f8bi_pin.py` — #18 evidence/repro: `VLLM_XPU_FA2_PIN_SPLITS=N`
  pin at the flash_attn max_num_splits site. Documents cap-not-exact.
- `v34_e5m2_guard.py` — #19 evidence/repro: env-gates the upstream
  e5m2 guard (`VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1`). Mount + run pre-serve.
- Host helpers: `/root/build/ccllat.py` (xccl allreduce RTT),
  `/root/build/tXXk.py` (parameterized long-ctx decode: ctx_k, tokens),
  `/root/build/f8ref.py` (3-prompt greedy hashes),
  `/root/build/aggk.py` (py-spy raw aggregator), `bootp.sh` (boot ->
  v32 patches -> restart -> health).

## Boot recipes (v34)

- PROD (unchanged): `bootp.sh nospec '' prodvXX ''` (image v31.1,
  turboquant_4bit_nc defaults). Verified this window: hashes
  0ce080630035 / f167d905a10b / 87d640ad2ed6, coh in band.
- fp8+spec k1 + v34: `bootp.sh '{"method":"mtp","num_speculative_tokens":1}' \
  'VLLM_XPU_SPEC_DRAFT_BARRIER=0' vXXf8s '--kv-cache-dtype fp8_e4m3'`,
  then apply v33_mq3d_triton.py + v33_scalefold.py + v34_shim_opt.py,
  restart.
- spec k2/k3 (documented opt-in): bootp with the spec JSON k=2/3 +
  `VLLM_XPU_SPEC_DRAFT_BARRIER=0` (k3: best single-stream long-ctx;
  hash-divergent vs nospec — do NOT use where bit-stability matters;
  never for concurrent serving).

## Addendum (same day, second pass): remaining local levers exhausted

Two further arms closed under an improvements-only gate:

- **Chunked-prefill size (max_num_batched_tokens) 16384 vs 8192**:
  hashes = prod refs exact; 2k 33.47, 65k warm 22.28, conc16@8k
  cold 46.0 / warm 114.8, TTFT 51.8s, 126k cold 124.3s — ALL parity
  with 8192 (33.49 / 22.29 / 47.2+114.5 / 52.8s / 124.4s). Prefill is
  compute/bandwidth-bound, not scheduling-bound; chunk size is a null
  axis. 8192 stands; MNBT32 not worth booting.
- **GDN build memo: approach FALSIFIED** (not just v1's implementation)
  — the build's derived tensors (state indices, query offsets,
  block-table slices) encode per-step acceptance/context state; the
  per-step build IS the freshness mechanism. A "correct" memo would
  recompute the varying fields = the build itself. The ~4.4ms/step is
  only removable inside P3 (frozen + replayed graph). See
  KNOWN_ISSUES #17 addendum.

**fp8-e4m3 KV nospec is a documented perf opt-in**: 65k warm 23.99 vs
prod 4bit 22.28 (+7.7%), 2k parity — for workloads that do not need
bit-stable greedy output (#18 bimodality is intrinsic). Not prod
because bit-stability gates prod.

**Concurrency guidance (measured)**: agg @8k saturates ~85-115
(cache-state dependent) — raise client streams to ~32; beyond is
latency-only. max-num-seqs=64 is not binding.

With that, every locally-implementable lever is either shipped
(v32/v33/v34 keeper patches, barrier-off, k1/k3 opt-ins) or measured
null/poison (GDN memo, MNBT, pin-splits, BATCH_INVARIANT, e5m2).
Maximum-token headroom now lives entirely in P3 (draft graph capture)
and P2 (worker-resident acceptance loop) — upstream-scale, both
blocked on the per-position draft metadata rebuild, both carrying
#04/#12-class corruption risk that demands upstream engineering.

## Addendum 2 (same day, third pass): P3-lite (graphed drafter) closed by direct test

The one remaining scoped increment — static-buffer drafter then capture
— was executed as arm **v35dp1**: `VLLM_XPU_MTP_EAGER_HEAD=0` (the v22
rollback env) restores the stock PIECEWISE drafter, and the static-input
machinery it needs already exists upstream (`self.input_ids` /
`self.hidden_states` / `_slot_mapping_buffer`, activated exactly when
the drafter is in graph mode — `llm_base_proposer._propose_impl`).
Findings:

1. **#09 is GONE on the current tree**: the v21-era oneCCL allgatherv
   segfault no longer fires — v35dp1 boots clean, zero segfaults, decode
   graphs 18/18. The head's `forward` was restructured since v21 to
   return hidden states ONLY; `compute_logits`/`get_top_tokens` are
   called by the proposer's `_greedy_sample` (eager by construction),
   so the logits allgather left the would-be-compiled region.
2. **The v31.1 guard owns the decision anyway**: `xpu.py:344` sets
   `TORCH_COMPILE_DISABLE=1` for spec+TP2 (the #11 fix posture) — the
   head is compile-eligible but never compiled; the target keeps
   whole-step XPU capture; the drafter registers PIECEWISE keys with no
   pieces behind them.
3. **The A/B gate REJECTED the config on its own merits**: f8ref
   `c77d4c73ba1b / f167d905a10b / b78b0a33f97f` — P1/P3 diverge from
   refs, P2 invariant (the familiar 2-of-3 #18-class signature; Paris
   logprob -0.452 vs ref -0.451). Perf parity at best: 2k warm 25.34 vs
   25.25, 65k warm 23.82 vs 23.73, conc16 19.7/30.5 vs 21.92 — no
   capture jump. The PIECEWISE registration merely reroutes draft
   inputs through the static-buffer path (`direct_eager_inputs` off):
   numerics change, zero benefit.
4. **Re-enabling compile for the drafter is convicted, not open**: the
   v31 discriminator matrix wedged compile+capture @1 chunk AND
   compile-no-capture @563 chunks at 65k, surviving every
   splitting-op variant including all-custom-ops-split. The eager draft
   is the measured price of the only clean posture
   (capture-without-compile, v31.1).

Verdict: no local path to a graphed/compiled drafter exists. Draft
capture is upstream-blocked on #11 (compiled-piece × spec livelock,
oneCCL IPC exchange path). P2 remains fork-scale as previously
assessed.

## Posture

Prod unchanged: **v31.1, NOSPEC, turboquant_4bit_nc** (conc16 decides).
Opt-in lanes: 4bit+spec k1 (lossless, flattest small-k curve), k3
(single-stream <=~100k records, hash-divergent; nospec wins >=126k).
fp8+spec carries the v34 shim. Open levers: P3 draft graph capture is
now LOCALLY CLOSED (v35dp1 falsified the only reachable config; the
v31 matrix convicts every compiled variant — upstream-blocked on #11);
P2 (worker-resident acceptance loop) remains fork-scale.
