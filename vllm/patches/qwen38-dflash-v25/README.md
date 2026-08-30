# qwen38-dflash-v25 — every-step pre-drafter drain + draft collective shrink
(#11 wedge) — **DID NOT FIX THE WEDGE** (validated 2026-08-30)

`llm-scaler-vllm-adv:v25` = validated v22 image + two env-gated overlays
aimed at the **>=32k long-context MTP wedge** (KNOWN_ISSUES #11):

1. `VLLM_XPU_SPEC_DRAFT_BARRIER` (default 1; =0 restores v22) with
   `VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX` (default 0 = every spec step):
   `torch.xpu.synchronize()` before invoking the drafter in
   `gpu_model_runner.propose_draft_token_ids`, gated by max running context.
2. `VLLM_XPU_MTP_LOCAL_ARGMAX` (default 1; =0 restores v22): the drafter's
   `_greedy_sample` uses the tiny `[bs, topk]` top-token gather for ALL
   batch sizes (incl. bs=1) instead of the full-vocab (248,320) all-gather.

## VERDICT (2026-08-30): the overlays are harmless but NOT a fix

The full arm matrix (KNOWN_ISSUES #11 final update) shows the wedge is a
**device-side kernel spin in the GDN spec-state path** — both GPU compute
and copy engines report 100% busy while EU utilization sits at 11-22% and
both ranks' host threads spin-block in full in-flight queues. Host-side
barriers, collective shrinking, graph removal (NG), draft-collective
removal (DTP1), the align-mode D2H bypass (`--mamba-cache-mode all`), and
prefill-chunk resizing (MNBT 4096) ALL still wedge at k=4.

Calibration note: the probe filler is 15.7443 tokens/repetition; the arm
rows labeled "32k" below were actually ~65k prompts (probe bug, fixed
2026-08-30). k=4 wedges at true 32k as well (DTP1/MALL rows).

## Evidence chain (2026-08-30, host 10.20.3.64, block512 + mamba-fp16, MTP k4)

| arm | config | wedge rate |
|---|---|---|
| E0 | user baseline | **3/3** (~65k real) |
| E2 | `use_local_argmax_reduction:true` (bs=1 probes; guard keeps full-vocab path) | **2/3** |
| E2-bs2 | same boot, 2 concurrent streams — tiny `get_top_tokens` gather ACTIVE | **5/8** (+1 instant-EOS degenerate, see #13) |
| E1 | `CCL_ENABLE_SYCL_KERNELS=0` | **2/3+** |
| v25 | both overlays ON (this image) | **3/3** |
| NG | `cudagraph_mode NONE` | **2/3** |
| DTP1 | `draft_tensor_parallel_size=1` | true-32k 1/3, 64k 0/2, 131k 1/2 |
| MALL | `--mamba-cache-mode all` | **2/3** |
| E4 | MNBT 4096 | **1/3** |
| K1 | `num_speculative_tokens=1` | 32k **0/7**, 64k 1/3, 131k 1/3 |

Live captures: v25 boot py-spy — both ranks at the align-mode `.cpu()`
submit (`gpu_model_runner.py:1489`), identical at +20 s/+40 s; DTP1 boot
py-spy — both ranks blocked enqueueing `torch.ops._xpu_C.gdn_attention`
(`_xpu_ops.py:183`), wedge formed during a PREFILL chunk. The moving
"stuck line" confirms the spin is upstream in device code; the host
blocks wherever the saturated queue catches it.

## What v25 IS good for

- The tiny-gather default (`VLLM_XPU_MTP_LOCAL_ARGMAX=1`) is a real
  per-step win when MTP is used at all: it removes a full-vocab (248,320
  value) all-gather and logits materialization per draft step (~128x less
  collective traffic), and it is required plumbing for the k=1 posture.
- The barrier overlay is inert overhead otherwise; `=0` restores v22.

## Validation (k=1 posture on this image — the usable MTP config)

| cell | spec | probes | result |
|---|---|---|---|
| v25 32k sequential | mtp k1 | 4/4 + 3/3 | **0 wedges**, 27.9-30.3 tok/s (nospec parity 27.9) |
| v25 64k sequential | mtp k1 | 2 + 1 | 1 WEDGE (~33%) |
| v25 131k sequential | mtp k1 | 1 + 2 | 2 WEDGE (~66%) |
| v25 concurrent 32k x2 (x2 rounds) | mtp k1 | 4 streams | 0 wedges; throughput valid, content = #13 filler artifact |
| v25 concurrent 131k x2 | mtp k1 | 2 streams | 0 wedges; 15.1 tok/s survivor |
| canonical car-game | mtp k1 | 512 tok | 45.4 tok/s (vs 33.2 nospec), correct `<think>` + HTML |
| acceptance | mtp k1 | serve log | E[len] 1.67-1.95, P0 rate 0.80-0.95 |

**Ship guidance:** full 262k envelope -> serve WITHOUT
`--speculative-config` (0 wedges at every length, faster than k=4
everywhere, +26% KV capacity). Envelope capped at 32k ->
`{"method":"mtp","num_speculative_tokens":1}` on this image (+37% short
ctx). A true >=64k MTP fix requires GPU-side kernel debugging of the GDN
spec-state path, not vLLM-side changes.

## Notes

- The align-mode `.cpu()` D2H remains (victim, not cause; upstream TODO
  "Remove .cpu() sync"). Deferring it by one step reorders mamba state
  shifting after the next forward — not safe; revisit only with a graphed
  propose segment (v20 L1 follow-up).
- The target sampler's own full-vocab logits gather (needed for
  temperature/top-k sampling) is unchanged and unavoidable.
