# Qwen3.8-27B FP8 + DSpark Drafter — B70 XPU Serving Stack

Achieves **72.2 tok/s median (85.9 peak)** on the isolated C1 benchmark
(temp=0, pp=0, n=16) on 2× Intel Arc Pro B70 (TP=2), vs 32.4 tok/s FP8
no-spec and 54.67 tok/s FP8+MTP2 (dspark image, graphs=0).

## Contents

- `Dockerfile` — retrofit build: copies the five fixed DSpark/DFlash files
  into the `qwen36-b70-vllm:b3-maxperf-final-v7` base image.
- `qwen3_dflash.py`, `registry.py` — DSpark draft model classes
  (`DSparkDraftModel`, markov + confidence heads).
- `dflash.py`, `utils.py` — **kernel readout fix** + eager-loop allocation
  fix (see below).
- `gpu_model_runner.py` — adaptive-mode list handling.
- `serve.sh` — reference serving command (image-conditional, see below).

All five python files are byte-identical to the validated
`llm-scaler-vllm-adv:v8` image contents.

## Validated configurations (2x Arc Pro B70, TP=2, drafter-fp8-v5)

| Image | graphs | gmu | len | seqs | dtype | Result |
|---|---|---|---|---|---|---|
| `llm-scaler-vllm-adv:v8` | 1 | 0.8 | 64000 | 64 | bf16 | coherent greedy; acceptance 59.5-65%, mean accepted length 3.38, ~63 tok/s single stream (512 tok / 8.1 s) |
| `qwen38-fp8-dspark:v8` | 0 | 0.90 | 8192 | 1 | bf16 | coherent greedy (rmacy serve.sh @6e63e9e verbatim) |

`serve.sh` picks the row matching `IMAGE` automatically.

The `llm-scaler` image additionally supports XPU graphs (PIECEWISE) with
this stack: the dflash eager drafting loop reuses a grow-only bias scratch
buffer (`dflash.py`) so it no longer churns the XPU caching allocator and
wedges the xe engines during piecewise capture — the historical reason
`VLLM_XPU_ENABLE_XPU_GRAPH=0` was required no longer applies to it.

## The kernel readout fix (why this matters)

SpecForge DSpark trains output position j to predict token anchor+j+1
(LM-style). The stock vLLM dflash kernel sampled draft outputs at query
offsets 1..k (BERT-style), so every draft was off by one position and
acceptance collapsed to ~24% for any SpecForge-trained drafter.

Fix (3 lines):
- `dflash.py`: `num_query_per_req = k` (was `k+1`)
- `utils.py`: `is_sample = is_query` (was `is_query & (query_off > 0)`)
- `utils.py`: `sample_out_idx = req*k + off` (was `req*k + off - 1`)

Effect on the released RadixArk drafter: 24% → 66% pos-0 acceptance.
With our fine-tuned drafter-fp8-v5: per-position acceptance 0.838 / 0.676 /
0.541 / 0.324 at k=4 (draft-position conditional rates), overall acceptance
59.5-65%, mean acceptance length 3.38.

## ESIMD page-attention gate (bf16 / eager correctness)

The XPU paged-attention decode shortcut
(`custom_esimd_kernels_vllm.eagle_ops.page_attn_decode`, inserted in
`vllm/v1/attention/backends/flash_attn.py`) is only correct for fp16
queries under XPU graphs on this build:

- bf16 query: `TORCH_CHECK(query.scalar_type() == torch::kHalf)` at
  `custom-esimd-kernels-vllm/csrc/eagle/eagle.sycl:444` — bf16 target-only
  (no drafter) serving crashes on the first decode step.
- fp16 query in eager mode (`VLLM_XPU_ENABLE_XPU_GRAPH` unset/0): the
  kernel returns garbage on the llm-scaler build (qwen3.8-27b eager fp16
  decode produces incoherent text while graph-replayed steps with identical
  weights/config are correct).
- Speculative decoding hides both: the verify pass runs with
  `max_query_len = k+1 > 1`, which bypasses this code path entirely.

Fixed in `vllm/patches/vllm_for_multi_arc.patch` (`PAGED_ATTN_ESIMD_INSERTED_v1`
gate): the insert now additionally requires `query.dtype == torch.float16`
and `VLLM_XPU_ENABLE_XPU_GRAPH` enabled; everything else routes to
`flash_attn_varlen_func`. `DISABLE_ESIMD_PAGE_ATTN=1` remains a manual
opt-out.

## Quality

Greedy spec decode is lossless by construction. Verified earlier: 4/5
byte-identical outputs vs target-only baseline; the one divergence is a
dtype tie-break (fp16 vs bf16), not a drafter effect. Cross-checked
against an independent bf16 reference endpoint: drafter adds zero
divergence.

## Requirements

- 2× Intel Arc Pro B70 (Battlemage G31, 8086:e223) or equivalent XPU
- intel/level-zero driver stack matching the image
- Qwen/Qwen3.8-27B-FP8 checkpoint (block FP8 e4m3, [128,128])
- `/models/drafter-fp8-v5` (1.36B DSpark drafter, HF format)

## Known limitations

- On builds without the ESIMD gate fix (this repo prior to the fix commit,
  and stock dspark images): bf16 target-only serving crashes at the ESIMD
  kernel assert; eager fp16 decode returns garbage on the llm-scaler
  lineage. Use the fixed image, or spec decoding, or
  `DISABLE_ESIMD_PAGE_ATTN=1`.
- Adaptive block truncation (DSPARK_ADAPTIVE_BLOCK=1) runs but is slower
  than fixed k=4 on single-request workloads.
