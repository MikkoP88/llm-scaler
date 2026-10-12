#!/bin/bash
# fn8 DEC40 D6 — D5 shape MINUS --enforce-eager PLUS FULL_DECODE_ONLY
# graphs (the ONLY delta vs D5; single-delta law).
#   + VLLM_XPU_MTP_EAGER_HEAD=1 (fork recipe: MTP/drafter head stays eager
#     under FULL_DECODE_ONLY — qwen3_5_mtp ESIMD norm + spec head don't
#     need to be capture-safe)
# Capture-risk register for this model (vs D3's tier crash — tier GONE here):
#   * no PLE (qwen3_5 has none) -> no 04-series .item() hot path
#   * gdn_attn.py builder HAS use_full_cuda_graph branches by design
#   * unknowns: mamba align-mode state copies, esimd_rms_norm under capture
# D5 eager floor: prose 24.056 / code 31.704 tok/s, step_ms p50 111,
# tok/step 2.67/3.51.  40 tok/s needs step <= 87 ms at 3.5 tok/step.
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export VLLM_XPU_MTP_EAGER_HEAD=1

mkdir -p /stage/bank
exec vllm serve --model /models/qwen3.8-27b-fp8 \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --block-size 64 \
  --dtype float16 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  --port 8321 > /stage/serve_d6.log 2>&1
