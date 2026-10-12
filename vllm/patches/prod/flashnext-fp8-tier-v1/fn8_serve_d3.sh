#!/bin/bash
# fn8 DEC40 D3 — GRAPHS leg: D2 shape MINUS --enforce-eager PLUS
# FULL_DECODE_ONLY graphs.  Single-delta law vs D2.
#   --async-scheduling (kept, D2 stack) + VLLM_XPU_ALLOW_ASYNC=1
#   + VLLM_XPU_MTP_EAGER_HEAD=1 (default-on in fork; explicit here for the
#     record: keeps lm_head/drafter head eager under FULL_DECODE_ONLY)
# Known risk (accepted, diagnostic-first): patch_04g _ple_forward_spec has
# .item() CPU syncs — if capture rejects them the boot dies with a clear
# traceback and the path gets a capture-safe rewrite (fixed-shape padded
# variant; max_len == s_slots.shape[1] is already static for uniform
# k+1 spec steps).
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export VLLM_XPU_ALLOW_ASYNC=1
export VLLM_XPU_MTP_EAGER_HEAD=1
export FN8_TIERED=1
export FN8_TIER_SLOTS=72
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export V1X_TIER_DUMP=/stage/bank/tier_d3.json
export V1X_TIER_FLUSH=200
export V1X_TIER_PREFETCH=1
export V1X_TIER_PF_LAYERS=4
export V1X_TIER_PF_BUDGET=8
export V1X_TIER_PF_STREAM=1

mkdir -p /stage/bank
exec vllm serve --model /models/flash-next-fp8 \
  --served-model-name flash-next-fp8 \
  --enable-expert-parallel \
  --async-scheduling \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --cpu-offload-gb 56 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --block-size 64 \
  --dtype float16 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  --port 8321 > /stage/serve_d3.log 2>&1
