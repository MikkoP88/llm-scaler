#!/bin/bash
# fn8 DEC40 D2 — ASYNC-ON leg: D1 shape (P3 + spec mtp x4 + patch_04b) with
# async scheduling ENABLED (the ONLY delta vs D1).
#   --no-async-scheduling  ->  --async-scheduling
#   + VLLM_XPU_ALLOW_ASYNC=1 (XPU fork async gate)
#   + VLLM_V60_ZOMBIE_STRIKES=3 (P84 wide-wave abort law; strikes couple to
#     async — never ship async without them)
# Still --enforce-eager (graphs = D3).
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export VLLM_XPU_ALLOW_ASYNC=1
export VLLM_V60_ZOMBIE_STRIKES=3
export FN8_TIERED=1
export FN8_TIER_SLOTS=72
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export V1X_TIER_DUMP=/stage/bank/tier_d2.json
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
  --enforce-eager \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --trust-remote-code \
  --port 8321 > /stage/serve_d2.log 2>&1
