#!/bin/bash
# fn8 DEC40 D4 — MICRO-FORENSICS leg: EXACT D1 posture (eager + MTP x4 +
# tier + patch_04g PLE-spec stack, 04b/04d..04g installed on qwen4_exp.py)
# with fresh log/bank paths.  Purpose: py-spy decomposition of the 250 ms
# decode step (where does wall time go: tier tolist sync / ensure host loop
# / python layer overhead / xpu kernel waits).
# Facts driving D4 (2026-10-12): B70 = 32GB each (64GB total) -> fp8 125B
# (~130GB) and even Q4 (~70GB) can NEVER be fully resident: tier+offload is
# forced; graphs stay blocked by tier capture-hostility; so the only road to
# 40 tok/s on Flash-Next 125B = cut the eager step itself.
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export FN8_TIERED=1
export FN8_TIER_SLOTS=72
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export V1X_TIER_DUMP=/stage/bank/tier_d4.json
export V1X_TIER_FLUSH=200
export V1X_TIER_PREFETCH=1
export V1X_TIER_PF_LAYERS=4
export V1X_TIER_PF_BUDGET=8
export V1X_TIER_PF_STREAM=1

mkdir -p /stage/bank
exec vllm serve --model /models/flash-next-fp8 \
  --served-model-name flash-next-fp8 \
  --enable-expert-parallel \
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
  --port 8321 > /stage/serve_d4.log 2>&1
