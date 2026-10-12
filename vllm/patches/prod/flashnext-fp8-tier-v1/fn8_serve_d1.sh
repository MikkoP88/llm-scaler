#!/bin/bash
# fn8 DEC40 D1 — MTP-ON leg: P3-certified shape + patch_04b proposer unwrap
# + speculative-config (the ONLY functional delta vs P3; patch_04 draft
# wiring already baked in v1.3.0, dormant until spec is requested).
# Still --no-async-scheduling --enforce-eager (single-delta law; async = D2,
# graphs = D3).
# Boot-log asserts:
#   - spec init lines (mtp, num_speculative_tokens=4) near boot
#   - "Detected MTP model. Sharing target model embedding weights" (04b proof)
#   - draft weight load: 0 "not found in params_dict" for mtp.* stream
#   - EP line: expert parallel size 2, local experts 256 (whole-expert law)
#   - "FN8_02A store layer N: .../72 slots pre-warmed from seed" x48
#   - NO "FN8_03 prefetch DISABLED" lines
export PYTHONHASHSEED=0
# P3-winning memory law: PIN the 56GB offload region (pinned lifts H2D BW
# 3-5x); DISABLE_UVA stays 1 (explicit copy_ path).
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export FN8_TIERED=1
export FN8_TIER_SLOTS=72
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export V1X_TIER_DUMP=/stage/bank/tier_d1.json
export V1X_TIER_FLUSH=200
# -- P3 knobs (patch_03) ----------------------------------------------------
export V1X_TIER_PREFETCH=1
export V1X_TIER_PF_LAYERS=4
export V1X_TIER_PF_BUDGET=8
export V1X_TIER_PF_STREAM=1

mkdir -p /stage/bank
exec vllm serve --model /models/flash-next-fp8 \
  --served-model-name flash-next-fp8 \
  --enable-expert-parallel \
  --no-async-scheduling \
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
  --port 8321 > /stage/serve_d1.log 2>&1
