#!/bin/bash
# fn8 P2 TIERED serve — TP2+EP2 TARGET shape (the whole-expert law).
# Whole experts per rank (never N-sliced intermediate: block-128 law),
# each rank owns 256 globals via the fork's contiguous determine_expert_map;
# the store seeds rank-locally from the SAME routing_hist.json (owned
# filter).  VRAM/rank (32 GB): dense ~8.7 GiB + pool 72*3.93MB*48 =
# 13.3 GiB -> 22 GiB; gmu 0.85 (27.2) leaves ~5 GiB KV.  Host mirror
# 48.3 GiB/rank -> --cpu-offload-gb 56 (was 104 at TP1).
# Boot-log asserts for this window (PHASES watch items):
#   - EP line: expert parallel size 2, local experts 256 (whole experts,
#     intermediate NOT /2 vs TP — the load-time shape assert)
#   - "FN8_02A store layer N: .../72 slots pre-warmed from seed" x48
#   - "Total CPU offloaded parameters" ~48.3 GiB (whitelist only)
# Rest of the law: same as fn8_serve_p2.sh (EP flag, tiered envs, sync
# scheduling, no tap).  Patches BEFORE serve: 00a/00b/00c/00d + 02a + 02b.
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export FN8_TIERED=1
export FN8_TIER_SLOTS=72
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export V1X_TIER_DUMP=/stage/bank/tier_tp2.json
export V1X_TIER_FLUSH=200

mkdir -p /stage/bank
exec vllm serve --model /models/flash-next-fp8 \
  --served-model-name flash-next-fp8 \
  --enable-expert-parallel \
  --no-async-scheduling \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --cpu-offload-gb 56 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 4 \
  --block-size 64 \
  --dtype float16 \
  --enforce-eager \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --trust-remote-code \
  --port 8321 > /stage/serve_p2t2.log 2>&1
