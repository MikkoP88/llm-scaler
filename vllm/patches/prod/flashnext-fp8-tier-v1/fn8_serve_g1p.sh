#!/bin/bash
# fn8 P1 PROFILING serve — routing-capture window on fn8-g1.
# Same P0 reference shape as fn8_serve_g1.sh (pageable offload, TP1, gmu 0.8)
# plus the three P1 knobs (PHASES Round 6):
#   --enable-return-routed-experts  initializes the capturer + binds the
#                                   per-layer buffers (EngineArgs field,
#                                   no patch needed)
#   DISABLE_ESIMD_MOE=1             ESIMD fused MoE does topk in-kernel and
#                                   bypasses moe_runner -> capture buffer
#                                   never written on the decode fast path;
#                                   this env restores the upstream modular
#                                   path 1:1 (esimd_utils toggle law)
#   V1X_ROUTING_DUMP                patch_01 histogram tap output
# patch_01_routing_dump.py must be applied BEFORE this serve (XPU sync-D2H
# fallback + the tap).  Requires: patch_00a/00b/00c/00d also applied.
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export DISABLE_ESIMD_MOE=1
export V1X_ROUTING_DUMP=/stage/bank/routing_hist.json
export V1X_ROUTING_FLUSH=25   # small corpus at P0 speed: flush often

mkdir -p /stage/bank
exec vllm serve --model /models/flash-next-fp8 \
  --served-model-name flash-next-fp8 \
  --enable-return-routed-experts \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --cpu-offload-gb 104 \
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
  --port 8319 > /stage/serve_g1p.log 2>&1
