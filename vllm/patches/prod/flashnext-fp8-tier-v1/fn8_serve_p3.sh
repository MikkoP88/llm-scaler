#!/bin/bash
# fn8 P3 PREFETCH serve — P2-TP2 shape + V1X_TIER_PREFETCH=1 (the ONLY delta).
# Identical pool/slots/offload/TP config to fn8_serve_p2t2.sh so the P3 gate
# (prompt tok/s >=600 @4K, no TTFT regression vs prefill_p2ref.json) reads as
# a pure prefetch effect.  patch_03 must be applied before this serve.
# Boot-log asserts for this window:
#   - EP line: expert parallel size 2, local experts 256 (whole-expert law)
#   - "FN8_02A store layer N: .../72 slots pre-warmed from seed" x48
#   - NO "FN8_03 prefetch DISABLED" lines (any = loud per-store disable —
#     triage before trusting perf legs)
#   - tier_p3.json pf_fills/pf_promotes/pf_waits > 0 (machinery live)
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export FN8_TIERED=1
export FN8_TIER_SLOTS=72
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export V1X_TIER_DUMP=/stage/bank/tier_p3.json
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
  --port 8321 > /stage/serve_p3.log 2>&1
