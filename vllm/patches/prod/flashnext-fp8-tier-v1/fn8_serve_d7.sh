#!/bin/bash
# fn8 DEC40 D7 — PATCHED-STACK + ASYNC leg on Flash-Next 125B (the gate
# model; user correction 2026-10-12: base = Qwen3.8-Flash-Next 125B).
# Posture = D4 EXACTLY (eager + MTP x4 + tier SLOTS=72 seed + PF + offload
# 56GB + util 0.8 + TP2+EP2) with two deltas:
#   1. patch_05  (qwen4_exp.py):  PLE spec group static sync-free rewrite
#      (kills up to 4 .item() syncs x 12 layers x 250 ms step; py-spy D4:
#      PLE 13-14% of wall) + telemetry capture gate.
#      patch_05b (parameter.py):  Parameter __torch_function__ opt-out
#      (py-spy D4: 9.5% of wall).
#   2. --async-scheduling + VLLM_V60_ZOMBIE_STRIKES=3 (user law 2026-10-12:
#      eager ON kills async's benefit -> --enforce-eager REMOVED; verified
#      in fork code: async_scheduling is an independent scheduler switch,
#      TP2+spec forces TORCH_COMPILE_DISABLE + XPU graphs stay env-gated
#      OFF (VLLM_XPU_ENABLE_XPU_GRAPH unset) -> same kernel path/numerics
#      as D4, AsyncScheduler active.  Graphs arrive in D8 with patch_06 —
#      D3's capture crash root-caused: tier tolist()/ensure_resident host
#      code sits inside the captured piece (fused_moe_modular_method.py
#      dispatch()); ESIMD moe_splits protect only the moe_ops call below
#      it, NOT the tier python above it).
# If this lands < 40 tok/s, D7b isolates async-off (attribution).
# WEDGE LAW: host reboot + drop_caches + util 0.8 before EVERY 125B
# offload boot (wedge #6 = skipped drop_caches).
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export VLLM_V60_ZOMBIE_STRIKES=3
export FN8_TIERED=1
export FN8_TIER_SLOTS=72
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export V1X_TIER_DUMP=/stage/bank/tier_d7.json
export V1X_TIER_FLUSH=200
export V1X_TIER_PREFETCH=1
export V1X_TIER_PF_LAYERS=4
export V1X_TIER_PF_BUDGET=8
export V1X_TIER_PF_STREAM=1

mkdir -p /stage/bank
exec vllm serve --model /models/flash-next-fp8 \
  --served-model-name flash-next-fp8 \
  --async-scheduling \
  --enable-expert-parallel \
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
  --trust-remote-code \
  --port 8321 > /stage/serve_d7.log 2>&1
