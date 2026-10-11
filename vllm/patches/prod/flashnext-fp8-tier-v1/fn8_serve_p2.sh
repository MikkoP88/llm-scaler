#!/bin/bash
# fn8 P2 TIERED serve — TP1 smoke shape (correctness + single-GPU perf).
# P0 reference shape (fn8_serve_g1.sh) + the P2 knobs (PHASES Round 9):
#   --enable-expert-parallel    ESIMD monolithic self-disable (wedge-law
#                               compliant kill of every weights-direct
#                               fast path; apply() becomes the sole
#                               weights consumer) — REQUIRED for the store
#   FN8_TIERED=1                patch_02a store master gate (+ patch_02b
#                               uva no-wrap + expert-only offload)
#   FN8_TIER_SLOTS=40           TP1/32GB VRAM math: dense-resident 17.45
#                               GiB + pool 40*3.93MB*48 = 7.4 GiB -> 24.9
#                               GiB; gmu 0.9 (28.8) leaves ~3.9 GiB KV
#   FN8_TIER_SEED               routing_hist.json from the P1 corpus
#                               (store reads counts[], filters per-layer
#                               hot lists by rank ownership automatically)
#   V1X_TIER_DUMP               store telemetry (patch_01 atomic law)
# No routing tap this window (no --enable-return-routed-experts, no
# V1X_ROUTING_DUMP): the store's own counters are the measurement.
# Patches BEFORE serve: 00a/00b/00c/00d + patch_02a + patch_02b
# (patch_01 NOT needed).  Sync scheduling (single-variable law — async
# + store is the P2b step).  Smoke S=40 is memory-driven, not coverage-
# driven; the real shape is fn8_serve_p2t2.sh (TP2+EP2, S=72).
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export FN8_TIERED=1
export FN8_TIER_SLOTS=40
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export V1X_TIER_DUMP=/stage/bank/tier_tp1.json
export V1X_TIER_FLUSH=200

mkdir -p /stage/bank
exec vllm serve --model /models/flash-next-fp8 \
  --served-model-name flash-next-fp8 \
  --enable-expert-parallel \
  --no-async-scheduling \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
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
  --port 8321 > /stage/serve_p2.log 2>&1
