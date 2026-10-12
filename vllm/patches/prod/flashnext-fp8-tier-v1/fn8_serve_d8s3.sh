#!/bin/bash
# fn8 DEC40 D8s3 — D8s2 gate probe (PLE safe-stub graphs) on FN8_07
# safe-D2H boot infra.  IDENTICAL posture to d8s2 (FN8_PLE_STUB=2,
# graphs FULL_DECODE_ONLY, tier stack, MTPx4) except:
#   + FN8_UVA_SAFE_D2H=1  — polled chunked offloader D2H; a bcs engine
#     reset mid-load becomes detect/retry/fast-fail(87) instead of the
#     infinite wedge that killed d8s/d8s2 attempts 1-3 (2026-10-12).
# Launched via /stage/fn8_boot_retry.sh (bounded boot retries) — do NOT
# exec this directly unless debugging.
# Gate (unchanged from d8s2):
#   graphed step_ms <= ~80 (>= 3.2 tok/step)  -> build 06f prestage
#   graphed step_ms still ~250                -> graphs lever dead, pivot
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export VLLM_V60_ZOMBIE_STRIKES=3
export VLLM_XPU_ALLOW_ASYNC=1
export VLLM_XPU_MTP_EAGER_HEAD=1
export FN8_TIERED=1
export FN8_TIER_SLOTS=72
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export FN8_TIER_GRAPHS=1
export FN8_TIER_NO_SYNC=1
export FN8_PLE_STUB=2
export V1X_TIER_DUMP=/stage/bank/tier_d8s3.json
export V1X_TIER_FLUSH=200
export V1X_TIER_PREFETCH=1
export V1X_TIER_PF_LAYERS=4
export V1X_TIER_PF_BUDGET=8
export V1X_TIER_PF_STREAM=1
# FN8_07 boot-infra (see patch_07_offload_safe_d2h.py):
export FN8_UVA_SAFE_D2H=1
export FN8_UVA_D2H_CHUNK_MB=128
export FN8_UVA_D2H_TIMEOUT_S=60
# inert in this posture (kept for the record; see patch_06d header)
export VLLM_XPU_SPEC_EXTRA_SPLITS="vllm::gdn_attention_core,vllm::gdn_attention_core_xpu,vllm::gdn_attention,vllm::ple_ngram_gather"

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
  --port 8321 > /stage/serve_d8s3.log 2>&1
