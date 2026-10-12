#!/bin/bash
# fn8 DEC40 D9 — 06f PLE prestage + 06g runner hook (REAL rows) on the
# d8s3 posture.  The d8s3 stub gate is DEAD as a measurement: crash-5
# reproduced WITH the 06d2/06d4 real-row-0 constant (SYCL IndexKernel
# OOB assertion, worker death at decode token ~18, xe ccs+bcs reset
# 11:25:51 2026-10-12) — ANY constant PLE embedding diverges the model
# within 3-4 decode steps (garbage-but-finite activations -> gather
# index UB).  So the gate must run with REAL per-step rows = 06f.
#
# Deltas vs d8s3:
#   - FN8_PLE_STUB REMOVED (fast path returns before the stub branch
#     anyway; keep the op path for prefill/group=None)
#   + FN8_PLE_PRESTAGE=1 (06f banks at assemble-tail, pre-profile/
#     capture; runner hook 06g stages real rows EAGER pre-dispatch;
#     captured graphs slice+view the fixed device bank)
# Boot via /stage/fn8_boot_retry.sh (FN8_07 safe-D2H retries).
# Gate: step_ms <= ~80 (>= 3.2 tok/step) -> 40 tok/s in reach;
#       ~250 -> graphs lever dead -> pivot Q4-of-125B.
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
export V1X_TIER_DUMP=/stage/bank/tier_d9.json
export V1X_TIER_FLUSH=200
export V1X_TIER_PREFETCH=1
export V1X_TIER_PF_LAYERS=4
export V1X_TIER_PF_BUDGET=8
export V1X_TIER_PF_STREAM=1
# FN8_06f/06g (PLE prestage):
export FN8_PLE_PRESTAGE=1
export FN8_PLE_PRESTAGE_MAX_TOKENS=4096
# FN8_07 boot-infra (patch_07_offload_safe_d2h.py):
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
  --port 8321 > /stage/serve_d9.log 2>&1
