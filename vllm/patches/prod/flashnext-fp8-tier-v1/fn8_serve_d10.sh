#!/bin/bash
# fn8 DEC40 D10 = D9 (certified 55.578 tok/s code median) + the
# high-reasoning chat template for LIVE testing.  Only deltas vs d9:
#   + --chat-template /root/chat_template_qwen38_high.jinja
#     (template docker-cp'd into the container from
#      /root/build/chat_template_qwen38_high.jinja md5 eb69f1c4 — same
#      pattern as the lane chain's per-boot docker cp; a -v mount would
#      require re-creating fn8-d1 and destroying the live 06l-06o patch
#      layer)
#   + log -> /stage/serve_d10.log
# Everything else byte-identical to d9 (TP2+EP, fp16 compute, kv fp8_e4m3,
# mamba fp16, MTPx4 own drafter, FULL_DECODE_ONLY graphs, FN8 tiered,
# PLE prestage, async, port 8321).
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
  --chat-template /root/chat_template_qwen38_high.jinja \
  --trust-remote-code \
  --port 8321 > /stage/serve_d10.log 2>&1
