#!/bin/bash
# serve_user_e5m2_262k_alias.sh — runs INSIDE lsv-test. Lane P:
# Boot-O-validated config (262144 + NOSPEC + e5m2 + async — 180 min
# clean soak, BOOTO_NO_FIRE_180MIN) + high->xhigh chat-template alias
# (WEDGE_PLAN §12). Spec stays OFF: 262144+mtp4 convicted (latest live
# crash 2026-09-13 ~11:02 UTC at ~49 min, dual ccs guc 22/32 + bcs
# teardown resets).
cd /llm-scaler/vllm
exec vllm serve \
  --model /models/target \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --async-scheduling --gpu-memory-utilization 0.9 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --block-size 512 \
  --dtype float16 \
  --kv-cache-dtype fp8_e5m2 \
  --mamba-ssm-cache-dtype float16 \
  \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --chat-template /root/chat_template_qwen38_high.jinja \
  --default-chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 8000 \
  >> /root/serve_full.log 2>&1
