#!/bin/bash
# serve_user.sh — runs INSIDE lsv-test. User's exact serve config,
# stdout/stderr captured to /root/serve_full.log (telemetry lesson from
# crash-4: never let serve output die with an interactive terminal).
cd /llm-scaler/vllm
exec vllm serve \
  --model /models/target \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --block-size 64 \
  --dtype float16 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 8000 \
  >> /root/serve_full.log 2>&1
