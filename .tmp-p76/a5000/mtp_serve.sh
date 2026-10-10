#!/bin/sh
# A5000 CUDA reference lane — stock vLLM (cu129-nightly) + qwen3.8-27B int4 GPTQ + MTP spec decode
# Host: 10.100.8.6 (NEVER REBOOT — litellm gateway :4004 lives here). Serve port 8100 (host net).
# Restore original service later: docker start vllm-server (re-downloads Intel/Qwen3.6-27B-int4-AutoRound).
set -x
docker rm -f vllm-mtp 2>/dev/null
docker run -d --name vllm-mtp --restart unless-stopped --network host --gpus all \
  -v /home/root/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm/vllm-openai:cu129-nightly \
  --model SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --served-model-name qwen3.8-27b-int4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 32 \
  --enforce-eager \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser deepseek_r1 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --host 0.0.0.0 --port 8100
