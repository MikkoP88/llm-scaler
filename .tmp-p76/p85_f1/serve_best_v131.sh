#!/bin/bash
# serve_best_v131.sh — certified R1 spec×4 posture on llm-scaler-exp:v1.2.31
# (P84-certified: async + ZOMBIE_STRIKES arrive via docker-run env; NO P77/P83
# bench levers — POOL_SHAVE/HEADROOM stay inert by default).
# CLI corrected vs the two failed v1.2.32 attempts:
#   - backslash present after --speculative-config (the killer typo)
#   - --chat-template / --async-scheduling / --port 8000 actually delivered
# Log goes to /root/serve_full.log (truncates the stale Oct-2 image artifact;
# LIVE-WATCH tick greps read this exact path).
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
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,30,35,40,45,48,50,55,60,64,65,70,75,80,85,90,95,96,100,105,110,112,115,120,125,128,130,135,140,145,150,155,160,165,170,175,180,185,190,192,195,200,205,210,215,220,224,225,230,235,240,245,250,255,256,260,265,270,275,280,285,290,295,300,305,310,315,320]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --chat-template /root/chat_template_qwen38_high.jinja \
  --async-scheduling \
  --port 8000 \
  > /root/serve_full.log 2>&1
