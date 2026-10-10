#!/bin/bash
# llm-scaler BENCH boot (P83 close-out, 2026-10-05): spec mtp x4, TP2, gmu
# 0.8, FULL_DECODE_ONLY graphs -- IDENTICAL serve line across the v1.2.30 /
# v1.2.29 / v1.2.26 benchmark legs; the IMAGE is the only variable. The
# P83/P83b knobs are exported on every leg: inert on unpatched images (no
# code reads them), live on v1.2.30 (patchers baked + serve_user.sh carries
# the same defaults for chain boots). No P80/P82 instruments on any leg --
# pure image comparison. Certified reference (instrumented, same code
# content as v1.2.30): ok=108/108 x3, walls 1235/1231/1214s.
export DFLASH2_EMIT_K=1
export VLLM_XPU_MTP_TP1=engine
export VLLM_XPU_HEADROOM_MB=1024
export VLLM_XPU_POOL_SHAVE_MB=2048
export VLLM_XPU_VIA_POOL_MAX_ROWS=8192
exec /opt/venv/bin/vllm serve --model /models/target --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.8 --max-model-len 262144 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 64 --dtype float16 \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-cache-dtype float16 --enable-prefix-caching \
  --trust-remote-code --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,30,35,40,45,48,50,55,60,64,65,70,75,80,85,90,95,96,100,105,110,112,115,120,125,128,130,135,140,145,150,155,160,165,170,175,180,185,190,192,195,200,205,210,215,220,224,225,230,235,240,245,250,255,256,260,265,270,275,280,285,290,295,300,305,310,315,320]}' \
  --chat-template /root/chat_template_qwen38_high.jinja \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --async-scheduling --port 8000 > /root/serve_full.log 2>&1
