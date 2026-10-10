#!/bin/bash
# llm-scaler P80 boot = boot_v79_x4.sh (v79a posture: width-2 + drafter
# loop-skip via DFLASH2_EMIT_K=1 + host-bind proposer) + P80_TRACE=1 event
# ladder. Wedge repro vehicle: minimal kernel stream (drafter 1 forward),
# zero guc resets at width-2, wedge reproduced at depth band (3428s leg).
# Ladder files: /root/p80_ladder.<pid> per worker (line-buffered, readable
# mid-run). Tracer patch must be applied BEFORE this boot.
export DFLASH2_EMIT_K=1
export VLLM_SPEC_TIMING=1
export VLLM_V52K_TIMER=1
export VLLM_XPU_MTP_TP1=engine
export P80_TRACE=1
# invalidate stale bytecode for the bind-mounted + P80-patched modules
rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/__pycache__/gpu_model_runner.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/__pycache__/llm_base_proposer.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/platforms/__pycache__/interface.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/__pycache__/logits_processor.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/__pycache__/mamba_utils.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/__pycache__/gdn_linear_attn.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/sample/__pycache__/rejection_sampler.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/__pycache__/attention.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/__pycache__/_p80_tracer.*.pyc
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
