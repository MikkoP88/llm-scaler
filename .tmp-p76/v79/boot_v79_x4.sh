#!/bin/bash
# v79 width-2 emission posture = boot_v78_x4.sh + DFLASH2_EMIT_K=1.
#
# llm-scaler v79 (2026-10-04): depth-hazard lever. The fork's v42-era
# width machinery (async_scheduler.py v42c placeholder cap, compilation.py
# capture-lattice multiple-of-udql, gpu_model_runner.py v42d
# uniform_decode_query_len) is env-armed end-to-end on this stack; the
# Qwen3_5MTP drafter (base-class loop) still drafts all 4 positions and
# the scheduler trims each row to 1 draft — the daily near-finish-trim
# contract. Effect: verify steps run uniform width 2 (graphed at udql=2),
# acceptance ~1.7 tok/step -> 1.18 verify rows per emitted token at depth
# vs 2.35 for full k=4 (and 5.0 for the stock zeros-gate path, which is
# why that gate was rejected for this lane). Depth exposure drops to
# ~nospec grade; speculative-config stays k=4 MTP x4 (spec supported).
export DFLASH2_EMIT_K=1
export VLLM_SPEC_TIMING=1
export VLLM_V52K_TIMER=1
export VLLM_XPU_MTP_TP1=engine
# invalidate stale bytecode for the bind-mounted modules (belt and braces; mtime check should already recompile)
rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/__pycache__/gpu_model_runner.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/__pycache__/llm_base_proposer.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/platforms/__pycache__/interface.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/__pycache__/logits_processor.*.pyc \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/__pycache__/mamba_utils.*.pyc
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
