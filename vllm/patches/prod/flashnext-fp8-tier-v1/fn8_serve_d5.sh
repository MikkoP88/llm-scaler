#!/bin/bash
# fn8 DEC40 D5 — ALT-MODEL leg (user-authorized: "alternative model path is
# allowed so you may use also Q4 (you can select best one) version next to
# FP8").  Model = /models/qwen3.8-27b-fp8:
#   * qwen3_5 arch, fork-NATIVE (qwen3_5.py + qwen3_5_mtp.py + registry),
#     own MTP head (mtp_num_hidden_layers=1 -> "model has own MTP not need
#     to external drafter")
#   * DENSE 27B hybrid GDN (64 layers, hidden 5120, full_attn every 4th,
#     NO PLE -> the 04-series problem class does not exist here)
#   * 29G fp8 -> ~14.5G/rank under TP2 = FULLY RESIDENT on dual B70:
#     NO --cpu-offload-gb, NO FN8_TIERED tier, NO --enable-expert-parallel
# Why this path: D0-D3 law — flash-next step wall = eager 48-layer MoE
# verify, and the fn8 tier module is capture-hostile (Events + tolist under
# capture) so graphs are blocked there; a resident 27B removes BOTH walls
# and matches the Strata reference class (27B + spec 4 = 40.8 code tok/s).
# Posture: EAGER + MTP x4 (D1-equivalent posture; graphs = D6 single-delta).
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1

mkdir -p /stage/bank
exec vllm serve --model /models/qwen3.8-27b-fp8 \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --block-size 64 \
  --dtype float16 \
  --enforce-eager \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --trust-remote-code \
  --port 8321 > /stage/serve_d5.log 2>&1
