#!/bin/bash
# fn8_serve_d11.sh — d11 = d10 fixes after the 2026-10-12 16:00 live death.
#
# deltas vs d10:
#   - --chat-template now points at /stage/chat_template_flashnext_cc.jinja
#     (md5 510ad8ee): flash-next's OWN native template as base (correct
#     <think> scaffolding — the qwen38_high override never opened a think
#     block, so flash-next leaked reasoning into content = the live
#     "garbage") + CC compat: reasoning_effort accepts high as alias for
#     xhigh (CC's scale), minimal/none -> low, default xhigh;
#     string-arguments fallback; content-part hardening.
#     Render-tested 6/6 in-container (test_render_cc_template.py).
#     The old qwen38 template stays mounted at
#     /root/chat_template_qwen38_high.jinja for A/B (relaunch d10).
#   - --enable-auto-tool-choice --tool-call-parser qwen3_coder: the
#     template's <tool_call><function=...> format is the Qwen3-Coder
#     tool dialect; without the flags any tools= request 400s.
#   - VLLM_V63_CONTENDED_BUDGET=0, VLLM_V64_DECODE_INTERLEAVE=0: the
#     lane interleave budgets created tiny mixed prefill+spec steps
#     (fatal one: T=8 = 5 spec + 3 chunk tail) — uncertified on this
#     model, and 06p now degrades instead of dying anyway.
#   - VLLM_V60_ZOMBIE_STRIKES=0 (was 3): the 27B-lane NaN-zombie guard
#     misfired strike-1 on a real 5904-token chunked-prefill request
#     (POISONED-ROW 15:57:49); no zombie history on this lane.
#   - requires patch_06p (mixed-step fallback) in the container image
#     (llm-scaler-exp:fn8-d11).
# everything else byte-identical to certified d9/d10.

export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1

# lane-residue scheduler machinery: OFF on this model (see header)
export VLLM_V60_ZOMBIE_STRIKES=0
export VLLM_V63_CONTENDED_BUDGET=0
export VLLM_V64_DECODE_INTERLEAVE=0

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

export FN8_PLE_PRESTAGE=1
export FN8_PLE_PRESTAGE_MAX_TOKENS=4096
export FN8_UVA_SAFE_D2H=1
export FN8_UVA_D2H_CHUNK_MB=128
export FN8_UVA_D2H_TIMEOUT_S=60

export VLLM_XPU_SPEC_EXTRA_SPLITS="vllm::gdn_attention_core,vllm::gdn_attention_core_xpu,vllm::gdn_attention,vllm::ple_ngram_gather"

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
  --chat-template /stage/chat_template_flashnext_cc.jinja \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --port 8321 > /stage/serve_d11.log 2>&1
