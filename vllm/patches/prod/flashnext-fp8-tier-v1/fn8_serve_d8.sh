#!/bin/bash
# fn8 DEC40 D8 — GRAPHS + CAPTURE-SAFE TIER leg on Flash-Next 125B (the
# gate model; user correction 2026-10-12: base = Qwen3.8-Flash-Next 125B).
# D7 verdict: step_ms ~250 invariant under patch_05 (PLE sync-free) +
# patch_05b (Parameter hook) + --async-scheduling; tier_d7.json h2d_ms
# law (drain syncs return instantly = device idle at miss points) +
# py-spy (96.9% inside forward, _ops.py 41% presence) => HOST per-op
# python dispatch is the wall => FULL_DECODE_ONLY graph replay is the
# lever.  D3 proved the graph machinery captures on this stack and died
# exactly in the tier python (tolist/ensure inside the piece) — patch_06
# makes the tier capture-legal:
#   capture branch: async D2H of topk_ids into a pinned int16 ring
#     (graph node; every replay refreshes with CURRENT ids) + the same
#     capture-legal slot_map_dev gather;
#   post-replay hook (both replay sites): ONE xpu sync per STEP, then
#     ensure_resident + _sync_slot_map on the default stream (ordered
#     before next replay).  Residency stale-by-one on newly-routed
#     experts (dummy for one step) — parity suite gates the quality.
# Env = D3's graph block (VLLM_XPU_ALLOW_ASYNC=1 + MTP_EAGER_HEAD=1 +
# cudagraph_mode FULL_DECODE_ONLY — exactly what reached capture in D3)
# + D7's certified stack (no-eager law, strikes, util 0.8, offload 56).
# WEDGE LAW: host reboot + drop_caches + util 0.8 before EVERY 125B
# offload boot (wedge #7: drop_caches alone does NOT protect — only ONE
# clean 125B offload boot per host reboot).
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
# FN8_06c: PLE n-gram gather joins the piecewise split list (eager between
# captured pieces; moe/GDN precedent; "unmatched names are harmless").
export VLLM_XPU_SPEC_EXTRA_SPLITS="vllm::gdn_attention_core,vllm::gdn_attention_core_xpu,vllm::gdn_attention,vllm::ple_ngram_gather"
export V1X_TIER_DUMP=/stage/bank/tier_d8.json
export V1X_TIER_FLUSH=200
export V1X_TIER_PREFETCH=1
export V1X_TIER_PF_LAYERS=4
export V1X_TIER_PF_BUDGET=8
export V1X_TIER_PF_STREAM=1

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
  --trust-remote-code \
  --port 8321 > /stage/serve_d8.log 2>&1
