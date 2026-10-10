#!/bin/bash
# serve_lmc_v133.sh — LMCache leg. v132 baseline config + local LMCache
# (CPU tier, no external server) via LMCacheConnectorV1.
cd /llm-scaler/vllm

# --- LMCache local configuration (env API) ---
# CPU tier for both engines (attn fp8 KV + mamba bytes), LRU eviction.
# 2 ranks x 100GiB CPU pool within host 376GiB (sized 2026-10-10: ~76GiB headroom after ~100GiB non-pool; max safe 110).
export LMCACHE_USE_EXPERIMENTAL_API=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=100
export LMCACHE_LOCAL_CPU=True
export LMCACHE_EVICT_POLICY=lr_u
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_SAVE_DECODE_CACHE=True
export LMCACHE_MIN_RETRIEVE_TOKENS=1024
export LMCACHE_LOG_LEVEL=INFO
# SYCL USM pinned host alloc segfaults above ~16-32GiB per process on this
# stack (probed 2026-10-08); run the CPU pools pageable.
export LMCACHE_DISABLE_PINNED_MEMORY=1
# CRITICAL: lmcache chunk keys chain from Python's builtin hash (default
# pre_caching_hash_algorithm). Without a fixed seed, EngineCore (scheduler
# lookups) and the TP workers (stores) derive different hashes/NONE_HASH per
# process -> 100% lookup misses despite clean stores. Seed must be fixed
# before any python starts (inherited by all spawned processes).
export PYTHONHASHSEED=0
# TEMP: byte-level round-trip forensics (per-chunk pool md5 store vs load)
# TEMP: poll stashed chunk rows at every step (when do writes finalize?)
# TEMP: wrap connector to_gpu/from_gpu (content-vs-placement verdict)

exec vllm serve --model /models/target \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --block-size 64 \
  --dtype float16 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,"lmcache.extra_config":{"mamba_pool_ratio":2}},"engine_id":"lsv65"}' \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,30,35,40,45,48,50,55,60,64,65,70,75,80,85,90,95,96,100,105,110,112,115,120,125,128,130,135,140,145,150,155,160,165,170,175,180,185,190,192,195,200,205,210,215,220,224,225,230,235,240,245,250,255,256,260,265,270,275,280,285,290,295,300,305,310,315,320]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --chat-template /root/chat_template_qwen38_high.jinja \
  --async-scheduling \
  --no-disable-hybrid-kv-cache-manager \
  --port 8000 > /root/serve_full.log 2>&1
