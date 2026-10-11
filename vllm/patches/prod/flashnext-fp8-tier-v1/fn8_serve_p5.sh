#!/bin/bash
# fn8 P5 ADAPT+PERSIST serve — P3 shape + V1X_TIER_ADAPT=1 + persist files
# (the ONLY deltas; NO speculative config — P4 verdict, spec stays dormant).
# Identical pool/slots/offload/TP config to fn8_serve_p3.sh so hit-rate
# deltas read as a pure adapt/persist effect.  patch_05 must be applied
# (after 02a/02c/03) before this serve.
# Boot-log asserts for this window:
#   - EP line: expert parallel size 2, local experts 256 (whole-expert law)
#   - "FN8_02A store layer N: .../72 slots pre-warmed from seed" x48
#     (boot-17: pre-warm order reflects the MERGED seed = base + .r files)
#   - NO "FN8_03 prefetch DISABLED" and NO "FN8_05 adapt DISABLED" lines
#   - tier_p5.json adapt_windows/adapt_swaps > 0, persist .r0/.r1 exist
# Control boot (boot-15, seed-only baseline): launch with FN8_P5_CONTROL=1
# in the docker exec env — adapt OFF, no persist files written.
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export FN8_TIERED=1
export FN8_TIER_SLOTS=72
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export V1X_TIER_DUMP=/stage/bank/tier_p5.json
export V1X_TIER_FLUSH=200
# -- P3 knobs (patch_03) ----------------------------------------------------
export V1X_TIER_PREFETCH=1
export V1X_TIER_PF_LAYERS=4
export V1X_TIER_PF_BUDGET=8
export V1X_TIER_PF_STREAM=1
# -- P5 knobs (patch_05) ----------------------------------------------------
if [ "${FN8_P5_CONTROL:-0}" = "1" ]; then
  export V1X_TIER_ADAPT=0
  export V1X_TIER_PERSIST=""
else
  export V1X_TIER_ADAPT=1
  export V1X_TIER_ADAPT_EVERY=200
  export V1X_TIER_ADAPT_SWAPS=4
  export V1X_TIER_ADAPT_DECAY=0.995
  export V1X_TIER_ADAPT_MIN_SCORE=3.0
  export V1X_TIER_PERSIST=/stage/bank/routing_hist_p5.json
  export V1X_TIER_PERSIST_EVERY=2
fi

mkdir -p /stage/bank
exec vllm serve --model /models/flash-next-fp8 \
  --served-model-name flash-next-fp8 \
  --enable-expert-parallel \
  --no-async-scheduling \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --cpu-offload-gb 56 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 4 \
  --block-size 64 \
  --dtype float16 \
  --enforce-eager \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --trust-remote-code \
  --port 8321 > /stage/serve_p5.log 2>&1
