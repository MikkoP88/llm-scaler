#!/bin/bash
# fn8 P4 MTP serve — P3-winning shape + speculative-config (the ONLY delta).
# P3 stack stays byte-identical: pinned offload (no DISABLE_PIN_MEMORY),
# UVA off, patch_03 v4 (_load_many + MIN_ROWS decode gate), prefetch on.
# Spec shape copied from the lsv lane V1227 certified serve (A5000 law:
# method mtp, num_speculative_tokens 4).  Acceptance is measured client-side
# as decode tok/s ratio vs the P3 nospec legs (4.975/5.243 tok/s, banks
# p3_run12.json) — the same A/B law as the 2026-10-07 MTP-vs-nospec verdict.
# Boot-log asserts for this window:
#   - spec init lines (mtp, num_speculative_tokens=4) near boot
#   - EP line: expert parallel size 2, local experts 256 (whole-expert law)
#   - "FN8_02A store layer N: .../72 slots pre-warmed from seed" x48
#   - NO "FN8_03 prefetch DISABLED" lines
#   - tier_p4.json pf_fills/pf_promotes > 0 (machinery live)
export PYTHONHASHSEED=0
# P3-winning memory law: PIN the 56GB offload region (v4: TTFT == transfer;
# pinned lifts H2D BW 3-5x).  DISABLE_UVA stays 1 (explicit copy_ path).
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export FN8_TIERED=1
export FN8_TIER_SLOTS=72
export FN8_TIER_SEED=/stage/bank/routing_hist.json
export V1X_TIER_DUMP=/stage/bank/tier_p4.json
export V1X_TIER_FLUSH=200
# -- P3 knobs (patch_03) ----------------------------------------------------
export V1X_TIER_PREFETCH=1
export V1X_TIER_PF_LAYERS=4
export V1X_TIER_PF_BUDGET=8
export V1X_TIER_PF_STREAM=1

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
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --trust-remote-code \
  --port 8321 > /stage/serve_p4.log 2>&1
