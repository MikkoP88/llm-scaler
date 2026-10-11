#!/bin/bash
# fn8 GPU smoke serve — flashnext-fp8-tier-v1 P0 reference shape.
# Throwaway container fn8-g1 (llm-scaler-exp:v1.2.35 + patch_00a/00b applied).
#
# Offload mode (decided by fn8_pin_probe.py, 2026-10-11): torch pin_memory
# SEGFAULTS between 16 and 20 GiB cumulative per process on this stack (same
# band as the LMCache USM pinned law) — the 48 GiB/rank UVA path is dead.
#   FN8_OFFLOAD_MODE=pageable no pin, functional_call copy-per-forward
#                             (correct reference path; PCIe-bound, ~5s/token)
#   FN8_OFFLOAD_MODE=uva      pin + UVA views — DO NOT use above ~14 GiB
# BANKED as the P0 reference for the P2 greedy-parity gate.
export PYTHONHASHSEED=0

MODE="${FN8_OFFLOAD_MODE:-pageable}"
if [ "$MODE" = "pageable" ]; then
  export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
  export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
  echo "FN8_OFFLOAD_MODE=pageable (pin+uva disabled)"
else
  echo "FN8_OFFLOAD_MODE=uva (pin + UVA views)"
fi

# TP1 LAW (2026-10-11 G1 leg-1): block-128 fp8 expert scales cannot be
# tensor-sharded — TP2 splits intermediate 640->320 = 2.5 blocks, loader
# rejects.  Reference run is TP1 (blocks intact).  P2 must use expert
# partitioning (whole experts per rank), never tensor-slicing.
#
# gmu 0.8 (user directive 2026-10-11, post leg-2 wedge + leg-3 OOM crash +
# host reboot): leaves ~6.4 GiB/card outside the engine pool so the offloader
# repack round-trips (patch_00c keeps them per-module) never floor-hit the
# card.  Note: does NOT address the 47.69 GiB PLE-table H2D (single tensor
# can never fit 32 GiB — needs the host-resident load path fix).
exec vllm serve --model /models/flash-next-fp8 \
  --served-model-name flash-next-fp8 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --cpu-offload-gb 104 \
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
  --port 8319 > /stage/serve_g1.log 2>&1
