#!/bin/bash
# fn8 P1 PROFILING serve — routing-capture window on fn8-g1.
# Same P0 reference shape as fn8_serve_g1.sh (pageable offload, TP1, gmu 0.8)
# plus the P1 knobs (PHASES Rounds 6/8):
#   --enable-return-routed-experts  initializes the capturer + binds the
#                                   per-layer buffers (EngineArgs field,
#                                   no patch needed)
#   V1X_ROUTING_DUMP                patch_01 histogram tap output
# patch_01_routing_dump.py must be applied BEFORE this serve (XPU sync-D2H
# fallback + the tap).  Requires: patch_00a/00b/00c/00d also applied.
# --no-async-scheduling: the fork refuses --enable-return-routed-experts
# with async scheduling (unvalidated combo, pydantic gate at boot).  The
# profiling window measures ROUTING CONTENT, not speed — per-token router
# decisions are batch-composition-independent, so sync scheduling is
# profile-equivalent (13:19 2026-10-11 boot failure → fix).
#
# DISABLE_ESIMD_MOE=1 IS FORBIDDEN HERE — WEDGE LAW (2026-10-11 bisect):
# the env wedges the boot-time _dummy_run (profile_run /
# determine_available_memory): EngineCore spins at 150-200% CPU inside
# at::native::xpu::_copy_xpu (userspace L0 poll, zero syscalls, zero disk
# I/O) forever; P0's identical phase takes 35 s.  Single-variable bisect:
# removing ONLY this env (capturer flag + --no-async-scheduling + patch_01
# all retained) boots in ~5-6 min.  The capturer is NOT implicated —
# init_routed_experts_capturer runs in _initialize_kv_caches AFTER
# determine_available_memory, so it does not even exist during the dummy
# run (no init log lines during the wedge; verified in the 24-min hang).
# Consequence: ESIMD stays enabled -> decode-path routing (<=12-token
# batches, monolithic fast path) is NOT captured; prefill (>12 tok,
# modular path) captures cleanly.  Corpus is prefill-dominant (~6:1) and
# stale decode rows echo prefill ids (distribution-preserving ~25%
# contamination), so the coverage curve stays sound for the P2 seed.
export PYTHONHASHSEED=0
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
export V1X_ROUTING_DUMP=/stage/bank/routing_hist.json
export V1X_ROUTING_FLUSH=25   # small corpus at P0 speed: flush often

mkdir -p /stage/bank
exec vllm serve --model /models/flash-next-fp8 \
  --served-model-name flash-next-fp8 \
  --enable-return-routed-experts \
  --no-async-scheduling \
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
  --port 8319 > /stage/serve_g1p.log 2>&1
