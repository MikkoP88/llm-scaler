#!/bin/bash
# dt_dflash2_serve.sh — boot lsv-test for the DFlash2 era-3 lane.
# The container idles in a sleep until /root/.dflash2_patched exists
# (docker cp + touch by the boot orchestrator); after docker restart the
# wrapper execs the serve. Same flags as serve_boot_var.sh (block 512,
# turboquant_4bit_nc, prefix caching) so the MTP comparison lane is
# apples-to-apples; only the speculative config differs.
# v42: DFLASH2_EMIT_K passthrough (default 7 = no truncation, byte-
# identical v41 behavior). Prefix the boot script caller:
#   DFLASH2_EMIT_K=4 bash /root/build/dt_dflash2_boot.sh
# v42b: DFLASH2_SYNC=1 forces --no-async-scheduling. REQUIRED for emit_k < 7:
# under async scheduling the engine skips post_step -> update_draft_token_ids,
# so the scheduler never sees the variable-width draft lists and schedules
# config-width spec slots with placeholder zeros (measured: D3 probe showed
# total_num_spec_tokens=7 with k=4; positions 4-6 zero, never accepted).
# Sync scheduling consumes true widths: verify runs k+1 target rows.
# NOTE: async_scheduling is TRI-STATE (bool | None, default None) and the
# engine auto-ENABLES async when None (config/vllm.py auto-decision block),
# so merely omitting --async-scheduling does NOT disable it; the explicit
# --no-async-scheduling negation is the only way to reach the sync path.
set -u
ASYNC_FLAG="--async-scheduling"
if [ "${DFLASH2_SYNC:-0}" = "1" ]; then ASYNC_FLAG="--no-async-scheduling"; fi
# v42f guard: k=4 is CONVICTED corrupt under graphs (NaN hidden states ->
# token-0 flood, 100% garbage acceptance; boundary EXACTLY k=4 — k=3/k=5
# bit-clean, matching the v29c MTP piece-capture k4 conviction). Refuse the
# combination unless graphs are explicitly disabled.
if [ "${DFLASH2_EMIT_K:-7}" = "4" ] && [ "${DFLASH2_CGMODE:-FULL_DECODE_ONLY}" != "NONE" ]; then
  echo "FATAL: DFLASH2_EMIT_K=4 corrupts numerics under graphs (v42 conviction)." \
       "Use DFLASH2_CGMODE=NONE (eager, 2-5x slower) or k=3 / k=5."
  exit 2
fi
docker rm -f lsv-test >/dev/null 2>&1
docker run -d --name lsv-test \
  --device /dev/dri -v /dev/dri:/dev/dri --network host --ipc host \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=${DFLASH2_XPUGRAPH:-1} \
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e CCL_TOPO_P2P_ACCESS=1 \
  -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
  -e CCL_ENABLE_SYCL_KERNELS=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CCL_ZE_IPC_EXCHANGE=drmfd \
  -e CCL_SYCL_ALLREDUCE_TMP_BUF=0 \
  -e ZE_AFFINITY_MASK=0,1 \
  -e VLLM_ALLOW_LONG_MODEL_LEN=1 \
  -e VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
  -e VLLM_USE_AOT_COMPILE=0 \
  -e CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_QUANTIZE_Q40_LIB=/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e PYTORCH_XPU_ALLOC_CONF=expandable_segments:True \
  -e DFLASH_DEBUG=1 \
  -e DFLASH2_EMIT_K=${DFLASH2_EMIT_K:-7} \
  -e DFLASH2_CGMODE=${DFLASH2_CGMODE:-FULL_DECODE_ONLY} \
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -v /models/drafter-fp8-v5:/models/drafter:ro \
  -v /models/qwen3.8-27b-dflash2:/models/dflash2:ro \
  -w /llm-scaler/vllm \
  --entrypoint /bin/bash \
  llm-scaler-prod:v1 -c "if [ -f /root/.dflash2_patched ]; then exec /opt/venv/bin/python3 /opt/venv/bin/vllm serve \
    --model /models/target \
    --served-model-name qwen3.8-27b-fp8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --block-size 512 \
    --dtype float16 \
    --kv-cache-dtype turboquant_4bit_nc \
    --mamba-ssm-cache-dtype float16 \
  $ASYNC_FLAG \
    --enable-prefix-caching \
    --trust-remote-code \
    --reasoning-parser deepseek_r1 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --speculative-config '{\"method\":\"dflash\",\"model\":\"/models/dflash2\",\"num_speculative_tokens\":7}' \
    --compilation-config "{\\\"cudagraph_mode\\\":\\\"\$DFLASH2_CGMODE\\\"}" \
    --port 8000 > /root/dt_dflash2.log 2>&1; else echo AWAIT_DFLASH2_PATCH; sleep 900; fi"
echo "booted dt_dflash2 (await-patch wrapper) log=/root/dt_dflash2.log emit_k=${DFLASH2_EMIT_K:-7}"
