#!/bin/bash
# repro_bootQ7.sh — Boot Q7: IPC-handle cache discriminator =
# CCL_ZE_CACHE_OPEN_IPC_HANDLES=0 under pidfd. EXACT Boot Q lane
# (e5m2@262144 + mtp4, pidfd, v1.2.10, oneCCL reduce path,
# kernel mode ON) + patch_f15b v3 (transition watchdog, ring 4096).
# NO arfunnel, NO arstage — single variable vs the convicted lane:
# IPC handles reopened per-op instead of cached open across ops.
# Q5 falsified cross-stream ordering (funnel, verified identical
# host AR order, no protection). Q6 falsified CCL_ENABLE_SYCL_KERNELS=0
# as a path (numerics-corrupt, DOA). J/J2 tested drmfd exchange only —
# pidfd + CACHE_OPEN_IPC_HANDLES=0 is the untested cell.
# Graduation: >= 61 min clean under the full armature (= 2x boot Q TTF)
# -> IPC-handle cache convicted + fix candidate (numerics battery +
# perf parity gates apply); wedge -> devcoredump salvage poller (in the
# armature) captures the batch-level GPU state for vendor escalation.
set -eu
MODE="${1:?usage: repro_bootQ7.sh Q7}"
docker rm -f lsv-test >/dev/null 2>&1 || true

# verbose GuC capture on any engine reset
for c in /sys/kernel/debug/dri/0/guc_log_level /sys/kernel/debug/dri/1/guc_log_level; do
  echo 5 > "$c" 2>/dev/null || true
done

docker run -td --privileged --name lsv-test \
    --device /dev/dri -v /dev/dri:/dev/dri --network host --ipc host \
    -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e CCL_TOPO_P2P_ACCESS=1 \
    -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
    -e CCL_ENABLE_SYCL_KERNELS=1 \
    -e CCL_ZE_IPC_EXCHANGE=pidfd \
    -e CCL_ZE_CACHE_OPEN_IPC_HANDLES=0 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
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
    -e VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1 \
    -e VLLM_V55_EVENT_TIMEOUT_S=600 \
    -e VLLM_F15B=1 -e VLLM_F15B_STALL_S=45 \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.10

echo "BOOT_$MODE container launched $(date +%H:%M:%S)"

docker cp /root/build/patch_f15b.py lsv-test:/root/patch_f15b.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_f15b.py | tee /root/build/lce1/f15b_apply_bootQ7.log
docker exec lsv-test grep -c "llm-scaler f15b" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker exec lsv-test /opt/venv/bin/pip install -q py-spy 2>/dev/null || true
docker cp /root/build/serve_user.sh lsv-test:/root/serve_user.sh
docker exec lsv-test sh -c 'chmod +x /root/serve_user.sh && : > /root/serve_full.log'
docker exec -d lsv-test bash /root/serve_user.sh
echo "BOOT_$MODE serve started $(date +%H:%M:%S)"

for i in $(seq 1 60); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo "BOOT_$MODE HEALTH_OK after ~$((i*10))s"; exit 0; }
  docker ps --format '{{.Names}}' | grep -q '^lsv-test$' || { echo "BOOT_$MODE CONTAINER DIED"; exit 7; }
done
echo "BOOT_$MODE HEALTH TIMEOUT"; exit 8
