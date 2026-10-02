#!/bin/bash
# repro_bootAB.sh A|B — Phase D A/B validation boots (e4m3+mtp4 lane).
#   A = control:    unpatched v1.2.9, NO DEBUG logging -> wedge expected
#                   under sustain pressure (trigger re-validation).
#   B = treatment:  v57 mamba buffer-isolation patcher applied to
#                   site-packages BEFORE serve start -> must stay clean.
# Both keep VLLM_V55_EVENT_TIMEOUT_S=60 for fast wedge detection and use
# the user's serve flags (serve_user_e4m3.sh).
set -eu
MODE="${1:?usage: repro_bootAB.sh A|B}"
docker rm -f lsv-test >/dev/null 2>&1 || true

docker run -td --privileged --name lsv-test \
    --device /dev/dri -v /dev/dri:/dev/dri --network host --ipc host \
    -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
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
    -e VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1 \
    -e VLLM_V55_EVENT_TIMEOUT_S=60 \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.9

echo "BOOT_$MODE container launched $(date +%H:%M:%S)"

if [ "$MODE" = "B" ]; then
  docker cp /root/build/patch_v57_mamba_buf_isolation.py lsv-test:/root/patch_v57.py
  docker exec lsv-test /opt/venv/bin/python3 /root/patch_v57.py | tee /root/build/lce1/v57_apply_bootB.log
  docker exec lsv-test grep -c '_get_mamba_copy_bufs_deferred' \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py
fi

docker cp /root/build/serve_user_e4m3.sh lsv-test:/root/serve_user_e4m3.sh
docker exec lsv-test sh -c 'chmod +x /root/serve_user_e4m3.sh && : > /root/serve_full.log'
docker exec -d lsv-test bash /root/serve_user_e4m3.sh
echo "BOOT_$MODE serve started $(date +%H:%M:%S)"

for i in $(seq 1 60); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo "BOOT_$MODE HEALTH_OK after ~$((i*10))s"; exit 0; }
  docker ps --format '{{.Names}}' | grep -q '^lsv-test$' || { echo "BOOT_$MODE CONTAINER DIED"; exit 7; }
done
echo "BOOT_$MODE HEALTH TIMEOUT"; exit 8
