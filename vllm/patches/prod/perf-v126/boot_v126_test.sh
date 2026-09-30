#!/bin/bash
# boot_v126_test.sh — v126 P28 lane-level gate: boot a TEST lane with the
# dual-bridge fix and an fp8 SSM state pool, for the P23F quality probe.
# Usage: boot_v126_test.sh <fp8_e4m3|fp8_e5m2|float16> [port]
# Base image: llm-scaler-exp:v1.2.25-raw (the v1.2.26 bake base).
# Chain = repro_bootV1225_prod.sh lineage (V1223+ async-REQUIRED posture)
# + patch_v126_dualbridge.py + --mamba-ssm-cache-dtype swap in serve_user.sh.
set -eu
DTYPE="${1:?usage: boot_v126_test.sh <fp8_e4m3|fp8_e5m2|float16> [port]}"
PORT="${2:-8000}"
TAG=$(echo "$DTYPE" | tr '._' '__')
STAMP=$(date +%H%M%S)
LOGD=/root/build/lce1
mkdir -p "$LOGD"

docker rm -f lsv-test >/dev/null 2>&1 || true

docker run -td --privileged --name lsv-test \
    --device /dev/dri -v /dev/dri:/dev/dri --network host --ipc host \
    -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e CCL_TOPO_P2P_ACCESS=1 \
    -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
    -e CCL_ENABLE_SYCL_KERNELS=1 \
    -e CCL_ZE_IPC_EXCHANGE=pidfd \
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
    -e VLLM_V55_DECODE_TIMEOUT_S=30 -e VLLM_XPU_SPEC_DRAFT_BARRIER=0 -e VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX=0 -e VLLM_RPC_TIMEOUT=60000 \
    -e VLLM_F15B=1 -e VLLM_F15B_STALL_S=45 -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
    -e VLLM_GUIDED_DECODING_BACKEND=xgrammar \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.25-raw

echo "BOOT_v126_${TAG} container launched (v1.2.25-raw + dualbridge) $(date +%H:%M:%S)"

# --- standing boot patcher chain (V1223+ lineage) ---------------------------
docker cp /root/build/patch_f15b.py lsv-test:/root/patch_f15b.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_f15b.py | tee "$LOGD/f15b_apply_v126.log"
docker cp /root/build/patch_v55_3.py lsv-test:/root/patch_v55_3.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v55_3.py | tee "$LOGD/v55_3_apply_v126.log"
docker exec lsv-test grep -c "llm-scaler v55.3" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker cp /root/build/patch_arstage.py lsv-test:/root/patch_arstage.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_arstage.py | tee "$LOGD/arstage_apply_v126.log"
docker exec lsv-test grep -c "llm-scaler arstage" /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py
docker cp /root/build/patch_v58_p1.py lsv-test:/root/patch_v58_p1.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v58_p1.py | tee "$LOGD/v58p1_apply_v126.log"
docker cp /root/build/patch_stall_v59.py lsv-test:/root/patch_stall_v59.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_stall_v59.py | tee "$LOGD/stall59_apply_v126.log"
docker cp /root/build/patch_wedge_v60.py lsv-test:/root/patch_wedge_v60.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_wedge_v60.py | tee "$LOGD/v60_apply_v126.log"
docker cp /root/build/patch_wedge_v62b_bake.py lsv-test:/root/patch_wedge_v62b_bake.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_wedge_v62b_bake.py --apply | tee "$LOGD/v62b_apply_v126.log"
docker exec lsv-test grep -c "llm-scaler v62 (WEDGEFIX-G): mode dispatch" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker cp /root/build/patch_sched_v63_bake.py lsv-test:/root/patch_sched_v63_bake.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_sched_v63_bake.py --apply | tee "$LOGD/v63_apply_v126.log"
docker exec lsv-test grep -c "llm-scaler v63 TTFTFIX" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
docker cp /root/build/patch_sched_v64_bake.py lsv-test:/root/patch_sched_v64_bake.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_sched_v64_bake.py --apply | tee "$LOGD/v64_apply_v126.log"
docker exec lsv-test grep -c "llm-scaler v64 TTFTFIX-2" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
docker exec lsv-test pip install -q py-spy 2>/dev/null || true
docker exec lsv-test sed -i "s@logger\.debug(@logger.info(@" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py

# --- v126: dual-bridge root fix ---------------------------------------------
docker cp /root/build/patch_v126_dualbridge.py lsv-test:/root/patch_v126_dualbridge.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v126_dualbridge.py | tee "$LOGD/v126_dualbridge_apply_${TAG}.log"
docker exec lsv-test grep -c "llm-scaler v126 DUAL BRIDGE" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py
docker exec lsv-test grep -c "llm-scaler v126 C7'" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py

# --- serve config: swap the ssm state dtype ---------------------------------
docker exec lsv-test test -f /root/.llm_scaler_exp_v1225_baked || { echo "BOOT_v126 MARKER MISSING (raw image?)"; exit 9; }
if docker exec lsv-test grep -q -- '--gpu-memory-utilization 0.8' /root/serve_user.sh \
   && docker exec lsv-test grep -q -- '--kv-cache-dtype fp8_e4m3' /root/serve_user.sh \
   && docker exec lsv-test grep -q -- '--async-scheduling' /root/serve_user.sh \
   && docker exec lsv-test grep -q -- "'{\"method\":\"mtp\",\"num_speculative_tokens\":4}'" /root/serve_user.sh; then
  echo "BOOT_v126 baked config verified (gmu0.8 bs64 e4m3KV pc-ON async-ON mtpx4)"
else
  echo "BOOT_v126 BAKED_SERVE_CONFIG_WRONG"; exit 10
fi
docker exec lsv-test sed -i "s@--mamba-ssm-cache-dtype float16@--mamba-ssm-cache-dtype $DTYPE@" /root/serve_user.sh
docker exec lsv-test grep -- "--mamba-ssm-cache-dtype" /root/serve_user.sh
if [ "$PORT" != "8000" ]; then
  docker exec lsv-test sed -i "s@--port 8000@--port $PORT@" /root/serve_user.sh
fi
docker exec lsv-test sh -c "chmod +x /root/serve_user.sh && : > /root/serve_full.log"
docker exec -d lsv-test bash /root/serve_user.sh
echo "BOOT_v126_${TAG} serve started (ssm=$DTYPE port=$PORT) $(date +%H:%M:%S)"

for i in $(seq 1 90); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 "http://localhost:$PORT/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "BOOT_v126_${TAG} HEALTH_OK after ~$((i*10))s"
    R=$(dmesg | grep -ac 'Engine reset' || true)
    echo "live_resets=$R"
    sleep 3
    docker exec lsv-test grep -m1 "v126 DUAL_FP8_BRIDGE engaged" /root/serve_full.log || true
    docker exec lsv-test grep -c "JIT compilation during inference" /root/serve_full.log || true
    echo "V126_TEST_LANE_UP_${TAG} $(date -u +%H:%M:%S)"
    exit 0
  fi
  docker ps --format '{{.Names}}' | grep -q '^lsv-test$' || { echo "BOOT_v126 CONTAINER DIED"; docker logs --tail 40 lsv-test 2>&1 | tail -40; exit 7; }
done
echo "BOOT_v126_${TAG} HEALTH TIMEOUT"; docker exec lsv-test tail -60 /root/serve_full.log 2>/dev/null || true; exit 8
