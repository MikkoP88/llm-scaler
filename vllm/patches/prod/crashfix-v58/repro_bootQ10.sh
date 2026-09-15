#!/bin/bash
# repro_bootQ10.sh — Boot Q10: eager-AR VOLUME discriminator =
# VLLM_XPU_ENABLE_XPU_GRAPH=0 (single env delta vs the convicted
# lane; util back at 0.9). Convicted lane otherwise (e5m2@262144 +
# mtp4, pidfd, v1.2.10, oneCCL kernel mode ON, expandable_segments
# ON) + f15b v4 + arstage v2 census.
# Q8/Q9 convicted UNIDIRECTIONAL eager-AR loss (rank1 DONE / rank0
# PENDING same collective; 100% staged; host order identical x4;
# Q9: flat 267/675 MiB free — no squeeze event, margin falsified).
# Surviving models: (a) per-eager-AR rare race ~1e-6/AR (~350-500k
# eager ARs to wedge; nospec immune = 0 eager ARs/step — Boot O), or
# (b) wall-time periodic hazard (~25 min). Graphs-off exposes target
# decode ARs too: ~148/step = 10.6x volume.
# Predictions: (a) wedge ~2-3 min post-health → fix = eliminate
# eager ARs (graph the drafter propose + deferred postprocess AR);
# (b) unchanged 23-33 min → in-repo elimination can't fully fix →
# vendor pack is the root-cause path.
# Armature = Q9's: AUTO-ARM at health, salvage via arm script, mem
# logger from boot. JIT-pause f15b dumps expected (benign class).
set -eu
MODE="${1:?usage: repro_bootQ10.sh Q10}"
docker rm -f lsv-test >/dev/null 2>&1 || true

docker run -td --privileged --name lsv-test \
    --device /dev/dri -v /dev/dri:/dev/dri --network host --ipc host \
    -e VLLM_XPU_ENABLE_XPU_GRAPH=0 \
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
    -e VLLM_F15B=1 -e VLLM_F15B_STALL_S=45 \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.10

echo "BOOT_$MODE container launched $(date +%H:%M:%S)"

nohup bash /root/build/mem_growth_logger.sh /root/build/lce1/bootQ10_memgrowth.log > /dev/null 2>&1 < /dev/null &

docker cp /root/build/patch_f15b.py lsv-test:/root/patch_f15b.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_f15b.py | tee /root/build/lce1/f15b_apply_bootQ10.log
docker cp /root/build/patch_arstage.py lsv-test:/root/patch_arstage.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_arstage.py | tee /root/build/lce1/arstage_apply_bootQ10.log
docker exec lsv-test grep -c "llm-scaler f15b" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker exec lsv-test grep -c "llm-scaler arstage" /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py
docker exec lsv-test /opt/venv/bin/pip install -q py-spy 2>/dev/null || true
docker cp /root/build/serve_user.sh lsv-test:/root/serve_user.sh
docker exec lsv-test sh -c 'chmod +x /root/serve_user.sh && : > /root/serve_full.log'
docker exec -d lsv-test bash /root/serve_user.sh
echo "BOOT_$MODE serve started (graphs OFF, util 0.9) $(date +%H:%M:%S)"

for i in $(seq 1 60); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "BOOT_$MODE HEALTH_OK after ~$((i*10))s"
    R=$(dmesg | grep -ac 'Engine reset')
    sed -i "s/^BASE=.*/BASE=$R/" /root/build/poll_bootQ10.sh
    echo "live_resets=$R"
    nohup bash /root/build/arm_bootQ10.sh > /root/build/lce1/bootQ10_arm.out 2>&1 &
    sleep 3
    tail -3 /root/build/lce1/bootQ10_arm.out
    nohup bash /root/build/poll_bootQ10.sh > /root/build/lce1/bootQ10_poll.out 2>&1 < /dev/null &
    echo "ARMED_AND_POLLED $(date -u +%H:%M:%S)"
    exit 0
  fi
  docker ps --format '{{.Names}}' | grep -q '^lsv-test$' || { echo "BOOT_$MODE CONTAINER DIED"; exit 7; }
done
echo "BOOT_$MODE HEALTH TIMEOUT"; exit 8
