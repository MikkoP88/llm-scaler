#!/bin/bash
# §24 S probe lane: repro_bootBS64_E4M3_G85NS_MB4K + patch_v24pc2 arena COH v2.
# Identical to repro_bootBS64_E4M3_G85NS_MB4K.sh (e4m3 bs64 gmu0.85 mnbt4096
# no-async MTP4 PC-ON) with patch_v24pc2 (arena copy-on-hit) applied after
# patch_v58_p1 and VLLM_COH2=1 in the container env. Patcher failure aborts
# the boot (set -e + grep verifies on all 4 patched files) -> BOOT_FAIL
# visible in the runner.
set -eu
MODE="${1:?usage: repro_bootBS64_E4M3_G85NS_MB4K_COH2.sh COH2}"
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
    -e VLLM_F15B=1 -e VLLM_F15B_STALL_S=45 -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
    -e VLLM_COH2=1 -e VLLM_COH2_ARENA=160 -e VLLM_COH2_MIN_BLOCKS=16 \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.10

echo "BOOT_$MODE container launched $(date +%H:%M:%S)"

nohup bash /root/build/mem_growth_logger.sh /root/build/lce1/bootQ21b_memgrowth.log > /dev/null 2>&1 < /dev/null &

docker cp /root/build/patch_f15b.py lsv-test:/root/patch_f15b.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_f15b.py | tee /root/build/lce1/f15b_apply_bootQ21b.log
docker cp /root/build/patch_arstage.py lsv-test:/root/patch_arstage.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_arstage.py | tee /root/build/lce1/arstage_apply_bootQ21b.log
docker exec lsv-test grep -c "llm-scaler f15b" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker exec lsv-test grep -c "llm-scaler arstage" /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py
docker cp /root/build/patch_v58_p1.py lsv-test:/root/patch_v58_p1.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v58_p1.py | tee /root/build/lce1/v58p1_apply_bootQ21b.log
docker cp /root/build/patch_v24pc2.py lsv-test:/root/patch_v24pc2.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v24pc2.py | tee /root/build/lce1/v24pc2_apply_bootCOH2.log
docker exec lsv-test grep -c "llm-scaler v24pc2" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/single_type_kv_cache_manager.py
docker exec lsv-test grep -c "llm-scaler v24pc2" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker exec lsv-test grep -c "llm-scaler v24pc2" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/output.py
docker exec lsv-test grep -c "llm-scaler v24pc2" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
docker exec lsv-test /opt/venv/bin/pip install -q py-spy 2>/dev/null || true
docker exec lsv-test sed -i "s@logger\.debug(@logger.info(@" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py
docker cp /root/build/serve_bs64_e4m3_g85ns_mb4k.sh lsv-test:/root/serve_user.sh
docker exec lsv-test sh -c 'chmod +x /root/serve_user.sh && : > /root/serve_full.log'
docker exec -d lsv-test bash /root/serve_user.sh
echo "BOOT_$MODE serve started (e4m3 bs64 PC-ON + v24pc2 arena COH v2) $(date +%H:%M:%S)"

for i in $(seq 1 60); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "BOOT_$MODE HEALTH_OK after ~$((i*10))s"
    R=$(dmesg | grep -ac 'Engine reset')
    echo "live_resets=$R"
    sleep 3
    echo "PROBE_BOOT_READY_NO_ARM $(date -u +%H:%M:%S)"
    exit 0
  fi
  docker ps --format '{{.Names}}' | grep -q '^lsv-test$' || { echo "BOOT_$MODE CONTAINER DIED"; exit 7; }
done
echo "BOOT_$MODE HEALTH TIMEOUT"; exit 8
