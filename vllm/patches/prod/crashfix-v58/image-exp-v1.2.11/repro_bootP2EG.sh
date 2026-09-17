#!/bin/bash
# repro_bootNV.sh — NEO-version screen boot (crashfix-v58 §24 F bisect).
# Q21b/Q31 lineage EXACTLY (kernel 6.17.0-1010-intel, fw 3.1, stock env,
# f15b+arstage+v58p1 patchers, graphs ON convicted, util 0.9, VIA=1) with
# a PARAMETRIZED NEO deb overlay: installs every *.deb found in $2.
#   usage: repro_bootNV.sh <MODE> <DEBDIR> <FORCEDEPS yes|no>
# S1 = neodl_S1   (NEO 26.18 runtime only, stock IGC 2.32.7, force)
# S2 = neodl_2622 (NEO 26.22.38646.4 + IGC 2.36.3 full, no force)
# S3 = neodl_S3   (IGC 2.34.4 only, stock NEO 26.14, force)
# Purpose: split the Q31 -24% decode cost between NEO runtime (submission)
# and IGC codegen; + screen the only untested release 26.22. Gates follow
# via nv_screen.sh (f8ref MUST-equal + q17 + bench3 + clock mode-guard).
set -eu
MODE="${1:?mode}"
DEBDIR="${2:?debdir}"
FORCE="${3:-no}"
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
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.10

echo "BOOT_$MODE container launched $(date +%H:%M:%S)"

echo "--- NEO OVERLAY from $DEBDIR (before any GPU use), FORCE=$FORCE:"
ls -la "$DEBDIR" | grep -c '\.deb' || true
docker exec lsv-test mkdir -p /root/neodl
for deb in "$DEBDIR"/*.deb; do
  docker cp "$deb" "lsv-test:/root/neodl/$(basename $deb)"
done
if [ "$FORCE" = "yes" ]; then
  docker exec lsv-test sh -c 'dpkg -i --force-depends /root/neodl/*.deb' 2>&1 | tail -8
else
  docker exec lsv-test sh -c 'dpkg -i /root/neodl/*.deb' 2>&1 | tail -8
fi
echo "--- NEO CENSUS (verify the mix actually landed):"
docker exec lsv-test sh -c "dpkg -l | grep -E 'intel-opencl-icd|intel-igc|libigdgmm|libze-intel-gpu1' | awk '{print \$2, \$3}'"

nohup bash /root/build/mem_growth_logger.sh /root/build/lce1/boot${MODE}_memgrowth.log > /dev/null 2>&1 < /dev/null &

docker cp /root/build/patch_f15b.py lsv-test:/root/patch_f15b.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_f15b.py | tee /root/build/lce1/f15b_apply_boot${MODE}.log
docker cp /root/build/patch_arstage.py lsv-test:/root/patch_arstage.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_arstage.py | tee /root/build/lce1/arstage_apply_boot${MODE}.log
docker exec lsv-test grep -c "llm-scaler f15b" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker exec lsv-test grep -c "llm-scaler arstage" /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py
docker cp /root/build/patch_v58_p1.py lsv-test:/root/patch_v58_p1.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v58_p1.py | tee /root/build/lce1/v58p1_apply_boot${MODE}.log
docker exec lsv-test sh -c 'pip install -q py-spy 2>/dev/null' || true
docker exec lsv-test sed -i "s@logger\.debug(@logger.info(@" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py
docker exec lsv-test find /opt/venv/lib/python3.12/site-packages/vllm -name __pycache__ -type d -prune -exec rm -rf {} +
docker cp /root/build/serve_user.sh lsv-test:/root/serve_user.sh
docker exec lsv-test sh -c 'chmod +x /root/serve_user.sh && : > /root/serve_full.log'
docker exec -d lsv-test bash /root/serve_user.sh
echo "BOOT_$MODE serve started (kernel 6.17.0-1010-intel, overlay=$DEBDIR force=$FORCE) $(date +%H:%M:%S)"

for i in $(seq 1 60); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "BOOT_$MODE HEALTH_OK after ~$((i*10))s"
    R=$(dmesg | grep -ac 'Engine reset')
    echo "live_resets=$R"
    sleep 3
    echo "PROBE_BOOT_READY $(date -u +%H:%M:%S)"
    exit 0
  fi
  docker ps --format '{{.Names}}' | grep -q '^lsv-test$' || { echo "BOOT_$MODE CONTAINER DIED"; exit 7; }
done
echo "BOOT_$MODE HEALTH TIMEOUT"; exit 8
