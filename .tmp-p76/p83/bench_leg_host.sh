#!/bin/bash
# llm-scaler BENCH leg driver (P83 close-out, 2026-10-05): bring up lsv-test
# from $1 (image tag) with the exact chain docker-run posture (v1227 ckpt,
# V63/V66 inert), cp the chat template + identical bench serve script,
# launch serve detached, wait for health 200. NO patchers applied -- the
# image runs exactly as baked; the image is the only variable between legs.
# Usage: bench_leg_host.sh <image-tag> <leg-name>
set -u
IMAGE="${1:?image tag}"
LEG="${2:?leg name}"
STAGE=/root/build/p83_bench
mkdir -p "$STAGE"

docker rm -f lsv-test >/dev/null 2>&1

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
    -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
    -e VLLM_GUIDED_DECODING_BACKEND=xgrammar \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    "$IMAGE" || { echo "LEG_$LEG RUN_FAIL image=$IMAGE"; exit 1; }

docker cp /root/build/v127_stage/chat_template_qwen38_high.jinja lsv-test:/root/chat_template_qwen38_high.jinja
docker cp "$STAGE/bench_boot.sh" lsv-test:/root/bench_boot.sh
docker exec -d lsv-test bash /root/bench_boot.sh

# health wait: up to 20 min (weight load + compile + cudagraph capture)
i=0
while [ "$i" -lt 120 ]; do
  i=$((i + 1))
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)
  if [ "$code" = "200" ]; then
    echo "LEG_$LEG READY after $((i * 10))s image=$IMAGE"
    exit 0
  fi
  if ! docker inspect -f '{{.State.Running}}' lsv-test 2>/dev/null | grep -q true; then
    echo "LEG_$LEG CONTAINER_DIED after $((i * 10))s image=$IMAGE"
    exit 2
  fi
done
echo "LEG_$LEG HEALTH_TIMEOUT image=$IMAGE"
exit 3
