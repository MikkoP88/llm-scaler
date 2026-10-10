#!/bin/bash
# P84 leg boot (2026-10-06): bring up lsv-test from $1 (image tag) with the
# EXACT certified bench posture (bench_leg_host.sh env block, template from
# v127_stage, identical bench_boot.sh serve line) plus MODE-selected extra
# env. The image + extra env are the only variables between legs.
#   MODE clean       = certified storm posture (no instruments)
#   MODE inst        = + VLLM_SPEC_TIMING=1 + VLLM_V52K_TIMER=1 (decode legs)
#   MODE async       = + VLLM_XPU_ALLOW_ASYNC=1 (FIX-2 async legs)
#   MODE async_inst  = async + instruments (async decode leg)
#   MODE asyncs      = async + VLLM_V60_ZOMBIE_STRIKES=3 (AS2 probe: the v60c
#                      poisoned-row guard strikes on transient async-overlap
#                      empty rows at the final-prefill-chunk -> decode
#                      transition; strikes are consecutive so 3 tolerates
#                      1-2 step lag while keeping NaN-zombie defense)
# Usage: boot_p84.sh <image-tag> <leg-name> <mode> [max-model-len]
#   Optional 4th arg overrides --max-model-len in the serve line (R1-lineage
#   images need <=163840: sharded-view drafter head consumes ~2.7 GiB of the
#   KV pool, 262144 no longer fits — lsv-test v78 R1b bring-up law).
set -u
IMAGE="${1:?image tag}"
LEG="${2:?leg name}"
MODE="${3:?mode}"
MAXLEN="${4:-262144}"
STAGE=/root/build/p84_stage
mkdir -p "$STAGE"

case "$MODE" in
  clean)      EXTRA="" ;;
  inst)       EXTRA="-e VLLM_SPEC_TIMING=1 -e VLLM_V52K_TIMER=1" ;;
  async)      EXTRA="-e VLLM_XPU_ALLOW_ASYNC=1" ;;
  async_inst) EXTRA="-e VLLM_XPU_ALLOW_ASYNC=1 -e VLLM_SPEC_TIMING=1 -e VLLM_V52K_TIMER=1" ;;
  asyncs)     EXTRA="-e VLLM_XPU_ALLOW_ASYNC=1 -e VLLM_V60_ZOMBIE_STRIKES=3" ;;
  *) echo "BOOT_P84 BAD_MODE=$MODE"; exit 1 ;;
esac

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
    $EXTRA \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    "$IMAGE" || { echo "BOOT_P84_$LEG RUN_FAIL image=$IMAGE mode=$MODE"; exit 1; }

docker cp /root/build/v127_stage/chat_template_qwen38_high.jinja lsv-test:/root/chat_template_qwen38_high.jinja
docker cp /root/build/p83_bench/bench_boot.sh lsv-test:/root/bench_boot.sh
if [ "$MAXLEN" != "262144" ]; then
  docker exec lsv-test sed -i "s/--max-model-len 262144/--max-model-len $MAXLEN/" /root/bench_boot.sh
fi
docker exec -d lsv-test bash /root/bench_boot.sh

i=0
while [ "$i" -lt 120 ]; do
  i=$((i + 1))
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)
  if [ "$code" = "200" ]; then
    echo "BOOT_P84_$LEG READY after $((i * 10))s image=$IMAGE mode=$MODE"
    exit 0
  fi
  if ! docker inspect -f '{{.State.Running}}' lsv-test 2>/dev/null | grep -q true; then
    echo "BOOT_P84_$LEG CONTAINER_DIED after $((i * 10))s image=$IMAGE mode=$MODE"
    exit 2
  fi
done
echo "BOOT_P84_$LEG HEALTH_TIMEOUT image=$IMAGE mode=$MODE"
exit 3
