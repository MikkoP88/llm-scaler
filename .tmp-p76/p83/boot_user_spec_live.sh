#!/bin/bash
# USER-DIRECTED LIVE SPEC WINDOW (2026-10-06 ~13:57 UTC): v1.2.30 + MTP spec
# x4 + chat_template_qwen38_high mounted READ-ONLY from /root/build per the
# user's exact -v form, served with --chat-template
# /root/chat_template_qwen38_high.jinja. Serve line = the proven bench_boot.sh
# MTP posture (P83/P83b knobs exported, FULL_DECODE_ONLY graphs, port 8000,
# serve log /root/serve_full.log). Watchdog PAUSED by design for this window
# (lane_watchdog.paused armed BEFORE the swap). No reboot: prior lane idle.
set -u
IMAGE=llm-scaler-exp:v1.2.30
STAGE=/root/build/p83_bench

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
    -v /root/build/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    "$IMAGE" || { echo "USER_SPEC_LIVE RUN_FAIL image=$IMAGE"; exit 1; }

# serve script: exact bench MTP serve line (spec x4 + --chat-template +
# P83/P83b knobs); template comes from the user's -v mount above.
docker cp "$STAGE/bench_boot.sh" lsv-test:/root/bench_boot.sh
docker exec -d lsv-test bash /root/bench_boot.sh

# health wait: up to 20 min (weight load + compile + cudagraph capture)
i=0
while [ "$i" -lt 120 ]; do
  i=$((i + 1))
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)
  if [ "$code" = "200" ]; then
    echo "USER_SPEC_LIVE READY after $((i * 10))s image=$IMAGE spec=mtpx4 template=chat_template_qwen38_high"
    exit 0
  fi
  if ! docker inspect -f '{{.State.Running}}' lsv-test 2>/dev/null | grep -q true; then
    echo "USER_SPEC_LIVE CONTAINER_DIED after $((i * 10))s image=$IMAGE"
    exit 2
  fi
done
echo "USER_SPEC_LIVE HEALTH_TIMEOUT image=$IMAGE"
exit 3
