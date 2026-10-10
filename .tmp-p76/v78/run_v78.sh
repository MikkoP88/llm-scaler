#!/bin/bash
# v78 engine-variant container deploy — recreates lsv-test on llm-scaler-exp:v1.2.29
# with the 4 v78 code binds + v52k mamba instrument bind, then launches the serve.
# Rollback = same script minus the $CODE binds and the VLLM_XPU_MTP_TP1 env (R1b-fallback posture).
set -e
IMG=llm-scaler-exp:v1.2.29
SP=/opt/venv/lib/python3.12/site-packages/vllm
CODE=/root/build/p77/code

docker rm -f lsv-test 2>/dev/null || true

docker run -d --name lsv-test \
  --privileged --network host --ipc host --shm-size 64m \
  --entrypoint /bin/bash \
  -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
  -e CCL_SYSCALL_ALLREDUCE_TMP_BUF=0 \
  -e VLLM_ALLOW_LONG_MODEL_LEN=1 \
  -e PYTORCH_XPU_ALLOC_CONF=expandable_segments:True \
  -e VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e CCL_TOPO_P2P_ACCESS=1 \
  -e CCL_SYCL_ALLREDUCE_TMP_BUF=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_AOT_COMPILE=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_QUANTIZE_Q40_LIB=/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so \
  -e VLLM_GUIDED_DECODING_BACKEND=xgrammar \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_V55_DECODE_TIMEOUT_S=30 \
  -e CCL_ENABLE_SYCL_KERNELS=1 \
  -e ZE_AFFINITY_MASK=0,1 \
  -e CCL_ZE_IPC_EXCHANGE=pidfd \
  -e CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 \
  -e VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX=0 \
  -e VLLM_RPC_TIMEOUT=60000 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 \
  -e VLLM_V55_EVENT_TIMEOUT_S=600 \
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
  -e VLLM_XPU_SPEC_DRAFT_BARRIER=0 \
  -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
  -e VLLM_XPU_FP8_MQ=1 \
  -e VLLM_F15B=1 \
  -e VLLM_F15B_STALL_S=45 \
  -e VLLM_SPEC_TIMING=1 \
  -e VLLM_V52K_TIMER=1 \
  -e VLLM_XPU_MTP_TP1=engine \
  -v /dev/dri:/dev/dri \
  -v /root/build/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro \
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -v /models/drafter-fp8-v5:/models/drafter:ro \
  -v /models/dflash2:/models/dflash2:ro \
  -v $CODE/gpu_model_runner.py:$SP/v1/worker/gpu_model_runner.py:ro \
  -v $CODE/llm_base_proposer.py:$SP/v1/spec_decode/llm_base_proposer.py:ro \
  -v $CODE/interface.py:$SP/platforms/interface.py:ro \
  -v $CODE/logits_processor.py:$SP/model_executor/layers/logits_processor.py:ro \
  -v /root/build/p77/mamba_utils_v52k.py:$SP/v1/worker/mamba_utils.py:ro \
  -v /root/build/p77/boot_v78.sh:/root/boot_v78.sh:ro \
  $IMG -c "sleep infinity"

docker exec -d lsv-test bash /root/boot_v78.sh
echo "LAUNCHED $(date '+%F %T')"
