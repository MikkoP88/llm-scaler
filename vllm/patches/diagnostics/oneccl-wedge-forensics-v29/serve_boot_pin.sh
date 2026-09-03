#!/bin/bash
# llm-scaler v29 P7 B3b: serve_boot_var.sh + NUMA pin. Both XPUs report
# numa_node=1 (node1 = odd CPUs on this host), and the P4 DMA suite proved the
# GPU<->GPU fabric is HOST-MEDIATED (P2P 9.5 GB/s < via-host 12.2 GB/s), so
# pinning serve CPU+memory to node 1 is a plausible comm-latency/jitter lever.
# Same arg layout as serve_boot_var.sh; image default v29.
DEF_SPEC='{"method":"mtp","num_speculative_tokens":4}'
SPECJSON="${1:-$DEF_SPEC}"
EXTRAENV="${2:-}"
LOG="${3:-pinboot}"
EXTRAFLAGS="${4:-}"
BLOCK="${5:-512}"
IMAGE="${6:-llm-scaler-vllm-adv:v29}"
CPUS=$(seq 1 2 63 | paste -sd,)
# NOTE: --cpuset-mems 1 is deliberately OMITTED: oneCCL worker threads fail
# to start under a single-node membind (pthread_create EINVAL, base_thread.cpp,
# "no membind support for NUMA node 1") — B3b crash log v29b3b_crash.log.
docker rm -f lsv-test >/dev/null 2>&1
ENVPART=()
for kv in $EXTRAENV; do ENVPART+=(-e "${kv%%=*}=${kv#*=}"); done
docker run -d --name lsv-test \
  --cpuset-cpus "$CPUS" \
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
  "${ENVPART[@]}" \
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -v /models/drafter-fp8-v5:/models/drafter:ro \
  -w /llm-scaler/vllm \
  --entrypoint /bin/bash \
  "${IMAGE}" -c "exec /opt/venv/bin/python3 /opt/venv/bin/vllm serve \
    --model /models/target \
    --served-model-name qwen3.8-27b-fp8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --block-size ${BLOCK} \
    --dtype float16 \
    --kv-cache-dtype turboquant_4bit_nc \
    --mamba-ssm-cache-dtype float16 \
    ${EXTRAFLAGS} \
    --async-scheduling \
    --enable-prefix-caching \
    --trust-remote-code \
    --reasoning-parser deepseek_r1 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --speculative-config '${SPECJSON}' \
    --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
    --port 8000 > /root/${LOG}.log 2>&1"
echo "booted PIN cpus=$CPUS mems=1 spec='${SPECJSON}' env='${EXTRAENV}' log=/root/${LOG}.log"
