#!/bin/bash
# serve_bench.sh — parameterized boot for baseline matrix (2026-09-07).
# Args: 1=kv-dtype 2=specjson|nospec 3=gmu 4=maxlen 5=logname 6=extraenv 7=extraflags 8=image
set -u
KV="${1:?kv dtype}"
SPEC="${2:?specjson|nospec}"
GMU="${3:-0.9}"
MAXLEN="${4:-262144}"
LOG="${5:-benchboot}"
EXTRAENV="${6:-}"
EXTRAFLAGS="${7:-}"
IMG="${8:-llm-scaler-exp:v1.2.5}"
docker rm -f lsv-test >/dev/null 2>&1
ENVPART=()
for kv in $EXTRAENV; do ENVPART+=(-e "${kv%%=*}=${kv#*=}"); done
SPECARG=""
if [ "$SPEC" != "nospec" ]; then SPECARG="--speculative-config '$SPEC'"; fi
docker run -d --name lsv-test \
  --device /dev/dri -v /dev/dri:/dev/dri --network host --ipc host \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e CCL_TOPO_P2P_ACCESS=1 -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
  -e CCL_ENABLE_SYCL_KERNELS=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e CCL_ZE_IPC_EXCHANGE=drmfd \
  -e CCL_SYCL_ALLREDUCE_TMP_BUF=0 -e ZE_AFFINITY_MASK=0,1 \
  -e VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
  -e VLLM_USE_AOT_COMPILE=0 -e CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 -e CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_QUANTIZE_Q40_LIB=/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e PYTORCH_XPU_ALLOC_CONF=expandable_segments:True \
  "${ENVPART[@]}" \
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -v /models/drafter-fp8-v5:/models/drafter:ro \
  -v /models/qwen3.8-27b-dflash2:/models/dflash2:ro \
  -w /llm-scaler/vllm --entrypoint /bin/bash "${IMG}" \
  -c "exec /opt/venv/bin/python3 /opt/venv/bin/vllm serve \
    --model /models/target --served-model-name qwen3.8-27b-fp8 \
    --tensor-parallel-size 2 --gpu-memory-utilization ${GMU} \
    --max-model-len ${MAXLEN} --max-num-batched-tokens 8192 --max-num-seqs 64 \
    --block-size 512 --dtype float16 \
    --kv-cache-dtype ${KV} --mamba-ssm-cache-dtype float16 \
    --async-scheduling --enable-prefix-caching --trust-remote-code \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --default-chat-template-kwargs '{\"preserve_thinking\":true,\"reasoning_effort\":\"xhigh\"}' \
    ${SPECARG} \
    --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
    ${EXTRAFLAGS} \
    --port 8000 > /root/${LOG}.log 2>&1"
for i in $(seq 1 80); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null)
  if [ "$code" = "200" ]; then echo "HEALTH OK after ~$((i*10))s"; echo "BENCH_BOOT_OK $LOG"; exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "CONTAINER DIED"; docker cp lsv-test:/root/${LOG}.log /tmp/${LOG}_dead.log >/dev/null 2>&1
    tail -30 /tmp/${LOG}_dead.log; exit 7
  fi
done
echo "HEALTH TIMEOUT"; docker cp lsv-test:/root/${LOG}.log /tmp/${LOG}_to.log >/dev/null 2>&1; tail -25 /tmp/${LOG}_to.log; exit 8
