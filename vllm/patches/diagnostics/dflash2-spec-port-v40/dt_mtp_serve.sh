#!/bin/bash
# dt_mtp_serve.sh — boot lsv-test as a STOCK-fork MTP lane at bf16.
# Reference lane for the DFlash2 comparison: same flags as the dflash
# lanes (block 512, turboquant_4bit_nc, prefix caching, bf16, gmu 0.85)
# but no era-3 patches (stock site-packages) and the fork's native MTP
# spec method. k=3 stays in the fork's known bit-stable region (k<=3).
set -u
docker rm -f lsv-test >/dev/null 2>&1
docker run -d --name lsv-test \
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
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -w /llm-scaler/vllm \
  --entrypoint /bin/bash \
  llm-scaler-prod:v1 -c "exec /opt/venv/bin/python3 /opt/venv/bin/vllm serve \
    --model /models/target \
    --served-model-name qwen3.8-27b-fp8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --block-size 512 \
    --dtype float16 \
    --kv-cache-dtype turboquant_4bit_nc \
    --mamba-ssm-cache-dtype float16 \
    --async-scheduling \
    --enable-prefix-caching \
    --trust-remote-code \
    --reasoning-parser deepseek_r1 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
    --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \
    --port 8000 > /root/dt_mtp.log 2>&1"
echo "booted dt_mtp (stock fork, bf16, k=3) log=/root/dt_mtp.log"
