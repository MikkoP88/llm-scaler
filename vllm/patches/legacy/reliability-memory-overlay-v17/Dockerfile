FROM llm-scaler-vllm-adv:v14
# llm-scaler-vllm-adv:v17 - reliability + memory fixes over v14 (see README.md):
#   #03  warmup watchdog (VLLM_XPU_WARMUP_TIMEOUT_S) + /opt/serve_supervised.sh
#   #05b TQ-safe post-capture warmup (VLLM_XPU_TQ_SAFE_WARMUP) + spec
#        auto-disable for turboquant KV dtypes (VLLM_ALLOW_TQ_SPEC to opt back in)
#   #05d check_stop ignore_eos hard guard + FINISH_DIAG diagnostics
#   #05e oneCCL scratch arena pre-allocation before KV pool sizing
#        (VLLM_XPU_PREALLOC_CCL_ARENA) - restores --gpu-memory-utilization 0.8
#   dflash.py: cold-start stall guard + latent scratch fix (commit a31915f)
#   monitor3.sh baked for per-arm telemetry (also shipped to /root/telemetry)
# Base file provenance: gpu_model_runner.py carries the v14 image's TurboQuant
# padded-page stride fix (md5 cb931a3b lineage); gpu_worker.py / sched utils /
# config verified md5-identical to the v14 image before editing.

COPY dflash.py /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/dflash.py
COPY gpu_model_runner.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
COPY gpu_worker.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py
COPY sched_utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/utils.py
COPY config_vllm.py /opt/venv/lib/python3.12/site-packages/vllm/config/vllm.py
COPY serve_supervised.sh /opt/serve_supervised.sh
COPY monitor3.sh /opt/monitor3.sh

# Canonical XPU serving env (the proven patches/qwen38-dflash serve.sh block)
# baked as image defaults; override per-run with -e. expandable_segments
# addresses the #05(a) fragmentation-driven mid-decode alloc stalls.
ENV ZE_AFFINITY_MASK=0,1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ALLOW_LONG_MODEL_LEN=1 \
    VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_USE_AOT_COMPILE=0 \
    VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    CCL_TOPO_P2P_ACCESS=1 \
    CCL_ZE_IPC_EXCHANGE=drmfd \
    CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
    CCL_SYCL_ALLREDUCE_TMP_BUF=0 \
    CCL_ENABLE_SYCL_KERNELS=1 \
    CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 \
    CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_XPU_ALLOC_CONF=expandable_segments:True

RUN chmod +x /opt/serve_supervised.sh /opt/monitor3.sh && \
    find /opt/venv/lib/python3.12/site-packages/vllm -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    SP=/opt/venv/lib/python3.12/site-packages && \
    grep -q '^import os' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q '^import faulthandler' $SP/vllm/v1/worker/gpu_worker.py && \
    python -m py_compile \
        $SP/vllm/v1/spec_decode/dflash.py \
        $SP/vllm/v1/worker/gpu_model_runner.py \
        $SP/vllm/v1/worker/gpu_worker.py \
        $SP/vllm/v1/core/sched/utils.py \
        $SP/vllm/config/vllm.py && \
    python -c "from vllm.v1.spec_decode.dflash import DFlashProposer; print('dflash import OK')" && \
    python -c "from vllm.v1.core.sched.utils import check_stop; print('sched utils import OK')" && \
    python -c "from vllm.config.vllm import VllmConfig; print('config import OK')" && \
    python -c "from vllm.v1.worker.gpu_worker import Worker; print('gpu_worker import OK')" && \
    python -c "from vllm.v1.worker.gpu_model_runner import GPUModelRunner; print('model runner import OK')"
