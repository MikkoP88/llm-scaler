FROM llm-scaler-vllm-adv:v17
# llm-scaler-vllm-adv:v18 - single-file fix over v17:
#   #03 boot-fault ROOT CAUSE + FIX: the v17 image ENV bakes
#   VLLM_XPU_ENABLE_XPU_GRAPH=1 (first enabled by the v17 rebuild).
#   With XPU graphs ON + the dflash drafter, the eager post-capture
#   _dummy_run (max_num_reqs x (k+1) spec tokens - a never-captured
#   shape) drives TP collectives that fault the xe ccs engine 4/4
#   launches (v17 battery a8/a9/a9r/a9r3: guc_id 24/38, ~60s
#   post-capture, survives warm AND cold host reboots; graphs-off +
#   spec boots healthy - pre-rebuild arms, rmacy v16, adv:v14 - and
#   graphs-on + nospec boots healthy - a4-a7). Fix: extend the #05b
#   post-capture skip to spec + graphs (VLLM_XPU_SPEC_SAFE_WARMUP,
#   default 1): synthesize hidden states for the sampler warmup
#   instead of the eager forward; the captured graphs already
#   exercised the kernels. Only gpu_worker.py differs from v17.
#
# v17 lineage (inherited, see ../qwen38-dflash-v17/Dockerfile):
#   #03 warmup watchdog, #05b TQ-safe warmup + spec auto-disable,
#   #05d check_stop guards, #05e oneCCL arena pre-alloc,
#   dflash.py cold-start stall guard, canonical XPU serving ENV.

COPY gpu_worker.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    grep -q '^import faulthandler' $SP/vllm/v1/worker/gpu_worker.py && \
    find $SP/vllm/v1/worker -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    python -m py_compile $SP/vllm/v1/worker/gpu_worker.py && \
    python -c "from vllm.v1.worker.gpu_worker import Worker; print('gpu_worker import OK')"
