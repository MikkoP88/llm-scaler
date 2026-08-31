# llm-scaler vllm-adv:v29 — graphs+MTP wedge fix arms (KNOWN_ISSUES #11
# residual, the piece-replay x eager-collective x allocator boundary).
# Base is v28dbg (flight recorder live). Three changes:
#
# FIX-1  xpu_communicator.py: VIA-allgather stable output buffer — the
#        per-call clone (~tens of allocs/spec-decode step) is replaced by a
#        per-shape cached buffer for decode-shaped contiguous tensors
#        (row cap 512; ragged prefill keeps the clone). Kills the allocator
#        churn that runs against expandable_segments:True exactly where
#        piecewise-captured graphs place their buffers.
#        VLLM_XPU_VIA_STABLEBUF=0 restores clone-per-call.
# FIX-2  gpu_worker.py: allocator reset (synchronize + empty_cache) before
#        the compile/warmup sequence and again before cudagraph capture —
#        boot-lottery lever: wedge susceptibility is decided at capture
#        time, so make every boot start capture from a defragmented pool.
#        VLLM_XPU_CAPTURE_STAB=0 disables.
# EVID   llm_base_proposer.py: propose() becomes a flight-recorder marker
#        wrapper (drafter region) — with the v28dbg AR markers this splits
#        each step's eager collectives drafter-vs-target in /tmp/fr_*.log.
FROM llm-scaler-vllm-adv:v28dbg

COPY xpu_communicator.py /tmp/v29/xpu_communicator.py
COPY gpu_worker.py /tmp/v29/gpu_worker.py
COPY llm_base_proposer.py /tmp/v29/llm_base_proposer.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    cp /tmp/v29/xpu_communicator.py $SP/vllm/distributed/device_communicators/xpu_communicator.py && \
    cp /tmp/v29/gpu_worker.py $SP/vllm/v1/worker/gpu_worker.py && \
    cp /tmp/v29/llm_base_proposer.py $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    grep -q 'llm-scaler v29 FIX-1' $SP/vllm/distributed/device_communicators/xpu_communicator.py && \
    grep -q 'VLLM_XPU_VIA_STABLEBUF' $SP/vllm/distributed/device_communicators/xpu_communicator.py && \
    grep -q 'llm-scaler v29 FIX-2' $SP/vllm/v1/worker/gpu_worker.py && \
    grep -q 'VLLM_XPU_CAPTURE_STAB' $SP/vllm/v1/worker/gpu_worker.py && \
    grep -q 'llm-scaler v29 (flight-recorder step markers)' $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    test "$(grep -c 'def propose' $SP/vllm/v1/spec_decode/llm_base_proposer.py)" = "1" && \
    test "$(grep -c 'def _propose_impl' $SP/vllm/v1/spec_decode/llm_base_proposer.py)" = "1" && \
    test "$(grep -c '@with_drafter_communicator' $SP/vllm/v1/spec_decode/llm_base_proposer.py)" = "2" && \
    test "$(grep -c '_xpu_capture_stab' $SP/vllm/v1/worker/gpu_worker.py)" = "3" && \
    /opt/venv/bin/python -m py_compile \
      $SP/vllm/distributed/device_communicators/xpu_communicator.py \
      $SP/vllm/v1/worker/gpu_worker.py \
      $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    find $SP/vllm -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    rm -rf /tmp/v29 && \
    echo "v29 overlay OK"
