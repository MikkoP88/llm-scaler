FROM llm-scaler-vllm-adv:v19
# llm-scaler-vllm-adv:v20 - spec step-cost reduction over v19 (Phase A2 dbg:
# env-gated segment timing; later phases add host cuts + the full propose
# graph). Files overlayed here (Phase A2):
#
#   1. vllm/v1/spec_decode/spec_timing.py (NEW): VLLM_SPEC_TIMING=1 env-gated
#      per-segment host+device timing (xpu event ring, one sync per 200
#      steps; zero-cost null ctx when off). Segments: step_wall, tforward
#      (verify replay), tlogits, propose, precompute, dforward, greedy.
#
#   2. vllm/v1/worker/gpu_model_runner.py: timing hooks only - spec_step()
#      at propose_draft_token_ids entry (step wall), spec_seg("tforward")
#      around _model_forward, spec_seg("tlogits") around compute_logits,
#      spec_seg("propose") around the dflash drafter.propose call.
#
#   3. vllm/v1/spec_decode/dflash.py: timing hooks only - spec_seg
#      ("precompute") around precompute_and_store_context_kv, _greedy_sample
#      split into a timed wrapper + _greedy_sample_impl.
#
#   4. vllm/v1/spec_decode/llm_base_proposer.py: timing hook - spec_seg
#      ("dforward") around the drafter forward under set_forward_context.
#
# All timing code is inert unless VLLM_SPEC_TIMING=1 (default 0); the serving
# behavior is bit-identical to v19 with the env unset.
#
# v19 lineage (inherited): #05b revision (TQ x spec allowed), MQ verify
# kernel + dispatch, #06 replay-safe verify graphs, #07 fp8 KV scale capture
# fix, #03 SPEC_SAFE_WARMUP, #05d/#05e, dflash cold-start guard.

COPY spec_timing.py /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/spec_timing.py
COPY gpu_model_runner.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
COPY dflash.py /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/dflash.py
COPY llm_base_proposer.py /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/llm_base_proposer.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    grep -q 'def spec_seg' $SP/vllm/v1/spec_decode/spec_timing.py && \
    grep -q 'spec_step()' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q 'spec_seg("precompute")' $SP/vllm/v1/spec_decode/dflash.py && \
    grep -q 'spec_seg("dforward")' $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    find $SP/vllm/v1/worker $SP/vllm/v1/spec_decode -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    python -m py_compile \
      $SP/vllm/v1/spec_decode/spec_timing.py \
      $SP/vllm/v1/worker/gpu_model_runner.py \
      $SP/vllm/v1/spec_decode/dflash.py \
      $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    python -c "from vllm.v1.spec_decode.spec_timing import spec_seg, spec_step; from vllm.v1.worker.gpu_model_runner import GPUModelRunner; from vllm.v1.spec_decode.dflash import DFlashProposer; print('v20 A2 imports OK')"
