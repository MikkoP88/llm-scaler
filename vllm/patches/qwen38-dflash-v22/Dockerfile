FROM llm-scaler-vllm-adv:v21
# llm-scaler-vllm-adv:v22 - MTP with cudagraphs on XPU: keep the MTP eagle
# head eager while the target keeps FULL_DECODE_ONLY XPU decode graphs.
#
# Bug (KNOWN_ISSUES #09): with --speculative-config method=mtp and
# --compilation-config cudagraph_mode=FULL_DECODE_ONLY (or PIECEWISE), TP=2
# boot dies during the eagle_head torch.compile warmup:
#   sycl invoke_barrier -> ccl allgatherv_large_su_ring<half> -> ... ->
#   c10d ProcessGroupXCCL::allgather_into_tensor_coalesced ->
#   all_gather_into_tensor -> pythonFallback (dynamo_eval_custom_code)
#   !!!!!!! Segfault encountered !!!!!!!  (both ranks) -> VllmWorker died.
# The v21 isolation matrix proved the trigger is compile-warmup x oneCCL:
# graphs mode crashed at k=4 AND k=1 (KV dtype/overlays/parsers
# irrelevant); --enforce-eager always booted (same collectives, eager,
# E[len] 3.38 at k=4). Raising CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD did not
# help. The head's sampling path issues the faulting collectives (full-vocab
# fp16 logits gather via LogitsProcessor._get_logits/_gather_logits ->
# compute_logits().argmax(); and the padded [batch, 64] gather in
# LogitsProcessor.get_top_tokens) - eager invocation is safe, dynamo-
# evaluated invocation segfaults (#05 family: eager collectives x compiled
# regions on oneCCL/xe).
#
# Fix (two coordinated, env-gated overlays; VLLM_XPU_MTP_EAGER_HEAD,
# default 1 on XPU, =0 restores stock):
#
#   1. vllm/model_executor/models/qwen3_5_mtp.py: module-tail
#      ignore_torch_compile() on Qwen3_5MultiTokenPredictor, Qwen3_5MTP and
#      Qwen3_5MoeMTP when on XPU - the head never enters torch.compile.
#      BOTH decorated classes must be opted out (ignoring only the outer
#      Qwen3_5MTP leaves the inner self.model compiled: the decorators
#      machinery only skips the exact class carrying the ignore key).
#      Non-XPU platforms are untouched.
#
#   2. vllm/v1/spec_decode/llm_base_proposer.py: initialize_cudagraph_keys
#      forces the drafter to CUDAGraphMode.NONE (instead of PIECEWISE) for
#      method=="mtp" on XPU when the env is on: an eager head has no
#      compiled subgraphs to capture piecewise, and NONE selects the
#      existing direct_eager_inputs propose fast path - exactly the code
#      path the eager validation cells exercised. Method-gated so dflash
#      and eagle drafters keep their validated PIECEWISE behavior.
#
# Effect: --speculative-config {"method":"mtp",...} now boots WITH
# --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}: the target
# backbone keeps its XPU decode graphs (the decode-loop win), only the
# single-layer MTP head runs eager.
#
# No other deltas vs the validated v21 image.

COPY qwen3_5_mtp.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py
COPY llm_base_proposer.py /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/llm_base_proposer.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    grep -q 'VLLM_XPU_MTP_EAGER_HEAD' $SP/vllm/model_executor/models/qwen3_5_mtp.py && \
    grep -q 'def _mtp_eager_head_enabled' $SP/vllm/model_executor/models/qwen3_5_mtp.py && \
    grep -q 'ignore_torch_compile(Qwen3_5MultiTokenPredictor)' $SP/vllm/model_executor/models/qwen3_5_mtp.py && \
    grep -q 'ignore_torch_compile(Qwen3_5MoeMTP)' $SP/vllm/model_executor/models/qwen3_5_mtp.py && \
    grep -q 'llm-scaler v22' $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    grep -q 'VLLM_XPU_MTP_EAGER_HEAD' $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    grep -q 'import os' $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    find $SP/vllm/model_executor/models $SP/vllm/v1/spec_decode -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    python -m py_compile \
      $SP/vllm/model_executor/models/qwen3_5_mtp.py \
      $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    python -c "from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP, Qwen3_5MoeMTP, Qwen3_5MultiTokenPredictor; from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer; from vllm.compilation.decorators import ignore_torch_compile, _should_ignore_torch_compile; print('v22 imports OK')"
