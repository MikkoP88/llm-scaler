FROM llm-scaler-vllm-adv:v22
# llm-scaler-vllm-adv:v25 - MTP draft-step collective shrink: batch=1 uses
# the tiny top-token gather (KNOWN_ISSUES #11 long-context wedge).
#
# Bug (KNOWN_ISSUES #11, deterministic at >=32k prompts): with MTP k=4,
# TP=2, FULL_DECODE_ONLY target graphs and the eager MTP head (v22), the
# drafter's _greedy_sample issues a FULL-vocab (248,320) oneCCL all-gather
# on EVERY draft step whenever hidden_states.shape[0] == 1 - i.e. every
# single-stream request - because the stock guard
#     use_local_argmax_reduction and hidden_states.shape[0] > 1
# only routes to LogitsProcessor.get_top_tokens (local argmax + padded
# [bs, 64] all-gather) for batch > 1. That non-preemptible oneCCL ring
# kernel interleaved with the graphed verify replays clogs the UR
# in-flight queue; both ranks then freeze in the align-mode .cpu() D2H
# (v24 py-spy: ur_command_list_manager::appendUSMMemcpy never submits).
# Neither a pre-drain (v23) nor a peer rendezvous (v24) upstream of the
# drafter could reach the clog - it forms in the collectives themselves,
# so the fix must shrink the collective.
#
# Evidence chain (2026-08-30, this session):
#   E0 user baseline (block512 + mamba-fp16, MTP k4): 3/3 WEDGE at 32k
#   E2 spec-config use_local_argmax_reduction=true:   2/3 WEDGE at 32k
#      (no surprise: the shape[0] > 1 guard keeps bs=1 on the full-vocab
#       path - config alone cannot fix single-stream; code change needed)
#   bs=2 concurrent probe on the E2 boot: get_top_tokens path active ->
#      see README.md for the outcome (the mechanism discriminator).
#
# Fix (two env-gated overlays):
#
#   1. VLLM_XPU_MTP_LOCAL_ARGMAX (default 1; =0 restores v22), in
#      vllm/v1/spec_decode/llm_base_proposer.py:
#      a. __init__: use_local_argmax_reduction defaults ON for
#         method=="mtp" on XPU (False by default upstream).
#      b. _greedy_sample: the stock shape[0] > 1 guard is lifted, so
#         batch=1 also uses get_top_tokens: the per-step draft collective
#         shrinks from 248,320 to 64 values per rank (~0.03%). (Measured
#         alone this does NOT stop the wedge - bs=2 probes with the tiny
#         gather active still wedged 5/8 - but it cuts the eager
#         collective traffic the barrier must keep symmetric.)
#
#   2. VLLM_XPU_SPEC_DRAFT_BARRIER (default 1; =0 restores v22) with
#      VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX (default 0 = every spec step),
#      in vllm/v1/worker/gpu_model_runner.py
#      propose_draft_token_ids: torch.xpu.synchronize() before invoking
#      the drafter on every spec step whose max running context is >=
#      MIN_CTX. v23/v24 barriers fired only on steps scheduling >= 512
#      tokens (prefill) - the >=32k wedge forms during DECODE steps
#      (bs*(k+1) tokens), which they never covered. Both ranks execute
#      the same step code, so both drain: the drafter's eager oneCCL
#      collectives always start from empty, symmetric queues, and no
#      stale spin kernel can straddle the .cpu() drain.
#
# No other deltas vs the validated v22 image.

COPY llm_base_proposer.py /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/llm_base_proposer.py
COPY gpu_model_runner.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    grep -q 'llm-scaler v25' $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    test "$(grep -c 'VLLM_XPU_MTP_LOCAL_ARGMAX' $SP/vllm/v1/spec_decode/llm_base_proposer.py)" = "4" && \
    grep -q 'VLLM_XPU_MTP_EAGER_HEAD' $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    grep -q 'import os' $SP/vllm/v1/spec_decode/llm_base_proposer.py && \
    grep -q 'llm-scaler v25' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q '_SPEC_DRAFT_BARRIER' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q '_SPEC_DRAFT_BARRIER_MIN_CTX' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q 'torch.xpu.synchronize()' $SP/vllm/v1/worker/gpu_model_runner.py && \
    test "$(grep -c 'max_seq_len >= _SPEC_DRAFT_BARRIER_MIN_CTX' $SP/vllm/v1/worker/gpu_model_runner.py)" = "1" && \
    test "$(grep -c '_SPEC_PREFILL_PEER_BARRIER' $SP/vllm/v1/worker/gpu_model_runner.py)" = "0" && \
    find $SP/vllm/v1/spec_decode $SP/vllm/v1/worker -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null && \
    python -m py_compile \
      $SP/vllm/v1/spec_decode/llm_base_proposer.py \
      $SP/vllm/v1/worker/gpu_model_runner.py && \
    python -c "from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer; import inspect, vllm.v1.spec_decode.llm_base_proposer as m; src=inspect.getsource(m.SpecDecodeBaseProposer._greedy_sample); assert 'VLLM_XPU_MTP_LOCAL_ARGMAX' in src and 'get_top_tokens' in src; print('v25 overlay guards OK')"
