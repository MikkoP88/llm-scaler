FROM llm-scaler-vllm-adv:v22
# llm-scaler-vllm-adv:v23 - prefill-step peer barrier for the spec drafter
# (KNOWN_ISSUES #11: large-prompt requests silently lost with MTP on).
#
# Bug: with --enable-prefix-caching + MTP (any k) + TP=2 on XPU, ~7.5% of
# requests with multi-block prompts (~4.8k tokens; block negotiated to 4096
# under turboquant_4bit_nc) wedge INSIDE the engine forever:
#   - client times out; vllm:request_success_total never moves (not late,
#     not aborted-counted until disconnect); request sits Running with KV
#     allocated until the client disconnect triggers the abort
#   - serve log completely clean, dmesg clean (no xe reset)
#   - py-spy during the wedge: both TP workers "active" - TP0 spinning in
#     ur_command_list_manager::appendUSMMemcpy (the align-mode .cpu() D2H
#     after sample_tokens), TP1 a full step ahead in _calc_spec_decode_metadata
#     polling urEventGetInfo, or inside the MTP draft (Triton TQ decode
#     attention launch path)
#   - xpu-smi during the wedge: BOTH GPUs Compute+Copy engines 100%, EU
#     ~11% active / ~56% stall - oneCCL spin-wait kernels burning both
#     engines with no competing work (so, unlike the deep-stream #05
#     variant, the 640ms GuC preempt timeout never fires: no reset, just an
#     eternal spin)
#   - 12-token prompts never wedge (0/200); the wedge always straddles the
#     sample -> propose seam of a large PREFILL step
#
# Isolation matrix (v22 image, ~4.8k-token prompts, 80 reqs/cell):
#   BASE (user flags)                 6 lost
#   --mamba-cache-mode all            2 lost   (align .cpu() not the trigger)
#   --async-scheduling off            6 lost   (async not the trigger)
#   --max-num-batched-tokens 4096     1 lost   (chunking does not fix)
#   --speculative-config off          0 lost   (MTP path IS the trigger)
#
# Mechanism: the MTP head issues its own TP collectives immediately after
# the large prefill forward; when the ranks reach the drafter skewed (one
# host still draining its prefill stream, the peer already submitting head
# collectives), oneCCL's non-preemptible spin-wait kernels miss the
# rendezvous - the documented #05 peer-late family, prefill variant.
#
# Fix (env-gated overlay; VLLM_SPEC_PREFILL_BARRIER, default 1; =0 restores
# v22 behavior): gpu_model_runner.propose_draft_token_ids drains this rank's
# XPU stream (torch.xpu.synchronize()) before running the drafter whenever
# the step scheduled >= 512 tokens. Both ranks execute the same step code,
# so both drain: the peer-late window collapses to host glue, and the head
# collectives start from empty, symmetric queues. Decode steps
# (max_num_seqs * (k+1) tokens - 320 at 64 seqs / k=4) skip the barrier and
# keep the async-scheduling fast path; the sync cost is only paid on large
# prefill steps (~2s) where losing tail overlap is negligible.
#
# No other deltas vs the validated v22 image.

COPY gpu_model_runner.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    grep -q '_SPEC_PREFILL_PEER_BARRIER' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q 'VLLM_SPEC_PREFILL_BARRIER' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q 'llm-scaler v23 (KNOWN_ISSUES #11)' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q 'torch.xpu.synchronize()' $SP/vllm/v1/worker/gpu_model_runner.py && \
    find $SP/vllm/v1/worker -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    python -m py_compile $SP/vllm/v1/worker/gpu_model_runner.py && \
    python -c "import ast, sys; \
        src = open('$SP/vllm/v1/worker/gpu_model_runner.py').read(); \
        ast.parse(src); \
        assert src.count('_SPEC_PREFILL_PEER_BARRIER') == 2, 'barrier flag occurrences'; \
        assert 'num_scheduled_tokens >= 512' in src, 'threshold'; \
        print('v23 overlay guards OK')"
