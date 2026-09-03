FROM llm-scaler-vllm-adv:v22
# llm-scaler-vllm-adv:v24 - prefill-step PEER RENDEZVOUS for the spec drafter
# (KNOWN_ISSUES #11: large-prompt requests silently lost with MTP on).
#
# v23 (one-sided torch.xpu.synchronize() before the drafter) was built,
# deployed, and DISPROVEN: py-spy during a v23 wedge showed
#   - TP0 stuck BEFORE the barrier in _update_states_after_model_execute
#     (align-mode .cpu() D2H): ur_command_list_manager::appendUSMMemcpy
#     cannot even be SUBMITTED - this rank's in-flight queue never drains;
#   - TP1 already PAST the barrier (its stream drained fast), inside the
#     drafter issuing the MTP head's vocab all-gather
#     (qwen3_5_mtp.py:476 -> _gather_logits -> ProcessGroupXCCL allgather).
# A one-sided drain cannot fix this: each rank drains its OWN stream and
# immediately proceeds, so the fast rank still enters the collective alone.
#
# Deadlock cycle (the #05 oneCCL peer-late family, prefill variant):
#   fast rank submits head all-gather -> its non-preemptible spin-wait
#   kernels monopolize the engines -> they starve that rank's own earlier
#   in-flight collective sends -> slow rank's pending receives never
#   complete -> slow rank's stream never drains -> its host blocks forever
#   in the .cpu() D2H submission before it can join the all-gather ->
#   eternal spin (no competing queue work => no GuC preempt reset; both
#   engines 100%, EU stall-dominant; dmesg clean; request stays Running
#   until the client aborts).
#
# Fix: TRUE host-level rendezvous before any drafter work on steps with
# >= 512 scheduled tokens: drain this rank's stream, then
# torch.distributed.barrier(group=tp_group.cpu_group) - the gloo cpu group
# every GroupCoordinator already builds (pure host barrier, no GPU kernels,
# cannot itself wedge). The fast rank now WAITS; the slow rank's stream
# drains (its completion needs only sends the fast rank already submitted);
# both ranks enter the drafter with empty, symmetric queues. Decode steps
# (max_num_seqs * (k+1) = 320 tokens at 64 seqs / k=4) skip it and keep the
# async-scheduling fast path. VLLM_SPEC_PREFILL_BARRIER=0 restores v22.
#
# No other deltas vs the validated v22 image.

COPY gpu_model_runner.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    grep -q 'llm-scaler v24 (KNOWN_ISSUES #11)' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q '_SPEC_PEER_RENDEZVOUS_GROUP' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q '_spec_peer_rendezvous_group' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q 'torch.distributed.barrier(group=_peer_group)' $SP/vllm/v1/worker/gpu_model_runner.py && \
    grep -q 'torch.xpu.synchronize()' $SP/vllm/v1/worker/gpu_model_runner.py && \
    find $SP/vllm/v1/worker -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    python -m py_compile $SP/vllm/v1/worker/gpu_model_runner.py && \
    python -c "import ast; \
        src = open('$SP/vllm/v1/worker/gpu_model_runner.py').read(); \
        ast.parse(src); \
        assert src.count('_SPEC_PEER_RENDEZVOUS_GROUP') >= 4, 'rendezvous refs'; \
        assert src.count('torch.distributed.barrier') == 1, 'one barrier call'; \
        assert 'num_scheduled_tokens >= 512' in src, 'threshold'; \
        print('v24 overlay guards OK')"
