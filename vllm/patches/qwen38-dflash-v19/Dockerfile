FROM llm-scaler-vllm-adv:v18
# llm-scaler-vllm-adv:v19 - TQ x drafter enablement + multi-query verify
# kernel over v18 (3 site-packages files differ):
#
#   1. vllm/config/vllm.py (#05b revision): VLLM_ALLOW_TQ_SPEC default
#      0 -> 1. The v14-era auto-disable of the dflash drafter under
#      turboquant KV is obsolete: the drafter's own KV is forced to
#      unquantized 'auto' (dflash _create_draft_vllm_config) and the
#      #05b/#03 warmup skips protect boot. Spec now serves WITH
#      turboquant KV by default; VLLM_ALLOW_TQ_SPEC=0 rolls back.
#
#   2. vllm/v1/attention/ops/triton_turboquant_decode.py (additive):
#      _tq_mq_decode_stage1 + _tq_mq_fwd_stage2 +
#      triton_turboquant_mq_decode_attention. A dflash verify step
#      (q_len = k+1 <= 8) previously hit the "synthetic decode" path:
#      q_len single-query kernel rows, EACH re-scanning the full
#      compressed context (~q_len x TQ KV bandwidth per step, cost
#      grows with depth - the TQ spec long-run collapse). The MQ
#      kernel scores all query rows per shared KV tile in ONE pass
#      (flash-decoding style, per-row causal limits q0+j, blind
#      split-combine via lse=-inf zero-weights). Existing kernels
#      untouched; VLLM_TQ_MQ_STAGE1_WARPS/STAGES tuning envs.
#
#   3. vllm/v1/attention/backends/turboquant_attn.py: continuation/
#      verify fast-branch dispatches 1 < q_len <= 8 to the MQ kernel
#      (single call, no per-row expand). VLLM_TQ_MQ_VERIFY=0 rolls
#      back to the v18 synthetic-decode path (kept verbatim).
#
# v18 lineage (inherited): #03 boot-fault fix (VLLM_XPU_SPEC_SAFE_WARMUP),
# #05b/#05d/#05e, dflash cold-start guard, canonical XPU serving ENV.

COPY config_vllm.py /opt/venv/lib/python3.12/site-packages/vllm/config/vllm.py
COPY turboquant_attn.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/turboquant_attn.py
COPY triton_turboquant_decode.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_turboquant_decode.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    grep -q 'VLLM_ALLOW_TQ_SPEC", "1"' $SP/vllm/config/vllm.py && \
    grep -q 'def triton_turboquant_mq_decode_attention' $SP/vllm/v1/attention/ops/triton_turboquant_decode.py && \
    grep -q 'triton_turboquant_mq_decode_attention' $SP/vllm/v1/attention/backends/turboquant_attn.py && \
    find $SP/vllm/v1/attention $SP/vllm/config -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    python -m py_compile \
      $SP/vllm/config/vllm.py \
      $SP/vllm/v1/attention/backends/turboquant_attn.py \
      $SP/vllm/v1/attention/ops/triton_turboquant_decode.py && \
    python -c "from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_mq_decode_attention; from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend; print('v19 imports OK')"
