# llm-scaler-vllm-adv:v37 - ALL keeper improvements BAKED (max-performance).
# Base: certified v31.1 (#11 fix posture: whole-step capture, no compile).
# = v36 recipe + v33_scalefold + v37 barrier-default-off.
# Baked (md5-gated against committed repo copies at build time):
#   v32_mq_regpatch.py    MQ stage1 per-row 2D accumulation   (active, prod)
#   v32_align_async_v2.py async counts + deferred postprocess (active, spec; nospec-inert)
#   v35_k4_unclamp.py     k>3 user-selectable                 (no-op nospec/k<=3)
#   v33_mq3d_triton.py    fp8-KV spec-verify Triton MQ3D path (fp8 lanes only)
#   v33_scalefold.py      S/P-tile fp8 scale folding, bit-identical  (fp8 lanes only)
#   v34_shim_opt.py       per-call overhead cache on v33 shim (AFTER v33+scalefold)
#   v34_e5m2_guard.py     e5m2+fp8-ckpt probe                 (env-gated, inert)
#   v34_f8bi_pin.py       FA2 split pin                       (env-gated, inert)
#   v37_barrier_default.py spec draft-barrier DEFAULT OFF     (spec lanes; v33 keeper state)
# EXCLUDED: v34_gdn_cache.py (REJECTED by A/B gate, -45% @65k);
#           v33_bm16.py (measured neutral, no gain);
#           MNBT/chunked-prefill 16384 (measured null).
# /root/.v37_baked marks the image so bootp.sh skips boot-time patching.
FROM llm-scaler-vllm-adv:v31.1

COPY v32_mq_regpatch.py v32_align_async_v2.py v35_k4_unclamp.py \
     v33_mq3d_triton.py v33_scalefold.py v34_shim_opt.py \
     v34_e5m2_guard.py v34_f8bi_pin.py v37_barrier_default.py \
     /tmp/v37/

RUN set -eux; \
    P=/opt/venv/bin/python; \
    $P /tmp/v37/v32_mq_regpatch.py; \
    $P /tmp/v37/v32_align_async_v2.py; \
    $P /tmp/v37/v35_k4_unclamp.py; \
    $P /tmp/v37/v33_mq3d_triton.py; \
    $P /tmp/v37/v33_scalefold.py; \
    $P /tmp/v37/v34_shim_opt.py; \
    $P /tmp/v37/v34_e5m2_guard.py; \
    $P /tmp/v37/v34_f8bi_pin.py; \
    $P /tmp/v37/v37_barrier_default.py; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    XPU=$SP/vllm/platforms/xpu.py; \
    GM=$SP/vllm/v1/worker/gpu_model_runner.py; \
    grep -q 'llm-scaler v35: #12 clamp removed' $XPU; \
    ! grep -q 'VLLM_XPU_ALLOW_K4_CAPTURE' $XPU; \
    grep -q 'llm-scaler v31.1 (#11 fix)' $XPU; \
    grep -q 'llm-scaler v32: per-row 2D score accumulation' \
        $SP/vllm/v1/attention/ops/triton_turboquant_decode.py; \
    grep -q 'llm-scaler v32: async accepted-count' $GM; \
    grep -q 'run_deferred_postprocess' \
        $SP/vllm/v1/worker/mamba_utils.py; \
    grep -q 'llm-scaler v33: allow the 3D split path' \
        $SP/vllm/v1/attention/ops/triton_unified_attention.py; \
    grep -q 'llm-scaler v33: hoist per-tensor fp8 scales' \
        $SP/vllm/v1/attention/ops/triton_unified_attention.py; \
    grep -q '_v33_mq3d_varlen' \
        $SP/vllm/v1/attention/backends/flash_attn.py; \
    grep -q 'llm-scaler v34: cached descale' \
        $SP/vllm/v1/attention/backends/flash_attn.py; \
    grep -q 'llm-scaler v34: pin FA2 KV split count' \
        $SP/vllm/v1/attention/backends/flash_attn.py; \
    grep -q 'llm-scaler v34: e5m2-with-fp8-ckpt opt-in' \
        $SP/vllm/model_executor/layers/attention/attention.py; \
    grep -q 'llm-scaler v37: draft-barrier default OFF' $GM; \
    grep -q 'VLLM_XPU_SPEC_DRAFT_BARRIER", "0"' $GM; \
    $P -m py_compile $XPU $GM \
        $SP/vllm/v1/attention/ops/triton_turboquant_decode.py \
        $SP/vllm/v1/worker/mamba_utils.py \
        $SP/vllm/v1/attention/ops/triton_unified_attention.py \
        $SP/vllm/v1/attention/backends/flash_attn.py \
        $SP/vllm/model_executor/layers/attention/attention.py; \
    $P -c "import ast; ast.parse(open('$GM').read()); print('ast OK')"; \
    rm -rf /tmp/v37; \
    find $SP/vllm -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    touch /root/.v37_baked; \
    echo "v37 bake OK (v32 MQ+async, v35 unclamp, v33 MQ3D+scalefold, v34 shim+guards, v37 barrier-off default; #12 clamp absent, v31.1 gate intact)"
