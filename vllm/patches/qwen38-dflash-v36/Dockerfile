# llm-scaler-vllm-adv:v36 - ALL keeper improvements BAKED.
# Base: certified v31.1 (#11 fix posture: whole-step capture, no compile).
# Baked (md5-gated against committed repo copies at build time):
#   v32_mq_regpatch.py    MQ stage1 per-row 2D accumulation   (active, prod)
#   v32_align_async_v2.py async counts + deferred postprocess (active, spec; nospec-inert)
#   v35_k4_unclamp.py     k>3 user-selectable                 (no-op nospec/k<=3)
#   v33_mq3d_triton.py    fp8-KV spec-verify Triton MQ3D path (fp8 lanes only)
#   v34_shim_opt.py       per-call overhead cache on v33 shim (AFTER v33)
#   v34_e5m2_guard.py     e5m2+fp8-ckpt probe                 (env-gated, inert)
#   v34_f8bi_pin.py       FA2 split pin                       (env-gated, inert)
# EXCLUDED: v34_gdn_cache.py - falsified at approach level (v34 addendum).
# Marker checks mirror Dockerfile.v31_1; the #12 clamp check is INVERTED.
# /root/.v36_baked marks the image so bootp.sh skips boot-time patching.
FROM llm-scaler-vllm-adv:v31.1

COPY v32_mq_regpatch.py v32_align_async_v2.py v35_k4_unclamp.py \
     v33_mq3d_triton.py v34_shim_opt.py v34_e5m2_guard.py v34_f8bi_pin.py \
     /tmp/v36/

RUN set -eux; \
    P=/opt/venv/bin/python; \
    $P /tmp/v36/v32_mq_regpatch.py; \
    $P /tmp/v36/v32_align_async_v2.py; \
    $P /tmp/v36/v35_k4_unclamp.py; \
    $P /tmp/v36/v33_mq3d_triton.py; \
    $P /tmp/v36/v34_shim_opt.py; \
    $P /tmp/v36/v34_e5m2_guard.py; \
    $P /tmp/v36/v34_f8bi_pin.py; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    XPU=$SP/vllm/platforms/xpu.py; \
    grep -q 'llm-scaler v35: #12 clamp removed' $XPU; \
    ! grep -q 'VLLM_XPU_ALLOW_K4_CAPTURE' $XPU; \
    grep -q 'llm-scaler v31.1 (#11 fix)' $XPU; \
    grep -q 'llm-scaler v32: per-row 2D score accumulation' \
        $SP/vllm/v1/attention/ops/triton_turboquant_decode.py; \
    grep -q 'llm-scaler v32: async accepted-count' \
        $SP/vllm/v1/worker/gpu_model_runner.py; \
    grep -q 'run_deferred_postprocess' \
        $SP/vllm/v1/worker/mamba_utils.py; \
    grep -q 'llm-scaler v33: allow the 3D split path' \
        $SP/vllm/v1/attention/ops/triton_unified_attention.py; \
    grep -q '_v33_mq3d_varlen' \
        $SP/vllm/v1/attention/backends/flash_attn.py; \
    grep -q 'llm-scaler v34: cached descale' \
        $SP/vllm/v1/attention/backends/flash_attn.py; \
    grep -q 'llm-scaler v34: pin FA2 KV split count' \
        $SP/vllm/v1/attention/backends/flash_attn.py; \
    grep -q 'llm-scaler v34: e5m2-with-fp8-ckpt opt-in' \
        $SP/vllm/model_executor/layers/attention/attention.py; \
    $P -m py_compile $XPU \
        $SP/vllm/v1/attention/ops/triton_turboquant_decode.py \
        $SP/vllm/v1/worker/gpu_model_runner.py \
        $SP/vllm/v1/worker/mamba_utils.py \
        $SP/vllm/v1/attention/ops/triton_unified_attention.py \
        $SP/vllm/v1/attention/backends/flash_attn.py \
        $SP/vllm/model_executor/layers/attention/attention.py; \
    $P -c "import ast; ast.parse(open('$XPU').read()); print('ast OK')"; \
    rm -rf /tmp/v36; \
    find $SP/vllm -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    touch /root/.v36_baked; \
    echo "v36 bake OK (v32 MQ+async, v35 unclamp, v33 MQ3D, v34 shim+guards; #12 clamp absent, v31.1 gate intact)"
