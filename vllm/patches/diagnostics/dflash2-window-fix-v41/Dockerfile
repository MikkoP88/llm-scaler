# llm-scaler-exp:dflash2-v41 — experimental DFlash2 lane image.
# Base: llm-scaler-prod:v1 (keepers-only certified lineage, v37/v38 baked).
# Adds: DFlash2 spec-decode port (v40, dflash2_edit.py — 10 hooks + 2 new
#       files) + read-side sliding-window fix (v41, dflash2_winfix_edit.py —
#       11 hooks on turboquant_attn.py / triton_turboquant_decode.py).
# Build context: /root/build/qwen38-dflash2 (patchers + new/ port files).
# The patchers run against THIS image's site-packages, byte-compile, and
# import-smoke-test themselves; grep gates below fail the build on any
# missing hook. /root/.dflash2_patched makes dt_dflash2_serve.sh exec the
# serve directly (no await-patch phase needed on a baked image).
FROM llm-scaler-prod:v1

COPY dflash2_edit.py dflash2_winfix_edit.py /root/dflash2_bake/
COPY new/qwen3_dflash2.py new/dflash2_proposer.py /root/dflash2_bake/new/

RUN mkdir -p /w/new \
    && cp /root/dflash2_bake/dflash2_edit.py /root/dflash2_bake/dflash2_winfix_edit.py /w/ \
    && cp /root/dflash2_bake/new/qwen3_dflash2.py /root/dflash2_bake/new/dflash2_proposer.py /w/new/ \
    && cd /w \
    && python3 /w/dflash2_edit.py \
    && python3 /w/dflash2_winfix_edit.py \
    && rm -rf /w

ARG SP=/opt/venv/lib/python3.12/site-packages
RUN SP=/opt/venv/lib/python3.12/site-packages \
    && grep -q "DFlash2DraftModel" $SP/vllm/model_executor/models/registry.py \
    && grep -q "DFlash2Proposer" $SP/vllm/v1/worker/gpu_model_runner.py \
    && grep -q "self.sliding_window = sliding_window" $SP/vllm/v1/attention/backends/turboquant_attn.py \
    && grep -q "win_start=mq_win" $SP/vllm/v1/attention/backends/turboquant_attn.py \
    && test "$(grep -c HAS_WINDOW $SP/vllm/v1/attention/ops/triton_turboquant_decode.py)" = 4 \
    && test -f $SP/vllm/model_executor/models/qwen3_dflash2.py \
    && test -f $SP/vllm/v1/spec_decode/dflash2.py \
    && python3 -m py_compile \
       $SP/vllm/model_executor/models/qwen3_dflash2.py \
       $SP/vllm/v1/spec_decode/dflash2.py \
       $SP/vllm/v1/attention/backends/turboquant_attn.py \
       $SP/vllm/v1/attention/ops/triton_turboquant_decode.py \
    && touch /root/.dflash2_patched /root/.dflash2_v41_baked \
    && echo DFLASH2_V41_BAKE_OK
