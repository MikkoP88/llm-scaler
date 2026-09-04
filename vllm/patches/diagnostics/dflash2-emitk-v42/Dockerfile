# llm-scaler-exp:dflash2-v42 — experimental DFlash2 k-adjustable lane image.
# Base: llm-scaler-prod:v1 (keepers-only certified lineage, v37/v38 baked).
# Adds the FULL DFlash2 chain on top:
#   v40 dflash2_edit.py        — DFlash2 port (10 hooks + 2 new files)
#   v41 dflash2_winfix_edit.py — TurboQuant sliding-window read fix
#   v42 dflash2_emitk_edit.py  — drafter emission cap + runner local-pad
#   v42c dflash2_schedk_edit.py— async placeholder width cap
#   v42d dflash2_udql_edit.py  — runner udql + capture-lattice rescale
#   v42d dflash2_disp_edit.py  — dispatcher width
# Default (DFLASH2_EMIT_K unset at serve time): byte-identical v41 behavior;
# the env gates evaluate at SERVE time, the patchers write them unconditionally.
# Build context: /root/build/qwen38-dflash2 (6 patchers + new/ port files).
# The v40/v41 patchers edit the installed tree in place; the v42 patchers
# write /w/patched ONLY — the explicit cp loop below mirrors boot phase 3
# (docker cp) so the baked tree is byte-identical to the runtime-patched
# lane tree (verified by md5 after build).
# /root/.dflash2_patched makes dt_dflash2_serve.sh exec the serve directly
# (no await-patch phase on a baked image); boot with
#   DFLASH2_IMG=llm-scaler-exp:dflash2-v42 bash dt_dflash2_serve.sh
FROM llm-scaler-prod:v1

COPY dflash2_edit.py dflash2_winfix_edit.py dflash2_emitk_edit.py \
     dflash2_schedk_edit.py dflash2_udql_edit.py dflash2_disp_edit.py \
     /root/dflash2_bake/
COPY new/qwen3_dflash2.py new/dflash2_proposer.py /root/dflash2_bake/new/

RUN mkdir -p /w/new \
    && cp /root/dflash2_bake/*.py /w/ \
    && cp /root/dflash2_bake/new/*.py /w/new/ \
    && cd /w \
    && python3 /w/dflash2_edit.py \
    && python3 /w/dflash2_winfix_edit.py \
    && python3 /w/dflash2_emitk_edit.py \
    && python3 /w/dflash2_schedk_edit.py \
    && python3 /w/dflash2_udql_edit.py \
    && python3 /w/dflash2_disp_edit.py \
    && SP=/opt/venv/lib/python3.12/site-packages \
    && for f in vllm/v1/spec_decode/utils.py vllm/v1/spec_decode/dflash.py \
         vllm/model_executor/models/qwen3_dflash.py \
         vllm/model_executor/models/registry.py \
         vllm/v1/core/sched/async_scheduler.py \
         vllm/config/vllm.py \
         vllm/v1/cudagraph_dispatcher.py \
         vllm/v1/worker/gpu_model_runner.py \
         vllm/v1/attention/backends/turboquant_attn.py \
         vllm/v1/attention/ops/triton_turboquant_decode.py; do \
         cp /w/patched/$f $SP/$f || exit 9; \
       done \
    && cp /w/new/qwen3_dflash2.py $SP/vllm/model_executor/models/qwen3_dflash2.py \
    && cp /w/new/dflash2_proposer.py $SP/vllm/v1/spec_decode/dflash2.py \
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
    && grep -q "llm-scaler v42: pad into a LOCAL tensor" $SP/vllm/v1/worker/gpu_model_runner.py \
    && grep -q "DFlash2 v42c async cap" $SP/vllm/v1/core/sched/async_scheduler.py \
    && grep -q "DFlash2 v42d: uniform_decode_query_len" $SP/vllm/v1/worker/gpu_model_runner.py \
    && grep -q "DFlash2 v42d: capture lattice rescaled" $SP/vllm/config/vllm.py \
    && grep -q "DFlash2 v42d dispatcher" $SP/vllm/v1/cudagraph_dispatcher.py \
    && grep -q "DFLASH2_EMIT_K" $SP/vllm/v1/spec_decode/dflash2.py \
    && python3 -m py_compile \
       $SP/vllm/model_executor/models/qwen3_dflash2.py \
       $SP/vllm/v1/spec_decode/dflash2.py \
       $SP/vllm/v1/attention/backends/turboquant_attn.py \
       $SP/vllm/v1/attention/ops/triton_turboquant_decode.py \
       $SP/vllm/v1/core/sched/async_scheduler.py \
       $SP/vllm/config/vllm.py \
       $SP/vllm/v1/cudagraph_dispatcher.py \
       $SP/vllm/v1/worker/gpu_model_runner.py \
    && touch /root/.dflash2_patched /root/.dflash2_v42_baked \
    && echo DFLASH2_V42_BAKE_OK
