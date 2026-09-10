# llm-scaler fp8-mtp4-v1 — bake validated fp8mq kernel defaults.
# Tiles stay at the shipped optimum (32/4/2 — retune REFUTED: 8/1/1 was
# 1.6-10x worse at long ctx). The collapse driver is SPLITS-scaled
# (mid scratch + per-step cost); SPLITS is the fix knob, set at build.
ARG BASE=llm-scaler-exp:v1.2.5
FROM ${BASE}
ARG SPLITS=16
ARG BLOCK_KV=32
ARG WARPS=4
ARG STAGES=2
COPY patch_fp8mq_defaults.py /tmp/patch_fp8mq_defaults.py
RUN /opt/venv/bin/python3 /tmp/patch_fp8mq_defaults.py \
      --splits ${SPLITS} --block-kv ${BLOCK_KV} --warps ${WARPS} --stages ${STAGES} \
 && /opt/venv/bin/python3 -m py_compile \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_fp8_mq.py \
 && grep -n 'os.environ.get("VLLM_FP8MQ' \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_fp8_mq.py \
 && rm /tmp/patch_fp8mq_defaults.py
