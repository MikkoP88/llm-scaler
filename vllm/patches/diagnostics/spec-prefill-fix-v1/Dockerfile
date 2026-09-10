# Build a separate image; do not patch a running engine.
ARG BASE_IMAGE=llm-scaler-exp:v1.2.6t2
FROM ${BASE_IMAGE}
COPY build_patch.py test_prefill_phase.py /tmp/spec-prefill-fix/
RUN /opt/venv/bin/python /tmp/spec-prefill-fix/build_patch.py \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched \
      /tmp/spec-prefill-fix/generated && \
    cp /tmp/spec-prefill-fix/generated/output.py \
      /tmp/spec-prefill-fix/generated/async_scheduler.py \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/ && \
    /opt/venv/bin/python /tmp/spec-prefill-fix/test_prefill_phase.py
