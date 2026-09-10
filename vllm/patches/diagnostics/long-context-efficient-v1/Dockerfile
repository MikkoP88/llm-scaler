# Optional candidate image; default cache OFF until end-to-end qualification.
FROM llm-scaler-exp:v1.2.5
COPY build_candidate.py fp32_cache.py /tmp/lce1-build/
RUN /opt/venv/bin/python /tmp/lce1-build/build_candidate.py \
      /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash2.py \
      /tmp/lce1-build/qwen3_dflash2.py && \
    cp /tmp/lce1-build/qwen3_dflash2.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash2.py && \
    cp /tmp/lce1-build/fp32_cache.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/lce1_fp32_cache.py && \
    /opt/venv/bin/python -m py_compile /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash2.py
