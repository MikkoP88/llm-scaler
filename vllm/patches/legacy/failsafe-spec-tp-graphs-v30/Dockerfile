# llm-scaler-vllm-adv:v30 - fail-safe for KNOWN_ISSUES #11.
FROM llm-scaler-vllm-adv:v29

COPY safe_spec_tp_graph.patch /tmp/v30/safe_spec_tp_graph.patch

RUN set -eux; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    XPU_PLATFORM=$SP/vllm/platforms; \
    cd $XPU_PLATFORM; \
    patch -p0 --forward --batch < /tmp/v30/safe_spec_tp_graph.patch; \
    grep -q 'llm-scaler v30 safety gate (KNOWN_ISSUES #11)' xpu.py; \
    grep -q 'VLLM_XPU_ALLOW_UNSAFE_SPEC_TP_GRAPH' xpu.py; \
    grep -q 'os.environ\["TORCH_COMPILE_DISABLE"\] = "1"' xpu.py; \
    test "$(grep -c 'Continuing with fully eager execution' xpu.py)" = "1"; \
    /opt/venv/bin/python -m py_compile xpu.py; \
    rm -rf /tmp/v30; \
    find $SP/vllm -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    echo "v30 spec-TP graph safety gate OK"
