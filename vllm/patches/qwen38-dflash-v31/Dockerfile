# llm-scaler-vllm-adv:v31 - graphs x speculative decode root-cause arms
# (#11 wedge candidate fix + #12 exposure clamp) on top of the v30 gate.
FROM llm-scaler-vllm-adv:v30

COPY spec_fixes.patch /tmp/v31/spec_fixes.patch

RUN set -eux; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    XPU_PLATFORM=$SP/vllm/platforms; \
    cd $XPU_PLATFORM; \
    patch -p0 --forward --batch < /tmp/v31/spec_fixes.patch; \
    grep -q 'llm-scaler v31 (#11 arm)' xpu.py; \
    grep -q 'llm-scaler v31 (#12 clamp)' xpu.py; \
    grep -q 'VLLM_XPU_SPEC_EAGER_GDN' xpu.py; \
    grep -q 'VLLM_XPU_SPEC_EXTRA_SPLITS' xpu.py; \
    grep -q 'VLLM_XPU_ALLOW_K4_CAPTURE' xpu.py; \
    grep -q '_spec.num_speculative_tokens = 3' xpu.py; \
    test "$(grep -c 'vllm::gdn_attention_core' xpu.py)" = "2"; \
    /opt/venv/bin/python -m py_compile xpu.py; \
    /opt/venv/bin/python -c "import ast; ast.parse(open('xpu.py').read()); print('ast OK')"; \
    rm -rf /tmp/v31; \
    find $SP/vllm -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    echo "v31 spec fixes OK (#11 GDN-split arm + #12 k-clamp)"
