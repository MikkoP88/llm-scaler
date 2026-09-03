# llm-scaler-prod:v1 — first PRODUCTION image under the new naming scheme.
# Content = EXACTLY the certified adv:v38 tree (keepers only; failed and
# diagnostics patches NOT included): FROM certified v37 (all keeper bakes)
# + the single v38 script (#18 ESIMD fp8 decode reroute).
# Post-build gate: 19/19 manifest md5s byte-identical to adv:v38
# (harmonized/MANIFEST.md5) — enforced outside the Dockerfile, see README.
# Bake marker: /root/.llm-scaler-prod_v1_baked (bootp.sh glob /root/.*_baked).
FROM llm-scaler-vllm-adv:v37

COPY v38_esimd_reroute.py /tmp/v38/

RUN set -eux; \
    P=/opt/venv/bin/python; \
    $P /tmp/v38/v38_esimd_reroute.py; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    FA=$SP/vllm/v1/attention/backends/flash_attn.py; \
    grep -q 'llm-scaler v38: fp8 decode must not use the ESIMD kernel' $FA; \
    grep -q 'VLLM_XPU_ALLOW_ESIMD_F8' $FA; \
    grep -q 'llm-scaler v37: draft-barrier default OFF' \
        $SP/vllm/v1/worker/gpu_model_runner.py; \
    grep -q '_v33_mq3d_varlen' $FA; \
    grep -q 'llm-scaler v34: cached descale' $FA; \
    $P -m py_compile $FA; \
    $P -c "import ast; ast.parse(open('$FA').read()); print('ast OK')"; \
    rm -rf /tmp/v38; \
    find $SP/vllm -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    touch /root/.llm-scaler-prod_v1_baked; \
    echo "llm-scaler-prod:v1 bake OK (tree == adv:v38: v37 keepers + v38 ESIMD fp8 reroute)"
