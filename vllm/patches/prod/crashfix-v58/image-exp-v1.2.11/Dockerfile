# llm-scaler-exp:v1.2.11 — CRASH-FREE CONTINGENCY production image (NEO 26.18).
# Base  = llm-scaler-exp:v1.2.10 (latest exp lineage; crashfix-v58 standing tree)
# Bake  = NEO 26.18 runtime deb overlay (hang-avoiding submission semantics,
#         GSD-12919/#939) + certified boot patchers (f15b + arstage + v58p1)
#         + MTP spec draft-3 serve config (26.18 tax-shifted optimum).
# Numerics contract: probes 1-2 EXACT vs 26.14 standing (cb8c3851b897 /
# 68332ec7c31b); probe-3 = 95e24129958b self-consistent (draft-len changes
# verify batch shape -> fp16 tiling -> one near-tie flip; see README).
# Bake marker: /root/.llm_scaler_exp_v1211_baked
FROM llm-scaler-exp:v1.2.10

COPY neodl/ /tmp/neodl/
COPY patch_f15b.py patch_arstage.py patch_v58_p1.py serve_user.sh /tmp/bake/

RUN set -eux; \
    dpkg -i --force-depends /tmp/neodl/*.deb; \
    P=/opt/venv/bin/python3; \
    $P /tmp/bake/patch_f15b.py; \
    $P /tmp/bake/patch_arstage.py; \
    $P /tmp/bake/patch_v58_p1.py; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    grep -q 'llm-scaler f15b'  $SP/vllm/v1/worker/gpu_model_runner.py; \
    grep -q 'llm-scaler arstage' $SP/vllm/distributed/device_communicators/xpu_communicator.py; \
    grep 'num_speculative_tokens":3' /tmp/bake/serve_user.sh; \
    sed -i 's@logger\.debug(@logger.info(@' $SP/vllm/v1/worker/mamba_utils.py || true; \
    /opt/venv/bin/pip install -q py-spy; \
    cp /tmp/bake/serve_user.sh /root/serve_user.sh; \
    chmod +x /root/serve_user.sh; \
    rm -rf /tmp/neodl /tmp/bake; \
    find $SP/vllm -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    touch /root/.llm_scaler_exp_v1211_baked; \
    dpkg -l | grep -E 'intel-opencl-icd|libze-intel-gpu1'; \
    echo 'llm-scaler-exp:v1.2.11 bake OK (v1.2.10 + NEO 26.18 + f15b/arstage/v58p1 + draft-3 serve)'
