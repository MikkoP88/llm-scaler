# llm-scaler-vllm-adv:v38 - #18 FIX BAKED: fp8-KV decode OFF the ESIMD fast path.
# Base: certified v37 (all keeper patches baked; prod 4bit lane).
# Delta vs v37 = EXACTLY ONE change:
#   v38_esimd_reroute.py   exclude fp8* kv_cache_dtype lanes from the
#                          PAGED_ATTN_ESIMD gate -> decode routes to the vxk
#                          FA2 varlen kernel (deterministic, tail-invariant,
#                          fp32-ref-correct to 1.4e-5, AND faster:
#                          65k warm 28.24 vs 23.99, 2k 33.7 vs 24.0).
#                          ROOT CAUSE of #18: eagle_ops.page_attn_decode is
#                          run-to-run nondeterministic on fp8 KV (multistable
#                          discrete outputs, 1-2 fp16 ULP, ~100-478 elems;
#                          fp8@511 969/998 calls, fp8@1024 988/998, graph
#                          @512 998/998; latent fp16@512 5/998). A/B
#                          force-back: VLLM_XPU_ALLOW_ESIMD_F8=1.
# NOT baked: v38_exact_kv_splits.py — hypothesis FALSIFIED (split machinery
#            inactive at short ctx: wkb=2 < 16 -> heuristic num_splits=1;
#            pin changed nothing). Committed as evidence/tooling only.
# /root/.v38_baked marks the image so bootp.sh skips boot-time patching.
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
    touch /root/.v38_baked; \
    echo "v38 bake OK (fp8 decode -> vxk FA2; ESIMD gate intact for fp16/auto lanes)"
