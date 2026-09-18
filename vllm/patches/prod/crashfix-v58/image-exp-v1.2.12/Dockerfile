# llm-scaler-exp:v1.2.12 — CRASH-AVOIDANCE production image (async OFF).
# Base  = llm-scaler-exp:v1.2.10 (STANDING runtime: NEO 26.14 + pidfd ENV;
#         the exact tree the §24-L G8NS battery validated — NOT v1.2.11,
#         which carries the NEO 26.18 decode tax, §21)
# Bake  = validated crash-avoidance serve config ONLY (thin layer):
#         e4m3 KV + --block-size 64 + --gpu-memory-utilization 0.8
#         + --async-scheduling REMOVED (overlap scheduler off).
# Evidence (§24 K/L, host 10.20.3.65, §20 F-bar batteries):
#   30/30 CLEAN 4h21m44s resets=0 (longest crash-free battery ever);
#   f8ref EXACT e4m3 set 52f598e7d38a/6b1c26403bfc/95e24129958b;
#   q17 solo 54.8 / conc4 153.7 (fastest solo ever; parity w/ async ON).
#   Ladder: async ON crashes 12m-1h13m at every gmu; async OFF crashes
#   at gmu 0.9 (36m58s, #15) and 0.85 (3h59m21s, #16); clean at 0.8 and
#   0.7. Threshold in (0.80, 0.85]. 16/16 events 'LR job cleanup'.
# Bake marker: /root/.llm_scaler_exp_v1212_baked
FROM llm-scaler-exp:v1.2.10

COPY serve_user.sh /tmp/bake_serve_user.sh

RUN set -eux; \
    test "$(grep -c 'async-scheduling' /tmp/bake_serve_user.sh)" = "0"; \
    grep -q -- '--block-size 64' /tmp/bake_serve_user.sh; \
    grep -q -- '--gpu-memory-utilization 0.8' /tmp/bake_serve_user.sh; \
    grep -q -- '--kv-cache-dtype fp8_e4m3' /tmp/bake_serve_user.sh; \
    cp /tmp/bake_serve_user.sh /root/serve_user.sh; \
    chmod +x /root/serve_user.sh; \
    rm -f /tmp/bake_serve_user.sh; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    find $SP/vllm -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    touch /root/.llm_scaler_exp_v1212_baked; \
    echo 'llm-scaler-exp:v1.2.12 bake OK (v1.2.10 standing + async-OFF crash-avoidance serve: e4m3 bs64 gmu0.8)'
