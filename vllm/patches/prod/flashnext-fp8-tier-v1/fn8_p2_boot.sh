#!/bin/bash
# fn8 P2 window boot (host-side, 10.20.3.65) — turnkey container + patches.
# Same shape as the P0/P1 fn8-g1 window: llm-scaler-exp:v1.2.35, host net,
# /dev/dri, /models ro, /root/fn8_stage -> /stage.
# Applies 00a/00b/00c/00d + patch_02a + patch_02b (idempotent markers;
# patch_01 NOT needed — no routing tap this window) and verifies the
# content gates before handing over to fn8_serve_p2.sh.
# PREREQ: corpus banked — /root/fn8_stage/bank/routing_hist.json present
# (the seed).  Usage: bash /root/fn8_stage/fn8_p2_boot.sh
set -e
NAME=fn8-p2
STAGE=/root/fn8_stage

[ -f "$STAGE/bank/routing_hist.json" ] || {
  echo "MISSING SEED $STAGE/bank/routing_hist.json — bank the corpus first"; exit 1; }
for f in patch_00a_qwen4exp_config.py patch_00b_qwen4exp_model.py \
         patch_00c_offload_repack_cache.py patch_00d_offload_forward_cache.py \
         patch_02a_tiered_experts.py patch_02b_uva_nowrap.py; do
  [ -f "$STAGE/$f" ] || { echo "MISSING $STAGE/$f"; exit 1; }
done

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --network host --device /dev/dri \
  -v /dev/dri:/dev/dri -v /models:/models:ro -v "$STAGE":/stage \
  --entrypoint /bin/bash llm-scaler-exp:v1.2.35 -lc "sleep infinity"

# apply in dependency order (idempotent; markers skip re-runs)
for p in patch_00a_qwen4exp_config patch_00b_qwen4exp_model \
         patch_00c_offload_repack_cache patch_00d_offload_forward_cache \
         patch_02a_tiered_experts patch_02b_uva_nowrap; do
  echo "== $p =="
  docker exec "$NAME" python3 "/stage/$p.py"
done

echo "== content gates =="
docker exec "$NAME" md5sum \
  /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py \
  /opt/venv/lib/python3.12/site-packages/vllm/model_executor/offloader/uva.py \
  | cut -c1-45
# expect modular_method a409e629... (Round 9 gate — only 02a touches it).
# uva.py md5 will NOT equal the t3 dry-run gate 98257355 here: patch_00d
# also edits uva.py (FN8_00D empty-cache block) before 02b — verify 02b by
# its FN8_02B marker count + boot-log lines instead.
docker exec "$NAME" grep -c FN8_02A \
  /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py
docker exec "$NAME" grep -c FN8_02B \
  /opt/venv/lib/python3.12/site-packages/vllm/model_executor/offloader/uva.py
echo "BOOT_OK — next: docker exec -d $NAME bash /stage/fn8_serve_p2.sh"
echo "(TP1 smoke first; for TP2 kill the serve tree, then"
echo " docker exec -d $NAME bash /stage/fn8_serve_p2t2.sh — venv carries,"
echo " patches are TP-shape independent)"
