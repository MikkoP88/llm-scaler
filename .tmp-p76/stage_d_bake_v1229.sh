#!/bin/bash
# stage_d_bake_v1229.sh — CRASHFIX-76 Stage D: bake llm-scaler-exp:v1.2.29
# from the live certified-clean v1.2.28 container. PYTHON-ONLY delta (law:
# .so sha unchanged): default posture becomes NOSPEC — strip
# --speculative-config from the IN-IMAGE /root/serve_user.sh. Spec stays
# SUPPORTED opt-in via the rebased chain knob V1227_SPEC=1 (B2 law: MTP
# deep-forward = the guc ccs crash trigger; nospec 2/2 clean at the death
# shape AND 2.2x faster than the fastest spec leg on the standard shape).
set -eu
MODE="${1:?usage: stage_d_bake_v1229.sh <MODE>}"

# 0) provenance: live lane must be the certified v1.2.28 surface
docker ps --format '{{.Names}} {{.Image}}' | grep -q '^lsv-test llm-scaler-exp:v1.2.28$' \
  || { echo "BAKE_$MODE WRONG_BASE_CONTAINER"; exit 30; }
docker exec lsv-test test -f /root/.llm_scaler_exp_v1226_baked \
  || { echo "BAKE_$MODE MARKER_MISSING"; exit 30; }

# 1) the single delta: default posture = nospec
PRE=$(docker exec lsv-test sh -c 'grep -c -- "--speculative-config" /root/serve_user.sh || true')
[ "$PRE" = "1" ] || { echo "BAKE_$MODE SPEC_LINE_UNEXPECTED count=$PRE"; exit 31; }
docker exec lsv-test sed -i '/--speculative-config/d' /root/serve_user.sh
POST=$(docker exec lsv-test sh -c 'grep -c -- "--speculative-config" /root/serve_user.sh || true')
[ "$POST" = "0" ] || { echo "BAKE_$MODE SPEC_LINE_STILL_THERE count=$POST"; exit 31; }

# certified flags intact after the strip
docker exec lsv-test grep -q -- '--kv-cache-dtype fp8_e4m3' /root/serve_user.sh \
  && docker exec lsv-test grep -q -- '--enable-prefix-caching' /root/serve_user.sh \
  && docker exec lsv-test grep -q -- '--async-scheduling' /root/serve_user.sh \
  && docker exec lsv-test grep -q -- '--tool-call-parser qwen3_coder' /root/serve_user.sh \
  && docker exec lsv-test grep -q -- '--reasoning-parser qwen3' /root/serve_user.sh \
  || { echo "BAKE_$MODE CERTIFIED_FLAGS_BROKEN"; exit 32; }

# 2) SHIP-MEASUREMENT LAW: python-only bake, .so byte-identical
docker exec lsv-test sha256sum \
  /opt/venv/lib/python3.12/site-packages/custom_esimd_kernels_vllm/custom_esimd_kernels_lgrf.cpython-312-x86_64-linux-gnu.so \
  | grep -q '^1d9dcf4e7a1c8db6e94b0673c51f58935d03de359dc109c41bbac6df00f85dff' \
  || { echo "BAKE_$MODE SO_SHA_DRIFT"; exit 33; }

# 3) commit
docker commit lsv-test llm-scaler-exp:v1.2.29 >/dev/null
docker images llm-scaler-exp:v1.2.29 --format 'BAKE_V1229_IMAGE {{.ID}} {{.Size}}'
docker run --rm --entrypoint sh llm-scaler-exp:v1.2.29 -c \
  'grep -c -- "--speculative-config" /root/serve_user.sh || true; grep -q -- "--async-scheduling" /root/serve_user.sh && echo V1229_SURFACE_OK'
echo "BAKE_V1229_DONE mode=$MODE $(date +%H:%M:%S)"
