#!/bin/bash
# watchdog_v2_drift_check.sh — v127 Layer 4: lane-config drift guard.
#
# The 2026-09-30 stall incident ran the lane OUTSIDE the boot chain with zero
# engine telemetry while health stayed 200 the whole time (STALL_ROOT_CAUSE.md
# Cause A). Health-only monitoring cannot catch that. This check can.
#
# Emits machine-greppable lines:
#   V1227_DRIFT_NONE              posture certified
#   DRIFT: <detail>               one line per violation
# Exit code: 0 always (WARN-only by design — never lets a monitor kill or
# restart the lane on drift; a human coordinates the restore window).
#
# Wire into lane_watchdog.sh inside its 60 s loop:
#   bash /root/build/v127_stage/watchdog_v2_drift_check.sh >> /root/build/lane_watchdog.log 2>&1
#
# POSTURE NOTE (updated 2026-10-10, P9b LMCACHE ERA): the standing surface is
# the LMCache-integrated lane — serve /root/serve_lmc_v133.sh (chain-rebased
# same day): LMCacheConnectorV1 kv_both ratio 2, --quantization fp8 ON the
# cmdline (the pre-P9b "flag = drift" rule is INVERTED now), mamba float16,
# kv fp8_e4m3, MTPx4, async, chat template flag, SAVE_DECODE_CACHE=True,
# LOG_LEVEL=WARNING, PYTHONHASHSEED=0. Image content v1.2.34 (a9f325d57e33).
# v1.2.33/v1.2.34 were docker commits of the LIVE container, so .Config.Image
# keeps the tag the container was CREATED from (v1.2.32) — the tag check is
# informational; the certified-CONTENT gates are the adapter md5 (dcb23fbe,
# v141 deployed on both adapter paths) and the gpu_model_runner md5
# (6520f916, v135 P10 queue-kick). Previous posture notes (v1.2.31 era:
# swift/ckpt/legacy branches, quantization-forbidden) retired with this
# rebase — see git history in the llm-scaler repo banks.
EXPECT_IMAGE="${V1227_EXPECT_IMAGE:-llm-scaler-exp:v1.2.35}"
# v141 adapter (double-unpin dedupe + per-step logging debug-demoted),
# deployed to BOTH the vllm connector path and the lmcache integration path
EXPECT_ADAPTER_MD5="${V1227_EXPECT_ADAPTER_MD5:-dcb23fbe0e8990a0b35d6367a8542f3f}"
EXPECT_ADAPTER_PATH="/opt/venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py"
# v135 P10: _p10_wait_counts queue-kick in the model runner (long-ctx
# chunked-prefill event-visibility stalls fixed 2026-10-10, baked v1.2.35)
EXPECT_GMR_MD5="${V1227_EXPECT_GMR_MD5:-6520f91686ac8db3d296595bb43c13e2}"
EXPECT_GMR_PATH="/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"
# fp16 SSM pool is certified (2.7-3.2x decode vs fp8 per the dtype law);
# flip via env only for an explicit fp8 leg window
EXPECT_SSM="${V1227_EXPECT_SSM:-float16}"
# 1 = spec MTPx4 standing posture; 0 = explicit nospec window
EXPECT_SPEC="${V1227_EXPECT_SPEC:-1}"
# 1 = --chat-template /root/chat_template_qwen38_high.jinja standing
EXPECT_TMPL="${V1227_EXPECT_TMPL:-1}"

if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
  echo "DRIFT: lsv-test container not running"
  exit 0
fi

IMG=$(docker inspect lsv-test --format '{{.Config.Image}}')
IMG_OK=0
if [ "$IMG" = "$EXPECT_IMAGE" ]; then
  IMG_OK=1
else
  # commit-of-running-container era: tag stays at the creation image —
  # acceptable ONLY while the adapter content md5 certifies the posture
  echo "DRIFT: image tag $IMG != $EXPECT_IMAGE (ok ONLY with certified adapter md5 below — commit-forward container)"
fi

# content gate (the real posture check in the commit-forward era)
ADAPTER_MD5=$(docker exec lsv-test md5sum $EXPECT_ADAPTER_PATH 2>/dev/null | awk '{print $1}')
ADAPTER_OK=0
if [ "$ADAPTER_MD5" = "$EXPECT_ADAPTER_MD5" ]; then
  ADAPTER_OK=1
else
  echo "DRIFT: adapter md5 '$ADAPTER_MD5' != $EXPECT_ADAPTER_MD5 (v141 certified content)"
fi

# content gate 2 (v135): P10 model-runner patch
GMR_MD5=$(docker exec lsv-test md5sum $EXPECT_GMR_PATH 2>/dev/null | awk '{print $1}')
GMR_OK=0
if [ "$GMR_MD5" = "$EXPECT_GMR_MD5" ]; then
  GMR_OK=1
else
  echo "DRIFT: gpu_model_runner md5 '$GMR_MD5' != $EXPECT_GMR_MD5 (v135 P10 certified content)"
fi

CM=$(docker exec lsv-test sh -c "ps -ef | grep 'vllm serve' | grep -v grep" 2>/dev/null)
if [ -z "$CM" ]; then
  echo "DRIFT: no vllm serve process found"
else
  echo "$CM" | grep -q -- "--model /models/target" || echo "DRIFT: serve model is not /models/target"
  echo "$CM" | grep -q -- "--served-model-name qwen3.8-27b-fp8" || echo "DRIFT: --served-model-name qwen3.8-27b-fp8 missing (litellm fleet mapping)"
  echo "$CM" | grep -q -- "--quantization fp8" || echo "DRIFT: --quantization fp8 missing (lmc-era surface carries the flag)"
  echo "$CM" | grep -q -- "LMCacheConnectorV1" || echo "DRIFT: LMCacheConnectorV1 missing from serve cmdline (lmc era)"
  echo "$CM" | grep -q -- "--async-scheduling" || echo "DRIFT: --async-scheduling missing (V1212-class posture)"
  echo "$CM" | grep -q -- "--kv-cache-dtype fp8_e4m3" || echo "DRIFT: fp8_e4m3 KV missing"
  echo "$CM" | grep -q -- "--mamba-ssm-cache-dtype $EXPECT_SSM" || echo "DRIFT: SSM pool dtype != $EXPECT_SSM"
  SPEC_OK=0
  if [ "$EXPECT_SPEC" = "1" ]; then
    echo "$CM" | grep -q "num_speculative_tokens.:4" && SPEC_OK=1 \
      || echo "DRIFT: spec MTPx4 missing (standing posture since P84)"
  else
    echo "$CM" | grep -q "num_speculative_tokens" \
      && echo "DRIFT: spec on the lane outside a nospec window" \
      || SPEC_OK=1
  fi
  TMPL_OK=0
  if [ "$EXPECT_TMPL" = "1" ]; then
    echo "$CM" | grep -q -- "--chat-template /root/chat_template_qwen38_high.jinja" && TMPL_OK=1 \
      || echo "DRIFT: --chat-template qwen38_high missing (standing posture since 2026-10-07)"
  else
    echo "$CM" | grep -q -- "--chat-template" \
      && echo "DRIFT: --chat-template override on serve cmdline (no-flag rollback window)" \
      || TMPL_OK=1
  fi
fi

# P84 async posture gate: async scheduling + zombie-strike tolerance are
# standing (strikes=3 couples to async — never ship it alone)
ASYNC_OK=0
if [ "${EXPECT_ASYNC:-1}" = "1" ]; then
  CENV=$(docker inspect lsv-test --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null)
  echo "$CENV" | grep -q '^VLLM_XPU_ALLOW_ASYNC=1$' \
    || echo "DRIFT: VLLM_XPU_ALLOW_ASYNC=1 missing (P84 async posture)"
  echo "$CENV" | grep -q '^VLLM_V60_ZOMBIE_STRIKES=3$' \
    || echo "DRIFT: VLLM_V60_ZOMBIE_STRIKES=3 missing (P84 wide-wave abort law)"
  if echo "$CENV" | grep -q '^VLLM_XPU_ALLOW_ASYNC=1$' \
     && echo "$CENV" | grep -q '^VLLM_V60_ZOMBIE_STRIKES=3$'; then
    ASYNC_OK=1
  fi
else
  ASYNC_OK=1
fi

# Telemetry heartbeat: the incident ran blind because serve stdout went to a
# dead interactive pts. serve_full.log must advance.
docker exec lsv-test sh -c 'test -s /root/serve_full.log' 2>/dev/null || echo "DRIFT: /root/serve_full.log empty (engine telemetry blind)"
docker exec lsv-test sh -c 'find /root/serve_full.log -mmin -10 | grep -q serve_full' 2>/dev/null || echo "DRIFT: /root/serve_full.log stale >10 min under traffic"

# Consolidated verdict: DRIFT lines above win; otherwise posture certified.
# Gate = content (adapter+GMR md5) + serve-cmdline floor + spec/tmpl/async +
# live telemetry. Image tag is NOT in the conjunction (commit-forward era).
if docker exec lsv-test sh -c 'test -s /root/serve_full.log' 2>/dev/null \
   && [ -n "$CM" ] \
   && [ "$ADAPTER_OK" = "1" ] \
   && [ "$GMR_OK" = "1" ] \
   && echo "$CM" | grep -q -- "--model /models/target" \
   && echo "$CM" | grep -q -- "--served-model-name qwen3.8-27b-fp8" \
   && echo "$CM" | grep -q -- "--quantization fp8" \
   && echo "$CM" | grep -q -- "LMCacheConnectorV1" \
   && echo "$CM" | grep -q -- "--async-scheduling" \
   && echo "$CM" | grep -q -- "--kv-cache-dtype fp8_e4m3" \
   && echo "$CM" | grep -q -- "--mamba-ssm-cache-dtype $EXPECT_SSM" \
   && [ "${SPEC_OK:-0}" = "1" ] \
   && [ "${ASYNC_OK:-1}" = "1" ] \
   && [ "${TMPL_OK:-1}" = "1" ]; then
  echo "V1227_DRIFT_NONE posture=certified-lmc-v135-spec-async-tmpl $(date -u +%H:%M:%S)"
fi
exit 0
