#!/bin/bash
# watchdog_v2_drift_check.sh — v127 Layer 4: lane-config drift guard.
#
# The 2026-09-30 stall incident ran the lane OUTSIDE the boot chain with zero
# engine telemetry while health stayed 200 the whole time (STALL_ROOT_CAUSE.md
# Cause A). Health-only monitoring cannot catch that. This check can.
#
# POSTURE NOTE (updated 2026-10-01 after the v1.2.26/v1.2.27 ships): the
# designated posture (user directive 2026-09-30: swift-qwen3.8-27b +
# --quantization fp8 + --chat-template chat_template_qwen38_high.jinja +
# --served-model-name qwen3.8-27b-fp8) was CERTIFIED by W1 and then FOLDED
# INTO THE CHECKPOINT by the bakes: the swift export was converted to a
# pre-quantized fp8 checkpoint (no runtime --quantization flag — the ship
# gates assert the flag's ABSENCE) with the chat template baked into
# tokenizer_config.json (no --chat-template flag — the baked_serve_config
# 'no-template' gate), mounted at /models/target (bind source
# qwen3.8-27b-fp8). Drift now means: serving anything OTHER than that
# certified surface (a runtime override flag is itself drift), or losing
# telemetry again.
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
# POSTURE NOTE (updated 2026-10-07 after the P84 ship): production =
# llm-scaler-exp:v1.2.31 (P83 guard + P83b via-pool + v52k parallel mamba
# carry baked) with ASYNC SCHEDULING STANDING (VLLM_XPU_ALLOW_ASYNC=1 +
# VLLM_V60_ZOMBIE_STRIKES=3; certified 2026-10-07: storms 3/3 ok=108/108
# @1211/1216/1236s, decode 27-28 tok/s/req con-4 deep vs 13.7-16.1 sync,
# solo 28.1; strikes=3 is the wide-wave abort law — the v60c poisoned-row
# guard strikes on transient async-overlap empty rows at the
# final-prefill-chunk->decode transition, strikes are consecutive so 3
# absorbs the 1-2 step lag; NEVER ship strikes=3 without async). SPEC
# MTPx4 is the standing posture since the user's live spec window was
# folded in (set V1227_EXPECT_SPEC=0 only for an explicit nospec window).
# POSTURE NOTE (updated 2026-10-07 later same day, user order): the
# high-reasoning qwen38 chat template is now served as an EXPLICIT runtime
# flag -- --chat-template /root/chat_template_qwen38_high.jinja on the
# serve cmdline, docker-cp'd from the stage dir by every chain boot
# (V1227_TMPL knob, default 1). The ckpt tokenizer keeps its baked
# chat_template (drift-relevant as the wrong-ckpt tell); the flag is the
# designated surface now (set V1227_EXPECT_TMPL=0 for a no-flag rollback
# window).
EXPECT_IMAGE="${V1227_EXPECT_IMAGE:-llm-scaler-exp:v1.2.31}"
EXPECT_POSTURE="${V1227_EXPECT_POSTURE:-swift}"
# fp16 SSM pool is certified; scaled-space validation legs flip it — tell the
# guard when a leg window is active instead of crying drift
EXPECT_SSM="${V1227_EXPECT_SSM:-float16}"
# 1 = spec MTPx4 standing posture (P84; the user's live spec window folded
# into production); 0 = explicit nospec window (chain V1227_SPEC=0 boot)
EXPECT_SPEC="${V1227_EXPECT_SPEC:-1}"
# 1 = --chat-template /root/chat_template_qwen38_high.jinja standing on
# the serve cmdline (2026-10-07 user order); 0 = no-flag rollback window
EXPECT_TMPL="${V1227_EXPECT_TMPL:-1}"
# certified checkpoint (pre-quantized fp8 swift export, template baked in),
# host-side path = the bind source of the /models/target mount
CKPT_DIR="${V1227_CKPT_DIR:-/models/qwen3.8-27b-fp8}"

if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
  echo "DRIFT: lsv-test container not running"
  exit 0
fi

IMG=$(docker inspect lsv-test --format '{{.Config.Image}}')
[ "$IMG" = "$EXPECT_IMAGE" ] || echo "DRIFT: image $IMG != $EXPECT_IMAGE"

BINDS=$(docker inspect lsv-test --format '{{json .HostConfig.Binds}}')
if [ "$EXPECT_POSTURE" = "swift" ]; then
  echo "$BINDS" | grep -q "qwen3.8-27b-fp8:/models/target" \
    || echo "DRIFT: certified fp8 swift ckpt not mounted at /models/target"
else
  echo "$BINDS" | grep -q "qwen3.8-27b-fp8:/models/target" || echo "DRIFT: legacy posture without the certified fp8 ckpt mount"
fi

CM=$(docker exec lsv-test sh -c "ps -ef | grep 'vllm serve' | grep -v grep" 2>/dev/null)
if [ -z "$CM" ]; then
  echo "DRIFT: no vllm serve process found"
else
  if [ "$EXPECT_POSTURE" = "swift" ]; then
    echo "$CM" | grep -q -- "--model /models/target" || echo "DRIFT: serve model is not /models/target (certified ckpt mount)"
    echo "$CM" | grep -q -- "--served-model-name qwen3.8-27b-fp8" || echo "DRIFT: --served-model-name qwen3.8-27b-fp8 missing (litellm fleet mapping)"
    # fp8 lives INSIDE the checkpoint (bake gates assert flag ABSENCE) —
    # a runtime override flag is itself drift (template moved to the
    # EXPECT_TMPL gate below, standing flag since 2026-10-07):
    echo "$CM" | grep -q -- "--quantization" && echo "DRIFT: --quantization override on serve cmdline (certified ckpt is pre-quantized)"
  else
    echo "$CM" | grep -q -- "--model /models/target" || echo "DRIFT: serve model is not the certified /models/target mount"
    echo "$CM" | grep -q "swift" && echo "DRIFT: swift model on the lane outside a swift-posture window"
    echo "$CM" | grep -q -- "--quantization" && echo "DRIFT: --quantization override on serve cmdline"
  fi
  # posture-independent certified floor (both postures)
  echo "$CM" | grep -q -- "--async-scheduling" || echo "DRIFT: --async-scheduling missing (V1212-class posture)"
  echo "$CM" | grep -q -- "--kv-cache-dtype fp8_e4m3" || echo "DRIFT: fp8_e4m3 KV missing"
  echo "$CM" | grep -q -- "--mamba-ssm-cache-dtype $EXPECT_SSM" || echo "DRIFT: SSM pool dtype != $EXPECT_SSM"
  # spec posture gate (inverted 2026-10-06): nospec is the certified
  # default; spec only on a sanctioned leg window
  SPEC_OK=0
  if [ "$EXPECT_SPEC" = "1" ]; then
    echo "$CM" | grep -q "num_speculative_tokens.:4" && SPEC_OK=1 \
      || echo "DRIFT: spec MTPx4 missing (leg window declared via V1227_EXPECT_SPEC=1)"
  else
    echo "$CM" | grep -q "num_speculative_tokens" \
      && echo "DRIFT: spec on the lane outside a leg window (nospec default since v1.2.29)" \
      || SPEC_OK=1
  fi
  # TMPL posture gate (2026-10-07 user order): the high-reasoning qwen38
  # template serves as an EXPLICIT runtime flag, standing default (the
  # chain's V1227_TMPL=1 sed injects it next to --async-scheduling)
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
# standing since the v1.2.31 ship (certified 2026-10-07: storms 3/3 clean,
# decode 2.3x; strikes=3 couples to async — never ship it alone)
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

# the checkpoint tokenizer keeps its BAKED chat_template (wrong-ckpt tell);
# since 2026-10-07 the user-designated serving surface is the runtime flag
# (chain docker-cp's the stage template into the container each boot) —
# that half moved to the TMPL_OK gate above; this half stays ckpt forensics
if [ "$EXPECT_POSTURE" = "swift" ]; then
  grep -q chat_template "$CKPT_DIR/tokenizer_config.json" 2>/dev/null \
    || echo "DRIFT: chat_template missing from $CKPT_DIR/tokenizer_config.json (wrong ckpt mounted?)"
fi

# Telemetry heartbeat: the incident ran blind because serve stdout went to a
# dead interactive pts. serve_full.log must advance.
docker exec lsv-test sh -c 'test -s /root/serve_full.log' 2>/dev/null || echo "DRIFT: /root/serve_full.log empty (engine telemetry blind)"
docker exec lsv-test sh -c 'find /root/serve_full.log -mmin -10 | grep -q serve_full' 2>/dev/null || echo "DRIFT: /root/serve_full.log stale >10 min under traffic"

# Consolidated verdict: DRIFT lines above win; otherwise posture certified.
TMPLTAG=""
[ "$EXPECT_TMPL" = "1" ] && TMPLTAG="-tmpl"
if docker exec lsv-test sh -c 'test -s /root/serve_full.log' 2>/dev/null \
   && [ -n "$CM" ] \
   && [ "$IMG" = "$EXPECT_IMAGE" ] \
   && echo "$CM" | grep -q -- "--async-scheduling" \
   && echo "$CM" | grep -q -- "--kv-cache-dtype fp8_e4m3" \
   && echo "$CM" | grep -q -- "--mamba-ssm-cache-dtype $EXPECT_SSM" \
   && [ "${SPEC_OK:-0}" = "1" ] \
   && [ "${ASYNC_OK:-1}" = "1" ] \
   && [ "${TMPL_OK:-1}" = "1" ]; then
  if [ "$EXPECT_POSTURE" = "swift" ] \
     && echo "$BINDS" | grep -q "qwen3.8-27b-fp8:/models/target" \
     && echo "$CM" | grep -q -- "--model /models/target" \
     && ! echo "$CM" | grep -q -- "--quantization" \
     && [ "${TMPL_OK:-0}" = "1" ] \
     && echo "$CM" | grep -q -- "--served-model-name qwen3.8-27b-fp8" \
     && grep -q chat_template "$CKPT_DIR/tokenizer_config.json" 2>/dev/null; then
    echo "V1227_DRIFT_NONE posture=certified-swift-fp8-spec-async${TMPLTAG} $(date -u +%H:%M:%S)"
  elif [ "$EXPECT_POSTURE" = "legacy" ] \
     && echo "$BINDS" | grep -q "qwen3.8-27b-fp8:/models/target" \
     && echo "$CM" | grep -q -- "--model /models/target" \
     && ! echo "$CM" | grep -q "swift" \
     && ! echo "$CM" | grep -q -- "--quantization" \
     && ! echo "$CM" | grep -q -- "--chat-template"; then
    echo "V1227_DRIFT_NONE posture=legacy $(date -u +%H:%M:%S)"
  fi
fi
exit 0
