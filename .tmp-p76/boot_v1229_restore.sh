#!/bin/bash
# boot_v1227_restore.sh — v127 Layer 1: RECERTIFY the lane on the
# user-designated posture (2026-09-30 directive) after the stall incident
# (STALL_ROOT_CAUSE.md) + knob legs for the CC-regime tuning round (Layer 2).
#
# DESIGNATED CERTIFIED POSTURE (user directive 2026-09-30):
#   model         /models/swift-qwen3.8-27b  (52 GB bf16 swift checkpoint)
#   quantization  --quantization fp8         (on-the-fly dynamic fp8)
#   template      --chat-template /root/chat_template_qwen38_high.jinja
#   served name   --served-model-name qwen3.8-27b-fp8
#                 (litellm local entries send "qwen3.8-27b-fp8" to :8000 —
#                 the fleet mapping MUST NOT change)
# Everything else stays certified v1226: image v1.2.26, full patcher chain,
# marker checks, async REQUIRED, spec MTPx4, XGrammar-2, fp8_e4m3 KV,
# fp16 SSM pool, barrier 0.
#
# What the 08:56 manual launch got WRONG is kept as law here: it ran OUTSIDE
# the boot chain (no patchers verified, no marker discipline) and BLIND
# (stdout to a dead pts — zero engine telemetry). This script fixes both:
# full certified chain + serve from a sed-generated /root/serve_user_v1227.sh
# with output >> /root/serve_full.log, then post-boot assertions that the
# designated posture is actually what is running.
#
# Postures (env V1227_POSTURE):
#   ckpt   (default) CERTIFIED since the v1.2.27 ship: pre-quantized fp8
#                     swift export mounted at /models/target, chat template
#                     baked into tokenizer_config.json, NO runtime
#                     --quantization/--chat-template flags (drift guard
#                     asserts their absence; = the script's else branch)
#   swift  explicit-legs-only PRE-certification designated posture:
#                     /models/swift bind + runtime quant/template flags
#   legacy            exact v1226 rollback (qwen3.8-27b-fp8 native ckpt, no
#                     quant/template flags) — one-command rollback window
#
# Knob legs (env; defaults = certified values): V1227_MNBT (8192),
# V1227_GMU (0.8), V1227_MAXSEQS (64), V1227_V63_BUDGET (unset = baked floor
# 1024), V1227_V66_STARVE (unset = baked 2.0 — VLLM_V66_PREFILL_STARVE_S
# passthrough for the D2 sweep leg), V1227_IMAGE (v1.2.27 since ship 10:35),
# V1227_PERREQ (0; 1 applies+arms the v128 D4 per-request split
# telemetry patch — see patch_perreq_v128.py),
# V1227_XGCOMPACT (0; 1 appends --structured-outputs-config
# {"backend":"xgrammar","disable_any_whitespace":true} to the serve
# cmdline — v128 WS-D fix candidate: guided-JSON grammars compiled
# whitespace-free, killing the whitespace-attractor degeneration class
# by construction; JSON output becomes compact),
# V1227_SCALEDSPACE (0; 1 arms the M1 scaled-space validation leg —
# requires v127_scales_e4m3.pt staged), V1227_M0LIVE (0; 1 applies+arms the
# M0-live runmax collector — pool stays fp16, harvest scales from traffic).
#
# NOTE swift boot is SLOW: 52 GB bf16 loads then quantizes on the fly
# (VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 is in the env below). Health wait is
# 25 min; boot duration + KV pool size are printed as certification evidence.
#
# WINDOW SCRIPT: stops/replaces the lsv-test container. Run ONLY in the
# maintenance window (standing directive: never stop the lane mid-fleet).
# usage: boot_v1229_restore.sh <MODE>
set -eu
MODE="${1:?usage: boot_v1229_restore.sh <MODE>}"
V1227_POSTURE="${V1227_POSTURE:-ckpt}"
# v1.2.29 (CRASHFIX-76): default posture NOSPEC — the in-image serve
# script ships without --speculative-config (B2 law: MTP deep-forward is
# the guc ccs crash trigger; nospec 2/2 clean at the death shape and
# 2.2x faster than the fastest spec leg on the standard shape). Spec
# stays SUPPORTED opt-in:
V1227_SPEC="${V1227_SPEC:-0}"
V1227_IMAGE="${V1227_IMAGE:-llm-scaler-exp:v1.2.29}"
V1227_MNBT="${V1227_MNBT:-8192}"
V1227_GMU="${V1227_GMU:-0.8}"
V1227_MAXSEQS="${V1227_MAXSEQS:-64}"
T0=$SECONDS

echo "BOOT_$MODE v129-recert posture=$V1227_POSTURE image=$V1227_IMAGE spec=$V1227_SPEC mnbt=$V1227_MNBT gmu=$V1227_GMU seqs=$V1227_MAXSEQS v63=${V1227_V63_BUDGET:-baked} v66s=${V1227_V66_STARVE:-baked} ss=${V1227_SCALEDSPACE:-0} m0live=${V1227_M0LIVE:-0} xgc=${V1227_XGCOMPACT:-0} perreq=${V1227_PERREQ:-0} $(date +%H:%M:%S)"

if [ "$V1227_POSTURE" = "swift" ]; then
  MODEL_MOUNT="-v /models/swift-qwen3.8-27b:/models/swift:ro"
  test -f /root/build/v127_stage/chat_template_qwen38_high.jinja \
    || { echo "BOOT_$MODE TEMPLATE MISSING (stage chat_template_qwen38_high.jinja to /root/build/v127_stage/)"; exit 6; }
else
  MODEL_MOUNT="-v /models/qwen3.8-27b-fp8:/models/target:ro"
fi

docker rm -f lsv-test >/dev/null 2>&1 || true

V63ENV=""
if [ -n "${V1227_V63_BUDGET:-}" ]; then
  V63ENV="-e V63_CONTENDED_BUDGET=${V1227_V63_BUDGET}"
fi

V66ENV=""
if [ -n "${V1227_V66_STARVE:-}" ]; then
  V66ENV="-e VLLM_V66_PREFILL_STARVE_S=${V1227_V66_STARVE}"
fi

docker run -td --privileged --name lsv-test \
    --device /dev/dri -v /dev/dri:/dev/dri --network host --ipc host \
    -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e CCL_TOPO_P2P_ACCESS=1 \
    -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
    -e CCL_ENABLE_SYCL_KERNELS=1 \
    -e CCL_ZE_IPC_EXCHANGE=pidfd \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e CCL_SYCL_ALLREDUCE_TMP_BUF=0 \
    -e ZE_AFFINITY_MASK=0,1 \
    -e VLLM_ALLOW_LONG_MODEL_LEN=1 \
    -e VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
    -e VLLM_USE_AOT_COMPILE=0 \
    -e CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 \
    -e VLLM_USE_V2_MODEL_RUNNER=0 \
    -e CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_QUANTIZE_Q40_LIB=/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e PYTORCH_XPU_ALLOC_CONF=expandable_segments:True \
    -e VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1 \
    -e VLLM_V55_EVENT_TIMEOUT_S=600 \
    -e VLLM_V55_DECODE_TIMEOUT_S=30 -e VLLM_XPU_SPEC_DRAFT_BARRIER=0 -e VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX=0 -e VLLM_RPC_TIMEOUT=60000 \
    -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
    -e VLLM_GUIDED_DECODING_BACKEND=xgrammar \
    $V63ENV \
    $V66ENV \
    $MODEL_MOUNT \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    "$V1227_IMAGE"

if [ "$V1227_POSTURE" = "swift" ]; then
  # template ships with the stage dir, NOT with the image (verified: absent
  # from v1.2.26) — the 08:56 launch had it only inside its own container
  docker cp /root/build/v127_stage/chat_template_qwen38_high.jinja lsv-test:/root/chat_template_qwen38_high.jinja
fi

nohup bash /root/build/mem_growth_logger.sh /root/build/lce1/bootV1212_memgrowth.log > /dev/null 2>&1 < /dev/null &

# --- v127 WS-B telemetry re-arm (host-side, read-only; idempotent so a
#     running recorder/scope from a previous window is never double-spawned;
#     recorder is error-tolerant while :8000 is down = boot itself recorded)
mkdir -p /root/build/telemetry /root/build/captures
if ! pgrep -f metrics_recorder.py >/dev/null; then
  nohup python3 /root/build/v127_stage/metrics_recorder.py >> /root/build/telemetry/recorder.log 2>&1 < /dev/null &
fi
if ! pgrep -f stall_scope.sh >/dev/null; then
  nohup /root/build/v127_stage/stall_scope.sh --watch >> /root/build/captures/scope_watch.log 2>&1 < /dev/null &
fi
echo "BOOT_$MODE telemetry armed (recorder 10s JSONL + stall_scope watch)"

# --- certified boot patcher chain (verbatim v1226 sequence) ---
docker exec lsv-test grep -c "llm-scaler v55.3" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker cp /root/build/patch_arstage.py lsv-test:/root/patch_arstage.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_arstage.py | tee /root/build/lce1/arstage_apply_bootV1227.log
docker exec lsv-test grep -c "llm-scaler arstage" /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py
docker cp /root/build/patch_stall_v59.py lsv-test:/root/patch_stall_v59.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_stall_v59.py | tee /root/build/lce1/stall59_apply_bootV1227.log
docker cp /root/build/patch_wedge_v60.py lsv-test:/root/patch_wedge_v60.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_wedge_v60.py | tee /root/build/lce1/v60_apply_bootV1227.log
docker cp /root/build/patch_wedge_v62b_bake.py lsv-test:/root/patch_wedge_v62b_bake.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_wedge_v62b_bake.py --apply | tee /root/build/lce1/v62b_apply_bootV1227.log
docker exec lsv-test grep -c "llm-scaler v62 (WEDGEFIX-G): mode dispatch" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker cp /root/build/patch_sched_v63_bake.py lsv-test:/root/patch_sched_v63_bake.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_sched_v63_bake.py --apply | tee /root/build/lce1/v63_apply_bootV1227.log
docker exec lsv-test grep -c "llm-scaler v63 TTFTFIX" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
docker cp /root/build/patch_sched_v64_bake.py lsv-test:/root/patch_sched_v64_bake.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_sched_v64_bake.py --apply | tee /root/build/lce1/v64_apply_bootV1227.log
docker exec lsv-test grep -c "llm-scaler v64 TTFTFIX-2" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
docker exec lsv-test pip install -q py-spy 2>/dev/null || true
docker exec lsv-test sed -i "s@logger\.debug(@logger.info(@" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py

# --- optional M1 scaled-space validation leg (dormant patch unless armed) ---
if [ "${V1227_SCALEDSPACE:-0}" = "1" ]; then
  docker cp /root/build/v127_stage/patch_scaled_space_v127.py lsv-test:/root/patch_scaled_space_v127.py
  docker exec lsv-test /opt/venv/bin/python3 /root/patch_scaled_space_v127.py | tee /root/build/lce1/v127_ss_apply.log
  docker exec lsv-test grep -c "llm-scaler v127 SCALEDSPACE" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py
  test -f /root/build/v127_stage/v127_scales_e4m3.pt || { echo "BOOT_$MODE V1227_SCALEDSPACE scales missing"; exit 11; }
  docker cp /root/build/v127_stage/v127_scales_e4m3.pt lsv-test:/root/v127_scales_e4m3.pt
fi

# --- optional M0-LIVE scale-collection leg (P29M-style runmax telemetry;
#     pool STAYS fp16 — we harvest live fp16 magnitudes -> e4m3 scales;
#     marker armed BEFORE serve start; harvest = merge m0live_runmax_*.pt +
#     calibrate_offline.py --stats. Never combine with SCALEDSPACE=1 in one
#     boot: separate windows per the one-knob-per-leg law) ---
if [ "${V1227_M0LIVE:-0}" = "1" ]; then
  docker cp /root/build/v127_stage/m0_live_collect_v127.py lsv-test:/root/m0_live_collect_v127.py
  docker exec lsv-test /opt/venv/bin/python3 /root/m0_live_collect_v127.py | tee /root/build/lce1/v127_m0live_apply.log
  docker exec lsv-test grep -c "llm-scaler v127 M0LIVE-COLLECTOR" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py
  docker exec lsv-test touch /root/.v127_m0live
  docker exec lsv-test rm -f /root/m0live_runmax_*.pt
fi

# --- optional v128 PERREQ leg (WS-D D4 per-request split telemetry;
#     dormant patch unless /root/.v128_perreq armed; the frontend appends
#     one JSON line per finished request to /root/v128_perreq.jsonl;
#     harvest = docker cp out after the leg, analyze with d4_split_v128) ---
if [ "${V1227_PERREQ:-0}" = "1" ]; then
  docker cp /root/build/v128_stage/patch_perreq_v128.py lsv-test:/root/patch_perreq_v128.py
  docker exec lsv-test /opt/venv/bin/python3 /root/patch_perreq_v128.py | tee /root/build/lce1/v128_perreq_apply.log
  docker exec lsv-test grep -c "llm-scaler v128 PERREQ" /opt/venv/lib/python3.12/site-packages/vllm/v1/engine/output_processor.py
  docker exec lsv-test touch /root/.v128_perreq
  docker exec lsv-test rm -f /root/v128_perreq.jsonl
fi

# --- baked-marker + dualbridge verification (v1226 certified) ---
docker exec lsv-test test -f /root/.llm_scaler_exp_v1226_baked || { echo "BOOT_$MODE MARKER MISSING"; exit 9; }
DBN=$(docker exec lsv-test sh -c 'grep -c "llm-scaler v126 DUAL BRIDGE" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py' | tr -d "[:space:]")
[ "$DBN" -ge 1 ] 2>/dev/null || { echo "BOOT V1226 DUALBRIDGE MISSING got=$DBN"; exit 9; }

# --- generate the v1227 serve copy: knob legs + posture flags ---
docker exec lsv-test sh -c "sed -e 's/--max-num-batched-tokens 8192/--max-num-batched-tokens $V1227_MNBT/' -e 's/--gpu-memory-utilization 0.8/--gpu-memory-utilization $V1227_GMU/' -e 's/--max-num-seqs 64/--max-num-seqs $V1227_MAXSEQS/' /root/serve_user.sh > /root/serve_user_v1227.sh"
if [ "$V1227_POSTURE" = "swift" ]; then
  docker exec lsv-test sed -i "s|--model /models/target|--model /models/swift --served-model-name qwen3.8-27b-fp8 --quantization fp8 --chat-template /root/chat_template_qwen38_high.jinja|" /root/serve_user_v1227.sh
fi
if [ "${V1227_SCALEDSPACE:-0}" = "1" ]; then
  docker exec lsv-test sed -i "s/--mamba-ssm-cache-dtype float16/--mamba-ssm-cache-dtype fp8_e4m3/" /root/serve_user_v1227.sh
fi

# --- optional v128 XGCOMPACT leg (WS-D engine-side fix candidate;
#     see fix_restore_v128b.py / xg_flag_probe_v128.py) ---
if [ "${V1227_XGCOMPACT:-0}" = "1" ]; then
  docker exec lsv-test sed -i "s|--async-scheduling|--structured-outputs-config '{\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}' --async-scheduling|" /root/serve_user_v1227.sh
  docker exec lsv-test grep -q -F -- "--structured-outputs-config '{\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}'" /root/serve_user_v1227.sh \
    || { echo "BOOT_$MODE XGCOMPACT_FLAG_MISSING"; exit 18; }
fi

# --- v1.2.29 SPEC knob: 0 (default) = certified NOSPEC posture — the
#     image ships without --speculative-config and the generated serve
#     script must stay without it (B2 law: MTP deep-forward = guc ccs
#     crash trigger). 1 = opt-in spec leg: re-append MTPx4 (hazard
#     documented in P76_GUC_CCS_UPSTREAM_HANDOFF.md) and assert live. ---
if [ "${V1227_SPEC:-0}" = "1" ]; then
  docker exec lsv-test sed -i "s|--async-scheduling|--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}' --async-scheduling|" /root/serve_user_v1227.sh
  docker exec lsv-test grep -q -F -- "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}'" /root/serve_user_v1227.sh \
    || { echo "BOOT_$MODE SPEC_FLAG_MISSING"; exit 19; }
else
  docker exec lsv-test grep -q -- "--speculative-config" /root/serve_user_v1227.sh \
    && { echo "BOOT_$MODE NOSPEC_DRIFT spec flag present"; exit 19; } || true
fi
docker exec lsv-test grep -q -- "--kv-cache-dtype fp8_e4m3" /root/serve_user_v1227.sh \
  && docker exec lsv-test grep -q -- "--enable-prefix-caching" /root/serve_user_v1227.sh \
  && docker exec lsv-test grep -q -- "--async-scheduling" /root/serve_user_v1227.sh \
  && docker exec lsv-test grep -q -- "--tool-call-parser qwen3_coder" /root/serve_user_v1227.sh \
  || { echo "BOOT_$MODE BAKED_SERVE_CONFIG_WRONG"; exit 10; }
if [ "$V1227_POSTURE" = "swift" ]; then
  docker exec lsv-test grep -q -- "--model /models/swift" /root/serve_user_v1227.sh \
    && docker exec lsv-test grep -q -- "--served-model-name qwen3.8-27b-fp8" /root/serve_user_v1227.sh \
    && docker exec lsv-test grep -q -- "--quantization fp8" /root/serve_user_v1227.sh \
    && docker exec lsv-test grep -q -- "--chat-template /root/chat_template_qwen38_high.jinja" /root/serve_user_v1227.sh \
    || { echo "BOOT_$MODE SWIFT_POSTURE_FLAGS_MISSING"; exit 10; }
fi
docker exec lsv-test sh -c 'chmod +x /root/serve_user_v1227.sh && : > /root/serve_full.log'
docker exec -d lsv-test bash /root/serve_user_v1227.sh
echo "BOOT_$MODE serve started (posture=$V1227_POSTURE, log=/root/serve_full.log) $(date +%H:%M:%S)"

# swift = 52 GB bf16 + on-the-fly fp8 quant -> boot is minutes slower than
# the native-fp8 ckpt; 25 min ceiling
for i in $(seq 1 150); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "BOOT_$MODE HEALTH_OK after ~$((i*10))s (script elapsed ${SECONDS}s)"
    # --- certification evidence: pool size + concurrency under this posture ---
    docker exec lsv-test grep -m1 "GPU KV cache size" /root/serve_full.log || true
    docker exec lsv-test grep -m1 "Maximum concurrency" /root/serve_full.log || true
    # --- posture assertions (what 08:56 got wrong can never recur silently) ---
    CM=$(docker exec lsv-test sh -c "ps -ef | grep 'vllm serve' | grep -v grep")
    if [ "$V1227_POSTURE" = "swift" ]; then
      echo "$CM" | grep -q -- "--model /models/swift" || { echo "BOOT_$MODE DRIFT model"; exit 12; }
      echo "$CM" | grep -q -- "--quantization fp8" || { echo "BOOT_$MODE DRIFT quantization missing"; exit 13; }
      echo "$CM" | grep -q -- "--chat-template /root/chat_template_qwen38_high.jinja" || { echo "BOOT_$MODE DRIFT chat-template missing"; exit 14; }
      echo "$CM" | grep -q -- "--served-model-name qwen3.8-27b-fp8" || { echo "BOOT_$MODE DRIFT served-model-name missing"; exit 15; }
    else
      echo "$CM" | grep -q -- "--model /models/target" || { echo "BOOT_$MODE DRIFT model"; exit 12; }
      echo "$CM" | grep -q "swift" && { echo "BOOT_$MODE DRIFT swift model"; exit 13; } || true
      echo "$CM" | grep -q -- "--quantization" && { echo "BOOT_$MODE DRIFT --quantization"; exit 14; } || true
      echo "$CM" | grep -q -- "--chat-template" && { echo "BOOT_$MODE DRIFT --chat-template"; exit 15; } || true
    fi
    docker exec lsv-test test -s /root/serve_full.log || { echo "BOOT_$MODE DRIFT serve log empty"; exit 16; }
    # --- v1.2.29 spec posture assert on the LIVE cmdline ---
    if [ "${V1227_SPEC:-0}" = "1" ]; then
      echo "$CM" | grep -q -- "--speculative-config" \
        || { echo "BOOT_$MODE DRIFT spec flag not live"; exit 17; }
    else
      echo "$CM" | grep -q -- "--speculative-config" \
        && { echo "BOOT_$MODE DRIFT speculative-config live in nospec posture"; exit 17; } || true
    fi
    echo "BOOT_$MODE posture $V1227_POSTURE verified on live cmdline, telemetry live"
    echo "BOOT_V1227_RESTORED posture=$V1227_POSTURE $(date -u +%H:%M:%S)"
    exit 0
  fi
  docker ps --format '{{.Names}}' | grep -q '^lsv-test$' || { echo "BOOT_$MODE CONTAINER DIED"; exit 7; }
done
echo "BOOT_$MODE HEALTH TIMEOUT"; exit 8
