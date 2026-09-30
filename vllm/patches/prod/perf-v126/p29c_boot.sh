#!/bin/bash
# p29c_boot.sh — v126 P29C lane boot: dual bridge + NATIVE fp8 ESIMD decode.
# Usage: p29c_boot.sh <fp8_e4m3|fp8_e5m2|float16> [port] [kv_dtype]
#   kv_dtype optional (default = baked fp8_e4m3); e5m2 full-format probe leg:
#     p29c_boot.sh fp8_e5m2 8000 fp8_e5m2
# Chain = boot_v126_test.sh (V1225 lineage on llm-scaler-exp:v1.2.25-raw)
#   + patch_v126_dualbridge.py (prefill/mixed bridge, P28-certified)
#   + patch_v126_poolpath.py  (fp8 SSM pools ride the ESIMD kernels natively)
#   + v131 .so (P29A fp8 register chain + P29B-FIX ring-row snapshot)
#   + p29b_check.py rev2 op-level gate BEFORE serve (clean GPUs)
# Hard gates: poolpath markers, .so sha e050551e..., IMPORT_OK, P29B PASS.
# Bisection switches (default all off): NO_POOLPATH=1, EAGER=1, ASYNC0=1,
#   PRODSO=1 (keep production .so — skip v131 swap + v131-specific gates).
set -eu
DTYPE="${1:?usage: p29c_boot.sh <fp8_e4m3|fp8_e5m2|float16> [port] [kv_dtype]}"
PORT="${2:-8000}"
KV_DTYPE="${3:-}"
TAG=$(echo "$DTYPE" | tr '._' '__')
KV_TAG=$(echo "${KV_DTYPE:-baked}" | tr '._' '__')
STAMP=$(date +%H%M%S)
LOGD=/root/build/lce1
mkdir -p "$LOGD"
PKG=/opt/venv/lib/python3.12/site-packages/custom_esimd_kernels_vllm
SO_NAME=custom_esimd_kernels_lgrf.cpython-312-x86_64-linux-gnu.so
SO_SHA_EXPECT=e050551ea29b729de4ab84af9f17f9b7d8f0ffcf80be302bb5fbc1bf46c7bea8

docker rm -f lsv-test >/dev/null 2>&1 || true

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
    -e VLLM_F15B=1 -e VLLM_F15B_STALL_S=45 -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
    -e VLLM_GUIDED_DECODING_BACKEND=xgrammar \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /models/drafter-fp8-v5:/models/drafter:ro \
    -v /models/dflash2:/models/dflash2:ro \
    -w /llm-scaler/vllm \
    --entrypoint /bin/bash \
    llm-scaler-exp:v1.2.25-raw

echo "BOOT_p29c_${TAG} container launched (v1.2.25-raw + bridge + poolpath + v131so) $(date +%H:%M:%S)"

# --- standing boot patcher chain (V1223+ lineage) ---------------------------
docker cp /root/build/patch_f15b.py lsv-test:/root/patch_f15b.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_f15b.py | tee "$LOGD/f15b_apply_p29c.log"
docker cp /root/build/patch_v55_3.py lsv-test:/root/patch_v55_3.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v55_3.py | tee "$LOGD/v55_3_apply_p29c.log"
docker exec lsv-test grep -c "llm-scaler v55.3" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker cp /root/build/patch_arstage.py lsv-test:/root/patch_arstage.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_arstage.py | tee "$LOGD/arstage_apply_p29c.log"
docker exec lsv-test grep -c "llm-scaler arstage" /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py
docker cp /root/build/patch_v58_p1.py lsv-test:/root/patch_v58_p1.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v58_p1.py | tee "$LOGD/v58p1_apply_p29c.log"
docker cp /root/build/patch_stall_v59.py lsv-test:/root/patch_stall_v59.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_stall_v59.py | tee "$LOGD/stall59_apply_p29c.log"
docker cp /root/build/patch_wedge_v60.py lsv-test:/root/patch_wedge_v60.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_wedge_v60.py | tee "$LOGD/v60_apply_p29c.log"
docker cp /root/build/patch_wedge_v62b_bake.py lsv-test:/root/patch_wedge_v62b_bake.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_wedge_v62b_bake.py --apply | tee "$LOGD/v62b_apply_p29c.log"
docker exec lsv-test grep -c "llm-scaler v62 (WEDGEFIX-G): mode dispatch" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
docker cp /root/build/patch_sched_v63_bake.py lsv-test:/root/patch_sched_v63_bake.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_sched_v63_bake.py --apply | tee "$LOGD/v63_apply_p29c.log"
docker exec lsv-test grep -c "llm-scaler v63 TTFTFIX" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
docker cp /root/build/patch_sched_v64_bake.py lsv-test:/root/patch_sched_v64_bake.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_sched_v64_bake.py --apply | tee "$LOGD/v64_apply_p29c.log"
docker exec lsv-test grep -c "llm-scaler v64 TTFTFIX-2" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
docker exec lsv-test pip install -q py-spy 2>/dev/null || true
docker exec lsv-test sed -i "s@logger\.debug(@logger.info(@" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py

# --- v126: dual-bridge root fix (prefill/mixed) ------------------------------
docker cp /root/build/patch_v126_dualbridge.py lsv-test:/root/patch_v126_dualbridge.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v126_dualbridge.py | tee "$LOGD/v126_dualbridge_apply_p29c_${TAG}.log"
docker exec lsv-test grep -c "llm-scaler v126 DUAL BRIDGE" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py
docker exec lsv-test grep -c "llm-scaler v126 C7'" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py

# --- v126 P29C: native fp8 decode path + v131 kernels ------------------------
# NO_POOLPATH=1 (bisection leg): skip the poolpath patch entirely — the lane
# then runs the P28-certified dual bridge for ALL fp8 decode (fp16 unaffected).
if [ "${NO_POOLPATH:-0}" = "1" ]; then
  echo "NO_POOLPATH=1: dual-bridge posture (P28 class), native ride DISABLED"
else
POOLPATCH="${POOLPATCH:-/root/build/patch_v126_poolpath.py}"
POOLPATCH_BASE=$(basename "$POOLPATCH")
docker cp "$POOLPATCH" "lsv-test:/root/$POOLPATCH_BASE"
docker exec lsv-test /opt/venv/bin/python3 "/root/$POOLPATCH_BASE" | tee "$LOGD/v126_poolpath_apply_p29c_${TAG}.log"
POOLPATH_MARKS=$(docker exec lsv-test grep -c "v126 P29C" /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py)
echo "poolpath markers=$POOLPATH_MARKS"
[ "$POOLPATH_MARKS" -ge 5 ] || { echo "P29C_POOLPATH_MARKERS_LOW"; exit 11; }
fi

# PRODSO=1 (bisection leg): keep the PRODUCTION .so shipped inside
# v1.2.25-raw — skips the v131 swap + its sha gate + the v131-specific
# P29B op gate. Convicts/acquits the v131 .so for the concurrent-bridge
# wedge with every other variable held constant.
if [ "${PRODSO:-0}" = "1" ]; then
  echo "PRODSO=1: v131 .so swap SKIPPED (production .so kept — wedge bisection leg)"
else
  docker cp /root/build/v131_so/inplace.so "lsv-test:$PKG/$SO_NAME"
fi
SO_SHA=$(docker exec lsv-test sha256sum "$PKG/$SO_NAME" | awk '{print $1}')
echo "so sha=$SO_SHA"
if [ "${PRODSO:-0}" != "1" ]; then
  [ "$SO_SHA" = "$SO_SHA_EXPECT" ] || { echo "P29C_SO_SHA_MISMATCH"; exit 12; }
fi
docker exec lsv-test /opt/venv/bin/python -c \
  "from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq, esimd_gdn_conv_fused_seq_spec; print('IMPORT_OK')"

# --- P29B rev2 op-level gate (serve DOWN, clean GPUs) ------------------------
# (v131-specific; skipped under PRODSO=1 — the production .so predates
# the P29B ring-row snapshot semantics this gate asserts)
if [ "${PRODSO:-0}" = "1" ]; then
  echo "PRODSO=1: P29B op gate SKIPPED (production .so leg)"
else
docker cp /root/build/p29b_check.py lsv-test:/root/p29b_check.py
set +e
docker exec lsv-test /opt/venv/bin/python /root/p29b_check.py > "$LOGD/p29b_p29c_${TAG}.log" 2>&1
P29B_RC=$?
set -e
tail -25 "$LOGD/p29b_p29c_${TAG}.log"
[ $P29B_RC -eq 0 ] || { echo "P29B_CHECK_FAILED rc=$P29B_RC"; exit 13; }
grep -q "P29B_VERDICT: PASS" "$LOGD/p29b_p29c_${TAG}.log" || { echo P29B_VERDICT_BAD; exit 14; }
echo "P29B_PASS_${TAG}"
fi

# --- serve config: ssm dtype (+ optional kv dtype) + port --------------------
docker exec lsv-test test -f /root/.llm_scaler_exp_v1225_baked || { echo "P29C MARKER MISSING (raw image?)"; exit 9; }
if docker exec lsv-test grep -q -- '--gpu-memory-utilization 0.8' /root/serve_user.sh \
   && docker exec lsv-test grep -q -- '--kv-cache-dtype fp8_e4m3' /root/serve_user.sh \
   && docker exec lsv-test grep -q -- '--async-scheduling' /root/serve_user.sh \
   && docker exec lsv-test grep -q -- "'{\"method\":\"mtp\",\"num_speculative_tokens\":4}'" /root/serve_user.sh; then
  echo "P29C baked config verified (gmu0.8 bs64 e4m3KV pc-ON async-ON mtpx4)"
else
  echo "P29C BAKED_SERVE_CONFIG_WRONG"; exit 10
fi
docker exec lsv-test sed -i "s@--mamba-ssm-cache-dtype float16@--mamba-ssm-cache-dtype $DTYPE@" /root/serve_user.sh
if [ -n "$KV_DTYPE" ]; then
  docker exec lsv-test sed -i "s@--kv-cache-dtype fp8_e4m3@--kv-cache-dtype $KV_DTYPE@" /root/serve_user.sh
fi
# EAGER=1 (bisection leg): swap async-scheduling for enforce-eager — no XPU
# graph capture/replay; isolates native-path-under-capture from plain run.
if [ "${EAGER:-0}" = "1" ]; then
  docker exec lsv-test sed -i "s@--async-scheduling@--enforce-eager@" /root/serve_user.sh
  echo "EAGER=1: async-scheduling -> enforce-eager (capture isolation leg)"
fi
# ASYNC0=1 (bisection leg): drop --async-scheduling, KEEP XPU graphs —
# splits graph-capture corruption from async-runner interplay.
if [ "${ASYNC0:-0}" = "1" ]; then
  docker exec lsv-test sed -i "s@ --async-scheduling@@" /root/serve_user.sh
  echo "ASYNC0=1: async-scheduling removed, graphs kept"
fi
docker exec lsv-test grep -- "--mamba-ssm-cache-dtype" /root/serve_user.sh
docker exec lsv-test grep -- "--kv-cache-dtype" /root/serve_user.sh
if [ "$PORT" != "8000" ]; then
  docker exec lsv-test sed -i "s@--port 8000@--port $PORT@" /root/serve_user.sh
fi
docker exec lsv-test sh -c "chmod +x /root/serve_user.sh && : > /root/serve_full.log"
# P29H=1 (diagnostic leg): in-situ native-vs-bridge mirror, activated by a
# MARKER FILE. Plain env inheritance between vLLM v1 process tiers is broken
# on this build (v4: serve pid 402 had V126_P29H=1, EngineCore/TP workers
# did NOT — neither docker exec -d -e nor an export inside serve_user.sh
# survives the EngineCore spawn). /root/.v126_p29h is seen by every process.
if [ "${P29H:-0}" = "1" ]; then
  docker exec lsv-test touch /root/.v126_p29h
  docker exec lsv-test ls -la /root/.v126_p29h
  docker exec -d lsv-test bash /root/serve_user.sh
  echo "P29H=1: in-situ native-vs-bridge diagnostic armed (/root/p29h_diag_<pid>.pt)"
else
  docker exec lsv-test rm -f /root/.v126_p29h
  docker exec -d lsv-test bash /root/serve_user.sh
fi
echo "BOOT_p29c_${TAG} serve started (ssm=$DTYPE kv=${KV_DTYPE:-baked} port=$PORT) $(date +%H:%M:%S)"

for i in $(seq 1 90); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 "http://localhost:$PORT/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "BOOT_p29c_${TAG} HEALTH_OK after ~$((i*10))s"
    R=$(dmesg | grep -ac 'Engine reset' || true)
    echo "live_resets=$R"
    sleep 3
    docker exec lsv-test grep -m1 "v126 DUAL_FP8_BRIDGE engaged" /root/serve_full.log || true
    docker exec lsv-test grep -c "JIT compilation during inference" /root/serve_full.log || true
    echo "NOTE: 'v126 P29C: fp8 SSM decode NATIVE ESIMD' marker fires on the FIRST spec decode — check serve_full.log during the probe"
    echo "V126_P29C_LANE_UP_${TAG}_${KV_TAG} $(date -u +%H:%M:%S)"
    exit 0
  fi
  docker ps --format '{{.Names}}' | grep -q '^lsv-test$' || { echo "P29C CONTAINER DIED"; docker logs --tail 40 lsv-test 2>&1 | tail -40; exit 7; }
done
echo "BOOT_p29c_${TAG} HEALTH TIMEOUT"; docker exec lsv-test tail -60 /root/serve_full.log 2>/dev/null || true; exit 8
