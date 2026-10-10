#!/bin/bash
# pristine.sh N — leg 3: clean-room container from baked image llm-scaler-exp:v1.2.32
# (NO v133-v139 patches, NO LMCache). Baseline env + baseline serve shape
# (mamba float16 per user's original baseline). Stop lsv-test engine first
# (frees VRAM + port 8000), health-wait, nonce probe N, bank output.
set -u
N=${1:-40}
B=/root/build

echo "=== PRISTINE LEG START n=$N $(date -u +%H:%M:%S)"

# 1) stop lsv-test engine tree (container stays)
PIDS=$(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  for p in $PIDS; do docker exec lsv-test kill -TERM "$p" 2>/dev/null || true; done
  sleep 14
fi
PIDS=$(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo "TERM_LEFTOVERS=$PIDS"
  for p in $PIDS; do docker exec lsv-test kill -9 "$p" 2>/dev/null || true; done
  sleep 6
fi
PIDS=$(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null || true)
[ -n "$PIDS" ] && { echo "ENGINE_STILL_ALIVE=$PIDS ABORT"; exit 1; }
echo "lsv-test engine tree: dead (container up)"

# 2) fresh pristine container
docker rm -f lsv-pristine >/dev/null 2>&1 || true
docker run -dit --name lsv-pristine --privileged --network host --ipc host \
  -v /dev/dri:/dev/dri \
  -v /models/dflash2:/models/dflash2:ro \
  -v /models/drafter-fp8-v5:/models/drafter:ro \
  -v /models/swift-qwen3.8-27b:/models/target:ro \
  -v /root/build/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro \
  -w /llm-scaler/vllm --entrypoint bash \
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
  -e VLLM_V55_DECODE_TIMEOUT_S=30 \
  -e VLLM_XPU_SPEC_DRAFT_BARRIER=0 \
  -e VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX=0 \
  -e VLLM_RPC_TIMEOUT=60000 \
  -e VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 \
  -e VLLM_GUIDED_DECODING_BACKEND=xgrammar \
  -e VLLM_XPU_ALLOW_ASYNC=1 \
  -e VLLM_V60_ZOMBIE_STRIKES=3 \
  llm-scaler-exp:v1.2.32 || { echo DOCKER_RUN_FAIL; exit 1; }
echo "lsv-pristine container up"
echo "--- pristine image mamba dtype support:"
docker exec lsv-pristine sh -c "grep -rn \"mamba_ssm_cache_dtype\" /opt/venv/lib/python3.12/site-packages/vllm/config/ 2>/dev/null | head -5" || true

# 3) derive pristine serve script: strip ALL LMC/LMC_DBG/PYTHONHASHSEED env,
#    mamba dtype float16, assert no connector
python3 - /root/serve_lmc_v133.sh $B/pristine_serve.sh <<'PYEOF'
import sys
src, dst = sys.argv[1:3]
out = []
for l in open(src).read().split("\n"):
    ls = l.strip()
    if ls.startswith("export ") and (
        ls.split()[1].split("=")[0] in
        ("PYTHONHASHSEED",)
        or ls.split()[1].split("=")[0].startswith(("LMCACHE_", "LMC_DBG_"))
    ):
        continue
    if "--mamba-ssm-cache-dtype" in l:
        l = "  --mamba-ssm-cache-dtype float16 \\"
    out.append(l)
txt = "\n".join(out)
assert "kv-transfer-config" not in txt
assert "LMCACHE_" not in txt and "LMC_DBG" not in txt
open(dst, "w").write(txt)
print("SERVE_DERIVED_OK lines=%d" % len(out))
PYEOF
[ $? -ne 0 ] && { echo DERIVE_ABORT; exit 1; }
grep -n "mamba-ssm-cache-dtype" $B/pristine_serve.sh

docker cp $B/pristine_serve.sh lsv-pristine:/root/serve.sh

# 4) launch serve
docker exec -d lsv-pristine bash /root/serve.sh
echo "serve launched; cold-compile boot, waiting..."

ok=0
for i in $(seq 1 110); do
  code=$(docker exec lsv-pristine sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; echo "HEALTH_OK tick=$i $(date -u +%H:%M:%S)"; break; }
  sleep 10
done
[ "$ok" = "1" ] || { echo HEALTH_TIMEOUT; docker exec lsv-pristine sh -c "pgrep -af vllm | head; ps aux | grep -c python"; exit 1; }

# 5) probe
echo "--- nonce probe (warm-up 2 + record N):"
python3 $B/nonce_probe.py 2 | tail -n 3
python3 $B/nonce_probe.py $N > $B/nonce_pristine.out 2>&1
tail -n 3 $B/nonce_pristine.out
echo "degenerate=$(grep -c DEGENERATE $B/nonce_pristine.out || true)"
echo "slowwalls:"; grep -E "wall=[0-9]{2,}" $B/nonce_pristine.out || echo none
echo "=== PRISTINE LEG DONE $(date -u +%H:%M:%S)"
