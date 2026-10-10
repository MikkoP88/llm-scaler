#!/bin/bash
# leg_p9img.sh — P9 baseline leg: PRISTINE llm-scaler-exp:v1.2.32 container
# (no patches, no LMCache), mamba float16, native engine. Bench suite:
# bench_prefix x3 + bench_run7n x2. Banks under bench_img32_*. Ends with
# pristine engine stopped + container removed (port/VRAM back for lsv-test).
set -u
B=/root/build

echo "=== LEG_P9IMG START $(date -u +%H:%M:%S)"

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

# 3) derive pristine serve (strip LMC env + PYTHONHASHSEED, float16, no conn)
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
docker exec -d lsv-pristine bash -c "bash /root/serve.sh > /root/serve_full.log 2>&1"
echo "serve launched; cold-compile boot, waiting..."

ok=0
for i in $(seq 1 110); do
  code=$(docker exec lsv-pristine sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; echo "HEALTH_OK tick=$i $(date -u +%H:%M:%S)"; break; }
  sleep 10
done
[ "$ok" = "1" ] || { echo HEALTH_TIMEOUT; docker exec lsv-pristine sh -c "tail -20 /root/serve_full.log"; exit 1; }

# 4) bench suite
cd $B || exit 1
echo "--- prefix pass1 (warm-up) start $(date -u +%H:%M:%S)"
python3 bench_prefix.py > $B/bench_img32_prefix_p1.out 2>&1
echo "P1_EXIT=$? $(date -u +%H:%M:%S)"
echo "--- prefix pass2 (record 1) start $(date -u +%H:%M:%S)"
python3 bench_prefix.py > $B/bench_img32_prefix_p2.out 2>&1
echo "P2_EXIT=$? $(date -u +%H:%M:%S)"
echo "--- prefix pass3 (record 2) start $(date -u +%H:%M:%S)"
python3 bench_prefix.py > $B/bench_img32_prefix_p3.out 2>&1
echo "P3_EXIT=$? $(date -u +%H:%M:%S)"
echo "--- run7n x2 start $(date -u +%H:%M:%S)"
python3 bench_run7n.py > $B/bench_img32_run7n_1.out 2>&1
echo "R1_EXIT=$? $(date -u +%H:%M:%S)"
python3 bench_run7n.py > $B/bench_img32_run7n_2.out 2>&1
echo "R2_EXIT=$? $(date -u +%H:%M:%S)"

# 5) bank + teardown
docker exec lsv-pristine cp /root/serve_full.log /tmp/telem_img.log
docker cp lsv-pristine:/tmp/telem_img.log $B/telem_bench_img32_serve.log >/dev/null
md5sum $B/telem_bench_img32_serve.log
echo "--- prefix p2/p3 reports:"
tail -12 $B/bench_img32_prefix_p2.out
tail -12 $B/bench_img32_prefix_p3.out
echo "--- run7n reports:"
cat $B/bench_img32_run7n_1.out
cat $B/bench_img32_run7n_2.out

for p in $(docker exec lsv-pristine pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-pristine kill -TERM $p 2>/dev/null || true
done
sleep 14
for p in $(docker exec lsv-pristine pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-pristine kill -9 $p 2>/dev/null || true
done
docker rm -f lsv-pristine >/dev/null 2>&1 || true
echo "pristine engine stopped, container removed"
echo "=== LEG_P9IMG DONE $(date -u +%H:%M:%S)"
