#!/bin/bash
# freshleg.sh N — fresh container from image + ONLY the 8 patched engine
# files from lsv-test + lsv-test's container env. Clean caches/pip.
# Isolates: the 8-file patch set vs container-state corruption cause.
set -u
N=${1:-100}
B=/root/build
V=/opt/venv/lib/python3.12/site-packages/vllm
SP=/root/serve_lmc_v133.sh

echo "=== FRESHLEG START n=$N $(date -u +%H:%M:%S)"

# env dump of lsv-test (the env all corrupted legs ran under)
docker inspect lsv-test --format "{{range .Config.Env}}{{println .}}{{end}}" \
  > $B/fresh_env.txt || { echo ENV_DUMP_FAIL; exit 1; }
wc -l < $B/fresh_env.txt

docker rm -f lsv-fresh >/dev/null 2>&1 || true
# shellcheck disable=SC2046
docker run -dit --name lsv-fresh --privileged --network host --ipc host \
  -v /dev/dri:/dev/dri \
  -v /models/dflash2:/models/dflash2:ro \
  -v /models/drafter-fp8-v5:/models/drafter:ro \
  -v /models/swift-qwen3.8-27b:/models/target:ro \
  -v /root/build/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro \
  --env-file $B/fresh_env.txt \
  -w /llm-scaler/vllm --entrypoint bash \
  llm-scaler-exp:v1.2.32 || { echo DOCKER_RUN_FAIL; exit 1; }
echo "lsv-fresh up (fresh caches, lsv-test env)"

# copy ONLY the 8 patched engine files (current lsv-test state = leg7:
# pristine core + patched worker -> use ORIGINAL patched copies banked
# earlier in pdiff/patched_*)
for f in mamba_utils.py gpu_model_runner.py utils.py block_pool.py \
         kv_cache_manager.py scheduler.py output.py \
         single_type_kv_cache_manager.py; do
  case $f in
    mamba_utils.py|gpu_model_runner.py|utils.py) d=v1/worker ;;
    output.py|scheduler.py) d=v1/core/sched ;;
    *) d=v1/core ;;
  esac
  docker cp $B/pdiff/patched_$f lsv-fresh:$V/$d/$f || { echo CP_FAIL $f; exit 1; }
done
echo "8 patched engine files installed"
docker exec lsv-fresh sh -c "find $V -name __pycache__ -exec rm -rf {} + 2>/dev/null; true"

# serve script: as-is (float16, no connector, pfx on, dbg off)
docker cp $SP lsv-fresh:/root/serve.sh
docker exec -d lsv-fresh bash /root/serve.sh
ok=0
for i in $(seq 1 110); do
  code=$(docker exec lsv-fresh sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; echo "HEALTH_OK tick=$i $(date -u +%H:%M:%S)"; break; }
  sleep 10
done
[ "$ok" = "1" ] || { echo HEALTH_TIMEOUT; exit 1; }

echo "--- nonce probe:"
python3 $B/nonce_probe.py 2 | tail -n 2
python3 $B/nonce_probe.py $N > $B/nonce_fresh8.out 2>&1
tail -n 2 $B/nonce_fresh8.out
echo "degenerate=$(grep -c DEGENERATE $B/nonce_fresh8.out || true)"
echo "slowwalls:"; grep -E "wall=[0-9]{2,}" $B/nonce_fresh8.out || echo none
echo "=== FRESHLEG DONE $(date -u +%H:%M:%S)"
