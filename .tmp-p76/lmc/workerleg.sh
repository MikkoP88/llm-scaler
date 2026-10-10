#!/bin/bash
# workerleg.sh FILES N — restore pristine worker/engine files into lsv-test
# from /root/build/pdiff/pristine_<name>, reboot (float16/no-conn/pfx-on/dbg-off),
# nonce probe N. FILES = space-separated basenames, e.g. "mamba_utils.py"
# or "mamba_utils.py gpu_model_runner.py utils.py".
set -u
FILES=$1; N=${2:-100}
SP=/root/serve_lmc_v133.sh
B=/root/build
V=/opt/venv/lib/python3.12/site-packages/vllm
TAG="w_$(echo $FILES | tr " ." "__")"

echo "=== WORKERLEG START files=[$FILES] n=$N $(date -u +%H:%M:%S)"

DIRMAP='v1/worker/mamba_utils.py v1/worker/gpu_model_runner.py v1/worker/utils.py v1/core/block_pool.py v1/core/kv_cache_manager.py v1/core/sched/scheduler.py v1/core/sched/output.py v1/core/single_type_kv_cache_manager.py'
for f in $FILES; do
  hit=""
  for d in $DIRMAP; do
    [ "$(basename $d)" = "$f" ] && hit=$d
  done
  [ -z "$hit" ] && { echo "UNKNOWN_FILE $f"; exit 1; }
  docker cp $B/pdiff/pristine_$f lsv-test:$V/$hit || { echo CP_FAIL $f; exit 1; }
  echo "RESTORED $hit"
done
# drop stale bytecode so the restore actually takes effect
docker exec lsv-test sh -c "find $V -name __pycache__ -path "*v1*" -exec rm -rf {} + 2>/dev/null; find $V/distributed -name __pycache__ -exec rm -rf {} + 2>/dev/null; true"
echo "pycache cleared"

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
docker exec lsv-test rm -f /tmp/lmc_off /tmp/lmc_snap_off /tmp/lmc_store_off

docker cp $SP lsv-test:$SP
docker exec -d lsv-test bash $SP
ok=0
for i in $(seq 1 60); do
  code=$(docker exec lsv-test sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; echo "HEALTH_OK tick=$i"; break; }
  sleep 10
done
[ "$ok" = "1" ] || { echo HEALTH_TIMEOUT; exit 1; }

echo "--- nonce probe:"
python3 $B/nonce_probe.py 2 | tail -n 2
python3 $B/nonce_probe.py $N > $B/nonce_${TAG}.out 2>&1
tail -n 2 $B/nonce_${TAG}.out
echo "degenerate=$(grep -c DEGENERATE $B/nonce_${TAG}.out || true)"
echo "slowwalls:"; grep -E "wall=[0-9]{2,}" $B/nonce_${TAG}.out || echo none
echo "=== WORKERLEG DONE tag=$TAG $(date -u +%H:%M:%S)"
