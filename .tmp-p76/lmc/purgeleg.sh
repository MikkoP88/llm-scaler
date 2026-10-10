#!/bin/bash
# purgeleg.sh N — lsv-test back to full-patch standing state + ALL compile
# caches purged. If corruption stops -> poisoned cache artifact root cause.
set -u
N=${1:-100}
B=/root/build
V=/opt/venv/lib/python3.12/site-packages/vllm
SP=/root/serve_lmc_v133.sh

echo "=== PURGELEG START n=$N $(date -u +%H:%M:%S)"

# stop lsv-fresh engine + container (frees port + VRAM)
for p in $(docker exec lsv-fresh pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-fresh kill -TERM $p 2>/dev/null || true
done
sleep 14
for p in $(docker exec lsv-fresh pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-fresh kill -9 $p 2>/dev/null || true
done
docker rm -f lsv-fresh >/dev/null 2>&1 || true
echo "lsv-fresh removed"

# stop lsv-test engine (dead already, but be safe)
for p in $(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-test kill -TERM $p 2>/dev/null || true
done
sleep 8
for p in $(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-test kill -9 $p 2>/dev/null || true
done

# restore full-patch standing state
for f in mamba_utils.py gpu_model_runner.py utils.py block_pool.py \
         kv_cache_manager.py scheduler.py output.py \
         single_type_kv_cache_manager.py; do
  case $f in
    mamba_utils.py|gpu_model_runner.py|utils.py) d=v1/worker ;;
    output.py|scheduler.py) d=v1/core/sched ;;
    *) d=v1/core ;;
  esac
  docker cp $B/pdiff/patched_$f lsv-test:$V/$d/$f || { echo CP_FAIL $f; exit 1; }
done
echo "8 patched files restored (standing state)"

# PURGE all caches + stale bytecode + sentinels + tmp uuid dirs
docker exec lsv-test sh -c "rm -rf /root/.triton /root/.cache /tmp/torchinductor_* /root/fresh_env.txt; find /tmp -maxdepth 1 -type d -regextype posix-extended -regex \"/tmp/[0-9a-f]{8}-[0-9a-f]{4}-.*\" -exec rm -rf {} + 2>/dev/null; rm -f /tmp/lmc_off /tmp/lmc_snap_off /tmp/lmc_store_off; find $V -name __pycache__ -exec rm -rf {} + 2>/dev/null; true"
echo "caches purged"

docker cp $SP lsv-test:$SP
docker exec -d lsv-test bash $SP
ok=0
for i in $(seq 1 110); do
  code=$(docker exec lsv-test sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; echo "HEALTH_OK tick=$i $(date -u +%H:%M:%S)"; break; }
  sleep 10
done
[ "$ok" = "1" ] || { echo HEALTH_TIMEOUT; exit 1; }

echo "--- nonce probe:"
python3 $B/nonce_probe.py 2 | tail -n 2
python3 $B/nonce_probe.py $N > $B/nonce_purged.out 2>&1
tail -n 2 $B/nonce_purged.out
echo "degenerate=$(grep -c DEGENERATE $B/nonce_purged.out || true)"
echo "slowwalls:"; grep -E "wall=[0-9]{2,}" $B/nonce_purged.out || echo none
echo "=== PURGELEG DONE $(date -u +%H:%M:%S)"
