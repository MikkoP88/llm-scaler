#!/bin/bash
# legboot.sh TAG N — boot lsv-test (files assumed already staged) + probe.
set -u
TAG=$1; N=${2:-100}
B=/root/build
SP=/root/serve_lmc_v133.sh
V=/opt/venv/lib/python3.12/site-packages/vllm

echo "=== LEGBOOT START tag=$TAG n=$N $(date -u +%H:%M:%S)"
for p in $(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-test kill -TERM $p 2>/dev/null || true
done
sleep 14
for p in $(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-test kill -9 $p 2>/dev/null || true
done
sleep 4
docker exec lsv-test sh -c "find $V -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -f /tmp/lmc_off /tmp/lmc_snap_off /tmp/lmc_store_off; true"
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
python3 $B/nonce_probe.py $N > $B/nonce_${TAG}.out 2>&1
tail -n 2 $B/nonce_${TAG}.out
echo "degenerate=$(grep -c DEGENERATE $B/nonce_${TAG}.out || true)"
echo "slowwalls:"; grep -E "wall=[0-9]{2,}" $B/nonce_${TAG}.out || echo none
echo "=== LEGBOOT DONE tag=$TAG $(date -u +%H:%M:%S)"
