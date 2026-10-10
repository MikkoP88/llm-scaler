#!/bin/bash
# llm-scaler v136g LOAD-PATH validation via eviction pressure at the exact
# production boot shape (prefix caching ON). probe_evict.py banks A, cycles
# the GPU KV pool with 96 distinct fillers (LRU-evicting A natively), then
# resends A: the external load (attn KV + mamba seed) must fire and the
# output must match byte-for-byte.
SP=/root/serve_lmc_v133.sh
echo "=== EVICT_LEG START $(date -u +%H:%M:%S)"
CNT=$(docker exec lsv-test grep -c -- "--enable-prefix-caching" $SP)
echo "PREFIX_FLAG_COUNT=$CNT"
if [ "$CNT" != "1" ]; then echo "PREFIX_FLAG_MISSING_ABORT"; exit 1; fi

PIDS=$(docker exec lsv-test pgrep -f "vllm serve|VLLM::" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  for p in $PIDS; do docker exec lsv-test kill -TERM "$p" 2>/dev/null || true; done
  sleep 14
fi
PIDS=$(docker exec lsv-test pgrep -f "vllm serve|VLLM::" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo "TERM_LEFTOVERS=$PIDS -> kill -9"
  for p in $PIDS; do docker exec lsv-test kill -9 "$p" 2>/dev/null || true; done
  sleep 6
fi
docker exec lsv-test rm -f /tmp/lmc_off
docker exec -d lsv-test bash $SP

ok=0
for i in $(seq 1 60); do
  code=$(docker exec lsv-test sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  if [ "$code" = "200" ]; then ok=1; echo "HEALTH_OK tick=$i"; break; fi
  sleep 10
done
if [ "$ok" != "1" ]; then
  echo "HEALTH_TIMEOUT"
  docker exec lsv-test tail -12 /root/serve_full.log
  exit 1
fi
docker exec lsv-test grep -ao "mamba_ssm_cache_dtype.: .[a-z0-9_]*" /root/serve_full.log | head -1

cd /root/build || exit 1
python3 probe_evict.py
echo "--- MARKERS(parked, seedwrite, loadcomplete, mismatch, retrievefail, jointlookups):"
docker exec lsv-test sh -c "grep -ac \"LMC mamba seed parked\" /root/serve_full.log; grep -ac \"LMC seed write\" /root/serve_full.log; grep -ac \"LMC async load complete\" /root/serve_full.log; grep -ac \"LMC seed mismatch\" /root/serve_full.log; grep -ac \"MAMBA SEED RETRIEVE FAILED\" /root/serve_full.log; grep -ac \"LMC joint lookup\" /root/serve_full.log"
echo "--- LAST_JOINT:"
docker exec lsv-test sh -c "grep -a \"LMC joint lookup\" /root/serve_full.log | tail -3"
echo "--- TRACEBACKS:"
docker exec lsv-test sh -c "grep -ac Traceback /root/serve_full.log || true"
echo "=== EVICT_LEG DONE $(date -u +%H:%M:%S)"
