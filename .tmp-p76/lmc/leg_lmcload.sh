#!/bin/bash
# llm-scaler v136g LOAD-PATH validation: boot with prefix caching OFF
# (native cold) so warm resends exercise the full LMCache external load:
# attn KV chunks + mamba boundary seed. Serve flag restored on exit.
SP=/root/serve_lmc_v133.sh
echo "=== LMCLOAD_LEG START $(date -u +%H:%M:%S)"
restore() {
  docker exec lsv-test sh -c "grep -q -- --enable-prefix-caching $SP || sed -i 's/ --trust-remote-code/ --enable-prefix-caching --trust-remote-code/' $SP"
  echo -n "PREFIX_FLAG_RESTORED_COUNT="
  docker exec lsv-test grep -c -- "--enable-prefix-caching" $SP
}
trap restore EXIT
docker exec lsv-test sed -i "s/ --enable-prefix-caching//" $SP
echo -n "PREFIX_FLAG_COUNT_AFTER_SED="
docker exec lsv-test grep -c -- "--enable-prefix-caching" $SP

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
docker exec lsv-test sh -c "grep -ao \"enable_prefix_caching.: .[A-Za-z]*\" /root/serve_full.log | head -1"

cd /root/build || exit 1
python3 probe_lmcload.py
echo "--- MARKERS(parked, seedwrite, loadcomplete, mismatch, retrievefail):"
docker exec lsv-test sh -c "grep -ac \"LMC mamba seed parked\" /root/serve_full.log; grep -ac \"LMC seed write\" /root/serve_full.log; grep -ac \"LMC async load complete\" /root/serve_full.log; grep -ac \"LMC seed mismatch\" /root/serve_full.log; grep -ac \"MAMBA SEED RETRIEVE FAILED\" /root/serve_full.log"
echo "--- NONFINITE(total_fwd_lines, nonfinite_lines):"
docker exec lsv-test sh -c "grep -ac LMCDbgFwd /root/serve_full.log; grep -aE 'n[1-9][0-9]*/a' /root/serve_full.log | grep -ac LMCDbgFwd || true"
echo "--- TRACEBACKS:"
docker exec lsv-test sh -c "grep -ac Traceback /root/serve_full.log || true"
echo "=== LMCLOAD_LEG DONE $(date -u +%H:%M:%S)"
