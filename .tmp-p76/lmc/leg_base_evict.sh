#!/bin/bash
# llm-scaler v136g BASELINE discriminator: remove the kv-transfer flag
# entirely (serve line 57 deleted; backup/restore) so the stock fork boots
# with NO LMCache connector, then run the identical probe_evict.py hammer.
# If filler~21 also corrupts here, the bug is a stock fork eviction hazard,
# not LMCache. Acceptance: all 96 fillers + A2 coherent, 0 non-finite.
SP=/root/serve_lmc_v133.sh
echo "=== BASE_EVICT_LEG START $(date -u +%H:%M:%S)"
restore() {
  docker exec lsv-test cp $SP.bak_base $SP
  echo -n "KVFLAG_RESTORED_COUNT="
  docker exec lsv-test grep -c "kv-transfer-config" $SP
}
trap restore EXIT
docker exec lsv-test cp $SP $SP.bak_base
docker exec lsv-test sed -i "57d" $SP
CNT=$(docker exec lsv-test grep -c "kv-transfer-config" $SP)
echo "KVFLAG_COUNT_AFTER_SED=$CNT"
if [ "$CNT" != "0" ]; then echo "SED_HIT_WRONG_LINE_ABORT"; exit 1; fi

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
echo "--- NONFINITE(total_fwd_lines, nonfinite_lines):"
docker exec lsv-test sh -c "grep -ac LMCDbgFwd /root/serve_full.log; grep -aE 'n[1-9][0-9]*/a' /root/serve_full.log | grep -ac LMCDbgFwd || true"
echo "--- TRACEBACKS:"
docker exec lsv-test sh -c "grep -ac Traceback /root/serve_full.log || true"
echo "=== BASE_EVICT_LEG DONE $(date -u +%H:%M:%S)"
