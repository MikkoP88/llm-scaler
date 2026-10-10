#!/bin/bash
# llm-scaler v137t LEG A (race-window bisect): connector OFF + LMC_PJ=1
# (EngineCore journal) + LMC_BOUND_STATS=0 / LMC_BOUND_MD5=0 (worker-side
# boundary stats pause REMOVED — the prime race-window perturbation) +
# LMC_DBG_FWD=1 (read-side NaN detector). Hypothesis: removing the stats
# pause re-opens the async-overlap race window -> corruption at filler~21-31
# WITH full journal + detector active.
SP=/root/serve_lmc_v133.sh
echo "=== TELEM_LEG_A START $(date -u +%H:%M:%S)"
restore() {
  docker exec lsv-test cp $SP.bak_a $SP
  echo -n "KVFLAG_RESTORED_COUNT="
  docker exec lsv-test grep -c "kv-transfer-config" $SP || true
}
trap restore EXIT
docker exec lsv-test cp $SP $SP.bak_a
docker exec lsv-test sed -i "57d" $SP
docker exec lsv-test sed -i "/export LMC_DBG_FWD=1/a export LMC_PJ=1\nexport LMC_BOUND_STATS=0\nexport LMC_BOUND_MD5=0" $SP
C1=$(docker exec lsv-test grep -c "kv-transfer-config" $SP)
C2=$(docker exec lsv-test grep -c "export LMC_PJ=1" $SP)
C3=$(docker exec lsv-test grep -c "LMC_BOUND_STATS=0" $SP)
echo "KVFLAG=$C1 PJ=$C2 STATS0=$C3"
if [ "$C1" != "0" ] || [ "$C2" != "1" ] || [ "$C3" != "1" ]; then echo "SED_ABORT"; exit 1; fi

PIDS=$(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  for p in $PIDS; do docker exec lsv-test kill -TERM "$p" 2>/dev/null || true; done
  sleep 14
fi
PIDS=$(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo "TERM_LEFTOVERS=$PIDS -> kill -9"
  for p in $PIDS; do docker exec lsv-test kill -9 "$p" 2>/dev/null || true; done
  sleep 6
fi
docker exec lsv-test rm -f /tmp/lmc_off /tmp/lmc_snap_off /tmp/lmc_store_off
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

cd /root/build || exit 1
python3 probe_evict.py
echo "PY_EXIT=$?"
echo "--- bank serve log:"
docker exec lsv-test cp /root/serve_full.log /tmp/telem_a.log
docker cp lsv-test:/tmp/telem_a.log /root/build/telem_a_serve.log
md5sum /root/build/telem_a_serve.log
echo "--- LMCNANDETECT count + first 6:"
docker exec lsv-test sh -c "grep -ac LMCNANDETECT /root/serve_full.log || true; grep -a LMCNANDETECT /root/serve_full.log | head -6"
echo "--- LMCPJ ANOM count + all (head 40):"
docker exec lsv-test sh -c "grep -ac \"LMCPJ.*ANOM\" /root/serve_full.log || true; grep -a \"LMCPJ.*ANOM\" /root/serve_full.log | head -40"
echo "--- POISONED_ROWS:"
docker exec lsv-test sh -c "grep -ac POISONED-ROW /root/serve_full.log || true"
echo "--- TRACEBACKS:"
docker exec lsv-test sh -c "grep -ac Traceback /root/serve_full.log || true"
echo "=== TELEM_LEG_A DONE $(date -u +%H:%M:%S)"
