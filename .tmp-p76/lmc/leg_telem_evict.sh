#!/bin/bash
# llm-scaler v137t TELEMETRY leg: connector OFF (serve line 57 deleted) +
# LMC_PJ=1 (block lifecycle journal) + LMC_DBG_FWD=1 (read-side hashes +
# LMCNANDETECT) + LMCBOUND write-side stats. Reproduces the deterministic
# eviction-pressure NaN corruption under full instrumentation and banks the
# serve log for post-hoc page-lifecycle analysis. Restore via trap.
SP=/root/serve_lmc_v133.sh
echo "=== TELEM_EVICT_LEG START $(date -u +%H:%M:%S)"
restore() {
  docker exec lsv-test cp $SP.bak_telem $SP
  echo -n "KVFLAG_RESTORED_COUNT="
  docker exec lsv-test grep -c "kv-transfer-config" $SP || true
  echo -n "PJFLAG_RESTORED_COUNT="
  docker exec lsv-test grep -c "export LMC_PJ=1" $SP || true
}
trap restore EXIT
docker exec lsv-test cp $SP $SP.bak_telem
docker exec lsv-test sed -i "57d" $SP
docker exec lsv-test sed -i "/export LMC_DBG_FWD=1/a export LMC_PJ=1" $SP
C1=$(docker exec lsv-test grep -c "kv-transfer-config" $SP)
C2=$(docker exec lsv-test grep -c "export LMC_PJ=1" $SP)
echo "KVFLAG_COUNT_AFTER_SED=$C1 PJFLAG_COUNT_AFTER_SED=$C2"
if [ "$C1" != "0" ] || [ "$C2" != "1" ]; then echo "SED_ABORT"; exit 1; fi

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
docker exec lsv-test sh -c "grep -ao \"mamba_ssm_cache_dtype[^,]*\" /root/serve_full.log | head -1"
echo "--- LMCPOOL (worker-side pool truth):"
docker exec lsv-test sh -c "grep -a LMCPOOL /root/serve_full.log | sort -u | head -10"

cd /root/build || exit 1
python3 probe_evict.py
echo "PY_EXIT=$?"
echo "--- bank serve log for post-hoc:"
docker exec lsv-test cp /root/serve_full.log /tmp/telem_serve.log
docker cp lsv-test:/tmp/telem_serve.log /root/build/telem_evict_serve.log
md5sum /root/build/telem_evict_serve.log
echo "--- LMCNANDETECT count + first 6:"
docker exec lsv-test sh -c "grep -ac LMCNANDETECT /root/serve_full.log || true; grep -a LMCNANDETECT /root/serve_full.log | head -6"
echo "--- LMCPJ ANOM count + all (head 40):"
docker exec lsv-test sh -c "grep -ac \"LMCPJ.*ANOM\" /root/serve_full.log || true; grep -a \"LMCPJ.*ANOM\" /root/serve_full.log | head -40"
echo "--- LMCBOUND nonfinite (n[1-9]):"
docker exec lsv-test sh -c "grep -a LMCBOUND /root/serve_full.log | grep -acE \"/n[1-9]\" || true"
echo "--- LMCBOUND total:"
docker exec lsv-test sh -c "grep -ac LMCBOUND /root/serve_full.log || true"
echo "--- POISONED_ROWS:"
docker exec lsv-test sh -c "grep -ac POISONED-ROW /root/serve_full.log || true"
echo "--- TRACEBACKS:"
docker exec lsv-test sh -c "grep -ac Traceback /root/serve_full.log || true"
echo "=== TELEM_EVICT_LEG DONE $(date -u +%H:%M:%S)"
