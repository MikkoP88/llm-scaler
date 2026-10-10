#!/bin/bash
# llm-scaler leg_conn2: EXACT leg_conn conditions (connector ON, no PJ,
# natural probe, fp8_e4m3 mamba, v138b+c live) + py-spy STALL WATCHDOG:
# workers froze at 21:06:56 mid wait_for_save (last log = mamba-save,
# next op slot_mapping.to(device) H2D) with no dmesg reset; engine died
# at +300s RPC timeout. This leg dumps frozen worker stacks live.
SP=/root/serve_lmc_v133.sh
echo "=== LEG_CONN2 START $(date -u +%H:%M:%S)"
C1=$(docker exec lsv-test grep -c "kv-transfer-config" $SP)
C2=$(docker exec lsv-test grep -c "export LMC_PJ" $SP)
echo "KVFLAG=$C1 (want 1) PJ=$C2 (want 0)"
if [ "$C1" != "1" ] || [ "$C2" != "0" ]; then echo "STATE_ABORT"; exit 1; fi

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
rm -f /root/build/pyspy_stall.txt

# --- py-spy stall watchdog (host side) ---
NDUMPS=0
(
  while [ -f /root/build/leg_conn2.running ]; do
    sleep 15
    MT=$(docker exec lsv-test stat -c %Y /root/serve_full.log 2>/dev/null || echo 0)
    STALE=$(( $(date +%s) - MT ))
    if [ "$STALE" -ge 90 ] && [ "$STALE" -le 420 ]; then
      WP=$(docker exec lsv-test pgrep -f "VLLM::Worker_T[P]" 2>/dev/null || true)
      {
        echo "=== STALE=${STALE}s $(date -u +%H:%M:%S) workers=$WP"
        for p in $WP; do
          docker exec lsv-test /opt/venv/bin/py-spy dump --pid "$p" 2>&1
          echo "--- end dump pid=$p"
        done
      } >> /root/build/pyspy_stall.txt 2>&1
      NDUMPS=$((NDUMPS+1))
      if [ "$NDUMPS" -ge 6 ]; then echo "WATCHDOG_MAXDUMPS" >> /root/build/pyspy_stall.txt; break; fi
      sleep 45
    fi
  done
) &
WD=$!
touch /root/build/leg_conn2.running

docker exec -d lsv-test bash $SP
ok=0
for i in $(seq 1 60); do
  code=$(docker exec lsv-test sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  if [ "$code" = "200" ]; then ok=1; echo "HEALTH_OK tick=$i"; break; fi
  sleep 10
done
if [ "$ok" != "1" ]; then
  echo "HEALTH_TIMEOUT"
  docker exec lsv-test tail -20 /root/serve_full.log
  rm -f /root/build/leg_conn2.running
  kill $WD 2>/dev/null
  exit 1
fi

cd /root/build || exit 1
python3 probe_evict.py
echo "PY_EXIT=$?"
sleep 30
rm -f /root/build/leg_conn2.running
kill $WD 2>/dev/null
echo "--- pyspy stall dumps:"
wc -l /root/build/pyspy_stall.txt 2>/dev/null || echo "NO_STALL_FILE (no freeze captured)"
echo "--- bank serve log:"
docker exec lsv-test cp /root/serve_full.log /tmp/telem_conn2.log
docker cp lsv-test:/tmp/telem_conn2.log /root/build/telem_conn2_serve.log
md5sum /root/build/telem_conn2_serve.log
echo "--- LMCNANDETECT count:"
docker exec lsv-test sh -c "grep -ac LMCNANDETECT /root/serve_full.log || true"
echo "--- POISON MD5 in probe output:"
grep -c 98abe3a2 /root/build/telem_conn2_leg.log || true
echo "--- TRACEBACKS:"
docker exec lsv-test sh -c "grep -ac Traceback /root/serve_full.log || true"
echo "--- ENGINE ALIVE CHECK:"
docker exec lsv-test sh -c "curl -s -o /dev/null -w \"HEALTH=%{http_code}\n\" http://localhost:8000/health"
echo "=== LEG_CONN2 DONE $(date -u +%H:%M:%S)"
