#!/bin/bash
# llm-scaler v138b+c CONNECTOR-ON natural-rate leg: LMCache external load
# path live (serve line 57 kept), NO PJ, ORIGINAL probe (sleep 0.3).
# v138b per-group zeroers + v138c carry-sanitize live.
# Acceptance: 96/96 FILLERS_DONE + PY_EXIT=0, LMCNANDETECT 0, anan 0,
# TRACEBACKS 0, engine survives, A==A2 True, connector lines present
# (mamba seed parked / seed write / async load complete), BOUND stats finite.
SP=/root/serve_lmc_v133.sh
echo "=== LEG_CONN START $(date -u +%H:%M:%S)"
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
  exit 1
fi

cd /root/build || exit 1
python3 probe_evict.py
echo "PY_EXIT=$?"
echo "--- bank serve log:"
docker exec lsv-test cp /root/serve_full.log /tmp/telem_conn.log
docker cp lsv-test:/tmp/telem_conn.log /root/build/telem_conn_serve.log
md5sum /root/build/telem_conn_serve.log
echo "--- V138 ZEROERS BUILT:"
docker exec lsv-test sh -c "grep -ac \"v138 mamba zeroers\" /root/serve_full.log || true"
echo "--- CONNECTOR LINES:"
docker exec lsv-test sh -c "grep -a \"mamba seed parked\" /root/serve_full.log | head -2 | cut -c1-150"
docker exec lsv-test sh -c "grep -ac \"mamba seed parked\" /root/serve_full.log || true"
docker exec lsv-test sh -c "grep -a \"seed write\" /root/serve_full.log | head -2 | cut -c1-150"
docker exec lsv-test sh -c "grep -a \"async load complete\" /root/serve_full.log | head -2 | cut -c1-150"
echo "--- LMCBOUND STATS:"
docker exec lsv-test sh -c "grep -a LMCBOUND /root/serve_full.log | tail -3 | cut -c1-200"
echo "--- LMCNANDETECT count (acceptance 0):"
docker exec lsv-test sh -c "grep -ac LMCNANDETECT /root/serve_full.log || true"
echo "--- ANAN count anywhere (acceptance 0):"
docker exec lsv-test sh -c "grep -ac anan /root/serve_full.log || true"
echo "--- POISON MD5 98abe3a2 in probe output:"
grep -c 98abe3a2 /root/build/telem_conn_leg.log || true
echo "--- TRACEBACKS (acceptance 0):"
docker exec lsv-test sh -c "grep -ac Traceback /root/serve_full.log || true"
echo "--- ENGINE ALIVE CHECK:"
docker exec lsv-test sh -c "curl -s -o /dev/null -w \"HEALTH=%{http_code}\n\" http://localhost:8000/health"
echo "=== LEG_CONN DONE $(date -u +%H:%M:%S)"
