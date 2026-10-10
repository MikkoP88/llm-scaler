#!/bin/bash
# llm-scaler v138b+c NATURAL-RATE validation leg: connector OFF, NO PJ,
# ORIGINAL probe (sleep 0.3) — EXACT base_evict conditions (which CRASHED
# the engine at filler26 pre-fix, md5 98abe3a2 + RemoteDisconnected) but
# with v138b per-group padded-slot zeroers + v138c carry-sanitize live.
# Acceptance: 96/96 fillers clean, 0x 98abe3a2, LMCNANDETECT 0, anan 0,
# TRACEBACKS 0, engine survives, POISONED_ROWS ~98 normal guard noise.
SP=/root/serve_lmc_v133.sh
echo "=== LEG_NAT START $(date -u +%H:%M:%S)"
restore() {
  docker exec lsv-test cp $SP.bak_c $SP
  echo -n "KVFLAG_RESTORED_COUNT="
  docker exec lsv-test grep -c "kv-transfer-config" $SP || true
}
trap restore EXIT
docker exec lsv-test cp $SP $SP.bak_c
docker exec lsv-test sed -i "57d" $SP
C1=$(docker exec lsv-test grep -c "kv-transfer-config" $SP)
C2=$(docker exec lsv-test grep -c "export LMC_PJ" $SP)
echo "KVFLAG=$C1 (want 0) PJ=$C2 (want 0)"
if [ "$C1" != "0" ] || [ "$C2" != "0" ]; then echo "SED_ABORT"; exit 1; fi

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
docker exec lsv-test cp /root/serve_full.log /tmp/telem_nat.log
docker cp lsv-test:/tmp/telem_nat.log /root/build/telem_nat_serve.log
md5sum /root/build/telem_nat_serve.log
echo "--- V138 ZEROERS BUILT (expect group=0/1/2 2 classes each rank):"
docker exec lsv-test sh -c "grep -a \"v138 mamba zeroers\" /root/serve_full.log"
echo "--- LMCNANDETECT count (acceptance 0):"
docker exec lsv-test sh -c "grep -ac LMCNANDETECT /root/serve_full.log || true"
echo "--- ANAN count anywhere (acceptance 0):"
docker exec lsv-test sh -c "grep -ac anan /root/serve_full.log || true"
echo "--- POISON MD5 98abe3a2 in probe output (acceptance 0):"
grep -c 98abe3a2 /root/build/telem_nat_leg.log || true
echo "--- POISONED_ROWS:"
docker exec lsv-test sh -c "grep -ac POISONED-ROW /root/serve_full.log || true"
echo "--- TRACEBACKS (acceptance 0):"
docker exec lsv-test sh -c "grep -ac Traceback /root/serve_full.log || true"
echo "--- ENGINE ALIVE CHECK:"
docker exec lsv-test sh -c "curl -s -o /dev/null -w \"HEALTH=%{http_code}\n\" http://localhost:8000/health"
echo "=== LEG_NAT DONE $(date -u +%H:%M:%S)"
