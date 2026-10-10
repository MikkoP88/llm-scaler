#!/bin/bash
# llm-scaler v138 LEG C (fix validation): connector OFF + LMC_PJ=2 +
# zero-sleep hammer — IDENTICAL to Leg B (which corrupted at filler91,
# md5 98abe3a2 '!!!!') but with the v138 mamba alloc-zeroing fix live
# (VLLM_MAMBA_ZERO_NEW_BLOCKS default 1). Acceptance: v138 zeroers built,
# LMCNANDETECT 0, anan 0, no 98abe3a2, A==A2 True, 96/96 clean.
SP=/root/serve_lmc_v133.sh
echo "=== TELEM_LEG_C START $(date -u +%H:%M:%S)"
restore() {
  docker exec lsv-test cp $SP.bak_c $SP
  echo -n "KVFLAG_RESTORED_COUNT="
  docker exec lsv-test grep -c "kv-transfer-config" $SP || true
}
trap restore EXIT
docker exec lsv-test cp $SP $SP.bak_c
docker exec lsv-test sed -i "57d" $SP
docker exec lsv-test sed -i "/export LMC_DBG_FWD=1/a export LMC_PJ=2\nexport LMC_BOUND_STATS=0\nexport LMC_BOUND_MD5=0" $SP
C1=$(docker exec lsv-test grep -c "kv-transfer-config" $SP)
C2=$(docker exec lsv-test grep -c "export LMC_PJ=2" $SP)
echo "KVFLAG=$C1 PJ2=$C2"
if [ "$C1" != "0" ] || [ "$C2" != "1" ]; then echo "SED_ABORT"; exit 1; fi

sed "s/    time.sleep(0.3)/    time.sleep(0.0)/" /root/build/probe_evict.py > /root/build/probe_evict0.py
grep -n "time.sleep(0.0)" /root/build/probe_evict0.py | head -1

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
python3 probe_evict0.py
echo "PY_EXIT=$?"
echo "--- bank serve log:"
docker exec lsv-test cp /root/serve_full.log /tmp/telem_c.log
docker cp lsv-test:/tmp/telem_c.log /root/build/telem_c_serve.log
md5sum /root/build/telem_c_serve.log
echo "--- V138 ZEROERS BUILT (expect 2 classes: conv 98304w, ssm 17920w):"
docker exec lsv-test sh -c "grep -a \"v138 mamba zeroers\" /root/serve_full.log"
echo "--- LMCNANDETECT count (acceptance 0):"
docker exec lsv-test sh -c "grep -ac LMCNANDETECT /root/serve_full.log || true"
echo "--- ANAN count anywhere (acceptance 0):"
docker exec lsv-test sh -c "grep -ac anan /root/serve_full.log || true"
echo "--- POISON MD5 98abe3a2 in probe output (acceptance 0):"
grep -c 98abe3a2 /root/build/telem_c_leg.log || true
echo "--- LMCPJ ANOM count (acceptance 0):"
docker exec lsv-test sh -c "grep -ac \"LMCPJ.*ANOM\" /root/serve_full.log || true"
echo "--- POISONED_ROWS:"
docker exec lsv-test sh -c "grep -ac POISONED-ROW /root/serve_full.log || true"
echo "--- TRACEBACKS:"
docker exec lsv-test sh -c "grep -ac Traceback /root/serve_full.log || true"
echo "=== TELEM_LEG_C DONE $(date -u +%H:%M:%S)"
