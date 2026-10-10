#!/bin/bash
# llm-scaler v134 LMCache-ON dtype leg: boot with the connector ACTIVE
# (lmc_off removed) per --mamba-ssm-cache-dtype and run the gated
# connector probes + the killer battery. Acceptance: coherent outputs
# everywhere (no '!!!!', no EMPTY), zero non-finite mamba pages, zero
# tracebacks. Usage: bash dtype_leg_lmc.sh <dtype>
DT="$1"
SP=/root/serve_lmc_v133.sh
echo "=== DTYPE_LMC_LEG $DT START $(date -u +%H:%M:%S)"
docker exec lsv-test sed -i "s/--mamba-ssm-cache-dtype [a-z0-9_]*/--mamba-ssm-cache-dtype $DT/" "$SP"
docker exec lsv-test grep -o -- "--mamba-ssm-cache-dtype [a-z0-9_]*" "$SP"

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
docker exec -d lsv-test bash "$SP"

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
echo -n "MAMBA_ENGINE_INITS="
docker exec lsv-test grep -ac "Mamba engine CPU pool scaled" /root/serve_full.log || true

cd /root/build || exit 1
python3 probe5warm.py
python3 probe7.py
python3 probe12.py
python3 probe8.py
python3 probe14.py
echo "--- NONFINITE(total_fwd_lines, nonfinite_lines):"
docker exec lsv-test sh -c "grep -ac LMCDbgFwd /root/serve_full.log; grep -aE 'n[1-9][0-9]*/a' /root/serve_full.log | grep -ac LMCDbgFwd || true"
echo "--- TRACEBACKS:"
docker exec lsv-test sh -c "grep -ac Traceback /root/serve_full.log || true"
echo "=== DTYPE_LMC_LEG $DT DONE $(date -u +%H:%M:%S)"
