#!/bin/bash
# llm-scaler v134 MambaDType support leg: boot one --mamba-ssm-cache-dtype
# on lsv-test (v134b scheduler + probe10 instrumentation, lmc_off gate) and
# run the killer battery (probe12 warm-resume, probe8 periodic full-pages,
# probe14 discriminator). Usage: bash dtype_leg.sh <dtype>
DT="$1"
SP=/root/serve_lmc_v133.sh
echo "=== DTYPE_LEG $DT START $(date -u +%H:%M:%S)"
docker exec lsv-test sed -i "s/--mamba-ssm-cache-dtype [a-z0-9_]*/--mamba-ssm-cache-dtype $DT/" "$SP"
docker exec lsv-test grep -o -- "--mamba-ssm-cache-dtype [a-z0-9_]*" "$SP"

# graceful engine stop (full tree incl VLLM::Worker_TP/EngineCore)
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
docker exec lsv-test touch /tmp/lmc_off
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

cd /root/build || exit 1
python3 probe12.py
python3 probe8.py
python3 probe14.py
echo "--- NONFINITE:"
docker exec lsv-test sh -c "grep -ac LMCDbgFwd /root/serve_full.log; grep -aE 'n[1-9][0-9]*/a' /root/serve_full.log | grep -ac LMCDbgFwd || true"
echo "=== DTYPE_LEG $DT DONE $(date -u +%H:%M:%S)"
