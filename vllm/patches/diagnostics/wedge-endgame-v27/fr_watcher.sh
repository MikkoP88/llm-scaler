#!/bin/bash
# In-flight wedge watcher (host-side, run next to the lsv-test container).
# Polls the engine /metrics every 2 s; when success count AND generation
# tokens freeze for >50 s while a request is running, captures IMMEDIATELY
# (while the wedge is live — post-mortem py-spy is idle and useless):
#   py-spy dumps of both TP workers, xpu-smi engine stats, and the tails
#   of the v28dbg flight-recorder logs (/tmp/fr_<pid>.log in-container).
# Stops after first capture (artifacts in /root/build/wedge_cap/).
OUT=/root/build/wedge_cap
mkdir -p $OUT
LAST_OK=0; LAST_GEN=0; STALL=0; PYSPY=/root/.local/bin/py-spy
while true; do
  M=$(curl -s --max-time 8 http://127.0.0.1:8000/metrics)
  OK=$(echo "$M" | awk '/^vllm:request_success_total\{.*abort/ {next} /^vllm:request_success_total/ {s+=$NF} END {print int(s)}')
  GEN=$(echo "$M" | awk '/^vllm:generation_tokens_total/ {print int($NF)}')
  RUN=$(echo "$M" | awk '/^vllm:num_requests_running/ {print int($NF)}')
  NOW=$(date +%H:%M:%S)
  if [ "${RUN:-0}" -gt 0 ] && [ "${OK:-0}" = "$LAST_OK" ] && [ "${GEN:-0}" = "$LAST_GEN" ]; then
    STALL=$((STALL+2))
  else
    STALL=0
  fi
  LAST_OK=$OK; LAST_GEN=$GEN
  if [ $STALL -ge 50 ]; then
    echo "$NOW STALL detected (ok=$OK gen=$GEN run=$RUN) — capturing"
    for d in 0 1; do xpu-smi stats -d $d -e > $OUT/xpu$d.txt 2>&1; done
    PIDS=$(docker top lsv-test -eo pid,cmd | grep -E "EngineCore|from multiprocessing" | awk '{print $1}')
    docker exec lsv-test sh -c 'for f in /tmp/fr_*.log; do echo "== $f"; tail -n 40 "$f"; done' > $OUT/fr_tail.txt 2>&1
    for p in $PIDS; do $PYSPY dump --pid $p > $OUT/pyspy_$p.txt 2>&1; done
    docker exec lsv-test sh -c 'grep -a "loggers.py:273" /root/*.log | tail -5' > $OUT/engine_tail.txt 2>&1
    echo "$NOW capture done:"; ls -la $OUT
    exit 0
  fi
  sleep 2
done
