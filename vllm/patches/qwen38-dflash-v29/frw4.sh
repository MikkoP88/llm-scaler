#!/bin/bash
# llm-scaler v29 FIX-3: wedge watcher with optional boot-lottery RECOVERY.
# Polls engine /metrics every 2 s; on a 50 s stall with requests running:
#   1. CAPTURE live evidence (xpu-smi both GPUs, py-spy dumps of both TP
#      workers, /tmp/fr_*.log tails = the v28dbg flight recorder, engine
#      log tail) into /root/build/wedge_cap/
#   2. if $1 == "restart": docker restart lsv-test — wedge susceptibility
#      is decided per-boot at capture time (KNOWN_ISSUES #11 v28dbg), so a
#      restart is a FRESH ROLL of the capture layout. Combined with client
#      timeout+retry this converts a permanent hang into one lost batch +
#      ~7 min reboot, without human intervention.
# $2 = engine log name inside the container (default v29a0).
# Loops forever (re-arms after each event) unless $1 == "once" (single
# capture, no restart — the v28dbg frw3 behavior).
OUT=/root/build/wedge_cap
MODE="${1:-once}"
LOGNAME="${2:-v29a0}"
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
    TS=$(date +%H%M%S)
    EV=$OUT/w${TS}
    mkdir -p $EV
    echo "$NOW STALL (ok=$OK gen=$GEN run=$RUN) mode=$MODE - capturing"
    for d in 0 1; do xpu-smi stats -d $d -e > $EV/xpu$d.txt 2>&1; done
    # TP workers write the fr logs and spin in the livelock; EngineCore just
    # parks in shm get_response (v29 wedge #1 lesson: py-spy must hit Worker_TP).
    PIDS=$(docker top lsv-test -eo pid,cmd | grep "Worker_TP" | awk '{print $1}')
    docker exec lsv-test sh -c 'for f in /tmp/fr_*.log; do echo "== $f"; tail -n 60 "$f"; done' > $EV/fr_tail.txt 2>&1
    for p in $PIDS; do $PYSPY dump --pid $p > $EV/pyspy_$p.txt 2>&1; done
    docker exec lsv-test sh -c "tail -n 30 /root/${LOGNAME}.log" > $EV/engine_tail.txt 2>&1
    echo "$NOW capture done:"; ls -la $EV
    if [ "$MODE" = "restart" ]; then
      echo "$NOW RESTARTING lsv-test (fresh capture-layout roll)"
      docker restart -t 30 lsv-test
      STALL=0; LAST_OK=0; LAST_GEN=0
      sleep 60
    else
      exit 0
    fi
  fi
  sleep 2
done
