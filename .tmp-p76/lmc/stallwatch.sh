#!/bin/bash
# stallwatch.sh ROUNDS — nonce probe loop + wedge capture on decode freeze.
# Detects: running>0 AND spec drafts zero-advance for 6x5s ticks =>
# banks py-spy dumps of ALL engine pids (2 samples 3s apart) + metrics +
# dmesg into /root/build/WD_stall_<ts>/. NEVER kills/restarts anything.
R=${1:-30}
B=/root/build
LOG=$B/stallwatch_probe.log
: > $LOG
echo "=== STALLWATCH START rounds=$R $(date -u +%H:%M:%S)"

( for r in $(seq 1 $R); do
    echo "--- round $r $(date -u +%H:%M:%S)" >> $LOG
    python3 $B/nonce_probe.py 20 >> $LOG 2>&1
  done
  echo "STALLWATCH_PROBES_DONE $(date -u +%H:%M:%S)" >> $LOG ) &

PREV=-1; STALL=0; CAPS=0
while :; do
  grep -q STALLWATCH_PROBES_DONE $LOG 2>/dev/null && break
  M=$(docker exec lsv-test sh -c "curl -s http://localhost:8000/metrics" 2>/dev/null)
  RUN=$(echo "$M" | grep -o "^vllm:num_requests_running{[^}]*} [0-9.]*" | grep -o "[0-9.]*$")
  DR=$(echo "$M" | grep -o "^vllm:spec_decode_num_drafts_total{[^}]*} [0-9.]*" | grep -o "[0-9.]*$")
  [ -z "$RUN" ] && RUN=0
  [ -z "$DR" ] && DR=$PREV
  NOW=$(date -u +%H:%M:%S)
  if awk -v r="$RUN" -v d="$DR" -v p="$PREV" 'BEGIN{exit !(r>0 && p!=-1 && d==p)}'; then
    STALL=$((STALL+1))
  else
    STALL=0
  fi
  echo "$NOW run=$RUN drafts=$DR stall_ticks=$STALL" >> $B/stallwatch_ticks.log
  if [ "$STALL" -ge 6 ]; then
    TS=$(date +%s)
    D=$B/WD_stall_$TS
    mkdir -p $D
    CAPS=$((CAPS+1))
    echo "!!! STALL CAPTURE #$CAPS at $NOW (run=$RUN drafts=$DR frozen 30s)" | tee -a $D/event.txt
    for s in 1 2; do
      docker exec lsv-test sh -c "curl -s http://localhost:8000/metrics" > $D/metrics_s$s.txt 2>/dev/null
      for p in $(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::"); do
        nm=$(docker exec lsv-test sh -c "tr '\\0' ' ' < /proc/$p/cmdline" 2>/dev/null | head -c 40)
        docker exec lsv-test /opt/venv/bin/py-spy dump --pid $p --nonblocking \
          > "$D/pyspy_s${s}_pid${p}.txt" 2>&1 || true
        echo "pid=$p cmd=$nm" >> $D/pids.txt
      done
      [ $s -eq 1 ] && sleep 3
    done
    dmesg -T | tail -30 > $D/dmesg.txt 2>/dev/null || true
    cp $LOG $D/probe_so_far.log
    STALL=0
    [ "$CAPS" -ge 3 ] && { echo "3 captures banked, stopping watcher" ; break; }
  fi
  PREV=$DR
  sleep 5
done
echo "=== STALLWATCH DONE $(date -u +%H:%M:%S) captures=$CAPS"
echo "--- probe summary:"
grep -c "clean wall" $LOG || true
grep -c DEGENERATE $LOG || true
grep -E "wall=[0-9]{2,}\.[0-9]+s" $LOG | awk -F"wall=" "{print \$2, \$1}" | sort -rn | head -8
