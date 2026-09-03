#!/bin/bash
# llm-scaler v29 P3: FULL-TELEMETRY wedge watcher (frw4 + deep capture).
# On a 50 s /metrics stall with requests running, captures per event dir:
#   xpu<N>.txt        xpu-smi stats (single shot)
#   xpu<N>_s{1,2,3}.txt  3 stats samples 2 s apart (counter deltas = storm rate)
#   xpu<N>_dump.txt   xpu-smi dump (full telemetry block)
#   pyspy_<pid>_{a,b}.txt  py-spy dump per TP worker, 5 s apart (stack frozen?)
#   fr_tail.txt / engine_tail.txt
#   numastat.txt / interrupts.txt / dmesg.txt / links.txt / proc.txt
# $1 = once|restart (restart = docker restart re-roll), $2 = engine log name.
OUT=/root/build/wedge_cap
MODE="${1:-once}"
LOGNAME="${2:-v29}"
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
    echo "$NOW STALL (ok=$OK gen=$GEN run=$RUN) mode=$MODE - full capture"
    # device telemetry: 3 deltas + dump
    for d in 0 1; do
      xpu-smi stats -d $d -e > $EV/xpu$d.txt 2>&1
      xpu-smi dump  -d $d    > $EV/xpu${d}_dump.txt 2>&1
    done
    # worker stacks twice, 5 s apart (frozen vs moving)
    PIDS=$(docker top lsv-test -eo pid,cmd | grep "Worker_TP" | awk '{print $1}')
    echo "$PIDS" > $EV/worker_pids.txt
    for p in $PIDS; do $PYSPY dump --pid $p > $EV/pyspy_${p}_a.txt 2>&1; done
    for d in 0 1; do xpu-smi stats -d $d -e > $EV/xpu${d}_s1.txt 2>&1; done
    sleep 2
    for d in 0 1; do xpu-smi stats -d $d -e > $EV/xpu${d}_s2.txt 2>&1; done
    sleep 2
    for d in 0 1; do xpu-smi stats -d $d -e > $EV/xpu${d}_s3.txt 2>&1; done
    for p in $PIDS; do $PYSPY dump --pid $p > $EV/pyspy_${p}_b.txt 2>&1; done
    # host-side
    docker exec lsv-test sh -c 'for f in /tmp/fr_*.log; do echo "== $f"; tail -n 60 "$f"; done' > $EV/fr_tail.txt 2>&1
    docker exec lsv-test sh -c "tail -n 40 /root/${LOGNAME}.log" > $EV/engine_tail.txt 2>&1
    numastat -m > $EV/numastat.txt 2>&1
    cp /proc/interrupts $EV/interrupts.txt 2>&1
    dmesg | tail -n 40 > $EV/dmesg.txt 2>&1
    for d in /sys/bus/pci/devices/0000:b1:00.0 /sys/bus/pci/devices/0000:da:00.0; do
      echo "$d $(cat $d/current_link_speed 2>/dev/null)x$(cat $d/current_link_width 2>/dev/null)"
    done > $EV/links.txt 2>&1
    for p in $PIDS; do
      echo "== pid $p"
      grep -E 'voluntary_ctxt_switches|nonvoluntary' /proc/$p/status 2>/dev/null
      cat /proc/$p/stat 2>/dev/null | awk '{print "state="$3 " minflt="$10 " majflt="$12}'
    done > $EV/proc.txt 2>&1
    echo "$NOW capture done:"; ls -la $EV
    if [ "$MODE" = "restart" ]; then
      echo "$NOW RESTARTING lsv-test (fresh roll)"
      docker restart -t 30 lsv-test
      STALL=0; LAST_OK=0; LAST_GEN=0
      sleep 60
    else
      exit 0
    fi
  fi
  sleep 2
done
