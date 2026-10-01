#!/bin/sh
# quiet_legs_v127.sh — W1 timing legs under a QUIET lane (running==0 AND
# waiting==0): v66 admission probe (certified gate) + solo genspeed + 4-stream
# async genspeed. The two earlier admission runs (13:25 sanity, 13:52 direct)
# were load-adjudicated STARVED with 2/3 streams in the certified family —
# this script produces the quiet-lane verdict the gate is calibrated for.
# Host-side only; waits up to 40 min for the window; read-only vs the lane
# except for the probe traffic itself.
LOG=/root/build/lce1/quiet_legs_W1.log
echo "=== quiet_legs start $(date +%T) ===" >> $LOG
i=0
while [ $i -lt 240 ]; do
  R=$(curl -s -m 6 http://localhost:8000/metrics | grep -E '^vllm:num_requests_running\{|^vllm:num_requests_waiting\{' | tr -s ' ' | cut -d ' ' -f2 | paste -sd, -)
  if [ "$R" = "0.0,0.0" ]; then
    echo "QUIET WINDOW at $(date +%T) after ${i} polls" >> $LOG
    echo "== ADMISSION (gate: verdict=PASS) ==" >> $LOG
    docker exec lsv-test /opt/venv/bin/python3 /root/probe_admission_v66.py http://127.0.0.1:8000 qwen3.8-27b-fp8 661 4000 >> $LOG 2>&1
    echo "ADMISSION_RC=$?" >> $LOG
    sleep 5
    echo "== SOLO GENSPEED 1x1024 x3 (banked class ~74) ==" >> $LOG
    cd /root/build && python3 bench_genspeed.py 1 1024 3 >> $LOG 2>&1
    echo "SOLO_RC=$?" >> $LOG
    sleep 5
    echo "== ASYNC GENSPEED 4x1024 x1 (banked agg ~122-158) ==" >> $LOG
    cd /root/build && python3 bench_genspeed.py 4 1024 1 >> $LOG 2>&1
    echo "ASYNC_RC=$?" >> $LOG
    echo "QUIET_LEGS_DONE $(date +%T)" >> $LOG
    tail -5 $LOG
    exit 0
  fi
  i=$((i+1))
  sleep 10
done
echo "NO_QUIET_WINDOW_40MIN $(date +%T)" >> $LOG
exit 42
