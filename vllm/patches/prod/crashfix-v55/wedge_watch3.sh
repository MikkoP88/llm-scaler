#!/bin/bash
# wedge_watch3.sh <tag> — throughput-zero wedge detector for the 600s-
# fence diagnostic boot. SpecDecoding metrics lines stop at wedge onset;
# when the newest line is >150s old while pressure is running, capture
# worker stacks DURING the live fence wait (t0, +90s, +240s) plus GPU
# util + dmesg (guc watch). Needs pressure running to be meaningful.
set -u
TAG="${1:?usage: wedge_watch3.sh <tag>}"
FLAG=/root/build/lce1/${TAG}.flag

last_metrics_age() {
  T=$(docker exec lsv-test sh -c 'grep SpecDecoding /root/serve_full.log 2>/dev/null | tail -1' 2>/dev/null | sed -n 's/.*INFO [0-9-]* \([0-9:]*\) .*/\1/p')
  [ -z "$T" ] && { echo 9999; return; }
  A=$(date +%H:%M:%S)
  echo $(( $(date -d "$A" +%s) - $(date -d "$T" +%s) ))
}

snap() {
  local D=$1 lbl=$2
  mkdir -p "$D"
  for dev in 0 1; do
    xpu-smi dump --device $dev --metrics UTILIZATION,EU_ARRAY \
      --interval 1 --number 2 > "$D/xpu_d${dev}_${lbl}.txt" 2>&1
  done
  docker exec lsv-test sh -c '
    for pid in $(ps aux | grep -E "VLLM::Worker" | grep -v grep | awk "{print \$2}"); do
      echo "== pid $pid main: state=$(awk "{print \$3}" /proc/$pid/stat 2>/dev/null) wchan=$(cat /proc/$pid/wchan 2>/dev/null)"
      echo "   kernel-stack:"; cat /proc/$pid/stack 2>/dev/null | head -6
      echo "   R-threads:"
      for t in /proc/$pid/task/*; do
        st=$(awk "{print \$3}" $t/stat 2>/dev/null)
        [ "$st" = "R" ] && echo "     $(basename $t) RUNNING wchan=$(cat $t/wchan 2>/dev/null)"
      done
      grep -E "ctxt_switches" /proc/$pid/status 2>/dev/null
    done' > "$D/workers_${lbl}.txt" 2>&1
  docker exec lsv-test sh -c 'tail -30 /root/serve_full.log | grep -v "GET /health"' > "$D/serve_${lbl}.txt" 2>&1
  dmesg -T | tail -10 > "$D/dmesg_${lbl}.txt" 2>&1
  echo "--- snap $lbl done $(date +%H:%M:%S)"
}

# wait for engine to be actively producing metrics first
AGE=9999
for i in $(seq 1 90); do
  AGE=$(last_metrics_age)
  [ "$AGE" -lt 30 ] && break
  sleep 10
done
echo "watching from $(date +%H:%M:%S) (metrics-age=${AGE}s)"

while :; do
  AGE=$(last_metrics_age)
  if [ "$AGE" -gt 150 ]; then
    echo "WEDGE-ONSET metrics-age=${AGE}s at $(date +%H:%M:%S)"
    D=/root/build/lce1/${TAG}
    snap "$D" t0
    sleep 90
    snap "$D" t90
    sleep 150
    snap "$D" t240
    pkill -f repro_sustain 2>/dev/null
    pkill -f dt_warmup_v53 2>/dev/null
    pkill -f repro_loop 2>/dev/null
    echo "onset+~7min $(date)" > "$FLAG"
    echo "CAPTURED3 $FLAG"
    exit 2
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "CONTAINER GONE $(date +%H:%M:%S)"
    echo "dead $(date)" > "$FLAG"
    exit 3
  fi
  sleep 10
done
