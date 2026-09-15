#!/bin/bash
# wedge_watch4.sh <tag> — watch3 (metrics-gap onset detection) + FULL fr
# file rescue per snap (v58: Boot D lost fr content beyond tail-30).
set -u
TAG="${1:?usage: wedge_watch4.sh <tag>}"
FLAG=/root/build/lce1/${TAG}.flag

last_metrics_age() {
  T=$(docker exec lsv-test sh -c 'grep SpecDecoding /root/serve_full.log 2>/dev/null | tail -1' 2>/dev/null | sed -n 's/.*INFO [0-9-]* \([0-9:]*\) .*/\1/p')
  [ -z "$T" ] && { echo 9999; return; }
  A=$(date +%H:%M:%S)
  echo $(( $(date -d "$A" +%s) - $(date -d "$T" +%s) ))
}

# v58: pressure liveness — age of the last inference POST in the serve
# log. Boot F lesson: metrics gap alone false-positives on the idle gap
# between pressure waves (watch3 fired, captured, and pkilled the NEXT
# wave). Only declare onset when traffic was flowing recently.
last_post_age() {
  T=$(docker exec lsv-test sh -c 'grep "POST /v1" /root/serve_full.log 2>/dev/null | tail -1' 2>/dev/null | sed -n 's/.*INFO: *[^ ]* \([0-9:]*\) .*/\1/p')
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
  # v58: FULL fr rescue (not tails) — by live worker pids
  docker exec lsv-test sh -c '
    for pid in $(ps aux | grep -E "VLLM::Worker" | grep -v grep | awk "{print \$2}"); do
      f=/tmp/fr_${pid}.log
      [ -f "$f" ] && cp "$f" /tmp/fr_rescue_${pid}_${lbl:-x}.log 2>/dev/null
    done; true' > /dev/null 2>&1
  docker exec lsv-test sh -c '
    for pid in $(ps aux | grep -E "VLLM::Worker" | grep -v grep | awk "{print \$2}"); do
      f=/tmp/fr_${pid}.log
      [ -f "$f" ] && { echo "== $f (tail -400)"; tail -400 "$f"; }
    done' > "$D/fr_tail400_${lbl}.txt" 2>&1
  docker exec lsv-test sh -c 'grep -n "V58-TRIPWIRE\|ASYNC-EVENT-STALL" /root/serve_full.log | tail -20' > "$D/tripwire_${lbl}.txt" 2>&1
  docker exec lsv-test sh -c 'tail -30 /root/serve_full.log | grep -v "GET /health"' > "$D/serve_${lbl}.txt" 2>&1
  dmesg -T | tail -10 > "$D/dmesg_${lbl}.txt" 2>&1
  echo "--- snap $lbl done $(date +%H:%M:%S)"
}

AGE=9999
for i in $(seq 1 90); do
  AGE=$(last_metrics_age)
  [ "$AGE" -lt 30 ] && break
  sleep 10
done
echo "watching from $(date +%H:%M:%S) (metrics-age=${AGE}s)"

while :; do
  AGE=$(last_metrics_age)
  POSTAGE=$(last_post_age)
  if [ "$AGE" -gt 150 ] && [ "${POSTAGE:-9999}" -lt 300 ]; then
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
    echo "CAPTURED4 $FLAG"
    exit 2
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "CONTAINER GONE $(date +%H:%M:%S)"
    echo "dead $(date)" > "$FLAG"
    exit 3
  fi
  sleep 10
done
