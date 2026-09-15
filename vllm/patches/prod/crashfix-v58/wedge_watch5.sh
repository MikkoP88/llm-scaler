#!/bin/bash
# wedge_watch5.sh <tag> — watch4 fix. Boot G miss: POST-age guard cannot
# see a wedged IN-FLIGHT request (POST line never logs). New trigger:
#   A) metrics-gap (>150s) AND engine reports running requests >= 1
#      (in-flight frozen request = wedge; idle wave boundary = 0 running)
#   B) ASYNC-EVENT-STALL fence count increases above boot baseline
#      (immediate capture while workers still alive)
# Snap set: watch4 (GPU util/EU, worker stacks+R-threads, full fr
# tail-400 rescue, tripwire grep, serve tail, dmesg) + engine /metrics
# dump (running/waiting/KV usage — memory-hypothesis context).
set -u
TAG="${1:?usage: wedge_watch5.sh <tag>}"
PORT="${2:-8000}"
FLAG=/root/build/lce1/${TAG}.flag

last_metrics_age() {
  T=$(docker exec lsv-test sh -c 'grep SpecDecoding /root/serve_full.log 2>/dev/null | tail -1' 2>/dev/null | sed -n 's/.*INFO [0-9-]* \([0-9:]*\) .*/\1/p')
  [ -z "$T" ] && { echo 9999; return; }
  A=$(date +%H:%M:%S)
  echo $(( $(date -d "$A" +%s) - $(date -d "$T" +%s) ))
}

fence_count() {
  F=$(docker exec lsv-test sh -c 'grep -c ASYNC-EVENT-STALL /root/serve_full.log 2>/dev/null' 2>/dev/null)
  case "$F" in ''|*[!0-9]*) echo 0;; *) echo "$F";; esac
}

running_reqs() {
  # >=0 running requests; -1 = endpoint unreachable
  docker exec lsv-test python3 -c "
import urllib.request
try:
    m = urllib.request.urlopen('http://127.0.0.1:${PORT}/metrics', timeout=5).read().decode()
    for l in m.splitlines():
        if l.startswith('vllm:num_requests_running{'):
            print(int(float(l.rsplit(' ',1)[1]))); break
    else:
        print(-1)
except Exception:
    print(-1)
" 2>/dev/null | tail -1
}

metrics_dump() {
  local out=$1
  docker exec lsv-test python3 -c "
import urllib.request
try:
    m = urllib.request.urlopen('http://127.0.0.1:${PORT}/metrics', timeout=5).read().decode()
    for l in m.splitlines():
        if ('requests_running' in l or 'requests_waiting' in l
                or 'cache_usage' in l or 'preemption' in l.lower()
                or 'spec_num' in l or 'acceptance' in l):
            print(l)
except Exception as e:
    print('metrics-unreachable', e)
" > "$out" 2>&1
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
  docker exec lsv-test sh -c '
    for pid in $(ps aux | grep -E "VLLM::Worker" | grep -v grep | awk "{print \$2}"); do
      echo "===== pyspy $pid ====="
      /opt/venv/bin/py-spy dump --pid $pid 2>&1
    done' > "$D/pyspy_${lbl}.txt" 2>&1
  docker exec lsv-test sh -c 'grep -n "V58-TRIPWIRE\|ASYNC-EVENT-STALL" /root/serve_full.log | tail -20' > "$D/tripwire_${lbl}.txt" 2>&1
  docker exec lsv-test sh -c 'tail -30 /root/serve_full.log | grep -v "GET /health"' > "$D/serve_${lbl}.txt" 2>&1
  metrics_dump "$D/engine_metrics_${lbl}.txt"
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

BASE=$(fence_count)
echo "fence baseline=$BASE"

while :; do
  AGE=$(last_metrics_age)
  R=$(running_reqs)
  F=$(fence_count)
  REASON=""
  if [ "$F" -gt "$BASE" ] 2>/dev/null; then
    REASON="fence=$F>base=$BASE"
  elif [ "$AGE" -gt 150 ] && [ "$R" -ge 1 ] 2>/dev/null; then
    REASON="metrics-age=${AGE}s running=$R"
  fi
  if [ -n "$REASON" ]; then
    echo "WEDGE-ONSET $REASON at $(date +%H:%M:%S)"
    D=/root/build/lce1/${TAG}
    snap "$D" t0
    sleep 90
    snap "$D" t90
    sleep 150
    snap "$D" t240
    pkill -f repro_sustain 2>/dev/null
    pkill -f dt_warmup_v53 2>/dev/null
    pkill -f 'repro_loo[p]' 2>/dev/null
    echo "onset+~7min $REASON $(date)" > "$FLAG"
    echo "CAPTURED5 $FLAG"
    exit 2
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "CONTAINER GONE $(date +%H:%M:%S)"
    echo "dead $(date)" > "$FLAG"
    exit 3
  fi
  sleep 10
done
