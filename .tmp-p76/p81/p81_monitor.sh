#!/bin/bash
# llm-scaler P81 wedge monitor (host). Clone of p80_monitor.sh + banks the
# P81 argsnap files. Watches the lane during the P81 depth leg; on serve
# death or wedge window banks EVERYTHING fast (py-spy dumps of both workers
# + P80 ladder + P81 argsnap tails + dmesg + serve-log tail) to
# /root/build/lce1/P81_wedge_<ts>/. Workers vanish within minutes of the RPC
# timeout -- bank EARLY, bank OFTEN (3 rounds while tree alive).
MLOG=/root/build/lce1/p81_monitor.log
TS() { date +%H:%M:%S; }
log() { echo "$(TS) $*" >> "$MLOG"; }

bank() {
  local dir=$1 round=$2
  mkdir -p "$dir"
  docker exec lsv-test sh -c 'for d in /proc/[0-9]*/comm; do c=$(cat "$d"); case "$c" in VLLM::Worker*) p=${d%/comm}; echo "${p#/proc/}";; esac; done' 2>/dev/null > "$dir/worker_pids.txt"
  for W in $(cat "$dir/worker_pids.txt"); do
    docker exec lsv-test /opt/venv/bin/py-spy dump --pid "$W" >> "$dir/pyspy_r${round}.txt" 2>&1
  done
  docker cp lsv-test:/root/. "$dir/container_root_snapshot/" >/dev/null 2>&1
  docker exec lsv-test sh -c 'ls -la /root/p80_ladder.* /root/p81_argsnap.* 2>/dev/null; echo ---; for f in /root/p80_ladder.*; do [ -f "$f" ] && { echo "== $f"; tail -n 40 "$f"; }; done' > "$dir/ladder_tail_r${round}.txt" 2>&1
  docker exec lsv-test sh -c 'for f in /root/p81_argsnap.*; do [ -f "$f" ] && { echo "== $f"; tail -n 30 "$f"; }; done' > "$dir/argsnap_tail_r${round}.txt" 2>&1
  docker exec lsv-test sh -c 'tail -n 60 /root/serve_full.log' > "$dir/serve_tail_r${round}.txt" 2>&1
  dmesg -T | tail -n 30 > "$dir/dmesg_r${round}.txt" 2>&1
  log "banked round=$round -> $dir"
}

log "=== monitor start ==="
DEAD=0
while true; do
  H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:8000/health)
  if [ "$H" != "200" ]; then
    N=$(docker exec lsv-test sh -c 'grep -l -E "^(python3|VLLM::)" /proc/[0-9]*/comm 2>/dev/null | wc -l' 2>/dev/null)
    DIR="/root/build/lce1/P81_wedge_$(date +%H%M%S)"
    if [ "$N" != "0" ]; then
      log "WEDGE WINDOW health=$H tree=$N"
      for R in 1 2 3; do
        bank "$DIR" "$R"
        sleep 45
        N2=$(docker exec lsv-test sh -c 'grep -l -E "^(python3|VLLM::)" /proc/[0-9]*/comm 2>/dev/null | wc -l' 2>/dev/null)
        [ "$N2" = "0" ] && break
      done
      log "wedge banking complete tree was=$N"
      exit 0
    else
      bank "$DIR" dead
      log "SERVE DEAD health=$H tree=0 (banked)"
      exit 0
    fi
  fi
  # minute heartbeat: ladder + argsnap line counts (mid-run readable by design)
  M=$(( $(date +%s) % 60 ))
  if [ "$M" -lt 15 ]; then
    LC=$(docker exec lsv-test sh -c 'wc -l /root/p80_ladder.* /root/p81_argsnap.* 2>/dev/null | tail -n +1' 2>/dev/null | tr "\n" " ")
    log "alive health=200 ladder/argsnap: $LC"
  fi
  sleep 15
done
