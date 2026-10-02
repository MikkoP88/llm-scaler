#!/bin/bash
# watch_live.sh — full live telemetry for the lsv-test vLLM engine.
# Captures to /root/build/lce1/telemetry/:
#   watch.log   — timestamped health/container/census lines + events
#   hits.log    — every engine-log line matching crash/warning classes
# Artifacts auto-captured on engine death (docker logs, serve tail, fr tails).
set -u
D=/root/build/lce1/telemetry
mkdir -p "$D"
W=$D/watch.log
HITS=$D/hits.log
: > "$W"
: > "$HITS"
PAT='Unknown vLLM environment variable|DEVICE_LOST|TimeoutError|RPC call to|Watchdog|ASYNC-EVENT-STALL|GAP-FENCE|EngineDeadError|RuntimeError|jit_monitor|CRITICAL|Engine hit an exception|WorkerProc hit an exception|EngineCore encountered'
DM0=$(dmesg 2>/dev/null | wc -l)
L0=0
DOWN=0
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$W"; }

log "TELEMETRY_START (engine log: lsv-test:/root/serve_full.log)"
i=0
while true; do
  i=$((i+1))
  H=$(curl -s -o /dev/null -m 3 -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo 000)
  C=$(docker ps -a --format '{{.Status}}' --filter name=^/lsv-test$ | head -1)
  [ -z "$C" ] && C=gone
  LINES=$(docker exec lsv-test sh -c "wc -l < /root/serve_full.log" 2>/dev/null || echo 0)
  if [ "${LINES:-0}" -gt "$L0" ] 2>/dev/null; then
    A=$((L0+1))
    NEW=$(docker exec lsv-test sh -c "sed -n \"${A},${LINES}p\" /root/serve_full.log" 2>/dev/null \
      | grep -E "$PAT" | tee -a "$HITS" | wc -l)
    [ "${NEW:-0}" != "0" ] && log "ENG +${NEW} class-hit(s) -> $HITS (last: $(tail -1 "$HITS" | cut -c1-120))"
    L0=$LINES
  fi
  DM=$(dmesg 2>/dev/null | wc -l)
  if [ "$DM" -gt "$DM0" ]; then
    dmesg | sed -n "$((DM0+1)),${DM}p" | grep -iE "xe |gpu|coredump" | while read -r ln; do log "DMESG: $(echo "$ln" | cut -c1-170)"; done
    DM0=$DM
  fi
  log "health=$H container=$C englines=$L0"
  if [ $((i % 6)) -eq 0 ]; then
    XS=$(xpu-smi discovery 2>/dev/null | grep -c "Device State: normal")
    WC=$(docker exec lsv-test sh -c 'ps aux 2>/dev/null | grep -c "VLLM::Worker"' )
    CPU=$(docker exec lsv-test sh -c 'ps aux 2>/dev/null | grep "VLLM::Worker" | grep -v grep' | awk '{s+=$3} END {printf "%.0f", s}')
    FR=$(docker exec lsv-test sh -c 'ls -la /tmp/fr_*.log 2>/dev/null' | awk '{print $5":"$9}' | tr '\n' ' ')
    log "METRIC gpu_normal=${XS:-?}/2 workers=${WC:-0} worker_cpu=${CPU:-0}% fr[${FR}]"
  fi
  if [ "$H" = "000" ] && [ "$DOWN" = "0" ] && [[ "$C" != Up* ]]; then
    log "TELEMETRY_ENGINE_DOWN (container=$C) — capturing artifacts"
    docker logs --tail 50 lsv-test > "$D/engine_down_docker.log" 2>&1 || true
    docker exec lsv-test sh -c 'tail -80 /root/serve_full.log' > "$D/engine_down_serve.log" 2>&1 || true
    docker exec lsv-test sh -c 'for f in /tmp/fr_*.log; do echo "== $f"; tail -30 "$f"; done' > "$D/engine_down_fr.log" 2>&1 || true
    dmesg -T | tail -40 > "$D/engine_down_dmesg.log" 2>&1 || true
    DOWN=1
  fi
  [ "$H" = "200" ] && DOWN=0
  [ $((i % 360)) -eq 0 ] && log "HEARTBEAT still watching (cycle $i)"
  sleep 10
done
