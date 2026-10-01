#!/bin/bash
# stall_scope.sh — v127 WS-B: automatic stall-evidence capture.
#
# STANDING TELEMETRY DIRECTIVE (user, 2026-09-30): root issues must be
# capturable AT THE MOMENT they happen. This script fires a full evidence
# bundle when a stall signature is seen (or on demand) so the next incident
# is diagnosed from a tarball, not from memory:
#   - full /metrics snapshot (every vllm: line)
#   - /root/serve_full.log tail (engine log; only exists on chain-launched
#     lanes — the 08:56 manual serve had none, which is why P32 was blind)
#   - py-spy dumps of EngineCore + both TP workers (function stacks)
#   - xpu-smi utilization/memory for both devices
#   - dmesg tail (GPU resets), ps snapshot, litellm-proxy error tail
#   - last 5 minutes of metrics_recorder JSONL (the run-up to the stall)
#
# Signatures (any fires the capture):
#   waiting   >  WAIT_THRESH   for > WAIT_SUSTAIN seconds
#   kv_perc   >  KV_THRESH     for > KV_SUSTAIN seconds   (eviction pressure)
#   ttft_tail — disabled by default (needs bucket deltas; the recorder JSONL
#               covers it) — enable via TTFT_WATCH=1 reading the recorder file
# Cooldown per episode: 300 s (one bundle per storm, not per sample).
#
# WARN/READ-ONLY by design: never restarts or kills anything, never reads
# device state from the engine forward path (READOUT LAW — everything here
# is host-side observation).
#
# usage:  stall_scope.sh --now                 # manual capture
#         nohup stall_scope.sh --watch ... &   # daemon mode
set -u
HOST="${SCOPE_HOST:-http://localhost:8000}"
OUT="${SCOPE_OUT:-/root/build/captures}"
WAIT_THRESH="${SCOPE_WAIT_THRESH:-3}"
WAIT_SUSTAIN="${SCOPE_WAIT_SUSTAIN:-30}"
KV_THRESH="${SCOPE_KV_THRESH:-0.97}"
KV_SUSTAIN="${SCOPE_KV_SUSTAIN:-60}"
COOLDOWN=300

mkdir -p "$OUT"
capture() {
  local tag="$1" ts
  ts=$(date -u +%Y%m%dT%H%M%S)
  local dir="$OUT/${tag}_${ts}"
  mkdir -p "$dir"
  echo "STALL_SCOPE capture tag=$tag -> $dir"

  curl -s -m 8 "$HOST/metrics" > "$dir/metrics.txt" 2>/dev/null || echo "metrics scrape failed" > "$dir/metrics.err"
  docker exec lsv-test sh -c 'tail -c 200000 /root/serve_full.log' > "$dir/serve_log_tail.txt" 2>/dev/null \
    || echo "no serve_full.log (lane not chain-launched = telemetry blind)" > "$dir/serve_log_tail.txt"
  for pid in $(docker exec lsv-test sh -c "ps -ef | grep -E 'EngineCore|from multiprocessing' | grep -v grep" 2>/dev/null | tr -s ' ' | cut -d' ' -f2); do
    docker exec lsv-test /opt/venv/bin/py-spy dump --pid "$pid" > "$dir/pyspy_${pid}.txt" 2>/dev/null || true
  done
  for p in 0 1; do
    xpu-smi dump --device "$p" --metrics UTILIZATION,MEMORY --number 3 > "$dir/xpusmi_${p}.txt" 2>/dev/null || true
  done
  dmesg | tail -80 > "$dir/dmesg_tail.txt" 2>/dev/null || true
  ps -ef > "$dir/ps_ef.txt" 2>/dev/null || true
  docker logs litellm-proxy --since 30m > "$dir/litellm_30m.log" 2>&1 || true
  tail -40 /root/build/telemetry/metrics_*.jsonl > "$dir/recorder_tail.txt" 2>/dev/null || true

  tar -czf "$dir.tar.gz" -C "$dir" . >/dev/null 2>&1 && rm -rf "$dir"
  echo "STALL_SCOPE_DONE $dir.tar.gz"
}

get_val() {
  curl -s -m 6 "$HOST/metrics" 2>/dev/null | grep -m1 "^vllm:$1{" | tr -s ' ' | cut -d' ' -f2
}

case "${1:-}" in
  --now)
    capture manual
    exit 0
    ;;
  --watch)
    LAST_FIRE=0
    wait_since=0
    kv_since=0
    echo "STALL_SCOPE watch start wait>${WAIT_THRESH}for${WAIT_SUSTAIN}s kv>${KV_THRESH}for${KV_SUSTAIN}s host=$HOST"
    while sleep 10; do
      now=$(date +%s)
      w=$(get_val num_requests_waiting)
      k=$(get_val kv_cache_usage_perc)
      [ -z "$w" ] && w=-1
      [ -z "$k" ] && k=-1

      if [ "$(awk -v a="$w" -v b="$WAIT_THRESH" 'BEGIN{print (a>b)?1:0}')" = "1" ]; then
        [ "$wait_since" -eq 0 ] && wait_since=$now
      else
        wait_since=0
      fi
      if [ "$(awk -v a="$k" -v b="$KV_THRESH" 'BEGIN{print (a>b)?1:0}')" = "1" ]; then
        [ "$kv_since" -eq 0 ] && kv_since=$now
      else
        kv_since=0
      fi

      fire=""
      if [ "$wait_since" -gt 0 ] && [ $((now - wait_since)) -ge "$WAIT_SUSTAIN" ]; then fire="waiting_storm"; fi
      if [ -z "$fire" ] && [ "$kv_since" -gt 0 ] && [ $((now - kv_since)) -ge "$KV_SUSTAIN" ]; then fire="kv_pressure"; fi

      if [ -n "$fire" ] && [ $((now - LAST_FIRE)) -ge "$COOLDOWN" ]; then
        LAST_FIRE=$now
        capture "$fire"
      fi
    done
    ;;
  *)
    echo "usage: stall_scope.sh --now | --watch"
    echo "env: SCOPE_WAIT_THRESH($WAIT_THRESH) SCOPE_WAIT_SUSTAIN($WAIT_SUSTAIN) SCOPE_KV_THRESH($KV_THRESH) SCOPE_KV_SUSTAIN($KV_SUSTAIN)"
    exit 64
    ;;
esac
