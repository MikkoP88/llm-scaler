#!/bin/bash
# /root/telemetry/monitor3.sh - per-arm comprehensive telemetry (evolves monitor2.sh).
#
# Usage:  ARM=<name> [PORT=8000] [CONTAINER=lsv-test] \
#         [SERVELOG=/root/telemetry/serve_tel.log] ./monitor3.sh &
#
# Outputs under /root/telemetry/arms/<ARM>/ :
#   meta.txt          one-shot arm metadata (image, cmd, full env incl.
#                     baked image ENV - verifies VLLM_XPU_USE_SAMPLER_KERNEL,
#                     PYTORCH_XPU_ALLOC_CONF, CCL_* at a glance)
#   monitor.log       health flaps, token-rate ticks, tripwire events
#   proc_csv.log      ts,health,worker_cpus every 5 s
#   proc_snapshots.log vllm process tree every 15 s
#   xpu_smi.log       guarded xpu-smi dump (both GPUs) every 30 s
#   metrics_jsonl.log filtered /metrics scrape every 5 s when serving
#   serve_issues.log  streamed grep of serve log: ERROR/WARNING/Traceback/
#                     EngineDead/DEVICE_LOST/DFLASH_STALL/FINISH_DIAG/
#                     watchdog/UR error/OOM/arena/TurboQuant
#   dmesg_watch.log   streamed dmesg (iso timestamps, arm-tagged)
#
# Tripwires -> /root/telemetry/capture_once.sh (proven forensic snapshot):
#   T1 new xe "Engine reset" count increase
#   T2 3rd "No available shared memory broadcast" timeout in serve log
#   T3 spin-wedge: health DOWN >= 300 s with a VLLM::Worker >= 70% CPU
#   T4 warmup watchdog traceback in serve log
#
# Read-only w.r.t. GPUs and processes; single guarded xpu-smi loop.
set -u
ARM="${ARM:?set ARM}"
PORT="${PORT:-8000}"
CONTAINER="${CONTAINER:-lsv-test}"
SERVELOG="${SERVELOG:-/root/telemetry/serve_tel.log}"
T=/root/telemetry/arms/$ARM
LOG=$T/monitor.log
CSV=$T/proc_csv.log
mkdir -p "$T"
echo "monitor3 start arm=$ARM port=$PORT container=$CONTAINER serve=$SERVELOG $(date '+%F %T')" >> "$LOG"

# --- one-shot arm metadata
{
  echo "===== arm=$ARM $(date '+%F %T') host_uptime=[$(uptime -p)]"
  echo "--- image / cmd"
  docker inspect -f 'image: {{.Config.Image}}' "$CONTAINER" 2>/dev/null
  docker inspect -f 'cmd: {{join .Config.Cmd " "}}' "$CONTAINER" 2>/dev/null
  echo "--- relevant env (image ENV + docker -e merged)"
  docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null \
    | grep -aE '^(VLLM|CCL|ZE_|PYTORCH|HF_|TRANSFORMERS|PATH)' | sort
} > "$T/meta.txt" 2>&1

# --- serve issue grep stream (background; -n 0 = only NEW lines, no replay
# of pre-monitor history which would false-trip T2/T4)
touch "$T/serve_issues.log"
tail -n 0 -F "$SERVELOG" 2>/dev/null \
  | grep --line-buffered -aE 'ERROR|Error|Traceback|WARNING|EngineDead|DEVICE_LOST|Engine reset|DFLASH_STALL|FINISH_DIAG|atchdog|TimeoutError|UR error|OUT_OF_DEVICE_MEMORY|OOM|arena|TurboQuant' \
  >> "$T/serve_issues.log" &
TAILPID=$!

# --- dmesg stream (background)
(sh -c "dmesg --follow --time-format iso 2>/dev/null || exec dmesg -w" \
  | sed "s/^/[$ARM] /") >> "$T/dmesg_watch.log" 2>/dev/null &
DMPID=$!

cleanup() {
  kill "$TAILPID" "$DMPID" 2>/dev/null
  echo "monitor3 stop arm=$ARM $(date '+%F %T')" >> "$LOG"
}
trap cleanup EXIT INT TERM

LAST_HEALTH=INIT
DOWN_SINCE=0
LAST_RESETS=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true)
LAST_RESETS=${LAST_RESETS:-0}
LAST_TOKENS=""
LAST_TOK_TS=0
LAST_FD=0
LAST_DS=0
SERVED_ONCE=0
TRIPPED_SB=0
TRIPPED_SPIN=0
TRIPPED_WD=0
i=0
while true; do
  TS=$(date '+%F %T')
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/health"; then H=UP; else H=DOWN; fi
  if [ "$H" != "$LAST_HEALTH" ]; then
    echo "$TS health $LAST_HEALTH->$H" >> "$LOG"
    LAST_HEALTH=$H
    if [ "$H" = "DOWN" ]; then DOWN_SINCE=$SECONDS; TRIPPED_SPIN=0; else DOWN_SINCE=0; SERVED_ONCE=1; fi
  fi

  # worker cpu (host view of container processes)
  WC=$(ps -eo pcpu,args --sort=-pcpu 2>/dev/null | grep -a 'VLLM::Worker' | grep -av grep | awk '{printf "%s,", int($1)}' | sed 's/,$//')
  echo "$TS,$H,$WC" >> "$CSV"

  if [ $((i % 3)) -eq 0 ]; then
    ps -eo pid,ppid,stat,pcpu,etime,args --sort=-pcpu 2>/dev/null \
      | grep -aE 'VLLM::|vllm serve|EngineCore' | grep -av grep | cut -c1-150 >> "$T/proc_snapshots.log"
    echo "--- $TS" >> "$T/proc_snapshots.log"
  fi

  # /metrics scrape -> jsonl + token-rate tick
  if [ "$H" = "UP" ] && curl -s -m 3 "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
      | grep -aE '^(vllm:num_requests_(running|waiting)|vllm:generation_tokens_total|vllm:prompt_tokens_total|vllm:time_to_first_token_seconds_(count|sum)|vllm:request_success_total|vllm:e2e_request_latency_seconds_(count|sum)|vllm:spec_decode_(num_accepted_tokens|num_draft_tokens|acceptance_rate)_total|vllm:gpu_cache_usage_perc)' > "$T/.m.tmp"; then
    awk -v ts="$TS" '{print ts, $0}' "$T/.m.tmp" >> "$T/metrics_jsonl.log"
    # Prometheus counters are floats ("1216.0"); coerce via awk int().
    TOK=$(grep -a 'generation_tokens_total' "$T/.m.tmp" | grep -av '^#' | awk '{print int($2)}' | head -1)
    if [ -n "$TOK" ] && [ "$TOK" != "$LAST_TOKENS" ]; then
      if [ -n "$LAST_TOKENS" ] && [ "$LAST_TOK_TS" -gt 0 ]; then
        DT=$((SECONDS - LAST_TOK_TS)); [ "$DT" -le 0 ] && DT=1
        RUN=$(grep -a 'num_requests_running' "$T/.m.tmp" | grep -av '^#' | awk '{print int($2)}' | head -1)
        WAIT=$(grep -a 'num_requests_waiting' "$T/.m.tmp" | grep -av '^#' | awk '{print int($2)}' | head -1)
        RATE=$(awk -v a="$LAST_TOKENS" -v b="$TOK" -v dt="$DT" 'BEGIN{printf "%.1f", (b-a)/dt}')
        echo "$TS gen_rate=${RATE} tok/s total=$TOK running=${RUN:-?} waiting=${WAIT:-?}" >> "$LOG"
      fi
      LAST_TOKENS=$TOK
      LAST_TOK_TS=$SECONDS
    fi
  fi

  # xpu-smi every 6th tick (~30 s), guarded single loop, per-device
  if [ $((i % 6)) -eq 0 ]; then
    if ! pgrep -f 'xpu-smi dump' >/dev/null 2>&1; then
      { timeout 8 xpu-smi dump --device 0 --metrics pu 2>&1 | tail -n +2 | tail -3
        timeout 8 xpu-smi dump --device 1 --metrics pu 2>&1 | tail -n +2 | tail -3
        echo "--- $TS"; } >> "$T/xpu_smi.log" &
    fi
  fi

  # T1: new engine resets
  RESETS=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true)
  RESETS=${RESETS:-0}
  if [ "$RESETS" -gt "$LAST_RESETS" ]; then
    echo "$TS TRIPWIRE new engine resets $LAST_RESETS->$RESETS" >> "$LOG"
    bash /root/telemetry/capture_once.sh "reset${RESETS}_$ARM" >> "$LOG" 2>&1 &
    LAST_RESETS=$RESETS
  fi

  # T2: shm_broadcast timeouts in serve log (boot-time compile noise cannot
  # trip: only after the server was healthy at least once this arm)
  SB=0
  [ -f "$SERVELOG" ] && SB=$(grep -ac 'No available shared memory broadcast' "$SERVELOG" 2>/dev/null || true)
  SB=${SB:-0}
  if [ "$SB" -ge 3 ] && [ "$TRIPPED_SB" -eq 0 ] && [ "$SERVED_ONCE" -eq 1 ]; then
    TRIPPED_SB=1
    echo "$TS TRIPWIRE shm_broadcast timeouts=$SB" >> "$LOG"
    bash /root/telemetry/capture_once.sh "shmbc_$ARM" >> "$LOG" 2>&1 &
  fi

  # T3: spin-wedge (health down >=300 s with a hot worker)
  if [ "$H" = "DOWN" ] && [ "$DOWN_SINCE" -gt 0 ] && [ $((SECONDS - DOWN_SINCE)) -ge 300 ]; then
    HOT=$(echo "$WC" | tr ',' '\n' | awk '$1>=70' | wc -l)
    if [ "$HOT" -ge 1 ] && [ "$TRIPPED_SPIN" -eq 0 ]; then
      TRIPPED_SPIN=1
      echo "$TS TRIPWIRE spin_wedge health_down=$((SECONDS - DOWN_SINCE))s worker_cpu=[$WC]" >> "$LOG"
      bash /root/telemetry/capture_once.sh "spin_$ARM" >> "$LOG" 2>&1 &
    fi
  fi

  # T4: warmup watchdog actually FIRED (faulthandler dump header). The
  # "watchdog armed/disarmed" info lines must not trip this.
  if [ "$TRIPPED_WD" -eq 0 ] && grep -aqE 'Timeout \([0-9]+s?\)|faulthandler' "$T/serve_issues.log" 2>/dev/null; then
    TRIPPED_WD=1
    echo "$TS TRIPWIRE warmup_watchdog_fired" >> "$LOG"
    bash /root/telemetry/capture_once.sh "watchdog_$ARM" >> "$LOG" 2>&1 &
  fi

  # diagnostic counters: #05d FINISH_DIAG + dflash stall guard hits
  FD=$(grep -ac 'FINISH_DIAG' "$T/serve_issues.log" 2>/dev/null || true); FD=${FD:-0}
  DS=$(grep -ac 'DFLASH_STALL' "$T/serve_issues.log" 2>/dev/null || true); DS=${DS:-0}
  [ "$FD" -gt "$LAST_FD" ] && echo "$TS FINISH_DIAG count $LAST_FD->$FD" >> "$LOG"
  [ "$DS" -gt "$LAST_DS" ] && echo "$TS DFLASH_STALL count $LAST_DS->$DS" >> "$LOG"
  LAST_FD=$FD
  LAST_DS=$DS

  i=$((i + 1))
  sleep 5
done
