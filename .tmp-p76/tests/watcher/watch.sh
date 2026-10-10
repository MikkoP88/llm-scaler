#!/usr/bin/env bash
# watch.sh — WATCHER mission: continuous lane telemetry harvester (READ-ONLY).
#
# Lane: lsv-test @ 10.20.3.65 (vLLM serve, log inside container at
# /root/serve_full.log). This script NEVER writes to the remote, never
# restarts anything, and only issues: /health curls + `docker exec ... wc/tail`
# reads (no -d, no lifecycle commands).
#
# Harvest strategy: incremental line-marker reads (wc -l baseline, then
# `tail -n +K | head -n 5000`). Each physical log line is fetched exactly once,
# which is the SPECTIMING dedupe mechanism; fingerprint/error lines are logged
# with first-seen harvest timestamps.
#
# Outputs in this directory:
#   health.log        "ts code" per probe (post-death probes throttled to 1/60s)
#   spectiming.log    raw SPECTIMING lines (deduped by construction)
#   fingerprints.log  "ts <line>" for v52f PRE-COPY / v52f DEF-PP
#   errors.log        "ts <line>" for ERROR|Traceback|ccs|reset|DEVICE_LOST|EngineCore died
#   raw_new.log       full new-line feed (post-mortem value, size-guarded 300MB)
#   watcher_status.log per-tick status + events (rotation, death, plink failures)
#   crash_tail.txt    one-shot tail -300 on first 000x2 after boot grace
#   DEATH_TIME        timestamp file, created on death detection
#   summary.txt       end-of-watch aggregates
#
# Timeline: boot grace 12 min (000 expected, no action). Hard deadline ~93 min
# so the WATCH_LOG.md report fits the 100-minute mission budget.

cd "$(dirname "$0")" || exit 1

HOSTPW=<HOSTPW>  # redacted before commit; original on host only
HOST=root@10.20.3.65
BASE=http://10.20.3.65:8000

START=$(date +%s)
DEADLINE=$((START + 5600))   # ~93 min watch, report headroom within 100-min budget
GRACE_END=$((START + 720))   # 12 min boot grace
SEED_LINES=14293             # pre-flight baseline, taken 2026-10-03 01:36 (health already 200)

last_lines=$SEED_LINES
zg=0                         # consecutive 000 codes counted only AFTER boot grace
death_ts=""
crash_done=0
crash_tries=0
last_probe=0
ticks=0

log() { echo "[$(date '+%F %T')] $*" >> watcher_status.log; }

log "WATCHER start seed_lines=$SEED_LINES grace_end_epoch=$GRACE_END deadline_epoch=$DEADLINE"

next_tick=$START
while :; do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then
    log "deadline reached — exiting loop"
    break
  fi
  ts=$(date '+%F %T')

  # ---- 1. health ---------------------------------------------------------
  code="(throttled)"
  if [ -z "$death_ts" ] || [ $((now - last_probe)) -ge 60 ]; then
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 "$BASE/health")
    last_probe=$now
    echo "$ts $code" >> health.log
    if [ "$code" = "000" ]; then
      # boot grace: 000 is expected, logged, no action; count only post-grace
      if [ "$now" -ge "$GRACE_END" ]; then
        zg=$((zg + 1))
      fi
    else
      zg=0
    fi
    if [ -z "$death_ts" ] && [ "$zg" -ge 2 ]; then
      death_ts="$ts"
      echo "$death_ts" > DEATH_TIME
      log "DEATH_TIME marked (000 x2 after boot grace) death_ts=$death_ts"
    fi
  fi

  # ---- crash-evidence step: once, on first 000x2 --------------------------
  if [ -n "$death_ts" ] && [ "$crash_done" -eq 0 ] && [ "$crash_tries" -lt 10 ]; then
    crash_tries=$((crash_tries + 1))
    timeout 60 plink -batch -pw "$HOSTPW" "$HOST" \
      "docker exec lsv-test tail -300 /root/serve_full.log" \
      > crash_tail.txt 2>> watcher_status.log
    rc=$?
    if [ "$rc" -eq 0 ] && [ -s crash_tail.txt ]; then
      crash_done=1
      log "crash evidence banked -> crash_tail.txt ($(wc -l < crash_tail.txt | tr -d ' ') lines)"
    else
      log "crash-evidence attempt $crash_tries rc=$rc — retrying next tick"
    fi
  fi

  # ---- 2. incremental serve-log harvest (read-only) ------------------------
  got=0; new_st=0; new_fp=0; new_er=0
  total=$(timeout 45 plink -batch -pw "$HOSTPW" "$HOST" \
    "docker exec lsv-test wc -l /root/serve_full.log" 2>> watcher_status.log \
    | awk '{print $1; exit}')
  if [ -n "$total" ] && [ "$total" -eq "$total" ] 2>/dev/null; then
    if [ "$total" -lt "$last_lines" ]; then
      log "LOG ROTATION/RESET: total=$total < last=$last_lines — baseline reset to 0"
      last_lines=0
    fi
    if [ "$total" -gt "$last_lines" ]; then
      nextln=$((last_lines + 1))
      timeout 90 plink -batch -pw "$HOSTPW" "$HOST" \
        "docker exec lsv-test sh -c \"tail -n +${nextln} /root/serve_full.log | head -n 5000\"" \
        > fetch.tmp 2>> watcher_status.log
      rc=$?
      if [ "$rc" -eq 0 ]; then
        tr -d '\r' < fetch.tmp > fetch.clean
        # drop trailing partial line (log mid-write race) — refetched next tick
        if [ -n "$(tail -c 1 fetch.clean)" ]; then
          sed -i '$d' fetch.clean
        fi
        got=$(wc -l < fetch.clean | tr -d ' ')
        if [ "$got" -gt 0 ]; then
          rawsz=$(stat -c '%s' raw_new.log 2>/dev/null || echo 0)
          if [ "$rawsz" -lt 314572800 ]; then
            cat fetch.clean >> raw_new.log
          fi
          grep -a 'SPECTIMING' fetch.clean >> spectiming.log
          new_st=$(grep -ac 'SPECTIMING' fetch.clean)
          grep -aE 'v52f PRE-COPY|v52f DEF-PP' fetch.clean | sed "s/^/$ts /" >> fingerprints.log
          new_fp=$(grep -acE 'v52f PRE-COPY|v52f DEF-PP' fetch.clean)
          grep -aE 'ERROR|Traceback|ccs|reset|DEVICE_LOST|EngineCore died' fetch.clean \
            | sed "s/^/$ts /" >> errors.log
          new_er=$(grep -acE 'ERROR|Traceback|ccs|reset|DEVICE_LOST|EngineCore died' fetch.clean)
          last_lines=$((last_lines + got))
        fi
      else
        log "harvest plink rc=$rc — baseline held at $last_lines, resume next tick"
      fi
    fi
  else
    log "wc probe failed/empty (container gone or net blip) — baseline held at $last_lines"
  fi

  # ---- 3. per-tick status --------------------------------------------------
  ticks=$((ticks + 1))
  echo "$ts tick=$ticks health=$code lines=$last_lines(+$got) st=+$new_st fp=+$new_fp err=+$new_er" \
    >> watcher_status.log

  # ---- pacing: 20s aligned ticks -------------------------------------------
  next_tick=$((next_tick + 20))
  now=$(date +%s)
  sl=$((next_tick - now))
  [ "$sl" -lt 1 ] && sl=1
  sleep "$sl"
done

# ---- final aggregates --------------------------------------------------------
{
  echo "=== WATCHER summary generated $(date '+%F %T') ==="
  echo "ticks=$ticks"
  echo "-- health code tallies (code count) --"
  awk '{c[$2]++} END {for (k in c) print k, c[k]}' health.log
  echo "first_200 $(grep -m1 ' 200$' health.log || echo none)"
  echo "last_probe $(tail -n 1 health.log)"
  echo "-- spectiming --"
  echo "count $(wc -l < spectiming.log 2>/dev/null | tr -d ' ')"
  echo "-- fingerprints --"
  echo "count $(wc -l < fingerprints.log 2>/dev/null | tr -d ' ')"
  echo "-- errors --"
  echo "count $(wc -l < errors.log 2>/dev/null | tr -d ' ')"
  if [ -f DEATH_TIME ]; then echo "DEATH_TIME $(cat DEATH_TIME)"; fi
} > summary.txt 2>&1

log "watch complete — outputs finalized ($(date '+%F %T'))"
