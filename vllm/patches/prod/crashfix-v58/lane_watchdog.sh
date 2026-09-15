#!/bin/bash
# lane_watchdog.sh — containment watchdog (WEDGE_PLAN §22 C item 3,
# deployed on user sign-off). Scope: §14-class lane death (health-000
# while container up) and container-gone. On trigger: §18 evidence
# capture (q22_crash_capture.sh, best-effort, archived per event) then
# auto-relaunch of the standing lineage (repro_bootQ21b.sh WD_*).
# Controls:
#   touch /root/build/lane_watchdog.paused  -> PAUSE (experiments)
#   rm    /root/build/lane_watchdog.paused  -> ARM
# Guards: young-container grace (<420s old = booting, no trigger),
# 2 consecutive fails required, min 600s between relaunches, max 3
# relaunches/hour then self-pause (avoids reboot storm).
LOG=/root/build/lane_watchdog.log
FLAG=/root/build/lane_watchdog.paused
log(){ echo "$(date '+%F %T') $*" >> $LOG; }

LAST_LAUNCH=$(date +%s)
FAILS=0
RELAUNCH_HR=0
HR_START=$(date +%s)
CYCLES=0
log "watchdog start pid=$$ paused=$([ -f $FLAG ] && echo yes || echo no) lineage=repro_bootQ21b.sh"

while :; do
  sleep 60
  NOW=$(date +%s)
  CYCLES=$((CYCLES+1))
  if [ $((NOW-HR_START)) -ge 3600 ]; then HR_START=$NOW; RELAUNCH_HR=0; fi
  if [ -f $FLAG ]; then
    FAILS=0
    [ $((CYCLES % 10)) -eq 0 ] && log "alive paused code-skip cycles=$CYCLES"
    continue
  fi
  # grace right after a watchdog relaunch
  [ $((NOW-LAST_LAUNCH)) -lt 420 ] && continue

  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || echo 000)
  UP=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c '^lsv-test$')

  if [ "$code" = "200" ]; then
    FAILS=0
    [ $((CYCLES % 10)) -eq 0 ] && log "alive armed code=200 cycles=$CYCLES"
    continue
  fi

  # young-container grace: deliberate or watchdog boot still warming
  if [ "$UP" = "1" ]; then
    AGE=$(docker inspect -f '{{.State.StartedAt}}' lsv-test 2>/dev/null)
    AS=$(date -d "${AGE%.*}" +%s 2>/dev/null)
    if [ -n "$AS" ] && [ $((NOW-AS)) -lt 420 ]; then
      FAILS=0
      log "booting: container age $((NOW-AS))s code=$code (grace)"
      continue
    fi
  fi

  FAILS=$((FAILS+1))
  log "unhealthy code=$code up=$UP fails=$FAILS"

  if [ "$FAILS" -lt 2 ]; then continue; fi
  if [ $((NOW-LAST_LAUNCH)) -lt 600 ]; then log "TRIGGER suppressed (rate <600s)"; continue; fi
  if [ "$RELAUNCH_HR" -ge 3 ]; then
    log "TRIGGER 3/hour cap reached — SELF-PAUSING; rm $FLAG to re-arm"
    touch $FLAG
    continue
  fi

  log "TRIGGER capture+relaunch #$((RELAUNCH_HR+1)) this hour"
  timeout 180 bash /root/build/q22_crash_capture.sh >> $LOG 2>&1
  TS=$(date +%H%M%S)
  [ -d /root/build/lce1/Q22_crash ] && mv /root/build/lce1/Q22_crash /root/build/lce1/WD_crash_$TS
  LAST_LAUNCH=$(date +%s)
  FAILS=0
  cd /root/build
  nohup bash repro_bootQ21b.sh WD_$TS > lce1/boot_WD_$TS.out 2>&1 < /dev/null &
  log "relaunch WD_$TS fired"
  RELAUNCH_HR=$((RELAUNCH_HR+1))
done
