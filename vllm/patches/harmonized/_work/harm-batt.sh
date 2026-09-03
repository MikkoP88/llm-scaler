#!/bin/bash
# harm-batt.sh <lane> <image> <bootp|raw> — harmonization validation battery.
#   bootp : baked image -> bootp.sh nospec lane (marker gate skips patching)
#   raw   : authentic tree -> serve_user_nospec.sh DIRECTLY (no boot patches)
# Battery: probe10 (P1 x10 hashes) / 2k bench x2 / 65k x2 (warm) / conc16.
# Everything appended to /root/build/harm-batt-<lane>.out
set -u
LANE="$1"; IMG="$2"; MODE="$3"
OUT="/root/build/harm-batt-${LANE}.out"
: > "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }

log "LANE=$LANE IMG=$IMG MODE=$MODE"

if [ "$MODE" = "bootp" ]; then
  bash /root/build/bootp.sh nospec "" "hb_${LANE}" "" "$IMG" >> "$OUT" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "BOOT FAILED rc=$rc — battery aborted"; exit 1; fi
else
  bash /root/build/serve_user_nospec.sh '' "" "hb_${LANE}" "" 512 "$IMG" >> "$OUT" 2>&1
  code=000
  for i in $(seq 1 90); do
    sleep 10
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null)
    if [ "$code" = "200" ]; then log "HEALTH OK after ~$((i*10))s (raw boot)"; break; fi
    if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
      log "CONTAINER DIED (raw boot)"; exit 1
    fi
  done
  if [ "$code" != "200" ]; then log "HEALTH TIMEOUT (raw boot) — battery aborted"; exit 1; fi
fi

log "=== probe10 (P1 x10, temp0, mt160) ==="
python3 /root/build/harm_probe10.py 2>&1 | tee -a "$OUT"

log "=== 2k bench x2 ==="
python3 /root/build/bench_completions.py --max-tokens 2048 --tag "${LANE}-2k-a" 2>&1 | tee -a "$OUT" | grep TAG
python3 /root/build/bench_completions.py --max-tokens 2048 --tag "${LANE}-2k-b" 2>&1 | tee -a "$OUT" | grep TAG

log "=== 65k x2 (second = warm) ==="
python3 /root/build/t65k.py 2>&1 | tee -a "$OUT" | tail -2
python3 /root/build/t65k.py 2>&1 | tee -a "$OUT" | tail -2

log "=== conc16 ==="
python3 /root/build/harm_conc16.py 2>&1 | tee -a "$OUT"

log "BATTERY DONE: $LANE"
