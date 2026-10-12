#!/bin/bash
# fn8 DEC40 — bounded boot-retry wrapper (FN8_07 companion).
# Usage: sh /stage/fn8_boot_retry.sh /stage/fn8_serve_d8s3.sh
#
# FN8_07 fast-fails a dead D2H chunk (worker exit 87 -> EngineCore fatal
# -> serve exits); this wrapper RELAUNCHES the boot (up to 4 tries).
# Backstop: any try that produces no "Application startup complete"
# within 22 min is killed hard (covers loader-side hangs outside uva.py
# and any residual wedge mode) and retried.  A successful boot runs
# forever (serve never exits), so the loop simply never continues.
#
# NOTE: pkill patterns here are bracket-quoted; do NOT add a plain
# 'fn8_serve' pattern — it matches this wrapper's own sh -c cmdline and
# kills the wrapper itself (learned 2026-10-12).
SERVE="$1"
[ -x "$SERVE" ] || [ -f "$SERVE" ] || { echo "usage: $0 <serve-script>" ; exit 9; }
TAG=$(basename "$SERVE" .sh | sed 's/^fn8_serve_//')
LOG=/stage/boot_retry_$TAG.log
SLOG=/stage/serve_$TAG.log
: > "$LOG"
for i in 1 2 3 4; do
  echo "=== BOOT_TRY $i start $(date -u +%H:%M:%S) ===" >> "$LOG"
  sh "$SERVE" &
  PID=$!
  OK=0
  SECS=0
  while [ "$SECS" -lt 1320 ]; do
    sleep 5; SECS=$((SECS+5))
    if grep -q "Application startup complete" "$SLOG" 2>/dev/null; then OK=1; break; fi
    kill -0 "$PID" 2>/dev/null || break
  done
  if [ "$OK" -eq 1 ]; then
    echo "=== BOOT_TRY $i STARTUP_OK $(date -u +%H:%M:%S) ===" >> "$LOG"
    wait "$PID"
    echo "=== SERVE_EXITED rc=$? $(date -u +%H:%M:%S) ===" >> "$LOG"
    exit 0
  fi
  echo "=== BOOT_TRY $i KILLED $(date -u +%H:%M:%S) ===" >> "$LOG"
  kill -9 "$PID" 2>/dev/null
  pkill -9 -f 'VLLM[:][:]' 2>/dev/null
  pkill -9 -f 'EngineCor[e]' 2>/dev/null
  sleep 20
done
echo "=== ALL_BOOT_TRIES_FAILED $(date -u +%H:%M:%S) ===" >> "$LOG"
