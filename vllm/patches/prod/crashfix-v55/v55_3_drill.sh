#!/bin/bash
# v55.3 kill-drill — prove P2 (fast-clean death) end-to-end on the live lane.
#
# Method: boot the staged v55.3 boot script with the decode bound cranked to
# 0.001s (VLLM_V55_DECODE_TIMEOUT_S=0.001). Every real v55 wait on a pending
# event then times out instantly — the same code path a genuine device-dead
# event takes, without needing to reproduce the actual deadlock. Expected:
# engine dies during warmup/first step, /root/v55_3_crash.log written with
# wait name + bound + rec_step_tokens + f15b ring tail, whole process group
# killed (container exited, zero leaked EngineCore/spawn procs on host).
#
# Watchdog is PAUSED for the drill; restore is manual+verified afterwards.
# Evidence -> /root/build/lce1/v55_3_drill_<TS>/ .
set -uo pipefail
cd /root/build
TS=$(date -u +%Y%m%dT%H%M%S)
EV=/root/build/lce1/v55_3_drill_$TS
mkdir -p "$EV"
LOG=$EV/drill.log
exec > >(tee -a "$LOG") 2>&1

echo "[drill] start $(date -u) ts=$TS"
echo "[drill] pausing lane watchdog"
touch /root/build/lane_watchdog.paused
systemctl is-active lane-watchdog || true

echo "[drill] building drill boot (decode bound 0.001s)"
sed 's/VLLM_V55_DECODE_TIMEOUT_S=30/VLLM_V55_DECODE_TIMEOUT_S=0.001/' \
    /root/build/repro_bootV1212.sh > /root/build/repro_boot_drill.sh
grep -n 'V55_DECODE_TIMEOUT' /root/build/repro_boot_drill.sh

echo "[drill] removing current lane container"
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 5

echo "[drill] booting drill lane $(date -u)"
bash /root/build/repro_boot_drill.sh DRILL > "$EV/boot_drill.out" 2>&1
RC=$?
echo "[drill] boot script rc=$RC (7=CONTAINER DIED is the expected fast-kill)"
cp "$EV/boot_drill.out" "$EV/boot_drill.copy.out" 2>/dev/null || true

sleep 10
STATE=$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} fini={{.State.FinishedAt}}' lsv-test 2>/dev/null || echo "GONE")
echo "[drill] container state: $STATE"

# unexpected-but-possible: all warmup waits completed before the 1ms bound
# tripped and health went 200 — force a fresh step through a real request.
if [ "$RC" = "0" ] && docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
  echo "[drill] lane reached HEALTH_OK — firing trigger request $(date -u)"
  curl -s -m 30 http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
    || true
  for i in $(seq 1 12); do
    sleep 5
    if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
      echo "[drill] container died ${i}x5s after trigger"
      break
    fi
  done
  STATE=$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} fini={{.State.FinishedAt}}' lsv-test 2>/dev/null || echo "GONE")
  echo "[drill] container state now: $STATE"
fi

echo "[drill] leaked engine procs on host (expect none):"
if ps -ef | grep -E 'from multiprocessing|EngineCore|spawn_main' | grep -v grep | grep -v drill; then
  echo "[drill] LEAK WARNING — processes above survived"
else
  echo "[drill] none"
fi

echo "[drill] pulling crash evidence from dead container"
docker cp lsv-test:/root/v55_3_crash.log "$EV/v55_3_crash.log" 2>/dev/null \
  && cat "$EV/v55_3_crash.log" \
  || echo "[drill] NO CRASH LOG — investigate boot_drill.out"
docker cp lsv-test:/root/serve_full.log "$EV/serve_full_drill.log" 2>/dev/null || true
docker cp lsv-test:/root/f15b_dump_1.log "$EV/f15b_dump_1.log" 2>/dev/null || true

echo "[drill] VERDICT inputs:"
echo "  rc=$RC state=$STATE"
if [ -f "$EV/v55_3_crash.log" ]; then
  echo "[drill] PASS candidate: crash log exists — verify wait=/bound=/ring tail above"
else
  echo "[drill] INCONCLUSIVE: no crash log (engine may have died before first v55 wait; check boot_drill.out)"
fi
echo "[drill] NOTE: lane is DOWN and watchdog PAUSED — run restore next:"
echo "  bash /root/build/repro_bootV1212.sh DRILLRESTORE"
