#!/bin/bash
# p22c1_traj_run.sh — run p22c1_traj.py with the serve DOWN (the serve
# preallocates nearly all GPU memory; the traj process needs headroom).
# Mirrors p22c1_ctrl_probe.sh's proven teardown/restore patterns:
# plain-pattern pkills (ERE lesson), port-down proof before proceeding,
# fresh serve_full.log + spec_lines assert on restore, watchdog paused
# throughout and re-armed at the end. Runs AFTER v3 completes.
set -u
L=/root/build/lce1/p22c1_traj.log
touch /root/build/lane_watchdog.paused
{
echo "=== P22C1 TRAJ start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
echo "container image=$IMG"
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

echo "--- 1. .so posture (must be the rebuild, sha prefix 7c5fdcd) ---"
SO=/opt/venv/lib/python3.12/site-packages/custom_esimd_kernels_vllm/custom_esimd_kernels_lgrf.cpython-312-x86_64-linux-gnu.so
SH=$(docker exec lsv-test sha256sum $SO | cut -c1-7)
echo "lgrf .so sha7=$SH"
[ "$SH" = "7c5fdcd" ] || { echo "ABORT_NOT_REBUILD_SO"; exit 1; }

echo "--- 2. tear down serve (plain pkills + port-down proof) ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; echo "port 8000 down after ${i}x5s"; break; fi
done
[ "$PD" -eq 1 ] || { echo "ABORT_SERVE_STILL_UP"; exit 1; }
sleep 4

echo "--- 3. run trajectory differential (GPU free) ---"
docker exec lsv-test python3 /root/p22c1_traj.py
RC=$?
echo "traj rc=$RC"

echo "--- 4. restore certified spec-4 serve ---"
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
echo "restored spec_lines=$SPL (want >=1)"
[ "$SPL" -ge 1 ] || { echo "ABORT_RESTORE_NOT_SPEC4"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C1 TRAJ DONE $(date +%F' '%T) rc=$RC ==="
echo "P22C1_TRAJ_RUN_DONE rc=$RC"
} > "$L" 2>&1
tail -60 "$L"
echo P22C1_TRAJ_SCRIPT_DONE
