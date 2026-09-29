#!/bin/bash
# p22c3_run.sh — run the P22-C C3 ESIMD op microbench (p22c3_ops.py) in a
# serve-down window on lsv-test. The bench times esimd_gdn_conv_fused_seq
# + _seq_spec directly (pool dtype fp16/e4m3/e5m2 the single variable);
# PASS = every fp8 cell <= 1.10x fp16. Op timing needs the XPU idle, so:
# pause watchdog -> teardown serve -> bench -> restore certified serve
# (spec_lines=2, markers=0) -> re-arm. EXIT trap mirrors the restore.
# Prereq: P22C5_AGG2_DONE (this runs in the gap after the agg matrix).
set -u
L=/root/build/lce1/p22c3_run.log

restore_lane() {
  [ -f /root/build/lane_watchdog.paused ] || return 0
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true" >/dev/null 2>&1
  for i in $(seq 1 18); do sleep 5; curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || break; done
  sleep 4
  docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo trap-restore-launched" >> "$L" 2>&1
  for i in $(seq 1 48); do sleep 10; curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && break; done
  rm -f /root/build/lane_watchdog.paused
  echo "trap lane_watchdog re-armed ($(date +%T))" >> "$L" 2>&1
}
trap restore_lane EXIT

{
echo "=== P22C3 RUN start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
grep -q 'P22C5_AGG2_DONE' /root/build/lce1/p22c5_agg2.log || { echo "ABORT_AGG2_NOT_DONE"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

echo "--- teardown serve (bench needs idle XPU) ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; break; fi
done
[ "$PD" -eq 1 ] || { echo "ABORT_SERVE_STILL_UP"; exit 1; }
sleep 8

echo "--- run the op microbench ---"
docker cp /root/build/p22c3_ops.py lsv-test:/root/ || { echo "ABORT_CP"; exit 1; }
docker exec lsv-test /opt/venv/bin/python /root/p22c3_ops.py
RC=$?
echo "bench rc=$RC"

echo "--- restore certified serve ---"
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log; true")
MK=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|v125 root fix\|perf-v125 C1\|v125 P22C5\|v125 C5DUPD\|FP8_NATIVE ENGAGED' /root/serve_full.log; true")
echo "restored spec_lines=$SPL marker-lines=$MK (want 2 / 0)"
[ "$SPL" -eq 2 ] || { echo "ABORT_RESTORE_SPEC"; exit 1; }
[ "$MK" -eq 0 ] || { echo "ABORT_RESTORE_MARKERS"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
if [ "$RC" -eq 0 ]; then
  echo "=== P22C3 RUN DONE (PASS) $(date +%F' '%T) ==="
  echo "P22C3_RUN_DONE"
else
  echo "=== P22C3 RUN DONE (REGRESSION CELLS PRESENT) $(date +%F' '%T) ==="
  echo "P22C3_RUN_DONE_REGRESS"
fi
} > "$L" 2>&1
tail -70 "$L"
echo P22C3_RUN_SCRIPT_DONE
