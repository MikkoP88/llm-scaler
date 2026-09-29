#!/bin/bash
# p22a_dbg_run.sh — one debug window for the spec-op e4m3 defect:
# pause watchdog, serve down, swap in the P22A .so, run p22a_dbg_spec.py,
# restore the original .so, certified serve back up, jitwarm, re-arm.
set -u
L=/root/build/lce1/p22a_dbg.log
SP=/opt/venv/lib/python3.12/site-packages
NEW_SO=/root/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/custom_esimd_kernels_lgrf.cpython-312-x86_64-linux-gnu.so
OLD_SO=$SP/custom_esimd_kernels_vllm/custom_esimd_kernels_lgrf.cpython-312-x86_64-linux-gnu.so
M=qwen3.8-27b-fp8
touch /root/build/lane_watchdog.paused
{
echo "=== P22A DBG start $(date +%T) ==="

wait_health () {
  local i=0 t=${1:-480}
  while [ $i -lt $t ]; do
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && return 0
    sleep 5; i=$((i+5))
  done
  return 1
}

docker exec lsv-test pkill -f 'vllm serve' || true
sleep 8
docker cp "$NEW_SO" lsv-test:"$OLD_SO"
echo ".so swapped for debug"
docker cp /root/build/p22a_dbg_spec.py lsv-test:/root/p22a_dbg_spec.py
docker exec lsv-test /opt/venv/bin/python /root/p22a_dbg_spec.py
echo "dbg rc=$?"
docker exec lsv-test cp -a "$OLD_SO.v125bak" "$OLD_SO"
echo "original .so restored"
docker exec -d lsv-test bash /root/serve_user.sh
wait_health 480 && echo "certified serve back up ($(date +%T))"
python3 /root/build/probe_jitwarm_v63.py http://127.0.0.1:8000 $M 731 >/dev/null 2>&1 || true
echo "=== P22A DBG DONE $(date +%T) ==="
} > "$L" 2>&1
rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
tail -60 "$L"
echo P22A_DBG_SCRIPT_DONE
