#!/bin/sh
# ship_v1218.sh — end-to-end v1.2.18 promotion orchestrator (detached-safe).
# Waits for the validation drill, then: lane down -> stage5 bake -> gates
# -> fresh boot V1218 -> sanity -> watchdog lineage repoint.
LOG=/root/build/lce1/v64_ship.log
{
echo "=== SHIP_V1218 start $(date +%T) ==="
# 1. wait for validation chain (drill incl.)
for i in $(seq 1 400); do
  grep -q VALIDATE_V64_DONE /root/build/_v64_validate.txt 2>/dev/null && break
  sleep 10
done
grep -q VALIDATE_V64_DONE /root/build/_v64_validate.txt || { echo ABORT_NO_VALIDATE; exit 1; }
grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_v64_sustain.txt || { echo ABORT_DRILL_NOT_CLEAN; tail -5 /root/build/_v64_sustain.txt; exit 1; }
echo "drill CLEAN $(date +%T)"
# 2. lane down (bake needs :8000)
docker exec lsv-test sh -c "pkill -f 'vllm serve' || true"
for i in $(seq 1 30); do docker exec lsv-test sh -c "pgrep -f 'vllm serve' >/dev/null" || break; sleep 2; done
sleep 3
if curl -s -o /dev/null -m 3 http://localhost:8000/health; then echo ABORT_LANE_STILL_UP; exit 1; fi
echo "lane DOWN $(date +%T)"
# 3. bake v1.2.18
bash /root/build/stage5_bake_v1218.sh > /root/build/lce1/v64_stage5_run.log 2>&1
grep -q STAGE5_BAKE_V1218_DONE /root/build/lce1/v64_stage5_run.log || { echo ABORT_STAGE5; tail -20 /root/build/lce1/v64_bake.log; exit 1; }
if grep -q 'GATE-FAIL' /root/build/lce1/v64_bake.log; then echo ABORT_STAGE5_GATES; exit 1; fi
docker images llm-scaler-exp:v1.2.18 --format '{{.ID}}' | grep -q . || { echo ABORT_NO_IMAGE; exit 1; }
echo "baked $(date +%T)"
# 4. boot fresh lane from v1.2.18
bash /root/build/repro_bootV1218.sh V1218 > /root/build/lce1/v64_boot.log 2>&1 || { echo ABORT_BOOT; tail -20 /root/build/lce1/v64_boot.log; exit 1; }
HW=0
for i in $(seq 1 45); do
  sleep 10
  c=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/health 2>/dev/null || echo 000)
  if [ "$c" = "200" ]; then echo "boot HEALTH_OK ~$((i*10))s"; HW=1; break; fi
done
[ "$HW" = "1" ] || { echo ABORT_BOOT_HEALTH; exit 1; }
# 5. sanity
sh /root/build/sanity_v1218.sh > /dev/null 2>&1
grep -q SANITY_V1218_DONE /root/build/_v64_sanity.txt || { echo ABORT_SANITY; exit 1; }
grep -m1 'SANITY finish' /root/build/_v64_sanity.txt || true
# 6. watchdog lineage repoint
cp /root/build/repro_bootV1212.sh /root/build/repro_bootV1212.sh.pre_v1218
cat /root/build/repro_bootV1218.sh > /root/build/repro_bootV1212.sh
echo "watchdog repointed"
echo "=== SHIP_V1218 COMPLETE $(date +%T) ==="
echo SHIP_V1218_COMPLETE
} > "$LOG" 2>&1
