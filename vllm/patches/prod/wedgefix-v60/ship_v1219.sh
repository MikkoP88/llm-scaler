#!/bin/sh
# ship_v1219.sh — end-to-end v1.2.19 promotion orchestrator (detached-safe).
# v1.2.19 = v1.2.18 + extended in-bake JIT warm (zero first-traffic JIT).
# lane down -> stage5 bake (extended warm + gates + commit) -> fresh boot
# V1219 -> sanity (incl. JIT-zero acceptance gate) -> validation chain
# (solo cold + battery + 3x drill) -> watchdog lineage repoint.
LOG=/root/build/lce1/v1219_ship.log
{
echo "=== SHIP_V1219 start $(date +%T) ==="
# 1. lane down (bake needs :8000)
docker exec lsv-test sh -c "pkill -f 'vllm serve' || true"
for i in $(seq 1 30); do docker exec lsv-test sh -c "pgrep -f 'vllm serve' >/dev/null" || break; sleep 2; done
sleep 3
if curl -s -o /dev/null -m 3 http://localhost:8000/health; then echo ABORT_LANE_STILL_UP; exit 1; fi
echo "lane DOWN $(date +%T)"
# 2. bake v1.2.19
bash /root/build/stage5_bake_v1219.sh > /root/build/lce1/v1219_stage5_run.log 2>&1
grep -q STAGE5_BAKE_V1219_DONE /root/build/lce1/v1219_stage5_run.log || { echo ABORT_STAGE5; tail -20 /root/build/lce1/v1219_bake.log; exit 1; }
if grep -q 'GATE-FAIL' /root/build/lce1/v1219_bake.log; then echo ABORT_STAGE5_GATES; exit 1; fi
docker images llm-scaler-exp:v1.2.19 --format '{{.ID}}' | grep -q . || { echo ABORT_NO_IMAGE; exit 1; }
echo "baked $(date +%T)"
# 3. boot fresh lane from v1.2.19
bash /root/build/repro_bootV1219.sh V1219 > /root/build/lce1/v1219_boot.log 2>&1 || { echo ABORT_BOOT; tail -20 /root/build/lce1/v1219_boot.log; exit 1; }
HW=0
for i in $(seq 1 45); do
  sleep 10
  c=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/health 2>/dev/null || echo 000)
  if [ "$c" = "200" ]; then echo "boot HEALTH_OK ~$((i*10))s"; HW=1; break; fi
done
[ "$HW" = "1" ] || { echo ABORT_BOOT_HEALTH; exit 1; }
# 4. sanity (includes the JIT-zero acceptance gate on fresh-boot traffic)
sh /root/build/sanity_v1219.sh > /dev/null 2>&1
grep -q SANITY_V1219_DONE /root/build/_v1219_sanity.txt || { echo ABORT_SANITY; exit 1; }
grep -m1 'SANITY finish' /root/build/_v1219_sanity.txt || true
# 5. validation chain (solo cold + battery + 3x drill + JIT recheck)
sh /root/build/validate_v1219.sh > /dev/null 2>&1
grep -q VALIDATE_V1219_DONE /root/build/_v1219_validate.txt || { echo ABORT_VALIDATE; exit 1; }
grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_v1219_sustain.txt || { echo ABORT_DRILL_NOT_CLEAN; exit 1; }
echo "validate CLEAN $(date +%T)"
# 6. watchdog lineage repoint
cp /root/build/repro_bootV1212.sh /root/build/repro_bootV1212.sh.pre_v1219
cat /root/build/repro_bootV1219.sh > /root/build/repro_bootV1212.sh
echo "watchdog repointed"
echo "=== SHIP_V1219 COMPLETE $(date +%T) ==="
echo SHIP_V1219_COMPLETE
} > "$LOG" 2>&1
