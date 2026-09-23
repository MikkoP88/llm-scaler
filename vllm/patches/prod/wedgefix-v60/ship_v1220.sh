#!/bin/sh
# ship_v1220.sh — end-to-end v1.2.20 promotion orchestrator (detached-safe).
# v1.2.20 = v1.2.18 + v66 FAIRFIX (waiting-admission starvation bypass) +
# serve thinking-defaults rollback + extended warm w/ sampler round.
# lane down -> stage5 bake -> fresh boot V1220 -> sanity (incl. v66
# admission-PASS acceptance gate) -> validation chain (solo cold +
# battery + 3x drill + fairness) -> watchdog lineage repoint.
LOG=/root/build/lce1/v1220_ship.log
{
echo "=== SHIP_V1220 start $(date +%T) ==="
# 1. lane down (bake needs :8000)
docker exec lsv-test sh -c "pkill -f 'vllm serve' || true"
for i in $(seq 1 30); do docker exec lsv-test sh -c "pgrep -f 'vllm serve' >/dev/null" || break; sleep 2; done
sleep 3
if curl -s -o /dev/null -m 3 http://localhost:8000/health; then echo ABORT_LANE_STILL_UP; exit 1; fi
echo "lane DOWN $(date +%T)"
# 2. bake v1.2.20
bash /root/build/stage5_bake_v1220.sh > /root/build/lce1/v1220_stage5_run.log 2>&1
grep -q STAGE5_BAKE_V1220_DONE /root/build/lce1/v1220_stage5_run.log || { echo ABORT_STAGE5; tail -20 /root/build/lce1/v1220_bake.log; exit 1; }
if grep -q 'GATE-FAIL' /root/build/lce1/v1220_bake.log; then echo ABORT_STAGE5_GATES; exit 1; fi
docker images llm-scaler-exp:v1.2.20 --format '{{.ID}}' | grep -q . || { echo ABORT_NO_IMAGE; exit 1; }
echo "baked $(date +%T)"
# 3. boot fresh lane from v1.2.20
bash /root/build/repro_bootV1220.sh V1220 > /root/build/lce1/v1220_boot.log 2>&1 || { echo ABORT_BOOT; tail -20 /root/build/lce1/v1220_boot.log; exit 1; }
HW=0
for i in $(seq 1 45); do
  sleep 10
  c=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/health 2>/dev/null || echo 000)
  if [ "$c" = "200" ]; then echo "boot HEALTH_OK ~$((i*10))s"; HW=1; break; fi
done
[ "$HW" = "1" ] || { echo ABORT_BOOT_HEALTH; exit 1; }
# 4. sanity (includes the v66 admission acceptance gate on fresh-boot lane)
sh /root/build/sanity_v1220.sh > /dev/null 2>&1
grep -q SANITY_V1220_DONE /root/build/_v1220_sanity.txt || { echo ABORT_SANITY; exit 1; }
grep -m1 'SANITY finish' /root/build/_v1220_sanity.txt || true
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1220_sanity.txt || { echo ABORT_ADMISSION_NOT_PASS; exit 1; }
echo "sanity+admission PASS $(date +%T)"
# 5. validation chain (solo cold + battery + 3x drill + fairness + JIT recheck)
sh /root/build/validate_v1220.sh > /dev/null 2>&1
grep -q VALIDATE_V1220_DONE /root/build/_v1220_validate.txt || { echo ABORT_VALIDATE; exit 1; }
grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_v1220_sustain.txt || { echo ABORT_DRILL_NOT_CLEAN; exit 1; }
echo "validate CLEAN $(date +%T)"
# 6. watchdog lineage repoint
cp /root/build/repro_bootV1212.sh /root/build/repro_bootV1212.sh.pre_v1220
cat /root/build/repro_bootV1220.sh > /root/build/repro_bootV1212.sh
echo "watchdog repointed"
echo "=== SHIP_V1220 COMPLETE $(date +%T) ==="
echo SHIP_V1220_COMPLETE
} > "$LOG" 2>&1
