#!/bin/sh
# resume_v1220.sh — resume SHIP_V1220 after the sanity probe-cp fix.
# v1.2.20 is baked (e1d92193a106) and the lane is UP on it; run the
# remaining ship phases only: sanity (fixed) -> admission gate ->
# validate -> sustain gate -> watchdog repoint.
LOG=/root/build/lce1/v1220_ship.log
{
echo "=== RESUME_V1220 start $(date +%T) ==="
# lane must be up on v1.2.20
IMG=$(docker ps --format '{{.Names}} {{.Image}}' | grep '^lsv-test ' | awk '{print $2}')
[ "$IMG" = "llm-scaler-exp:v1.2.20" ] || { echo "ABORT_LANE_IMAGE=$IMG"; exit 1; }
curl -s -o /dev/null -m 4 http://127.0.0.1:8000/health || { echo ABORT_LANE_NOT_HEALTHY; exit 1; }
# 4. sanity (fixed: docker cp probe into lane before exec)
sh /root/build/sanity_v1220.sh > /dev/null 2>&1
grep -q SANITY_V1220_DONE /root/build/_v1220_sanity.txt || { echo ABORT_SANITY; exit 1; }
grep -m1 'SANITY finish' /root/build/_v1220_sanity.txt || true
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1220_sanity.txt || { echo ABORT_ADMISSION_NOT_PASS; tail -12 /root/build/_v1220_sanity.txt; exit 1; }
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
echo "=== SHIP_V1220 COMPLETE (resumed) $(date +%T) ==="
echo SHIP_V1220_COMPLETE
} >> "$LOG" 2>&1
