#!/bin/sh
# ship_v1221.sh — end-to-end v1.2.21 promotion orchestrator (detached-safe).
# v1.2.21 = v1.2.20 + v88 int64 pool-offset kernels wheel (tiny-step wedge
# root fix: GDN conv-state update kernels overflowed int32 at state ids
# >= 4097 on the padded 1-MiB-page mamba pool -> DEVICE_LOST wedge).
# lane down -> generate v1221 sanity/validate -> stage5 bake (repro gate) ->
# fresh boot V1221 -> sanity (incl. v66 admission gate + v88 wheel check) ->
# validation chain (solo cold + battery + WEDGE ACCEPTANCE serialized 24 +
# burst x2 + 3x drill + fairness) -> watchdog lineage repoint.
LOG=/root/build/lce1/v1221_ship.log
{
echo "=== SHIP_V1221 start $(date +%T) ==="
# 1. lane down (bake needs :8000)
docker exec lsv-test sh -c "pkill -f 'vllm serve' || true"
for i in $(seq 1 30); do docker exec lsv-test sh -c "pgrep -f 'vllm serve' >/dev/null" || break; sleep 2; done
sleep 3
if curl -s -o /dev/null -m 3 http://localhost:8000/health; then echo ABORT_LANE_STILL_UP; exit 1; fi
echo "lane DOWN $(date +%T)"
# 2. generate v1221 sanity/validate from the v1220 provenance
S=/root/build/sanity_v1221.sh
sed -e 's/v1220/v1221/g' -e 's/V1220/V1221/g' /root/build/sanity_v1220.sh > "$S"
printf '%s\n' \
  'echo "== V88 WHEEL CHECK ==" >> $R' \
  'docker exec lsv-test /opt/venv/bin/pip freeze 2>/dev/null | grep -i vllm-xpu-kernels >> $R 2>&1' \
  'docker exec lsv-test sh -c "ls /root/.llm_scaler_exp_v1220_baked /root/.llm_scaler_exp_v1221_baked 2>&1" >> $R 2>&1' \
  'echo "SANITY_V1221_EXTRA_DONE" >> $R' >> "$S"
grep -q SANITY_V1221_DONE "$S" || { echo ABORT_SANITY_GEN; exit 1; }
V=/root/build/validate_v1221.sh
sed -e 's/v1220/v1221/g' -e 's/V1220/V1221/g' /root/build/validate_v1220.sh | grep -v '^echo VALIDATE_V1221_DONE' > "$V"
printf '%s\n' \
  'echo "== V88 WEDGE ACCEPTANCE: serialized 24 (historically DEAD at 17) ==" >> $OUT' \
  'OK=0; FAIL=0; H="200"' \
  'for i in $(seq 1 24); do' \
  '  out=$(curl -s --max-time 20 -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"Count from 1 to 5.\",\"max_tokens\":24,\"temperature\":0}" -o /dev/null -w "%{http_code}")' \
  '  [ "$out" = "200" ] && OK=$((OK+1)) || FAIL=$((FAIL+1))' \
  '  sleep 6' \
  '  H=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:8000/health)' \
  '  [ "$H" != "200" ] && { echo "SERIALIZED DEAD cycle $i ok=$OK fail=$FAIL" >> $OUT; break; }' \
  'done' \
  '[ "$H" = "200" ] && echo "SERIALIZED SURVIVED ok=$OK fail=$FAIL" >> $OUT' \
  'echo "== V88 WEDGE ACCEPTANCE: burst_harsh x2 ==" >> $OUT' \
  'bash /root/build/burst_harsh.sh v1221_a 2>&1 | tail -2 >> $OUT' \
  'bash /root/build/burst_harsh.sh v1221_b 2>&1 | tail -2 >> $OUT' \
  'grep -q "SURVIVED" /root/build/lce1/burst_v1221_a.log || echo BURST_A_NOT_SURVIVED >> $OUT' \
  'grep -q "SURVIVED" /root/build/lce1/burst_v1221_b.log || echo BURST_B_NOT_SURVIVED >> $OUT' \
  'echo "== WEDGES: engine resets in dmesg (must be 0 new) ==" >> $OUT' \
  'dmesg | grep -c "Engine reset" >> $OUT 2>&1' \
  'echo VALIDATE_V1221_DONE >> $OUT' >> "$V"
grep -q VALIDATE_V1221_DONE "$V" || { echo ABORT_VALIDATE_GEN; exit 1; }
sh -n "$S" || { echo ABORT_SANITY_SYNTAX; exit 1; }
sh -n "$V" || { echo ABORT_VALIDATE_SYNTAX; exit 1; }
echo "v1221 sanity/validate generated $(date +%T)"
# 3. bake v1.2.21
bash /root/build/stage5_bake_v1221.sh > /root/build/lce1/v1221_stage5_run.log 2>&1
grep -q STAGE5_BAKE_V1221_DONE /root/build/lce1/v1221_stage5_run.log || { echo ABORT_STAGE5; tail -20 /root/build/lce1/v1221_bake.log; exit 1; }
if grep -q 'GATE-FAIL' /root/build/lce1/v1221_bake.log; then echo ABORT_STAGE5_GATES; exit 1; fi
docker images llm-scaler-exp:v1.2.21 --format '{{.ID}}' | grep -q . || { echo ABORT_NO_IMAGE; exit 1; }
echo "baked $(date +%T)"
# 4. boot fresh lane from v1.2.21
bash /root/build/repro_bootV1221.sh V1221 > /root/build/lce1/v1221_boot.log 2>&1 || { echo ABORT_BOOT; tail -20 /root/build/lce1/v1221_boot.log; exit 1; }
HW=0
for i in $(seq 1 45); do
  sleep 10
  c=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/health 2>/dev/null || echo 000)
  if [ "$c" = "200" ]; then echo "boot HEALTH_OK ~$((i*10))s"; HW=1; break; fi
done
[ "$HW" = "1" ] || { echo ABORT_BOOT_HEALTH; exit 1; }
# 5. sanity (includes v66 admission gate + v88 wheel check on fresh lane)
sh /root/build/sanity_v1221.sh > /dev/null 2>&1
grep -q SANITY_V1221_DONE /root/build/_v1221_sanity.txt || { echo ABORT_SANITY; exit 1; }
grep -m1 'SANITY finish' /root/build/_v1221_sanity.txt || true
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1221_sanity.txt || { echo ABORT_ADMISSION_NOT_PASS; exit 1; }
echo "sanity+admission PASS $(date +%T)"
# 6. validation chain (solo cold + battery + WEDGE ACCEPTANCE + 3x drill + fairness)
sh /root/build/validate_v1221.sh > /dev/null 2>&1
grep -q VALIDATE_V1221_DONE /root/build/_v1221_validate.txt || { echo ABORT_VALIDATE; exit 1; }
grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_v1221_sustain.txt || { echo ABORT_DRILL_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED SURVIVED ok=24' /root/build/_v1221_validate.txt || { echo ABORT_SERIALIZED_NOT_CLEAN; exit 1; }
grep -q 'BURST_A_NOT_SURVIVED\|BURST_B_NOT_SURVIVED' /root/build/_v1221_validate.txt && { echo ABORT_BURST_NOT_CLEAN; exit 1; } || true
echo "validate CLEAN $(date +%T)"
# 7. watchdog lineage repoint
cp /root/build/repro_bootV1212.sh /root/build/repro_bootV1212.sh.pre_v1221
cat /root/build/repro_bootV1221.sh > /root/build/repro_bootV1212.sh
echo "watchdog repointed"
echo "=== SHIP_V1221 COMPLETE $(date +%T) ==="
echo SHIP_V1221_COMPLETE
} > "$LOG" 2>&1
