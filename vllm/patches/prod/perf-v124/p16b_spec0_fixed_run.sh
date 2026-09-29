#!/bin/bash
# p16b_spec0_fixed_run.sh — P16-b: spec-OFF leg AFTER the v124 P16 compile-gate
# fix (xpu.py v31.1 gate extended from spec+TP>1 to ALL TP>1). Validates the
# fix (spec0 must boot; gate warning must now fire on the spec0 boot) and
# captures the spec0 solo + 4x1024 measurements for the two-equation
# spec_timing fit. Certified spec4 serve restored + verified at the end.
# Evidence chain kept: gate-line count in the FAILED serve_spec0.log is read
# BEFORE the relaunch overwrites it (expect 0 = gate had not fired = root cause).
set -u
L=/root/build/lce1/p16b_spec0_fixed.log
M=qwen3.8-27b-fp8
{
echo "=== P16B SPEC0-FIXED start $(date +%F' '%T) ==="

wait_health () {
  local i=0 t=${1:-300}
  while [ $i -lt $t ]; do
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && return 0
    sleep 5; i=$((i+5))
  done
  return 1
}

echo "--- 0. apply idempotent gate fix + evidence of the pre-fix state ---"
docker exec lsv-test test -f /root/serve_user_spec0.sh || { echo ABORT_NO_VARIANT_SCRIPT; exit 1; }
echo "evidence gate_in_FAILED_spec0_log=$(docker exec lsv-test sh -c "grep -c 'inductor compilation disabled' /root/serve_spec0.log" || true) (expect 0 = gate never fired pre-fix)"
echo "evidence dynamo_crash_in_FAILED_log=$(docker exec lsv-test sh -c "grep -c 'Unsupported hasattr call' /root/serve_spec0.log" || true) (expect >=1)"
docker exec lsv-test /opt/venv/bin/python /root/p16_fix_nonspec_gate.py
docker exec lsv-test /opt/venv/bin/python -c 'import ast; ast.parse(open("/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py").read()); print("SYNTAX_OK")'
GATE=$(docker exec lsv-test sh -c "grep -c 'v124 P16' /opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py" || true)
echo "gate_fix_marker_count=$GATE (expect >=2)"
[ "$GATE" -ge 2 ] 2>/dev/null || { echo ABORT_FIX_NOT_APPLIED; exit 1; }

echo "--- 1. launch spec-OFF variant (patched xpu.py) ---"
docker exec lsv-test pkill -f 'vllm serve' || true
sleep 8
docker exec -d lsv-test bash /root/serve_user_spec0.sh
echo "variant launched, waiting for health..."
wait_health 420 || { echo ABORT_SPEC0_BOOT; docker exec lsv-test tail -40 /root/serve_spec0.log; exit 1; }
sleep 5
SPECN=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_spec0.log" || true)
GATE2=$(docker exec lsv-test sh -c "grep -c 'inductor compilation disabled' /root/serve_spec0.log" || true)
echo "spec0_confirm_spec_lines=$SPECN (expect 0); gate_fired_now=$GATE2 (expect >=1)"
{ [ "$SPECN" = "0" ] && [ "$GATE2" -ge 1 ]; } 2>/dev/null || { echo ABORT_CONFIRM; exit 1; }
echo "SPEC0-BOOT FIXED: non-spec pipeline boots $(date +%T)"

cd /root/build
echo "== SPEC0 SOLO 1x1024 x3 ==";  python3 bench_genspeed.py 1 1024 3
echo "== SPEC0 ASYNC 4x1024 ==";   python3 bench_genspeed.py 4 1024 1

echo "--- 2. restore certified spec4 serve ---"
docker exec lsv-test pkill -f 'vllm serve' || true
sleep 8
docker exec -d lsv-test bash /root/serve_user.sh
wait_health 420 || { echo ABORT_RESTORE_BOOT; exit 1; }
sleep 5
REST=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log" || true)
RESTV=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens.:4' /root/serve_full.log" || true)
GATE3=$(docker exec lsv-test sh -c "grep -c 'inductor compilation disabled' /root/serve_full.log" || true)
echo "restored_spec_lines=$REST (expect >=1); value4_lines=$RESTV; gate=$GATE3 (expect >=1)"
[ "$REST" -ge 1 ] 2>/dev/null || { echo ABORT_RESTORE_NOT_SPEC4; exit 1; }
python3 /root/build/probe_jitwarm_v63.py http://127.0.0.1:8000 $M 731 >/dev/null 2>&1 || true
echo "certified serve restored + rewarm $(date +%T)"
echo "=== P16B DONE $(date +%T) ==="
} > "$L" 2>&1
tail -60 "$L"
echo P16B_DONE
