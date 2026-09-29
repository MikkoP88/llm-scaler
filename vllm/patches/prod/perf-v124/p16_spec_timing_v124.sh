#!/bin/bash
# p16_spec_timing_v124.sh — P16: spec_timing composition via spec-off A/B.
# Method: measure solo + 4x1024 throughput with spec MTP x4 (certified serve,
# numbers also in p15_baseline.log) and with spec OFF (variant serve), then fit
# the two-equation model:
#   spec4: E / R4 = 5f + s      (E=3.562 accepted/iter; 4 draft + 1 verify fwd)
#   spec0: 1  / R0 = f + s
#   -> f = (T4 - T0)/4 ; s = T0 - f
# Output: draft-share / verify-share / sampler+other share of the spec4 step.
# Posture: temporary EXPERIMENT serve on the running lane (no image change);
# certified serve_user.sh restored at the end + verified.
set -u
L=/root/build/lce1/p16_spec_timing.log
M=qwen3.8-27b-fp8
{
echo "=== P16 SPEC TIMING start $(date +%F' '%T) ==="

wait_health () {  # $1 = timeout_s
  local i=0 t=${1:-300}
  while [ $i -lt $t ]; do
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && return 0
    sleep 5; i=$((i+5))
  done
  return 1
}

echo "--- 1. spec4 (certified serve) solo + 4x1024 fresh pass ---"
wait_health 300 || { echo ABORT_NOT_UP; exit 1; }
cd /root/build
echo "== SPEC4 SOLO 1x1024 x3 ==";  python3 bench_genspeed.py 1 1024 3
echo "== SPEC4 ASYNC 4x1024 ==";   python3 bench_genspeed.py 4 1024 1

echo "--- 2. build + launch spec-OFF variant serve ---"
docker exec lsv-test sh -c "grep -v 'speculative-config' /root/serve_user.sh | sed 's|serve_full.log|serve_spec0.log|' > /root/serve_user_spec0.sh && chmod +x /root/serve_user_spec0.sh"
docker exec lsv-test grep -c 'speculative-config' /root/serve_user_spec0.sh | grep -qx 0 || { echo ABORT_VARIANT_STILL_HAS_SPEC; exit 1; }
docker exec lsv-test pkill -f 'vllm serve' || true
sleep 8
docker exec -d lsv-test bash /root/serve_user_spec0.sh
echo "variant launched, waiting for health..."
wait_health 420 || { echo ABORT_SPEC0_BOOT; docker exec lsv-test tail -30 /root/serve_spec0.log; exit 1; }
sleep 5
SPECN=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_spec0.log" || true)
echo "spec0_confirm_spec_lines=$SPECN (expect 0)"
[ "$SPECN" = "0" ] || { echo ABORT_SPEC0_NOT_CONFIRMED; exit 1; }
echo "== SPEC0 SOLO 1x1024 x3 =="; python3 bench_genspeed.py 1 1024 3
echo "== SPEC0 ASYNC 4x1024 ==";  python3 bench_genspeed.py 4 1024 1

echo "--- 3. restore certified spec4 serve ---"
docker exec lsv-test pkill -f 'vllm serve' || true
sleep 8
docker exec -d lsv-test bash /root/serve_user.sh
wait_health 420 || { echo ABORT_RESTORE_BOOT; exit 1; }
sleep 5
REST=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens.:4\|num_speculative_tokens\\\":4' /root/serve_full.log" || true)
echo "restored_spec4_lines=$REST (expect >=1)"
[ "$REST" -ge 1 ] 2>/dev/null || { echo ABORT_RESTORE_NOT_SPEC4; exit 1; }
python3 /root/build/probe_jitwarm_v63.py http://127.0.0.1:8000 $M 731 >/dev/null 2>&1 || true
echo "certified serve restored + rewarm $(date +%T)"

echo "=== P16 SPEC TIMING measurements DONE $(date +%T) ==="
} > "$L" 2>&1
tail -80 "$L"
echo P16_MEASURE_DONE
