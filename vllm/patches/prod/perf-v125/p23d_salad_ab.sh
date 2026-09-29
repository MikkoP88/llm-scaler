#!/bin/bash
# p23d_salad_ab.sh — v125 P23D: measure the tool-call salad RATE on
# three arms under an identical standardized prefix.
#
# Evidence so far (P23/P23B/P23C): post-prefix parser T1/T2 failures
# with 'The!!!...!' reasoning-channel salad on n2e5m2s4 (T1 twice +
# T2a; T2b recovered after 20 s) and the P23 n2e4m3s4 name=None abort
# (same signature, reasoning channel hidden by the v89 probe); all
# fresh-boot probes clean; fp16 batteries never failed. grammar_matcher
# warnings appear even in the e4m3 PASS leg (2x), so the warning is
# context noise — the salad is the discriminator. This study turns the
# anecdotes into rates:
#   arm cert16s4  = serve_user.sh           (fp16 spec control)
#   arm n2e4m3s4  = =2 pool-straight P22B   (e4m3)
#   arm n2e5m2s4  = =2 pool-straight P22B   (e5m2)
# Each arm: boot + posture gates -> prefix (1 sustain round + 1 burst)
# -> 30 alternating T1/T2 requests (p23d_probe.py) -> grammar_matcher
# delta. Teardown restores certified serve. EXIT trap mirrors.
set -u
L=/root/build/lce1/p23d_salad.log
R=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
CGP=/opt/venv/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py
XPU=/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py
XO=/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py
SP=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
SO=/root/build/vxk/build/temp/_xpu_C.abi3.so

restore_lane() {
  [ -f /root/build/lane_watchdog.paused ] || return 0
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true" >/dev/null 2>&1
  for i in $(seq 1 18); do sleep 5; curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || break; done
  sleep 4
  if docker exec lsv-test test -f $SP/_xpu_C.abi3.so.v125bak; then
    docker exec lsv-test cp -a $SP/_xpu_C.abi3.so.v125bak $SP/_xpu_C.abi3.so >> "$L" 2>&1
  fi
  docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU 2>/dev/null; cp $R.pre_v125_c1 $R 2>/dev/null; cp $CGP.pre_v125_diag $CGP 2>/dev/null; cp $XO.pre_v125_c5 $XO 2>/dev/null; python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo trap-clean-venv" >> "$L" 2>&1
  docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo trap-restore-launched" >> "$L" 2>&1
  for i in $(seq 1 48); do sleep 10; curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && break; done
  rm -f /root/build/lane_watchdog.paused
  echo "trap lane_watchdog re-armed ($(date +%T))" >> "$L" 2>&1
}
trap restore_lane EXIT

stop_serve() {
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
  for i in $(seq 1 18); do
    sleep 5
    curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || return 0
  done
  return 1
}

arm() {  # $1 tag, $2 serve script, $3 serve log, $4 mode (''/fp8/fp16)
  echo "=== P23DARM $1 $(date +%F' '%T) ==="
  stop_serve || { echo "ABORT_${1}_PORT"; return 1; }
  sleep 4
  docker exec lsv-test sh -c ": > $3; cd /root && nohup bash $2 > /dev/null 2>&1 & echo $1-launched"
  HK=0
  for i in $(seq 1 48); do
    sleep 10
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "$1 health OK after ${i}0s"; break; }
  done
  [ "$HK" -eq 1 ] || { echo "ABORT_${1}_HEALTH"; return 1; }
  sleep 15
  ENG=$(docker exec lsv-test sh -c "grep -c 'FP8_NATIVE ENGAGED' $3; true")
  C6=$(docker exec lsv-test sh -c "grep -c 'P22C6' $3; true")
  echo "[$1] engaged=$ENG c6=$C6 (mode '${4:-info}')"
  case "${4:-}" in
    fp8)  [ "$ENG" -ge 1 ] || { echo "ABORT_${1}_NO_ENGAGED"; return 1; }
          [ "$C6" -eq 0 ] || { echo "ABORT_${1}_C6"; return 1; } ;;
    fp16) [ "$ENG" -eq 0 ] || { echo "ABORT_${1}_ENGAGED"; return 1; } ;;
  esac

  echo "[$1] prefix: 1 sustain round + 1 burst"
  GM0=$(docker exec lsv-test sh -c "grep -c 'grammar_matcher' $3; true")
  bash /root/build/repro_sustain.sh 1 > /root/build/_p23d_${1}_sustain.txt 2>&1
  grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_p23d_${1}_sustain.txt || { echo "ABORT_${1}_PREFIX_SUSTAIN"; return 1; }
  bash /root/build/burst_harsh.sh "${1}_p23d" > /root/build/_p23d_${1}_burst.txt 2>&1
  grep -q "SURVIVED" /root/build/lce1/burst_${1}_p23d.log || { echo "ABORT_${1}_PREFIX_BURST"; return 1; }

  GM1=$(docker exec lsv-test sh -c "grep -c 'grammar_matcher' $3; true")
  echo "[$1] grammar_matcher after prefix: $GM1 (delta $((GM1-GM0)))"

  echo "[$1] 30-request tool-call probe"
  python3 /root/build/p23d_probe.py > /root/build/_p23d_${1}_probe.txt 2>&1
  grep -E 'SALAD|OTHER|ERROR' /root/build/_p23d_${1}_probe.txt | head -12
  grep 'SUMMARY' /root/build/_p23d_${1}_probe.txt
  GM2=$(docker exec lsv-test sh -c "grep -c 'grammar_matcher' $3; true")
  echo "[$1] grammar_matcher after probe: $GM2 (probe delta $((GM2-GM1)))"
  TB=$(docker exec lsv-test sh -c "grep -ciE 'traceback|EngineDeadError' $3; true")
  echo "[$1] tracebacks=$TB"
  echo "[$1] ARM COMPLETE $(date +%T)"
}

{
echo "=== P23D SALAD A/B start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
grep -q 'ABORT_n2e5m2s4_PARSER' /root/build/lce1/p23c_battery.log || { echo "ABORT_NO_P23C_EVIDENCE"; exit 1; }

echo "--- clean venv, apply opsall + C5 + C6, swap wheel ---"
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && (test -f $XO.pre_v125_c5 && cp $XO.pre_v125_c5 $XO || true) && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }
for PYP in patch_v125_opsall_fix.py patch_v125_s4fp8_bridge.py patch_v125_c6_staticraise.py p23d_probe.py; do
  docker cp /root/build/$PYP lsv-test:/root/ || { echo "ABORT_CP_$PYP"; exit 1; }
done
PF=$(docker exec lsv-test python3 /root/patch_v125_opsall_fix.py)
echo "$PF" | grep -q 'V125_OPSFIX_OK\|V125_OPSFIX_ALREADY' || { echo "ABORT_OPSFIX"; exit 1; }
PC=$(docker exec lsv-test python3 /root/patch_v125_s4fp8_bridge.py)
echo "$PC" | grep -q 'V125_C5_OK\|V125_C5_ALREADY' || { echo "ABORT_C5"; exit 1; }
PC6=$(docker exec lsv-test python3 /root/patch_v125_c6_staticraise.py)
echo "$PC6" | grep -q 'V125_C6_OK\|V125_C6_ALREADY' || { echo "ABORT_C6"; exit 1; }

docker exec lsv-test cp -a $SP/_xpu_C.abi3.so $SP/_xpu_C.abi3.so.v125bak 2>/dev/null || true
docker cp "$SO" lsv-test:$SP/_xpu_C.abi3.so || { echo "ABORT_WHEEL_CP"; exit 1; }
LV=$(docker exec lsv-test /opt/venv/bin/python -c "
import os, torch, vllm_xpu_kernels
so = os.path.join(os.path.dirname(vllm_xpu_kernels.__file__), '_xpu_C.abi3.so')
torch.ops.load_library(so)
print('WHEEL_LOAD_OK')
" 2>&1 | tail -2)
echo "$LV"
echo "$LV" | grep -q 'WHEEL_LOAD_OK' || { echo "ABORT_WHEEL_LOAD"; exit 1; }
docker exec lsv-test sh -c "test -x /root/serve_p22c5_n2_e4m3s4.sh && test -x /root/serve_p22c5_n2_e5m2s4.sh && echo N2_SCRIPTS_PRESENT" || { echo "ABORT_NO_N2_SCRIPTS"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

arm cert16s4  /root/serve_user.sh             /root/serve_full.log           fp16 || { echo "ARM_CERT_FAILED"; exit 1; }
arm n2e4m3s4  /root/serve_p22c5_n2_e4m3s4.sh  /root/serve_p22c5_n2_e4m3s4.log fp8  || { echo "ARM_E4M3_FAILED"; exit 1; }
arm n2e5m2s4  /root/serve_p22c5_n2_e5m2s4.sh  /root/serve_p22c5_n2_e5m2s4.log fp8  || { echo "ARM_E5M2_FAILED"; exit 1; }

echo "--- teardown: wheel revert + venv revert + certified serve ---"
stop_serve || { echo "ABORT_TEARDOWN_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test cp -a $SP/_xpu_C.abi3.so.v125bak $SP/_xpu_C.abi3.so || { echo "ABORT_WHEEL_REVERT"; exit 1; }
STRR=$(docker exec lsv-test bash -c "strings $SP/_xpu_C.abi3.so | grep -c 'fp8_e4m3fn'" || true)
echo "wheel reverted fp8_strings=$STRR (want 0)"
[ "$STRR" -eq 0 ] 2>/dev/null || { echo "ABORT_WHEEL_REVERT_PROOF"; exit 1; }
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && cp $XO.pre_v125_c5 $XO && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo REVERT_CLEAN_OK" || { echo "ABORT_REVERT"; exit 1; }
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK2=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK2=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK2" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
SPL2=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log; true")
INSTR2=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|v125 root fix\|perf-v125 C1\|v125 P22C5\|v125 C5DUPD\|FP8_NATIVE ENGAGED\|P22C6' /root/serve_full.log; true")
echo "restored spec_lines=$SPL2 marker-lines=$INSTR2 (want 2 / 0)"
[ "$SPL2" -eq 2 ] || { echo "ABORT_RESTORE_SPEC"; exit 1; }
[ "$INSTR2" -eq 0 ] || { echo "ABORT_RESTORE_MARKERS"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P23D SALAD A/B DONE $(date +%F' '%T) ==="
echo "P23D_DONE"
} > "$L" 2>&1
tail -90 "$L"
echo P23D_SCRIPT_DONE
