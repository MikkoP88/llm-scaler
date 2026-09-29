#!/bin/bash
# p23f_ns_quality.sh — v125 P23F: ns-path quality-RATE study.
#
# P23E aborted at its first gate: e4m3ns (=1 ESIMD fp8) returned one
# WRONG concurrent-math answer (x8d 7/8) on a FRESH boot, no prefix —
# the first direct ns-fp8 quality failure, contradicting the clean
# Run-7m n2val evidence (intermittent). The old probe printed only the
# count. This study measures rates on the SAME stack as P23E
# (opsall + C5 + C6 venv + P22B wheel):
#   arm fix16ns = serve_p22c4_16ns.sh   (fp16 ns control)
#   arm e4m3ns  = serve_p22c4_e4m3ns.sh (=1 ESIMD + v124 bridge)
#   arm e5m2ns  = serve_p22c4_e5m2ns.sh (=1 ESIMD + v124 bridge)
# Per arm: boot + posture gates -> PHASE A fresh-boot probe
# (10 concurrent math rounds + serial 80-question control + 30 tool
# calls) -> prefix (1 sustain + 1 burst, the P23D prefix) -> PHASE B
# probe (3 math rounds + 30 tool calls) -> grammar_matcher deltas,
# tracebacks. Quality failures DO NOT abort — this is a measurement.
# EXIT trap mirrors the full restore. Prereq: P23E abort evidence.
set -u
L=/root/build/lce1/p23f_quality.log
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

arm() {  # $1 tag, $2 serve script, $3 serve log — MEASUREMENT, no quality aborts
  echo "=== P23FARM $1 $(date +%F' '%T) ==="
  DM0=$(dmesg | grep -c "Engine reset" || true)
  stop_serve || { echo "ABORT_${1}_PORT_STILL_UP"; return 1; }
  sleep 4
  docker exec lsv-test sh -c ": > $3; cd /root && nohup bash $2 > /dev/null 2>&1 & echo $1-launched"
  HK=0
  for i in $(seq 1 48); do
    sleep 10
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "$1 health OK after ${i}0s"; break; }
  done
  [ "$HK" -eq 1 ] || { echo "ABORT_${1}_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' $3 | tail -6; true"; return 1; }
  sleep 15
  SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' $3; true")
  FX=$(docker exec lsv-test sh -c "grep -c 'v125 root fix' $3; true")
  ENG=$(docker exec lsv-test sh -c "grep -c 'FP8_NATIVE ENGAGED' $3; true")
  C5=$(docker exec lsv-test sh -c "grep -c 'v125 P22C5 STATIC_FP8_BRIDGE' $3; true")
  C6=$(docker exec lsv-test sh -c "grep -c 'P22C6' $3; true")
  echo "[$1] spec_lines=$SPL (want 0) v125fix=$FX engaged=$ENG c5=$C5 c6_raise=$C6"
  [ "$SPL" -eq 0 ] || { echo "ABORT_${1}_SPEC_LINES"; return 1; }
  [ "$FX" -ge 1 ] || { echo "ABORT_${1}_NO_FIX"; return 1; }

  echo "[$1] PHASE A: fresh-boot probe (10 conc rounds + 80 serial + 30 tool)"
  GM0=$(docker exec lsv-test sh -c "grep -c 'grammar_matcher' $3; true")
  python3 /root/build/p23f_probe.py A > /root/build/_p23f_${1}_A.txt 2>&1
  grep -E 'SUMMARY|FLIP' /root/build/_p23f_${1}_A.txt
  GM1=$(docker exec lsv-test sh -c "grep -c 'grammar_matcher' $3; true")
  echo "[$1] grammar_matcher after A: $GM1 (delta $((GM1-GM0)))"

  echo "[$1] prefix: 1 sustain round + 1 burst (the P23D prefix)"
  bash /root/build/repro_sustain.sh 1 > /root/build/_p23f_${1}_sustain.txt 2>&1
  grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_p23f_${1}_sustain.txt || { echo "ABORT_${1}_PREFIX_SUSTAIN"; return 1; }
  bash /root/build/burst_harsh.sh "${1}_p23f" > /root/build/_p23f_${1}_burst.txt 2>&1
  grep -q "SURVIVED" /root/build/lce1/burst_${1}_p23f.log || { echo "ABORT_${1}_PREFIX_BURST"; return 1; }

  echo "[$1] PHASE B: post-prefix probe (3 conc rounds + 30 tool)"
  python3 /root/build/p23f_probe.py B > /root/build/_p23f_${1}_B.txt 2>&1
  grep -E 'SUMMARY' /root/build/_p23f_${1}_B.txt
  GM2=$(docker exec lsv-test sh -c "grep -c 'grammar_matcher' $3; true")
  echo "[$1] grammar_matcher after B: $GM2 (B delta $((GM2-GM1)))"
  TB=$(docker exec lsv-test sh -c "grep -ciE 'traceback|EngineDeadError' $3; true")
  DM1=$(dmesg | grep -c "Engine reset" || true)
  echo "[$1] tracebacks=$TB resets_delta=$((DM1-DM0))"
  echo "[$1] ARM COMPLETE $(date +%T)"
  return 0
}

{
echo "=== P23F NS-QUALITY start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
grep -q 'LEG_E4M3NS_FAILED' /root/build/lce1/p23e_battery.log || { echo "ABORT_NO_P23E_EVIDENCE"; exit 1; }
grep -q 'P23D_DONE' /root/build/lce1/p23d_salad.log || { echo "ABORT_NO_P23D_PROOF"; exit 1; }

echo "--- apply opsall + C5 + C6, swap wheel (identical stack to P23E) ---"
docker start lsv-test || { echo "ABORT_CONTAINER_START"; exit 1; }
sleep 3
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && (test -f $XO.pre_v125_c5 && cp $XO.pre_v125_c5 $XO || true) && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }
docker cp /root/build/p23f_probe.py lsv-test:/root/ || { echo "ABORT_CP_PROBE"; exit 1; }
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
STR=$(docker exec lsv-test bash -c "strings $SP/_xpu_C.abi3.so | grep -c 'fp8_e4m3fn'" || true)
echo "wheel fp8_strings=$STR (want >=1)"
[ "$STR" -ge 1 ] 2>/dev/null || { echo "ABORT_WHEEL_STRINGS"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

arm fix16ns  /root/serve_p22c4_16ns.sh   /root/serve_p22c4_16ns.log    || { echo "ARM_FIX16NS_FAILED"; exit 1; }
arm e4m3ns   /root/serve_p22c4_e4m3ns.sh /root/serve_p22c4_e4m3ns.log  || { echo "ARM_E4M3NS_FAILED"; exit 1; }
arm e5m2ns   /root/serve_p22c4_e5m2ns.sh /root/serve_p22c4_e5m2ns.log  || { echo "ARM_E5M2NS_FAILED"; exit 1; }

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
echo "=== P23F NS-QUALITY DONE $(date +%F' '%T) ==="
echo "P23F_DONE"
} > "$L" 2>&1
tail -160 "$L"
echo P23F_SCRIPT_DONE
