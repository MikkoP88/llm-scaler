#!/bin/bash
# p23b_parser_diag.sh — discriminate the P23 parser failure on BOTH =2
# spec-fp8 postures (e4m3 AND e5m2 — the e5m2 parser leg never ran; the
# chain aborted at e4m3). Boot each =2 posture with the validated stack
# (opsall + C5 + C6 + P22B wheel), gate the posture markers, then run
# p23b_diag.py (stock/long/nothink variants of the exact T1/T2/T3
# bodies, full raw dumps). Restore certified serve at the end.
# Prereq: P23 battery log exists (raise-leg PASS = C6 stack validated).
set -u
L=/root/build/lce1/p23b_diag.log
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
    STR=$(docker exec lsv-test bash -c "strings $SP/_xpu_C.abi3.so | grep -c 'fp8_e4m3fn'" || true)
    echo "trap wheel-reverted fp8_strings=$STR (want 0)" >> "$L" 2>&1
  fi
  docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU 2>/dev/null; cp $R.pre_v125_c1 $R 2>/dev/null; cp $CGP.pre_v125_diag $CGP 2>/dev/null; cp $XO.pre_v125_c5 $XO 2>/dev/null; python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo trap-clean-venv" >> "$L" 2>&1
  docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo trap-restore-launched" >> "$L" 2>&1
  for i in $(seq 1 48); do sleep 10; curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && break; done
  rm -f /root/build/lane_watchdog.paused
  echo "trap lane_watchdog re-armed ($(date +%T))" >> "$L" 2>&1
}
trap restore_lane EXIT

diagleg() {  # $1 tag, $2 serve script, $3 serve log
  echo "=== P23BLEG $1 $(date +%F' '%T) ==="
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
  PD=0
  for i in $(seq 1 18); do
    sleep 5
    if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; break; fi
  done
  [ "$PD" -eq 1 ] || { echo "ABORT_${1}_PORT_STILL_UP"; exit 1; }
  sleep 4
  docker exec lsv-test sh -c ": > $3; cd /root && nohup bash $2 > /dev/null 2>&1 & echo $1-launched"
  HK=0
  for i in $(seq 1 48); do
    sleep 10
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "$1 health OK after ${i}0s"; break; }
  done
  [ "$HK" -eq 1 ] || { echo "ABORT_${1}_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' $3 | tail -5"; exit 1; }
  sleep 15
  SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' $3; true")
  FX=$(docker exec lsv-test sh -c "grep -c 'v125 root fix' $3; true")
  ENG=$(docker exec lsv-test sh -c "grep -c 'FP8_NATIVE ENGAGED' $3; true")
  C5=$(docker exec lsv-test sh -c "grep -c 'v125 P22C5 STATIC_FP8_BRIDGE' $3; true")
  C6=$(docker exec lsv-test sh -c "grep -c 'P22C6' $3; true")
  echo "[$1] spec_lines=$SPL (want 2) v125fix=$FX engaged=$ENG (want >=1) c5=$C5 (want 0) c6=$C6 (want 0)"
  [ "$SPL" -eq 2 ] || { echo "ABORT_${1}_SPEC_LINES"; exit 1; }
  [ "$ENG" -ge 1 ] || { echo "ABORT_${1}_NO_ENGAGED"; exit 1; }
  [ "$C5" -eq 0 ] || { echo "ABORT_${1}_C5_PRESENT"; exit 1; }
  [ "$C6" -eq 0 ] || { echo "ABORT_${1}_C6_RAISED"; exit 1; }
  python3 /root/build/p23b_diag.py
  echo "[$1] diag complete $(date +%T)"
}

{
echo "=== P23B DIAG start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
grep -q 'P23LEG cert16s4' /root/build/lce1/p23_battery.log || { echo "ABORT_NO_P23_LOG"; exit 1; }
docker exec lsv-test sh -c "test -x /root/serve_p22c5_n2_e4m3s4.sh && test -x /root/serve_p22c5_n2_e5m2s4.sh && echo N2_SCRIPTS_PRESENT" || { echo "ABORT_NO_N2_SCRIPTS"; exit 1; }

echo "--- clean venv, apply opsall + C5 + C6, swap wheel ---"
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && (test -f $XO.pre_v125_c5 && cp $XO.pre_v125_c5 $XO || true) && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }
docker cp /root/build/patch_v125_opsall_fix.py lsv-test:/root/ || { echo "ABORT_CP_FIX"; exit 1; }
PF=$(docker exec lsv-test python3 /root/patch_v125_opsall_fix.py)
echo "$PF" | grep -q 'V125_OPSFIX_OK\|V125_OPSFIX_ALREADY' || { echo "ABORT_OPSFIX_PATCH"; exit 1; }
docker cp /root/build/patch_v125_s4fp8_bridge.py lsv-test:/root/ || { echo "ABORT_CP_C5"; exit 1; }
PC=$(docker exec lsv-test python3 /root/patch_v125_s4fp8_bridge.py)
echo "$PC" | grep -q 'V125_C5_OK\|V125_C5_ALREADY' || { echo "ABORT_C5_PATCH"; exit 1; }
docker cp /root/build/patch_v125_c6_staticraise.py lsv-test:/root/ || { echo "ABORT_CP_C6"; exit 1; }
PC6=$(docker exec lsv-test python3 /root/patch_v125_c6_staticraise.py)
echo "$PC6" | grep -q 'V125_C6_OK\|V125_C6_ALREADY' || { echo "ABORT_C6_PATCH"; exit 1; }
docker cp /root/build/p23b_diag.py lsv-test:/root/ || { echo "ABORT_CP_DIAG"; exit 1; }

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

diagleg n2e4m3s4 /root/serve_p22c5_n2_e4m3s4.sh /root/serve_p22c5_n2_e4m3s4.log
diagleg n2e5m2s4 /root/serve_p22c5_n2_e5m2s4.sh /root/serve_p22c5_n2_e5m2s4.log

echo "--- teardown: wheel revert + venv revert + certified serve ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD2=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD2=1; break; fi
done
[ "$PD2" -eq 1 ] || { echo "ABORT_TEARDOWN_STILL_UP"; exit 1; }
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
echo "=== P23B DIAG DONE $(date +%F' '%T) ==="
echo "P23B_DIAG_DONE"
} > "$L" 2>&1
tail -80 "$L"
echo P23B_DIAG_SCRIPT_DONE
