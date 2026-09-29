#!/bin/bash
# p22c5_agg2.sh — v125 agg matrix on the VALIDATED spec-fp8 posture.
#
# Replaces the held p22c4_agg.sh (whose s4 legs would have run the
# convicted static bridge — garbage-in). Per Run 7m: spec fp8 = =2 +
# P22B wheel (pool-straight native, ALL_CLEAN at every n, both formats);
# ns fp8 = =1 ESIMD surface; cert16s4 = fp16 control. This chain:
# clean venv -> opsall + C5 -> WHEEL SWAP (backup + strings proof) ->
# six legs (fix16ns / e4m3ns / e5m2ns / cert16s4 / n2e4m3s4 / n2e5m2s4)
# each with agg 4x256 x2 + solo x2 -> wheel REVERT (strings=0 proof) +
# 4-file venv revert + certified serve + markers=0 + re-arm.
# EXIT trap mirrors the full restore.
set -u
L=/root/build/lce1/p22c5_agg2.log
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

probes() {  # $1 = tag
python3 - "$1" <<'PYEOF'
import json, sys, threading, time, urllib.request
tag = sys.argv[1]
M = "qwen3.8-27b-fp8"
URL = "http://127.0.0.1:8000/v1/chat/completions"

def call(mt, out):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Count from 1 to 1000."}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        n = r.get("usage", {}).get("completion_tokens", mt)
        out.append((n, time.perf_counter() - t0))
    except Exception as e:
        print(f"[{tag}] call_FAIL {type(e).__name__}: {e}")
        out.append((0, -1.0))

# agg 4x256 x2
for run in (1, 2):
    outs = [[] for _ in range(4)]
    ths = [threading.Thread(target=call, args=(256, o)) for o in outs]
    t0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.perf_counter() - t0
    recs = [o[0] for o in outs if o]
    ok = [n for n, dt in recs if dt > 0]
    toks = sum(ok)
    per = " ".join(f"{n}tok/{dt:.1f}s" for n, dt in recs)
    print(f"[{tag}] agg4x256#{run} wall={wall:.2f}s aggregate={toks/wall if wall>0 else 0:.2f} tok/s ok_streams={len(ok)}/4 [{per}]")

# solo x2
solos = []
for _ in range(2):
    o = []
    t0 = time.perf_counter()
    call(200, o)
    if o and o[0][1] > 0:
        solos.append(o[0][0] / o[0][1])
s = " ".join(f"{v:.2f}" for v in solos)
print(f"[{tag}] solo=[{s}] tok/s")
PYEOF
}

leg() {  # $1 tag, $2 serve script, $3 serve log, $4 want_spec_lines,
         # $5 mode: ''=marker-info-only (ns) / fp8=ENGAGED>=1+C5==0 (n2 s4)
         #          / fp16=ENGAGED==0+C5==0 (cert control)
  echo "=== AGG2LEG $1 $(date +%F' '%T) ==="
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
  echo "[$1] spec_lines=$SPL (want $4) v125fix=$FX (want >=1) engaged=$ENG c5_static=$C5 (mode '${5:-info}')"
  [ "$SPL" -eq "$4" ] || { echo "ABORT_${1}_SPEC_LINES"; exit 1; }
  [ "$FX" -ge 1 ] || { echo "ABORT_${1}_NO_FIX_WARNING"; exit 1; }
  case "${5:-}" in
    fp8)
      [ "$ENG" -ge 1 ] || { echo "ABORT_${1}_NO_ENGAGED"; exit 1; }
      [ "$C5" -eq 0 ] || { echo "ABORT_${1}_C5_PRESENT"; exit 1; } ;;
    fp16)
      [ "$ENG" -eq 0 ] || { echo "ABORT_${1}_UNEXPECTED_ENGAGED"; exit 1; }
      [ "$C5" -eq 0 ] || { echo "ABORT_${1}_C5_PRESENT"; exit 1; } ;;
  esac
  probes "$1"
}

{
echo "=== P22C5 AGG2 start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
grep -q 'P22C5_N2VAL_DONE' /root/build/lce1/p22c5_n2val.log || { echo "ABORT_N2VAL_NOT_DONE"; exit 1; }
docker exec lsv-test sh -c "for f in serve_p22c4_16ns.sh serve_p22c4_e4m3ns.sh serve_p22c4_e5m2ns.sh serve_p22c4_e4m3s4.sh serve_p22c4_e5m2s4.sh; do test -x /root/\$f || exit 1; done && echo LEG_SCRIPTS_PRESENT" || { echo "ABORT_NO_LEG_SCRIPTS"; exit 1; }

echo "--- clean venv, apply opsall + C5, swap wheel ---"
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && (test -f $XO.pre_v125_c5 && cp $XO.pre_v125_c5 $XO || true) && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }
docker cp /root/build/patch_v125_opsall_fix.py lsv-test:/root/ || { echo "ABORT_CP_FIX"; exit 1; }
PF=$(docker exec lsv-test python3 /root/patch_v125_opsall_fix.py)
echo "$PF" | grep -q 'V125_OPSFIX_OK\|V125_OPSFIX_ALREADY' || { echo "ABORT_OPSFIX_PATCH"; exit 1; }
docker cp /root/build/patch_v125_s4fp8_bridge.py lsv-test:/root/ || { echo "ABORT_CP_C5"; exit 1; }
PC=$(docker exec lsv-test python3 /root/patch_v125_s4fp8_bridge.py)
echo "$PC" | grep -q 'V125_C5_OK\|V125_C5_ALREADY' || { echo "ABORT_C5_PATCH"; exit 1; }

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

echo "--- write =2 serve scripts from the s4 leg scripts ---"
docker exec lsv-test sh -c "sed 's/VLLM_XPU_GDN_FP8_NATIVE=1/VLLM_XPU_GDN_FP8_NATIVE=2/' /root/serve_p22c4_e4m3s4.sh > /root/serve_p22c5_n2_e4m3s4.sh; sed 's/VLLM_XPU_GDN_FP8_NATIVE=1/VLLM_XPU_GDN_FP8_NATIVE=2/' /root/serve_p22c4_e5m2s4.sh > /root/serve_p22c5_n2_e5m2s4.sh; sed -i 's|>> /root/serve_p22c4_|>> /root/serve_p22c5_n2_|g' /root/serve_p22c5_n2_e4m3s4.sh /root/serve_p22c5_n2_e5m2s4.sh; chmod +x /root/serve_p22c5_n2_*.sh && echo N2_SCRIPTS_OK" || { echo "ABORT_N2_SCRIPTS"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

leg fix16ns  /root/serve_p22c4_16ns.sh       /root/serve_p22c4_16ns.log   0 ''
leg e4m3ns   /root/serve_p22c4_e4m3ns.sh     /root/serve_p22c4_e4m3ns.log 0 ''
leg e5m2ns   /root/serve_p22c4_e5m2ns.sh     /root/serve_p22c4_e5m2ns.log 0 ''
leg cert16s4 /root/serve_user.sh             /root/serve_full.log         2 fp16
leg n2e4m3s4 /root/serve_p22c5_n2_e4m3s4.sh  /root/serve_p22c5_n2_e4m3s4.log 2 fp8
leg n2e5m2s4 /root/serve_p22c5_n2_e5m2s4.sh  /root/serve_p22c5_n2_e5m2s4.log 2 fp8

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
INSTR2=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|v125 root fix\|perf-v125 C1\|v125 P22C5\|v125 C5DUPD\|FP8_NATIVE ENGAGED' /root/serve_full.log; true")
echo "restored spec_lines=$SPL2 fix-marker-lines=$INSTR2 (want 2 / 0)"
[ "$SPL2" -eq 2 ] || { echo "ABORT_RESTORE_SPEC"; exit 1; }
[ "$INSTR2" -eq 0 ] || { echo "ABORT_RESTORE_MARKERS"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C5 AGG2 DONE $(date +%F' '%T) ==="
echo "P22C5_AGG2_DONE"
} > "$L" 2>&1
tail -120 "$L"
echo P22C5_AGG2_SCRIPT_DONE
