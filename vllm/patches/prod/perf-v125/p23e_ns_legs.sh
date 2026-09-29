#!/bin/bash
# p23e_ns_legs.sh — v125 P23E: complete the ns-leg quality matrix.
#
# P23D convicted the =2 spec-fp8 postures: 20% post-prefix tool-call
# salad (6/30 both formats, poisoned-state windows of ~3 consecutive
# requests; fp16 control 0/30 on the same venv+wheel). The =1 ns
# postures (ESIMD decode + v124 unique-bridge prefill) never got their
# legs — this chain runs them with the FULL battery (v64, drill,
# serialized, burst, v125b instrumented parser, genspeed, agg) PLUS the
# P23D salad probe (30 alternating tool-call requests) appended after
# the parser step, so ns-fp8 quality is measured under the same
# standardized conditions. Legs: e4m3ns, e5m2ns, fix16ns (the ns fp16
# control, '' 1/1/1). Teardown identical to P23C.
# EXIT trap mirrors the full restore. Prereq: P23D_DONE.
set -u
L=/root/build/lce1/p23e_battery.log
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

boot_leg() {  # $1 tag, $2 serve script, $3 serve log
  stop_serve || { echo "ABORT_${1}_PORT_STILL_UP"; return 1; }
  sleep 4
  docker exec lsv-test sh -c ": > $3; cd /root && nohup bash $2 > /dev/null 2>&1 & echo $1-launched"
  HK=0
  for i in $(seq 1 48); do
    sleep 10
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "$1 health OK after ${i}0s"; break; }
  done
  [ "$HK" -eq 1 ] || { echo "ABORT_${1}_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' $3 | tail -6"; return 1; }
  sleep 15
  return 0
}

math_probe() {  # $1 tag — one n1 math call + one x8-distinct burst
python3 - "$1" <<'PYEOF'
import json, sys, threading, time, urllib.request
tag = sys.argv[1]
M, URL = "qwen3.8-27b-fp8", "http://127.0.0.1:8000/v1/chat/completions"
def call(content, mt, out, idx):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        out[idx] = (r["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        out[idx] = f"FAIL:{type(e).__name__}"
o = [None]; call("What is 123+268? Reply with just the number.", 600, o, 0)
ok1 = "391" in o[0]
print(f"[{tag}] n1-math {'OK' if ok1 else 'BAD: ' + o[0][:40]!r}")
a = [11 + (i % 19) for i in range(8)]
b = [3 + (i % 7) for i in range(8)]
pr = [f"What is {a[i]}*{b[i]}? Reply with just the number." for i in range(8)]
outs = [None] * 8
ths = [threading.Thread(target=call, args=(pr[i], 32, outs, i)) for i in range(8)]
for t in ths: t.start()
for t in ths: t.join()
ok = sum(1 for i in range(8) if outs[i] == str(a[i] * b[i]))
print(f"[{tag}] x8d {ok}/8 -> {'CLEAN' if ok == 8 else 'BAD'}")
sys.exit(0 if (ok1 and ok == 8) else 1)
PYEOF
}

battery() {  # $1 tag, $2 serve script, $3 serve log, $4 want_spec_lines,
             # $5 mode (''/fp8/fp16), $6 drill_runs, $7 serial_runs,
             # $8 burst_runs — parser step = INSTRUMENTED v125b
  echo "=== P23ELEG $1 $(date +%F' '%T) ==="
  DM0=$(dmesg | grep -c "Engine reset" || true)
  boot_leg "$1" "$2" "$3" || return 1
  SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' $3; true")
  FX=$(docker exec lsv-test sh -c "grep -c 'v125 root fix' $3; true")
  ENG=$(docker exec lsv-test sh -c "grep -c 'FP8_NATIVE ENGAGED' $3; true")
  C5=$(docker exec lsv-test sh -c "grep -c 'v125 P22C5 STATIC_FP8_BRIDGE' $3; true")
  C6=$(docker exec lsv-test sh -c "grep -c 'P22C6' $3; true")
  echo "[$1] spec_lines=$SPL (want $4) v125fix=$FX engaged=$ENG c5=$C5 c6_raise=$C6 (mode '${5:-info}')"
  [ "$SPL" -eq "$4" ] || { echo "ABORT_${1}_SPEC_LINES"; return 1; }
  [ "$FX" -ge 1 ] || { echo "ABORT_${1}_NO_FIX"; return 1; }
  case "${5:-}" in
    fp8)  [ "$ENG" -ge 1 ] || { echo "ABORT_${1}_NO_ENGAGED"; return 1; }
          [ "$C5" -eq 0 ] || { echo "ABORT_${1}_C5_PRESENT"; return 1; }
          [ "$C6" -eq 0 ] || { echo "ABORT_${1}_C6_RAISED"; return 1; } ;;
    fp16) [ "$ENG" -eq 0 ] || { echo "ABORT_${1}_UNEXPECTED_ENGAGED"; return 1; }
          [ "$C5" -eq 0 ] || { echo "ABORT_${1}_C5_PRESENT"; return 1; }
          [ "$C6" -eq 0 ] || { echo "ABORT_${1}_C6_RAISED"; return 1; } ;;
  esac
  math_probe "$1" || { echo "ABORT_${1}_MATH"; return 1; }

  OUT=/root/build/_p23e_$1.txt
  echo "== SOLO COLD seed114 ==" > $OUT
  python3 /root/build/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 114 >> $OUT 2>&1
  echo "== BATTERY_V64 ==" >> $OUT
  sh /root/build/battery_v64.sh >> $OUT 2>&1
  grep -q BATTERY_V64_DONE /root/build/_v64_battery.txt || { echo "ABORT_${1}_BATTERY"; return 1; }
  echo "== WEDGE DRILL x$6 ==" >> $OUT
  bash /root/build/repro_sustain.sh "$6" > /root/build/_p23e_${1}_sustain.txt 2>&1
  DRC=$?
  grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_p23e_${1}_sustain.txt || { echo "ABORT_${1}_DRILL rc=$DRC"; return 1; }
  echo "drill_rc=$DRC" >> $OUT
  S=1
  while [ "$S" -le "$7" ]; do
    echo "== SERIALIZED 24 leg $S ==" >> $OUT
    OK=0; FAIL=0; H="200"
    for i in $(seq 1 24); do
      out=$(curl -s --max-time 20 -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"Count from 1 to 5.\",\"max_tokens\":24,\"temperature\":0}" -o /dev/null -w "%{http_code}")
      [ "$out" = "200" ] && OK=$((OK+1)) || FAIL=$((FAIL+1))
      sleep 6
      H=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:8000/health)
      [ "$H" != "200" ] && break
    done
    echo "SERIALIZED$S ok=$OK fail=$FAIL survived=$H" >> $OUT
    [ "$H" = "200" ] && [ "$OK" -eq 24 ] || { echo "ABORT_${1}_SERIALIZED$S"; return 1; }
    S=$((S+1))
  done
  B=0
  while [ "$B" -lt "$8" ]; do
    T="${1}_$B"
    bash /root/build/burst_harsh.sh "$T" >> $OUT 2>&1
    grep -q "SURVIVED" /root/build/lce1/burst_$T.log || { echo "ABORT_${1}_BURST_$B"; return 1; }
    B=$((B+1))
  done
  echo "== PARSER BATTERY (v125b instrumented) ==" >> $OUT
  cd /root/build && python3 cc_parser_battery_v125b.py >> $OUT 2>&1
  PRC=$?
  if [ "$PRC" -ne 0 ]; then
    echo "[$1] parser DOUBLE-FAIL (rc=$PRC) — serve-log structured-output lines:"
    docker exec lsv-test sh -c "grep -iE 'structured|xgrammar|grammar|guidance|tool_parser' $3 | tail -12; true"
    echo "ABORT_${1}_PARSER"
    return 1
  fi
  grep -q 'RETRIED-PASS' $OUT && echo "[$1] NOTE: parser needed settle-retry (transient signature)"
  echo "== SALAD PROBE x30 ==" >> $OUT
  GMa=$(docker exec lsv-test sh -c "grep -c 'grammar_matcher' $3; true")
  python3 /root/build/p23d_probe.py >> $OUT 2>&1
  GMb=$(docker exec lsv-test sh -c "grep -c 'grammar_matcher' $3; true")
  SAL=$(grep -c ' SALAD ' $OUT || true)
  echo "[$1] salad_probe=$SAL/30 grammar_matcher_delta=$((GMb-GMa))"
  echo "== SOLO GENSPEED x3 ==" >> $OUT
  cd /root/build && python3 bench_genspeed.py 1 1024 3 >> $OUT 2>&1
  echo "== AGG 4x1024 x1 ==" >> $OUT
  cd /root/build && python3 bench_genspeed.py 4 1024 1 >> $OUT 2>&1
  TB=$(docker exec lsv-test sh -c "grep -ciE 'traceback|EngineDeadError' $3; true")
  DM1=$(dmesg | grep -c "Engine reset" || true)
  echo "[$1] tracebacks=$TB (want 0) resets_delta=$((DM1-DM0)) (want 0)"
  [ "$TB" -eq 0 ] || { echo "ABORT_${1}_TRACEBACKS"; return 1; }
  [ "$DM1" -eq "$DM0" ] || { echo "ABORT_${1}_RESETS"; return 1; }
  echo "[$1] LEG PASS $(date +%T)"
  return 0
}

{
echo "=== P23E NS-LEGS start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
grep -q 'P23D_DONE' /root/build/lce1/p23d_salad.log || { echo "ABORT_NO_P23D_PROOF"; exit 1; }
DM=$(dmesg | grep -c "Engine reset" || true)
echo "dmesg Engine resets at start=$DM (recorded, not gated — host carried P23+P23B traffic)"

echo "--- apply opsall + C5 + C6, swap wheel, write =2 scripts ---"
docker start lsv-test || { echo "ABORT_CONTAINER_START"; exit 1; }
sleep 3
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && (test -f $XO.pre_v125_c5 && cp $XO.pre_v125_c5 $XO || true) && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }
for PYP in patch_v125_opsall_fix.py patch_v125_s4fp8_bridge.py patch_v125_c6_staticraise.py cc_parser_battery_v125b.py; do
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
STR=$(docker exec lsv-test bash -c "strings $SP/_xpu_C.abi3.so | grep -c 'fp8_e4m3fn'" || true)
echo "wheel fp8_strings=$STR (want >=1)"
[ "$STR" -ge 1 ] 2>/dev/null || { echo "ABORT_WHEEL_STRINGS"; exit 1; }
docker exec lsv-test sh -c "sed 's/VLLM_XPU_GDN_FP8_NATIVE=1/VLLM_XPU_GDN_FP8_NATIVE=2/' /root/serve_p22c4_e4m3s4.sh > /root/serve_p22c5_n2_e4m3s4.sh; sed 's/VLLM_XPU_GDN_FP8_NATIVE=1/VLLM_XPU_GDN_FP8_NATIVE=2/' /root/serve_p22c4_e5m2s4.sh > /root/serve_p22c5_n2_e5m2s4.sh; sed -i 's|>> /root/serve_p22c4_|>> /root/serve_p22c5_n2_|g' /root/serve_p22c5_n2_e4m3s4.sh /root/serve_p22c5_n2_e5m2s4.sh; chmod +x /root/serve_p22c5_n2_*.sh && echo N2_SCRIPTS_OK" || { echo "ABORT_N2_SCRIPTS"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

echo "--- battery legs (parser = v125b instrumented, + salad probe) ---"
battery e4m3ns    /root/serve_p22c4_e4m3ns.sh     /root/serve_p22c4_e4m3ns.log    0 ''   1 1 1 || { echo "LEG_E4M3NS_FAILED"; exit 1; }
battery e5m2ns    /root/serve_p22c4_e5m2ns.sh     /root/serve_p22c4_e5m2ns.log    0 ''   1 1 1 || { echo "LEG_E5M2NS_FAILED"; exit 1; }
battery fix16ns   /root/serve_p22c4_16ns.sh       /root/serve_p22c4_16ns.log      0 ''   1 1 1 || { echo "LEG_FIX16NS_FAILED"; exit 1; }

echo "--- C. teardown: wheel revert + venv revert + certified serve ---"
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
echo "=== P23E NS-LEGS DONE $(date +%F' '%T) ==="
echo "P23E_BATTERY_DONE"
} > "$L" 2>&1
tail -150 "$L"
echo P23E_BATTERY_SCRIPT_DONE
