#!/bin/bash
# p22c5_fixval.sh — v125 P22C5 fix validation: boots the two C4 legs that
# died at the wheel's spec dtype guard (e4m3s4: RuntimeError ssm_state dtype
# ... Float8_e4m3fn at capture 0/64; e5m2s4 never reached) on the patched
# venv (opsall root fix + P22C5 static fp8 spec bridge) and runs gates +
# battery + solo speed per leg. Speed is RECORDED not gated (the bridge-tax
# verdict feeds the P24 wheel decision); correctness gates are hard.
# Ends: teardown + clean-venv restore (all four files) + certified serve +
# lane_watchdog re-arm. EXIT trap mirrors the restore.
set -u
L=/root/build/lce1/p22c5_fixval.log
R=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
CGP=/opt/venv/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py
XPU=/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py
XO=/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py

restore_lane() {
  [ -f /root/build/lane_watchdog.paused ] || return 0
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true" >/dev/null 2>&1
  for i in $(seq 1 18); do sleep 5; curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || break; done
  sleep 4
  docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU 2>/dev/null; cp $R.pre_v125_c1 $R 2>/dev/null; cp $CGP.pre_v125_diag $CGP 2>/dev/null; test -f $XO.pre_v125_c5 && cp $XO.pre_v125_c5 $XO; python3 -c \"import py_compile; py_compile.compile('$XPU', doraise=True); py_compile.compile('$R', doraise=True); py_compile.compile('$CGP', doraise=True); py_compile.compile('$XO', doraise=True)\" && echo trap-clean-venv" >> "$L" 2>&1
  docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo trap-restore-launched" >> "$L" 2>&1
  for i in $(seq 1 48); do sleep 10; curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && break; done
  rm -f /root/build/lane_watchdog.paused
  echo "trap lane_watchdog re-armed ($(date +%T))" >> "$L" 2>&1
}
trap restore_lane EXIT

battery() {  # $1 = tag
python3 - "$1" <<'PYEOF'
import json, sys, threading, time, urllib.request
tag = sys.argv[1]
M = "qwen3.8-27b-fp8"
URL = "http://127.0.0.1:8000/v1/chat/completions"

def call(body, out, idx):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        ch = r["choices"][0]
        msg = ch.get("message", {})
        out[idx] = (msg.get("content") or "") + "|" + (msg.get("reasoning_content") or "")
    except Exception as e:
        print(f"[{tag}] call_FAIL {type(e).__name__}: {e}")
        out[idx] = ""

def mk(content, mt, think=False):
    b = {"model": M, "max_tokens": mt, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": think},
         "messages": [{"role": "user", "content": content}]}
    return b

# MT8: 8 concurrent multiply checks
outs = [""] * 8
ths = [threading.Thread(target=call, args=(mk("What is 12*12? Reply with just the number.", 32, False), outs, i)) for i in range(8)]
for t in ths: t.start()
for t in ths: t.join()
mt8 = sum(1 for o in outs if "144" in o)

# MATH600
o = [""]; call(mk("What is 123+268? Reply with just the number.", 600, False), o, 0)
math391 = "391" in o[0]

# LIST300
o = [""]; call(mk("List the first 5 prime numbers, one per line, no other text.", 300, False), o, 0)
list5 = all(p in o[0] for p in ("2", "3", "5", "7", "11"))

# THINK600
o = [""]; call(mk("What is 17*23? Reply with just the number.", 600, True), o, 0)
think = len(o[0].split("|")) == 2 and len(o[0].split("|")[1].strip()) > 0

# solo speed x2 (200-token timed gen)
solos = []
for _ in range(2):
    body = mk("Count from 1 to 1000.", 200, False)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        n = r.get("usage", {}).get("completion_tokens", 200)
        solos.append(n / (time.perf_counter() - t0))
    except Exception as e:
        print(f"[{tag}] solo_FAIL {type(e).__name__}: {e}")
        solos.append(0.0)

s = " ".join(f"{v:.2f}" for v in solos)
print(f"[{tag}] BATTERY mt8={mt8}/8 math391={math391} list5={list5} think={think} solo=[{s}] tok/s")
if not (mt8 == 8 and math391 and list5 and think and all(v > 5 for v in solos)):
    print(f"[{tag}] BATTERY_FAIL")
PYEOF
}

leg() {  # $1 tag, $2 serve script, $3 serve log
  echo "=== C5LEG $1 $(date +%F' '%T) ==="
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
  PD=0
  for i in $(seq 1 18); do
    sleep 5
    if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; break; fi
  done
  [ "$PD" -eq 1 ] || { echo "ABORT_${1}_PORT_STILL_UP"; return 1; }
  sleep 4
  docker exec lsv-test sh -c ": > $3; cd /root && nohup bash $2 > /dev/null 2>&1 & echo $1-launched"
  HK=0
  for i in $(seq 1 48); do
    sleep 10
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "$1 health OK after ${i}0s"; break; }
  done
  [ "$HK" -eq 1 ] || { echo "ABORT_${1}_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' $3 | tail -8"; return 1; }
  sleep 15
  # tqdm carriage returns collapse the whole progress bar onto a couple of
  # log lines, so grep -c 'Capturing' counts LINES not graphs. The robust
  # completion signal is the runner's own post-capture line (health already
  # gates worker death; the legs chain calibrated the banner count as 1).
  CAP=$(docker exec lsv-test sh -c "grep -c 'Capturing' $3; true")
  CGF=$(docker exec lsv-test sh -c "grep -c 'Graph capturing finished' $3; true")
  SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' $3; true")
  FX=$(docker exec lsv-test sh -c "grep -c 'v125 root fix' $3; true")
  C5=$(docker exec lsv-test sh -c "grep -c 'v125 P22C5 STATIC_FP8_BRIDGE' $3; true")
  PA=$(docker exec lsv-test sh -c "grep -c 'v125 P22A' $3; true")
  echo "[$1] capture_finished=$CGF (want >=1; tqdm-lines=$CAP) spec_lines=$SPL (want 2) v125fix=$FX (want >=1) C5_static=$C5 (want >=1) P22A=$PA (info)"
  [ "$CGF" -ge 1 ] || { echo "ABORT_${1}_CAPTURE"; return 1; }
  [ "$SPL" -eq 2 ] || { echo "ABORT_${1}_SPEC_LINES"; return 1; }
  [ "$FX" -ge 1 ] || { echo "ABORT_${1}_NO_FIX_WARNING"; return 1; }
  [ "$C5" -ge 1 ] || { echo "ABORT_${1}_NO_C5_MARKER"; return 1; }
  battery "$1"
}

{
echo "=== P22C5 FIXVAL start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
docker exec lsv-test sh -c "test -x /root/serve_p22c4_e4m3s4.sh && test -x /root/serve_p22c4_e5m2s4.sh && echo LEG_SCRIPTS_PRESENT" || { echo "ABORT_NO_LEG_SCRIPTS"; exit 1; }

echo "--- clean venv first (certified posture), then apply opsall + P22C5 ---"
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && (test -f $XO.pre_v125_c5 && cp $XO.pre_v125_c5 $XO || true) && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }

docker cp /root/build/patch_v125_opsall_fix.py lsv-test:/root/ || { echo "ABORT_CP_FIX"; exit 1; }
PF=$(docker exec lsv-test python3 /root/patch_v125_opsall_fix.py)
echo "$PF"
echo "$PF" | grep -q 'V125_OPSFIX_OK\|V125_OPSFIX_ALREADY' || { echo "ABORT_OPSFIX_PATCH"; exit 1; }

docker cp /root/build/patch_v125_s4fp8_bridge.py lsv-test:/root/ || { echo "ABORT_CP_C5"; exit 1; }
PC=$(docker exec lsv-test python3 /root/patch_v125_s4fp8_bridge.py)
echo "$PC"
echo "$PC" | grep -q 'V125_C5_OK\|V125_C5_ALREADY' || { echo "ABORT_C5_PATCH"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

leg e4m3s4 /root/serve_p22c4_e4m3s4.sh /root/serve_p22c4_e4m3s4.log || exit 1
leg e5m2s4 /root/serve_p22c4_e5m2s4.sh /root/serve_p22c4_e5m2s4.log || exit 1

echo "--- teardown + clean-venv restore + certified serve ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD2=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD2=1; break; fi
done
[ "$PD2" -eq 1 ] || { echo "ABORT_TEARDOWN_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && cp $XO.pre_v125_c5 $XO && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo REVERT_CLEAN_OK" || { echo "ABORT_REVERT"; exit 1; }
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK2=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK2=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK2" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
SPL2=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log; true")
INSTR2=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|v125 root fix\|perf-v125 C1\|v125 P22C5' /root/serve_full.log; true")
echo "restored spec_lines=$SPL2 fix-marker-lines=$INSTR2 (want 2 / 0)"
[ "$SPL2" -eq 2 ] || { echo "ABORT_RESTORE_SPEC"; exit 1; }
[ "$INSTR2" -eq 0 ] || { echo "ABORT_RESTORE_MARKERS"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C5 FIXVAL DONE $(date +%F' '%T) ==="
echo "P22C5_FIXVAL_DONE"
} > "$L" 2>&1
tail -80 "$L"
echo P22C5_FIXVAL_SCRIPT_DONE
