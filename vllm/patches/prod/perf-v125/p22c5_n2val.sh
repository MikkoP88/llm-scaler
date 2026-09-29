#!/bin/bash
# p22c5_n2val.sh — v125 P22B wheel + VLLM_XPU_GDN_FP8_NATIVE=2 validation.
#
# The static bridge (P22C5) corrupts at n>1 (Run 7k conviction; n=1 never
# touched it — ESIMD takes num_spec_decodes==1). The P22B wheel makes =2
# pool-straight fp8 spec possible: the SYCL kernel owns state updates on
# the REAL pool with REAL cache_indices — byte-identical protocol to
# certified fp16, only StateT differs. This chain swaps the freshly built
# wheel into lsv-test (build_vxk_v125_buildonly.sh artifact), boots both
# s4 fp8 legs under =2, and runs the SAME correctness probe families that
# convicted the bridge (mt8 / x8-identical / x8-distinct / pair both) —
# =2 must be clean at n>1 where the bridge was not — plus solo x2 per leg.
#
# Gates per leg: capture_finished>=1, spec_lines=2, FP8_NATIVE ENGAGED
# >=1 (the =2 pool-straight marker). C5 STATIC marker must be ABSENT
# (bridge must not engage under =2).
#
# Ends: wheel REVERTED (.v125bak restored + strings proof back to the old
# reject literal), venv 4-file revert, certified serve, markers=0,
# watchdog re-arm. EXIT trap mirrors the full restore.
set -u
L=/root/build/lce1/p22c5_n2val.log
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
  # wheel revert FIRST (cert serve must load the original), then venv
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

probes() {
python3 - <<'PYEOF'
import json, sys, threading, time, urllib.request
M = "qwen3.8-27b-fp8"
URL = "http://127.0.0.1:8000/v1/chat/completions"

def call(content, mt, out, idx):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        m = r["choices"][0]["message"]
        out[idx] = ((m.get("content") or "").strip(),
                    r["choices"][0].get("finish_reason"),
                    r.get("usage", {}).get("completion_tokens"))
    except Exception as e:
        out[idx] = (f"FAIL:{type(e).__name__}", "err", -1)

def burst(prompts, mt, tag, expect):
    n = len(prompts)
    outs = [None] * n
    ths = [threading.Thread(target=call, args=(prompts[i], mt, outs, i)) for i in range(n)]
    for t in ths: t.start()
    for t in ths: t.join()
    ok = sum(1 for i, o in enumerate(outs) if o[0] == expect[i])
    bad = [(i, o[0][:24], expect[i]) for i, o in enumerate(outs) if o[0] != expect[i]]
    print(f"[{tag}] {ok}/{n} ok" + ("" if not bad else f" BAD={bad}"))
    return ok == n

ID8 = "What is 12*12? Reply with just the number."

# n=1 sanity (ESIMD spec op path, unchanged by =2)
o = [None]; call("What is 123+268? Reply with just the number.", 600, o, 0)
print(f"[n1-math] {o[0][0][:20]!r} -> {'OK' if '391' in o[0][0] else 'BAD'}")

# n>1 correctness — the regime where the bridge failed; =2 must be clean
mt_i = all(burst([ID8]*8, 32, f"mt8id-r{r}", expect=["144"]*8) for r in (1, 2))
time.sleep(2)
xi = sum(1 for r in range(1, 7)
         if burst(["What is 9*11? Reply with just the number."]*8, 32,
                  f"x8id-r{r}", expect=["99"]*8))
print(f"[x8-identical] {xi}/6 rounds all-99")
time.sleep(2)
xd = 0
for r in range(1, 7):
    a = [11 + ((r*8 + i) % 19) for i in range(8)]
    b = [3 + ((r*8 + i) % 7) for i in range(8)]
    pr = [f"What is {a[i]}*{b[i]}? Reply with just the number." for i in range(8)]
    if burst(pr, 32, f"x8di-r{r}", expect=[str(a[i]*b[i]) for i in range(8)]):
        xd += 1
print(f"[x8-distinct] {xd}/6 rounds all-correct")
time.sleep(2)
pi = sum(1 for r in range(1, 11)
         if burst(["What is 7*8? Reply with just the number."]*2, 32,
                  f"pid-r{r}", expect=["56", "56"]))
print(f"[pair-identical] {pi}/10 rounds both-56")
time.sleep(2)
pd = 0
for r in range(1, 11):
    a = [13 + ((r*2 + i) % 17) for i in range(2)]
    b = [4 + ((r*2 + i) % 5) for i in range(2)]
    pr = [f"What is {a[i]}*{b[i]}? Reply with just the number." for i in range(2)]
    if burst(pr, 32, f"pdi-r{r}", expect=[str(a[i]*b[i]) for i in range(2)]):
        pd += 1
print(f"[pair-distinct] {pd}/10 rounds both-correct")

# solo speed x2
solos = []
for _ in range(2):
    t0 = time.perf_counter()
    o = [None]; call("Count from 1 to 1000.", 200, o, 0)
    n = o[0][2] if o[0][2] and o[0][2] > 0 else 200
    solos.append(n / (time.perf_counter() - t0))
s = " ".join(f"{v:.2f}" for v in solos)
allok = (mt_i and xi == 6 and xd == 6 and pi == 10 and pd == 10)
print(f"[RESULT] mt8id={mt_i} x8id={xi}/6 x8di={xd}/6 pid={pi}/10 pdi={pd}/10 "
      f"solo=[{s}] tok/s -> {'ALL_CLEAN' if allok else 'DEFECT_PRESENT'}")
PYEOF
}

leg() {  # $1 tag, $2 serve script, $3 serve log
  echo "=== N2LEG $1 $(date +%F' '%T) ==="
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
  CGF=$(docker exec lsv-test sh -c "grep -c 'Graph capturing finished' $3; true")
  SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' $3; true")
  ENG=$(docker exec lsv-test sh -c "grep -c 'FP8_NATIVE ENGAGED' $3; true")
  C5=$(docker exec lsv-test sh -c "grep -c 'v125 P22C5 STATIC_FP8_BRIDGE' $3; true")
  echo "[$1] capture_finished=$CGF (want >=1) spec_lines=$SPL (want 2) engaged=$ENG (want >=1) c5_static=$C5 (want 0)"
  [ "$CGF" -ge 1 ] || { echo "ABORT_${1}_CAPTURE"; return 1; }
  [ "$SPL" -eq 2 ] || { echo "ABORT_${1}_SPEC_LINES"; return 1; }
  [ "$ENG" -ge 1 ] || { echo "ABORT_${1}_NO_ENGAGED"; return 1; }
  [ "$C5" -eq 0 ] || { echo "ABORT_${1}_C5_SHOULD_BE_ABSENT"; return 1; }
  probes
}

{
echo "=== P22C5 N2VAL start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
grep -q 'V125_BUILDONLY_DONE' /root/build/lce1/v125_buildonly.log || { echo "ABORT_WHEEL_NOT_BUILT"; exit 1; }
grep -q 'P22C5_DIAG2_DONE' /root/build/lce1/p22c5_diag2.log || { echo "ABORT_DIAG2_NOT_DONE"; exit 1; }

echo "--- clean venv, apply opsall + C5 (=2 gate lives in C5), swap wheel ---"
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && (test -f $XO.pre_v125_c5 && cp $XO.pre_v125_c5 $XO || true) && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }
docker cp /root/build/patch_v125_opsall_fix.py lsv-test:/root/ || { echo "ABORT_CP_FIX"; exit 1; }
PF=$(docker exec lsv-test python3 /root/patch_v125_opsall_fix.py)
echo "$PF" | grep -q 'V125_OPSFIX_OK\|V125_OPSFIX_ALREADY' || { echo "ABORT_OPSFIX_PATCH"; exit 1; }
docker cp /root/build/patch_v125_s4fp8_bridge.py lsv-test:/root/ || { echo "ABORT_CP_C5"; exit 1; }
PC=$(docker exec lsv-test python3 /root/patch_v125_s4fp8_bridge.py)
echo "$PC" | grep -q 'V125_C5_OK\|V125_C5_ALREADY' || { echo "ABORT_C5_PATCH"; exit 1; }

# wheel swap (backup current, copy new, load_library verify)
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

echo "--- write =2 serve scripts (pool-straight native fp8 spec) ---"
docker exec lsv-test sh -c "sed 's/VLLM_XPU_GDN_FP8_NATIVE=1/VLLM_XPU_GDN_FP8_NATIVE=2/' /root/serve_p22c4_e4m3s4.sh > /root/serve_p22c5_n2_e4m3s4.sh; sed 's/VLLM_XPU_GDN_FP8_NATIVE=1/VLLM_XPU_GDN_FP8_NATIVE=2/' /root/serve_p22c4_e5m2s4.sh > /root/serve_p22c5_n2_e5m2s4.sh; sed -i 's|>> /root/serve_p22c4_|>> /root/serve_p22c5_n2_|g' /root/serve_p22c5_n2_e4m3s4.sh /root/serve_p22c5_n2_e5m2s4.sh; chmod +x /root/serve_p22c5_n2_*.sh && echo N2_SCRIPTS_OK" || { echo "ABORT_N2_SCRIPTS"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

leg n2e4m3s4 /root/serve_p22c5_n2_e4m3s4.sh /root/serve_p22c5_n2_e4m3s4.log || { echo "LEG1_FAILED"; exit 1; }
leg n2e5m2s4 /root/serve_p22c5_n2_e5m2s4.sh /root/serve_p22c5_n2_e5m2s4.log || { echo "LEG2_FAILED"; exit 1; }

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
echo "=== P22C5 N2VAL DONE $(date +%F' '%T) ==="
echo "P22C5_N2VAL_DONE"
} > "$L" 2>&1
tail -70 "$L"
echo P22C5_N2VAL_SCRIPT_DONE
