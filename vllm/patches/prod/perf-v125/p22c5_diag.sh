#!/bin/bash
# p22c5_diag.sh — v125 P22C5 follow-up diagnosis. The fixval battery showed
# mt8=7/8 + think=False on BOTH s4 fp8 legs while cert probes are clean.
# think=False is already PROVEN a probe bug (this API surfaces reasoning as
# message.reasoning / stream deltas, NOT reasoning_content; cert content
# '\n\n391' with 53 completion tokens = thinking worked). mt8=7/8 needs
# conviction: MT8 is the ONLY battery probe exercising n>1 STATIC-BRIDGE
# batches (solo/math/list run at n=1 -> ESIMD spec op, no bridge; warm cert
# MT8 is 8/8 '144' @ 4 tok). This chain distinguishes a cold-first-batch
# warmup flake from a real bridge defect at n>1: per-stream dumps across
# repeated rounds + an identical-answer 8-way crosstalk check + a 2-stream
# x20 persistence check. Ends restored + re-armed (trap mirrors fixval).
set -u
L=/root/build/lce1/p22c5_diag.log
R=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
CGP=/opt/venv/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py
XPU=/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py
XO=/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py

restore_lane() {
  [ -f /root/build/lane_watchdog.paused ] || return 0
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true" >/dev/null 2>&1
  for i in $(seq 1 18); do sleep 5; curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || break; done
  sleep 4
  docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU 2>/dev/null; cp $R.pre_v125_c1 $R 2>/dev/null; cp $CGP.pre_v125_diag $CGP 2>/dev/null; test -f $XO.pre_v125_c5 && cp $XO.pre_v125_c5 $XO; python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo trap-clean-venv" >> "$L" 2>&1
  docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo trap-restore-launched" >> "$L" 2>&1
  for i in $(seq 1 48); do sleep 10; curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && break; done
  rm -f /root/build/lane_watchdog.paused
  echo "trap lane_watchdog re-armed ($(date +%T))" >> "$L" 2>&1
}
trap restore_lane EXIT

diag() {
python3 - <<'PYEOF'
import json, threading, time, urllib.request
M = "qwen3.8-27b-fp8"
URL = "http://127.0.0.1:8000/v1/chat/completions"

def call(content, mt, think, out, idx):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think},
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

def burst(n, content, mt, tag):
    outs = [None] * n
    ths = [threading.Thread(target=call, args=(content, mt, False, outs, i)) for i in range(n)]
    for t in ths: t.start()
    for t in ths: t.join()
    ok = sum(1 for o in outs if o[0] == "144")
    bad = [(i, o) for i, o in enumerate(outs) if o[0] != "144"]
    print(f"[{tag}] {ok}/{n} ok" + ("" if not bad else f" BAD={bad}"))
    return ok == n

# 1) MT8 x4 rounds (round 1 = cold first batch, the fixval-battery condition)
allok = True
for rnd in range(1, 5):
    allok &= burst(8, "What is 12*12? Reply with just the number.", 32, f"mt8-r{rnd}")
    time.sleep(2)

# 2) 8-way IDENTICAL x10 (crosstalk: all streams same Q, all must be 99)
xok = 0
for rnd in range(1, 11):
    outs = [None] * 8
    ths = [threading.Thread(target=call, args=("What is 9*11? Reply with just the number.", 32, False, outs, i)) for i in range(8)]
    for t in ths: t.start()
    for t in ths: t.join()
    good = sum(1 for o in outs if o[0] == "99")
    if good != 8:
        print(f"[x8-r{rnd}] {good}/8 BAD={[(i,o) for i,o in enumerate(outs) if o[0]!='99']}")
    else:
        xok += 1
print(f"[x8-identical] {xok}/10 rounds all-99")

# 3) pair x20 (2-stream persistence: both 56 every round)
pok = 0
for rnd in range(1, 21):
    outs = [None] * 2
    ths = [threading.Thread(target=call, args=("What is 7*8? Reply with just the number.", 32, False, outs, i)) for i in range(2)]
    for t in ths: t.start()
    for t in ths: t.join()
    if all(o[0] == "56" for o in outs):
        pok += 1
    else:
        print(f"[pair-r{rnd}] BAD={outs}")
print(f"[pair20] {pok}/20 rounds both-56")

# 4) THINK corrected check (reasoning via message.reasoning / token burn)
o = [None]; call("What is 17*23? Reply with just the number.", 600, True, o, 0)
c, fin, ntok = o[0]
print(f"[think] content={c[:30]!r} finish={fin} tokens={ntok} -> "
      f"{'OK' if ('391' in c and ntok and ntok > 20) else 'SUSPECT'}")
PYEOF
}

{
echo "=== P22C5 DIAG start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

docker cp /root/build/patch_v125_opsall_fix.py lsv-test:/root/ || { echo "ABORT_CP_FIX"; exit 1; }
PF=$(docker exec lsv-test python3 /root/patch_v125_opsall_fix.py)
echo "$PF" | grep -q 'V125_OPSFIX_OK\|V125_OPSFIX_ALREADY' || { echo "ABORT_OPSFIX_PATCH"; exit 1; }
docker cp /root/build/patch_v125_s4fp8_bridge.py lsv-test:/root/ || { echo "ABORT_CP_C5"; exit 1; }
PC=$(docker exec lsv-test python3 /root/patch_v125_s4fp8_bridge.py)
echo "$PC" | grep -q 'V125_C5_OK\|V125_C5_ALREADY' || { echo "ABORT_C5_PATCH"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

echo "--- boot e4m3s4 (spec-4 fp8 pool, static bridge) ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
for i in $(seq 1 18); do sleep 5; curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || break; done
sleep 4
docker exec lsv-test sh -c ": > /root/serve_p22c4_e4m3s4.log; cd /root && nohup bash serve_p22c4_e4m3s4.sh > /dev/null 2>&1 & echo e4m3s4-launched"
HK=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "e4m3s4 health OK after ${i}0s"; break; }
done
[ "$HK" -eq 1 ] || { echo "ABORT_HEALTH"; exit 1; }
sleep 15
CGF=$(docker exec lsv-test sh -c "grep -c 'Graph capturing finished' /root/serve_p22c4_e4m3s4.log; true")
C5=$(docker exec lsv-test sh -c "grep -c 'v125 P22C5 STATIC_FP8_BRIDGE' /root/serve_p22c4_e4m3s4.log; true")
echo "capture_finished=$CGF C5_static=$C5"
[ "$CGF" -ge 1 ] || { echo "ABORT_CAPTURE"; exit 1; }
[ "$C5" -ge 1 ] || { echo "ABORT_NO_C5"; exit 1; }

diag

echo "--- teardown + clean-venv restore + certified serve ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then break; fi
done
sleep 4
docker exec lsv-test sh -c "cp $XPU.pre_v125_opsfix $XPU && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && cp $XO.pre_v125_c5 $XO && python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ('$XPU', '$R', '$CGP', '$XO')]\" && echo REVERT_CLEAN_OK" || { echo "ABORT_REVERT"; exit 1; }
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK2=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK2=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK2" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C5 DIAG DONE $(date +%F' '%T) ==="
echo "P22C5_DIAG_DONE"
} > "$L" 2>&1
tail -60 "$L"
echo P22C5_DIAG_SCRIPT_DONE
