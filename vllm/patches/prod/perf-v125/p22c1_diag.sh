#!/bin/bash
# p22c1_diag.sh — one instrumented boot of the ns serve to see, at
# runtime, why the C1 dispatch exclusion did not prevent FULL-graph
# replay (fixval ABORT_V60G with the patch statically engaged).
# Applies patch_v125_diag.py on top of the C1 patch, boots the ctrl
# ns config, fires two probes, dumps v125DIAG + v60g lines, then
# restores the certified serve and re-arms the watchdog.
set -u
L=/root/build/lce1/p22c1_diag.log

{
echo "=== P22C1 DIAG start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

echo "--- 1. restore clean venv, re-apply C1 patch, apply diag instrument ---"
docker cp /root/build/patch_v125_nospec_no_full_graph.py lsv-test:/root/ || { echo "ABORT_CP1"; exit 1; }
docker cp /root/build/patch_v125_diag.py lsv-test:/root/ || { echo "ABORT_CP2"; exit 1; }
R=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
CGP=/opt/venv/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py
docker exec lsv-test sh -c "cp $R.pre_v125_c1 $R 2>/dev/null; cp $CGP.pre_v125_diag $CGP 2>/dev/null; python3 -c \"import py_compile; py_compile.compile('$R', doraise=True); py_compile.compile('$CGP', doraise=True)\" && echo RESTORED_CLEAN"
P1=$(docker exec lsv-test python3 /root/patch_v125_nospec_no_full_graph.py)
echo "$P1"
echo "$P1" | grep -q 'V125_C1_PATCH_OK\|V125_C1_ALREADY_PATCHED' || { echo "ABORT_C1_REAPPLY"; exit 1; }
P=$(docker exec lsv-test python3 /root/patch_v125_diag.py)
echo "$P"
echo "$P" | grep -q 'V125_DIAG_DONE ok=True,True' || { echo "ABORT_DIAG_PATCH"; exit 1; }

echo "--- 2. boot ns serve (ctrl config, C1+diag live) ---"
touch /root/build/lane_watchdog.paused
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; echo "port down after ${i}x5s"; break; fi
done
[ "$PD" -eq 1 ] || { echo "ABORT_SERVE_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c ": > /root/serve_p22c1_ctrl.log; cd /root && nohup bash serve_p22c1_ctrl.sh > /dev/null 2>&1 & echo ns-launched"
HK=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "ns health OK after ${i}0s"; break; }
done
[ "$HK" -eq 1 ] || { echo "ABORT_NS_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' /root/serve_p22c1_ctrl.log | tail -5"; exit 1; }
sleep 15

echo "--- 3. two probes ---"
python3 - <<'PYEOF'
import json, urllib.request
M = "qwen3.8-27b-fp8"
def ask(prompt, mt, label):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        vis = r["choices"][0]["message"].get("content") or ""
        print(f"[diag] {label} TEXT {vis[:120]!r}")
    except Exception as e:
        print(f"[diag] {label}_FAIL {type(e).__name__}: {e}")
ask("What is 17*23? Answer with the number only.", 12, "MT12")
ask("List the numbers from 1 to 5, comma separated, nothing else.", 40, "LIST40")
PYEOF

echo "--- 4. collect instrument output ---"
docker exec lsv-test sh -c "grep 'v125DIAG' /root/serve_p22c1_ctrl.log | head -90"
echo "--- v60g count:"
V6=$(docker exec lsv-test sh -c "grep -c 'v60g NO-DRAFT' /root/serve_p22c1_ctrl.log || echo 0")
echo "ns v60g count=$V6"

echo "--- 5. restore certified serve ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD2=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD2=1; echo "port down after ${i}x5s"; break; fi
done
[ "$PD2" -eq 1 ] || { echo "ABORT_TEARDOWN_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK2=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK2=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK2" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
SPL2=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
echo "restored spec_lines=$SPL2 (want >=1)"

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C1 DIAG DONE $(date +%F' '%T) ==="
echo "P22C1_DIAG_DONE"
} > "$L" 2>&1
tail -60 "$L"
echo P22C1_DIAG_SCRIPT_DONE
