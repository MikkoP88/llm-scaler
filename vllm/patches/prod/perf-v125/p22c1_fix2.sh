#!/bin/bash
# p22c1_fix2.sh — Run 7i root-fix validation (PHASES.md).
#
# Conviction (Run 7h D3 matrix): stale custom_ops='none' from pass-1
# VLLM_COMPILE resolution (left in place when the TP>1 platform gate
# force-disables compile late) poisons spec-off decode — native op
# fallback NaNs on single-token GDN decode forwards. Fix =
# patch_v125_opsall_fix.py: at the gate, custom_ops 'none'->'all'.
#
# Boot A (fixed-ns): serve_p22c1_nograph.sh ({"cudagraph_mode":"NONE"},
# NO explicit ops) on the fixed venv — validates the fix propagates to
# the worker model build. Coherent + v60g=0 + v125-fix warnings = PASS.
# Boot B (cert-on-fix): serve_user.sh (spec-4 + FULL graphs) on the
# fixed venv — structural check that the certified posture tolerates
# ops 'all' (full battery is P23). Capture 64 + spec_lines>=1 +
# fingerprints + v60g=0 = PASS.
# End: revert venv to clean (.pre_v125_opsfix), certified serve on the
# OLD posture (the fix stays staged; C4/P23 run on the fixed venv).
set -u
L=/root/build/lce1/p22c1_fix2.log
R=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
CGP=/opt/venv/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py

restore_lane() {
  [ -f /root/build/lane_watchdog.paused ] || return 0
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true" >/dev/null 2>&1
  for i in $(seq 1 18); do sleep 5; curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || break; done
  sleep 4
  docker exec lsv-test sh -c "cp $R.pre_v125_c1 $R 2>/dev/null; cp $CGP.pre_v125_diag $CGP 2>/dev/null; cp /opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py.pre_v125_opsfix /opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py 2>/dev/null; echo trap-clean-venv" >> "$L" 2>&1
  docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo trap-restore-launched" >> "$L" 2>&1
  for i in $(seq 1 48); do sleep 10; curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && break; done
  rm -f /root/build/lane_watchdog.paused
  echo "trap lane_watchdog re-armed ($(date +%T))" >> "$L" 2>&1
}
trap restore_lane EXIT

boot_and_wait() {  # $1 script name (serve_p22c1_nograph | serve_user), $2 log path
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
  PD=0
  for i in $(seq 1 18); do
    sleep 5
    if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; break; fi
  done
  [ "$PD" -eq 1 ] || { echo "ABORT_PORT_STILL_UP"; exit 1; }
  sleep 4
  docker exec lsv-test sh -c ": > $2; cd /root && nohup bash $1.sh > /dev/null 2>&1 & echo $1-launched"
  HK=0
  for i in $(seq 1 48); do
    sleep 10
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "$1 health OK after ${i}0s"; break; }
  done
  [ "$HK" -eq 1 ] || { echo "ABORT_$1_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' $2 | tail -5"; exit 1; }
  sleep 15
}

probes() {  # $1 tag
  python3 - "$1" <<'PYEOF'
import json, sys, time, urllib.request
TAG = sys.argv[1]
M = "qwen3.8-27b-fp8"
def ask(prompt, mt, label, think=False, time_it=False):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        vis = r["choices"][0]["message"].get("content") or ""
        think_blk = "<think>" in (r["choices"][0]["message"].get("reasoning_content") or "") or bool(r["choices"][0]["message"].get("reasoning_content"))
        extra = f" think={think_blk}" if think else ""
        dt = f" {time.time()-t0:.1f}s" if time_it else ""
        print(f"[{TAG}] {label} TEXT {vis[:100]!r}{extra}{dt}")
    except Exception as e:
        print(f"[{TAG}] {label}_FAIL {type(e).__name__}: {e}")
ask("What is 17*23? Answer with the number only.", 12, "MT12")
ask("List the numbers from 1 to 5, comma separated, nothing else.", 40, "LIST40")
ask("What is 12*12? Answer with the number only.", 8, "MT8")
ask("Count from 1 to 12, comma separated, nothing else.", 60, "LIST60")
ask("Think briefly, then answer: what is 19*3? Number only.", 400, "THINK", think=True)
ask("Write a 100-word story about a lighthouse.", 200, "GEN100", time_it=True)
PYEOF
}

collect_leg() {  # $1 log path, $2 want-capture
  CN=$(docker exec lsv-test sh -c "grep -c 'Capturing' $1 || echo 0")
  echo "capture lines=$CN (leg expectation $2)"
  FX=$(docker exec lsv-test sh -c "grep -c 'v125 root fix' $1 || echo 0")
  echo "v125-fix warnings=$FX"
  V6=$(docker exec lsv-test sh -c "grep -c 'v60g NO-DRAFT' $1 || echo 0")
  echo "v60g count=$V6 (want 0)"
  SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' $1 || echo 0")
  echo "spec_lines=$SPL"
  echo "--- init posture:"
  docker exec lsv-test sh -c "grep -m1 'Initializing a V1 LLM engine' $1 | grep -oE \"enforce_eager=[A-Za-z]+|'mode': <CompilationMode.[A-Z_]+: [0-9]+>|custom_ops': \\[[^]]*\\]\""
}

{
echo "=== P22C1 FIX2 start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

echo "--- 1. clean venv + apply opsfix ---"
docker exec lsv-test sh -c "cp $R.pre_v125_c1 $R 2>/dev/null; cp $CGP.pre_v125_diag $CGP 2>/dev/null; python3 -c \"import py_compile; py_compile.compile('$R', doraise=True); py_compile.compile('$CGP', doraise=True)\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }
docker cp /root/build/patch_v125_opsall_fix.py lsv-test:/root/ || { echo "ABORT_CP_FIX"; exit 1; }
PF=$(docker exec lsv-test python3 /root/patch_v125_opsall_fix.py)
echo "$PF"
echo "$PF" | grep -q 'V125_OPSFIX_OK\|V125_OPSFIX_ALREADY' || { echo "ABORT_OPSFIX_PATCH"; exit 1; }

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

echo "=== Boot A: fixed-ns (nograph script, NO explicit ops) ==="
boot_and_wait serve_p22c1_nograph /root/serve_p22c1_nograph.log
probes fixA
collect_leg /root/serve_p22c1_nograph.log 0

echo "=== Boot B: certified serve_user.sh on fixed venv ==="
boot_and_wait serve_user /root/serve_full.log
probes fixB
collect_leg /root/serve_full.log 64

echo "--- revert venv to clean, restore certified posture ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD2=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD2=1; break; fi
done
[ "$PD2" -eq 1 ] || { echo "ABORT_TEARDOWN_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c "cp /opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py.pre_v125_opsfix /opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py && cp $R.pre_v125_c1 $R && cp $CGP.pre_v125_diag $CGP && python3 -c \"import py_compile; py_compile.compile('/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py', doraise=True); py_compile.compile('$R', doraise=True); py_compile.compile('$CGP', doraise=True)\" && echo REVERT_CLEAN_OK"
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK2=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK2=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK2" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
SPL2=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
INSTR2=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|v125 root fix\|perf-v125 C1' /root/serve_full.log || echo 0")
echo "restored spec_lines=$SPL2 fix-marker-lines=$INSTR2 (want >=1 / 0)"

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C1 FIX2 DONE $(date +%F' '%T) ==="
echo "P22C1_FIX2_DONE"
} > "$L" 2>&1
tail -70 "$L"
echo P22C1_FIX2_SCRIPT_DONE
