#!/bin/bash
# p22c1_d3.sh — Run 7h discriminator chain (PHASES.md).
#
# Established (Run 7g + H12 + log archaeology 2026-09-29):
#  - Any non-enforce-eager ns boot degenerates (NaN decode hidden rows ->
#    v60g) with OR without graphs; enforce-eager ns is coherent.
#  - Engine-side there is a TWO-PASS resolution: pass 1 constructs
#    VLLM_COMPILE + custom_ops 'none' + rms_norm native + a torch.compile
#    window; the xpu.py:349 platform gate (TP>1, v31.1 #11 + v124 P16)
#    sets TORCH_COMPILE_DISABLE LATE, flipping mode to NONE but leaving
#    custom_ops/kernel priorities STALE ('none'/native). enforce_eager
#    resolves everything at pass 1 (mode NONE + ops 'all' + xpu_kernels).
#  - The certified spec-4 lane runs the SAME pass-1 'none'/native posture
#    and is coherent -> the poison is the conjunction (pass-1 compiled
#    posture) x (spec-off), not the posture alone.
#
# Three ns legs, all cudagraph_mode NONE (graphs excluded by H12), D3
# NaN-slice instrument in the venv:
#   A nograph : {"cudagraph_mode":"NONE"}                  (known-bad baseline; D3 NaN nature)
#   B modenone: {"mode":"NONE","cudagraph_mode":"NONE"}    (eager op-posture at pass 1, non-eager runner)
#   C opsall  : {"cudagraph_mode":"NONE","custom_ops":["all"]} (compile window, ops substituted)
# Verdicts:
#   B coherent -> poison inside pass-1 compiled posture; C then splits it:
#     C coherent  -> custom_ops 'none' (native ops NaN on ns decode); fix = ops 'all'
#     C degenerate-> the dynamo/compile window; fix = mode NONE at pass 1
#   B degenerate -> poison is enforce_eager-gated runner machinery (not
#     compilation config at all); next probe = D3 NaN pattern.
# Lane hygiene: the currently running certified serve was booted with the
# C1+diag venv still applied (v125DIAG lines in serve_full.log 22:24);
# this chain restores the CLEAN venv before D3 and re-boots the certified
# serve on clean code at the end. EXIT trap restores the lane on abort.
set -u
L=/root/build/lce1/p22c1_d3.log
R=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
CG=/opt/venv/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py

restore_lane() {
  # idempotent: only acts while this chain owns the lane
  [ -f /root/build/lane_watchdog.paused ] || return 0
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true" >/dev/null 2>&1
  for i in $(seq 1 18); do
    sleep 5
    curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health || break
  done
  sleep 4
  docker exec lsv-test sh -c "cp $R.pre_v125_c1 $R 2>/dev/null; cp $CG.pre_v125_diag $CG 2>/dev/null; python3 -c \"import py_compile; py_compile.compile('$R', doraise=True); py_compile.compile('$CG', doraise=True)\" && echo CLEAN_VENV_RESTOREED" >> "$L" 2>&1
  docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched" >> "$L" 2>&1
  for i in $(seq 1 48); do
    sleep 10
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && break
  done
  SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
  INSTR=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|perf-v125 C1' /root/serve_full.log || echo 0")
  echo "trap-restore spec_lines=$SPL instr_lines=$INSTR (want >=1 / 0)" >> "$L" 2>&1
  rm -f /root/build/lane_watchdog.paused
  echo "trap lane_watchdog re-armed ($(date +%T))" >> "$L" 2>&1
}
trap restore_lane EXIT

probes() {
  python3 - "$1" <<'PYEOF'
import json, sys, urllib.request
TAG = sys.argv[1]
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
        print(f"[{TAG}] {label} TEXT {vis[:120]!r}")
    except Exception as e:
        print(f"[{TAG}] {label}_FAIL {type(e).__name__}: {e}")
ask("What is 17*23? Answer with the number only.", 12, "MT12")
ask("List the numbers from 1 to 5, comma separated, nothing else.", 40, "LIST40")
PYEOF
}

collect() {  # $1 = leg log path
  CAPN=$(docker exec lsv-test sh -c "grep -c 'Capturing' $1 || echo 0")
  echo "capture lines=$CAPN (want 0)"
  echo "--- pass-1 config posture:"
  docker exec lsv-test sh -c "grep -m1 'Initializing a V1 LLM engine' $1 | grep -oE \"enforce_eager=[A-Za-z]+|'mode': <CompilationMode.[A-Z_]+: [0-9]+>|custom_ops': \\[[^]]*\\]|rms_norm=\\[[^]]*\\]\""
  echo "--- init engine line (compile window?):"
  docker exec lsv-test sh -c "grep -m1 'init engine' $1"
  echo "--- v60g count:"
  docker exec lsv-test sh -c "grep -c 'v60g NO-DRAFT' $1 || echo 0"
  echo "--- D3 NaN slice (first 26):"
  docker exec lsv-test sh -c "grep 'v125D3' $1 | head -26"
}

{
echo "=== P22C1 D3 start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

echo "--- 1. restore clean venv (also fixes instrumented certified lane), apply D3 ---"
docker exec lsv-test sh -c "cp $R.pre_v125_c1 $R 2>/dev/null; cp $CG.pre_v125_diag $CG 2>/dev/null; python3 -c \"import py_compile; py_compile.compile('$R', doraise=True); py_compile.compile('$CG', doraise=True)\" && echo CLEAN_VENV_OK" || { echo "ABORT_CLEAN_VENV"; exit 1; }
MK=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|perf-v125 C1' $R $CG || echo 0")
echo "marker lines after clean restore (want 0): $MK"
docker cp /root/build/patch_v125_d3.py lsv-test:/root/ || { echo "ABORT_CP_D3"; exit 1; }
P=$(docker exec lsv-test python3 /root/patch_v125_d3.py)
echo "$P"
echo "$P" | grep -q 'V125_D3_OK\|V125_D3_ALREADY' || { echo "ABORT_D3_PATCH"; exit 1; }

echo "--- 2. build three leg scripts from ctrl (single-variable changes) ---"
B="docker exec lsv-test sh -c"
$B "sed \"s|{\\\"cudagraph_mode\\\":\\\"FULL_DECODE_ONLY\\\",\\\"cudagraph_capture_sizes\\\":[^]]*]}|{\\\"cudagraph_mode\\\":\\\"NONE\\\"}|; s|serve_p22c1_ctrl|serve_p22c1_nograph|g\" /root/serve_p22c1_ctrl.sh > /root/serve_p22c1_nograph.sh" || { echo "ABORT_BUILD_A"; exit 1; }
$B "sed \"s|{\\\"cudagraph_mode\\\":\\\"FULL_DECODE_ONLY\\\",\\\"cudagraph_capture_sizes\\\":[^]]*]}|{\\\"mode\\\":\\\"NONE\\\",\\\"cudagraph_mode\\\":\\\"NONE\\\"}|; s|serve_p22c1_ctrl|serve_p22c1_modenone|g\" /root/serve_p22c1_ctrl.sh > /root/serve_p22c1_modenone.sh" || { echo "ABORT_BUILD_B"; exit 1; }
$B "sed \"s|{\\\"cudagraph_mode\\\":\\\"FULL_DECODE_ONLY\\\",\\\"cudagraph_capture_sizes\\\":[^]]*]}|{\\\"cudagraph_mode\\\":\\\"NONE\\\",\\\"custom_ops\\\":[\\\"all\\\"]}|; s|serve_p22c1_ctrl|serve_p22c1_opsall|g\" /root/serve_p22c1_ctrl.sh > /root/serve_p22c1_opsall.sh" || { echo "ABORT_BUILD_C"; exit 1; }
docker exec lsv-test sh -c "chmod +x /root/serve_p22c1_nograph.sh /root/serve_p22c1_modenone.sh /root/serve_p22c1_opsall.sh; grep -h compilation-config /root/serve_p22c1_nograph.sh /root/serve_p22c1_modenone.sh /root/serve_p22c1_opsall.sh"

touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused ($(date +%T))"

run_leg() {  # $1 leg name
  echo "=== LEG $1 boot $(date +%T) ==="
  docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
  PD=0
  for i in $(seq 1 18); do
    sleep 5
    if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; break; fi
  done
  [ "$PD" -eq 1 ] || { echo "ABORT_${1}_PORT_STILL_UP"; exit 1; }
  sleep 4
  docker exec lsv-test sh -c ": > /root/serve_p22c1_$1.log; cd /root && nohup bash serve_p22c1_$1.sh > /dev/null 2>&1 & echo $1-launched"
  HK=0
  for i in $(seq 1 48); do
    sleep 10
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "$1 health OK after ${i}0s"; break; }
  done
  [ "$HK" -eq 1 ] || { echo "ABORT_${1}_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' /root/serve_p22c1_$1.log | tail -5"; exit 1; }
  sleep 15
  probes "$1"
  collect "/root/serve_p22c1_$1.log"
}

run_leg nograph
run_leg modenone
run_leg opsall

echo "--- restore certified serve on CLEAN venv ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD2=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD2=1; break; fi
done
[ "$PD2" -eq 1 ] || { echo "ABORT_TEARDOWN_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c "cp $R.pre_v125_c1 $R 2>/dev/null; cp $CG.pre_v125_diag $CG 2>/dev/null; python3 -c \"import py_compile; py_compile.compile('$R', doraise=True); py_compile.compile('$CG', doraise=True)\" && echo CLEAN_VENV_FOR_RESTORE"
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK2=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK2=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK2" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
SPL2=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
INSTR2=$(docker exec lsv-test sh -c "grep -c 'v125DIAG\|v125D3\|perf-v125 C1' /root/serve_full.log || echo 0")
echo "restored spec_lines=$SPL2 instr_lines=$INSTR2 (want >=1 / 0)"

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C1 D3 DONE $(date +%F' '%T) ==="
echo "P22C1_D3_DONE"
} > "$L" 2>&1
tail -80 "$L"
echo P22C1_D3_SCRIPT_DONE
