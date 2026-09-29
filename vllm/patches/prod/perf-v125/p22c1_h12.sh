#!/bin/bash
# p22c1_h12.sh — H1/H2 discriminator for the C1 mystery (PHASES.md Run 7g).
#
# Established by diag round-3: on a FULL_DECODE_ONLY graphs boot, ns decode
# is degenerate BOTH via FULL replay (run 6/7) AND via C1 mode=NONE
# runnable-eager (round-3: zero live D2 wrapper entries with budget
# remaining => early-return to runnable ran; output still '3!!!', v60g x20,
# hidden states NaN). enforce-eager boot (eager16ns) is fully coherent and
# the two serve scripts differ ONLY by --enforce-eager + the
# cudagraph_mode=FULL_DECODE_ONLY compilation-config (diff verified).
#
# This boot: identical to ctrl16ns except cudagraph_mode=NONE — graphs are
# never captured, no capture dummy runs, but the boot is NOT enforce-eager
# (VLLM_COMPILE wrapper present, persistent buffers, profile run, async).
#   coherent   -> H1: the graphs boot (capture/persistent buffers) is what
#                 poisons ns decode globally.
#   degenerate -> H2: any non-enforce-eager boot poisons ns decode; root fix
#                 must equal enforce-eager semantics for spec-off serves.
# C1+diag remain applied in the venv: with config mode NONE the dispatcher
# can never return FULL, C1 is inert, and D1 (fresh process => fresh
# counters) will now show LIVE nospec=True dispatch lines (boot uses only
# ~2 dispatches, budget 40 survives).
set -u
L=/root/build/lce1/p22c1_h12.log

{
echo "=== P22C1 H12 start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

echo "--- 1. build nograph serve script from ctrl (single-variable change) ---"
docker exec lsv-test sh -c "sed \"s|{\\\"cudagraph_mode\\\":\\\"FULL_DECODE_ONLY\\\",\\\"cudagraph_capture_sizes\\\":[^]]*]}|{\\\"cudagraph_mode\\\":\\\"NONE\\\"}|; s|serve_p22c1_ctrl|serve_p22c1_nograph|g\" /root/serve_p22c1_ctrl.sh > /root/serve_p22c1_nograph.sh && chmod +x /root/serve_p22c1_nograph.sh && grep -E 'compilation-config|enforce' /root/serve_p22c1_nograph.sh" || { echo "ABORT_BUILD_SCRIPT"; exit 1; }

echo "--- 2. boot nograph serve ---"
touch /root/build/lane_watchdog.paused
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; echo "port down after ${i}x5s"; break; fi
done
[ "$PD" -eq 1 ] || { echo "ABORT_SERVE_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c ": > /root/serve_p22c1_nograph.log; cd /root && nohup bash serve_p22c1_nograph.sh > /dev/null 2>&1 & echo nograph-launched"
HK=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "nograph health OK after ${i}0s"; break; }
done
[ "$HK" -eq 1 ] || { echo "ABORT_NOGRAPH_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' /root/serve_p22c1_nograph.log | tail -5"; exit 1; }
sleep 15
CG=$(docker exec lsv-test sh -c "grep -c 'Capturing' /root/serve_p22c1_nograph.log || echo 0")
echo "capture lines=$CG (want 0 — graphs must NOT be captured)"
CM=$(docker exec lsv-test sh -c "grep -oE 'cudagraph_mode.: ?<CUDAGraphMode.[A-Z_]+' /root/serve_p22c1_nograph.log | head -2")
echo "resolved modes: $CM"

echo "--- 3. probes ---"
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
        print(f"[h12] {label} TEXT {vis[:120]!r}")
    except Exception as e:
        print(f"[h12] {label}_FAIL {type(e).__name__}: {e}")
ask("What is 17*23? Answer with the number only.", 12, "MT12")
ask("List the numbers from 1 to 5, comma separated, nothing else.", 40, "LIST40")
PYEOF

echo "--- 4. verdict inputs ---"
V6=$(docker exec lsv-test sh -c "grep -c 'v60g NO-DRAFT' /root/serve_p22c1_nograph.log || echo 0")
echo "nograph v60g count=$V6"
echo "--- live D1 dispatch lines (nospec=True expected):"
docker exec lsv-test sh -c "grep 'nospec=True' /root/serve_p22c1_nograph.log | head -6"
echo "--- D2 wrapper lines total (expect 0 — wrapper bypassed):"
docker exec lsv-test sh -c "grep -c 'v125DIAG wrapper' /root/serve_p22c1_nograph.log || echo 0"

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
echo "=== P22C1 H12 DONE $(date +%F' '%T) ==="
echo "P22C1_H12_DONE"
} > "$L" 2>&1
tail -60 "$L"
echo P22C1_H12_SCRIPT_DONE
