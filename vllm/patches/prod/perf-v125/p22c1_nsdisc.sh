#!/bin/bash
# p22c1_nsdisc.sh — non-spec surface defect discriminator (C1 follow-up).
#
# Established: fp16 non-spec serve degenerates (v3: 'Unit!!!' lockup on
# all four no-think probes) while the SAME no-think instrument on the
# certified spec-4 serve is fully coherent, and the seq kernel's fp8/fp16
# recurrence is op-level BOUNDED (traj PASS). So the defect lives in the
# spec-stripped SERVE surface with real weights. This script splits the
# remaining hypothesis space with three paired instruments, cert serve
# (part 1, live) vs ctrl non-spec serve (part 2, after boot):
#   A. short-mt sweep with logprobs — localize the FIRST divergent step:
#      sane at mt<=8 and degenerate later -> long-gen recurrence issue;
#      wrong at mt=2-4 -> per-step defect (state write/z gating).
#   B. think-ON mt=600 dual capture — the cell v2 lost: if non-spec
#      reasoning is coherent and answers correctly, only the no-think
#      combination is broken (template/sampling interaction, not kernels).
#   C. identical MATH prompt logprob comparison cert vs ctrl at the
#      first positions (printed side by side for diffing).
# Watchdog paused only across the ctrl boot; restored + re-armed after.
set -u
L=/root/build/lce1/p22c1_nsdisc.log

probe_battery() {  # $1 = tag, $2 = base url
python3 - "$1" "$2" <<'PYEOF'
import json, sys, urllib.request
tag, base = sys.argv[1], sys.argv[2]
M = "qwen3.8-27b-fp8"
def ask(prompt, mt, label, think=False, lp=False):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if lp:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        ch = r["choices"][0]
        m = ch["message"]
        vis = m.get("content") or ""
        rsn = m.get("reasoning_content") or ""
        print(f"[{tag}] {label} content_len={len(vis)} reasoning_len={len(rsn)} finish={ch.get('finish_reason')}")
        print(f"[{tag}] {label} TEXT {((vis + ' ||| ' + rsn) if rsn else vis)[:300]!r}")
        if lp and ch.get("logprobs"):
            for i, lpstep in enumerate(ch["logprobs"].get("content", [])[:6]):
                top = ",".join(f"{e['token']!r}:{e['logprob']:.2f}"
                               for e in lpstep.get("top_logprobs", [])[:3])
                print(f"[{tag}] {label} pos{i} {lpstep['token']!r} top: {top}")
        return vis, rsn
    except Exception as e:
        print(f"[{tag}] {label}_FAIL {type(e).__name__}: {e}")
        return "", ""

# A. short-mt sweep, no-think, first tokens with logprobs
for mt in (2, 4, 8, 16, 32, 64):
    ask("What is 17*23? Answer with the number only.", mt, f"MT{mt}", lp=True)
# B. think-ON long gen, dual capture
ask("What is 17*23? Answer with the number only.", 600, "THINK600", think=True)
ask("Write one paragraph (3 sentences) explaining why the sky is blue.", 400, "THINKPROSE", think=True)
PYEOF
}

{
echo "=== P22C1 NSDISC start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
echo "container image=$IMG"
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

echo "--- 1. cert16s4 battery (live serve) ---"
curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health || { echo "ABORT_NO_HEALTH"; exit 1; }
SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
echo "cert spec_lines=$SPL"
probe_battery cert16s4 http://127.0.0.1:8000

echo "--- 2. boot ctrl non-spec serve ---"
touch /root/build/lane_watchdog.paused
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; echo "port 8000 down after ${i}x5s"; break; fi
done
[ "$PD" -eq 1 ] || { echo "ABORT_SERVE_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c ": > /root/serve_p22c1_ctrl.log; cd /root && nohup bash serve_p22c1_ctrl.sh > /dev/null 2>&1 & echo ctrl-launched"
HK=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "ctrl health OK after ${i}0s"; break; }
done
[ "$HK" -eq 1 ] || { echo "ABORT_CTRL_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' /root/serve_p22c1_ctrl.log | tail -5"; exit 1; }
sleep 15
SPL2=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_p22c1_ctrl.log || echo 0")
echo "ctrl spec_lines=$SPL2 (want 0)"

echo "--- 3. ctrl16ns battery ---"
probe_battery ctrl16ns http://127.0.0.1:8000

echo "--- 4. restore certified spec-4 serve ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD2=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD2=1; echo "port 8000 down after ${i}x5s"; break; fi
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
SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
echo "restored spec_lines=$SPL (want >=1)"
[ "$SPL" -ge 1 ] || { echo "ABORT_RESTORE_NOT_SPEC4"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C1 NSDISC DONE $(date +%F' '%T) ==="
echo "P22C1_NSDISC_DONE"
} > "$L" 2>&1
tail -100 "$L"
echo P22C1_NSDISC_SCRIPT_DONE
