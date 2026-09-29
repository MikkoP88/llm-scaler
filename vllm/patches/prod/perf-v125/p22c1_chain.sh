#!/bin/bash
# p22c1_chain.sh — C1 verdict completion chain, launched while v3
# (p22c1_ctrl_probe.sh) is still restoring the certified serve.
#
# Step A: wait for v3's terminal marker (it re-arms the watchdog at the
#   very end; we re-pause immediately — the armed window is seconds on
#   a healthy serve, the normal lane state).
# Step B: verify the restored serve IS the certified spec-4 posture
#   (health + num_speculative_tokens lines in the fresh serve_full.log),
#   then run the SAME v3 no-think probes against it. This is the last
#   H2 discriminator cell: certified spec-4 + no-think coherent =>
#   degeneration is specific to the spec-stripped serving surface (or
#   its ESIMD non-spec kernel path), NOT fp8 and NOT the template.
# Step C: stage p22c1_traj.py into the container and run the op-level
#   trajectory differential with the serve DOWN (GPU headroom), then
#   restore the certified serve and re-arm (same patterns as v3).
set -u
L=/root/build/lce1/p22c1_chain.log

# ---- Step A: wait for v3 terminal marker (<=45 min)
i=0
while [ $i -lt 2700 ]; do
  grep -q "P22C1_CTRL_PROBE_DONE" /root/build/lce1/p22c1_ctrl_probe.log 2>/dev/null && break
  sleep 15; i=$((i+15))
done
grep -q "P22C1_CTRL_PROBE_DONE" /root/build/lce1/p22c1_ctrl_probe.log || { echo "ABORT_V3_NEVER_FINISHED" >> "$L"; exit 1; }

touch /root/build/lane_watchdog.paused
{
echo "=== P22C1 CHAIN start $(date +%F' '%T) ==="

echo "--- B1. certified serve posture ---"
HK=0
for i in $(seq 1 60); do
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; break; }
  sleep 10
done
[ "$HK" -eq 1 ] || { echo "ABORT_NO_HEALTH"; exit 1; }
echo "health OK"
SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
echo "cert spec_lines=$SPL (want >=1)"
[ "$SPL" -ge 1 ] || { echo "ABORT_NOT_SPEC4"; exit 1; }

echo "--- B2. no-think probes on certified spec-4 (the H2 last cell) ---"
python3 - <<'PYEOF'
import json, urllib.request
M = "qwen3.8-27b-fp8"
def ask(prompt, mt, label):
    body = json.dumps({"model": M, "max_tokens": mt, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        m = r["choices"][0]["message"]
        vis = m.get("content") or ""
        rsn = m.get("reasoning_content") or ""
        fin = r.get("choices")[0].get("finish_reason")
        print(f"[cert16s4] {label} content_len={len(vis)} reasoning_len={len(rsn)} finish={fin}")
        print(f"[cert16s4] {label} CONTENT {vis[:700]!r}")
        if rsn:
            print(f"[cert16s4] {label} REASONING-TAIL {rsn[-200:]!r}")
        return vis
    except Exception as e:
        print(f"[cert16s4] {label}_FAIL {type(e).__name__}: {e}")
        return ""
a = ask("What is 17*23? Answer with the number only.", 600, "MATH600")
b = ask("List the numbers from 1 to 5, comma separated, nothing else.", 300, "LIST300")
c = ask("Write one paragraph (3 sentences) explaining why the sky is blue.", 250, "PROSE250")
d = ask("Write a short story (5 sentences) about a robot learning to garden.", 400, "STORY400")
print(f"[cert16s4] VERDICT math391={'391' in a} list5={'5' in b} prose_len={len(c)} story_len={len(d)}")
PYEOF

echo "--- C1. stage traj into container ---"
docker cp /root/build/p22c1_traj.py lsv-test:/root/p22c1_traj.py || { echo "ABORT_DOCKER_CP"; exit 1; }

echo "--- C2. tear down serve (plain pkills + port-down proof) ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; echo "port 8000 down after ${i}x5s"; break; fi
done
[ "$PD" -eq 1 ] || { echo "ABORT_SERVE_STILL_UP"; exit 1; }
sleep 4

echo "--- C3. run trajectory differential (GPU free) ---"
docker exec lsv-test python3 /root/p22c1_traj.py
RC=$?
echo "traj rc=$RC"

echo "--- C4. restore certified spec-4 serve ---"
docker exec lsv-test sh -c ": > /root/serve_full.log; cd /root && nohup bash serve_user.sh > /dev/null 2>&1 & echo restore-launched"
HK=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "restored health OK after ${i}0s"; break; }
done
[ "$HK" -eq 1 ] || { echo "ABORT_RESTORE_HEALTH"; exit 1; }
SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
echo "restored spec_lines=$SPL (want >=1)"
[ "$SPL" -ge 1 ] || { echo "ABORT_RESTORE_NOT_SPEC4"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C1 CHAIN DONE $(date +%F' '%T) traj_rc=$RC ==="
echo "P22C1_CHAIN_DONE"
} > "$L" 2>&1
tail -80 "$L"
echo P22C1_CHAIN_SCRIPT_DONE
