#!/bin/bash
# p22c1_ctrl_probe.sh — C1 step 1: the H2 discriminator.
#
# Run 6 left an attribution gap: e4m3ns (ESIMD fp8) showed repetition
# collapse on PROSE250, but the fp16 non-spec CONTROL leg was never
# probed with corrected max_tokens (the runner's mt=40 probes are
# content-empty artifacts; the probe600 watcher only targeted fp8
# tags). Until the fp16 control is probed, we cannot separate:
#   H1: fp8 quantization noise through the state recurrence collapses
#       long generations (fp8-specific defect);
#   H2: the non-spec ESIMD path (or spec-stripped serving) degenerates
#       for long greedy generations REGARDLESS of pool dtype — in which
#   case run 6's e4m3 "failure" was not an fp8 signal at all.
# This script boots the exact ctrl-ns serve config (fp16 pool, spec
# stripped) on the CURRENT container, runs corrected probes, restores
# the certified spec-4 serve. Watchdog paused throughout.
# .so posture VERIFIED pre-launch: the rebuilt lgrf .so (sha 7c5fdcd…,
# 96.8 MB) is IN PLACE and the stock backup sits as .v125bak (1d9dcf4e…,
# 16.7 MB) — run 6 did NOT restore stock. For a fp16 pool stock and
# rebuild execute identical fp16 code paths, so ctrl fidelity holds.
set -u
L=/root/build/lce1/p22c1_ctrl_probe.log
touch /root/build/lane_watchdog.paused
{
echo "=== P22C1 CTRL PROBE start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
echo "container image=$IMG"
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

echo "--- 1. tear down certified serve (v2: ERE-safe kill + port-down proof) ---"
# v1 bug: pkill -f 'vllm serve\|from vllm' — pkill takes ERE, where \| is a
# LITERAL pipe -> matched nothing -> certified serve never died -> the fresh
# serve died at port-bind -> all probes measured the WRONG serve (the
# certified spec-4). v2 kills with plain patterns and PROVES :8000 is down
# before booting; the health wait then measures the new serve only.
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; echo "port 8000 down after ${i}x5s"; break; fi
done
[ "$PD" -eq 1 ] || { echo "ABORT_SERVE_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c "sed '/--speculative-config/d; s|serve_full.log|serve_p22c1_ctrl.log|' /root/serve_user.sh > /root/serve_p22c1_ctrl.sh && chmod +x /root/serve_p22c1_ctrl.sh; grep -c 'speculative-config' /root/serve_p22c1_ctrl.sh; grep -o 'mamba-ssm-cache-dtype [a-z0-9_]*' /root/serve_p22c1_ctrl.sh"
docker exec lsv-test sh -c ": > /root/serve_p22c1_ctrl.log"
docker exec -d lsv-test sh -c "cd /root && bash serve_p22c1_ctrl.sh"

echo "--- 2. health wait (<=480s) ---"
HK=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "health OK after ${i}0s"; break; }
done
[ "$HK" -eq 1 ] || { echo "ABORT_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' /root/serve_p22c1_ctrl.log | tail -5"; exit 1; }
sleep 15

echo "--- 3. posture marks ---"
docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_p22c1_ctrl.log || echo 0" | sed 's/^/spec_lines=/'
docker exec lsv-test sh -c "grep -o 'mamba_ssm_cache_dtype.: .[a-z0-9_]*.' /root/serve_p22c1_ctrl.log | head -1"
docker exec lsv-test sh -c "grep -c 'ESIMD_FP8_ELIGIBLE' /root/serve_p22c1_ctrl.log || echo 0" | sed 's/^=/ (want 0)/'

echo "--- 4. corrected numerics probes v3 (no-think + dual capture) ---"
# v2 lesson: content-only reads returned empty on the non-spec fp16 serve
# (qwen3 kept everything in reasoning_content at mt=600), while earlier
# probes concatenated the two — measurements weren't comparable. v3 forces
# enable_thinking=false (chat_template_kwargs) so the answer lands in
# content, AND prints both fields separately so nothing is ever hidden.
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
        print(f"[ctrl16ns] {label} content_len={len(vis)} reasoning_len={len(rsn)} finish={fin}")
        print(f"[ctrl16ns] {label} CONTENT {vis[:700]!r}")
        if rsn:
            print(f"[ctrl16ns] {label} REASONING-TAIL {rsn[-200:]!r}")
        return vis
    except Exception as e:
        print(f"[ctrl16ns] {label}_FAIL {type(e).__name__}: {e}")
        return ""
a = ask("What is 17*23? Answer with the number only.", 600, "MATH600")
b = ask("List the numbers from 1 to 5, comma separated, nothing else.", 300, "LIST300")
c = ask("Write one paragraph (3 sentences) explaining why the sky is blue.", 250, "PROSE250")
d = ask("Write a short story (5 sentences) about a robot learning to garden.", 400, "STORY400")
print(f"[ctrl16ns] VERDICT math391={'391' in a} list5={'5' in b} prose_len={len(c)} story_len={len(d)}")
PYEOF

echo "--- 5. restore certified spec-4 serve (v2: kill proof + fresh log) ---"
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
echo "=== P22C1 CTRL PROBE DONE $(date +%F' '%T) ==="
echo "P22C1_CTRL_PROBE_DONE"
} > "$L" 2>&1
tail -40 "$L"
echo P22C1_CTRL_PROBE_SCRIPT_DONE
