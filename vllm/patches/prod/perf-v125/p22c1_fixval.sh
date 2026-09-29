#!/bin/bash
# p22c1_fixval.sh — validate the v125 C1 root fix (no-draft batches never
# dispatch through the FULL nospec graph; eager fallback instead).
#
# Verdict matrix going in (PHASES.md Run 7d):
#   ctrl16ns  (graphs ON,  async, non-spec) = degenerate from pos1 + v60g
#   eager16ns (graphs OFF, async, non-spec) = fully coherent, v60g-0
# This leg reboots the EXACT ctrl config (graphs ON) with the dispatch
# patch applied and must now look like eager16ns: coherent output AND
# zero v60g firings. Then restores the certified spec-4 serve and proves
# the spec path untouched (MT8 logprobs coherent, spec_lines >= 1).
set -u
L=/root/build/lce1/p22c1_fixval.log

battery() {  # $1 = tag
python3 - "$1" <<'PYEOF'
import json, sys, urllib.request
tag = sys.argv[1]
M = "qwen3.8-27b-fp8"
def ask(prompt, mt, label, think=False, lp=False):
    body = {"model": M, "max_tokens": mt, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if lp:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
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
            for i, s in enumerate(ch["logprobs"].get("content", [])[:6]):
                top = ",".join(f"{e['token']!r}:{e['logprob']:.2f}"
                               for e in s.get("top_logprobs", [])[:3])
                print(f"[{tag}] {label} pos{i} {s['token']!r} top: {top}")
        return vis
    except Exception as e:
        print(f"[{tag}] {label}_FAIL {type(e).__name__}: {e}")
        return ""
for mt in (8, 64):
    ask("What is 17*23? Answer with the number only.", mt, f"MT{mt}", lp=True)
a = ask("What is 17*23? Answer with the number only.", 600, "MATH600")
b = ask("List the numbers from 1 to 5, comma separated, nothing else.", 300, "LIST300")
c = ask("Write one paragraph (3 sentences) explaining why the sky is blue.", 250, "PROSE250")
d = ask("Write a short story (5 sentences) about a robot learning to garden.", 400, "STORY400")
print(f"[{tag}] VERDICT math391={'391' in a} list5={'5' in b} prose_len={len(c)} story_len={len(d)}")
ask("What is 17*23? Answer with the number only.", 600, "THINK600", think=True)
PYEOF
}

{
echo "=== P22C1 FIXVAL start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
echo "container image=$IMG"
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

echo "--- 1. apply v125 C1 root-fix patch ---"
docker cp /root/build/patch_v125_nospec_no_full_graph.py lsv-test:/root/ || { echo "ABORT_CP"; exit 1; }
P=$(docker exec lsv-test python3 /root/patch_v125_nospec_no_full_graph.py)
echo "$P"
echo "$P" | grep -q 'V125_C1_PATCH_OK\|V125_C1_ALREADY_PATCHED' || { echo "ABORT_PATCH"; exit 1; }

echo "--- 2. boot non-spec serve (graphs ON, ctrl config) ---"
touch /root/build/lane_watchdog.paused
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; echo "port 8000 down after ${i}x5s"; break; fi
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
SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_p22c1_ctrl.log || echo 0")
echo "ns spec_lines=$SPL (want 0)"
CG=$(docker exec lsv-test sh -c "grep -c 'Capturing CUDA graphs' /root/serve_p22c1_ctrl.log || echo 0")
echo "ns boot graph-capture lines=$CG (graphs still captured; dispatch excludes them)"

echo "--- 3. non-spec battery (must be coherent, like eager16ns) ---"
battery fix16ns

echo "--- 4. v60g tripwire (must be 0) ---"
V6=$(docker exec lsv-test sh -c "grep -c 'v60g NO-DRAFT' /root/serve_p22c1_ctrl.log || echo 0")
echo "ns v60g count=$V6 (want 0)"
[ "$V6" -eq 0 ] || { echo "ABORT_V60G_FIRED output still degenerate"; docker exec lsv-test sh -c "grep 'v60g NO-DRAFT' /root/serve_p22c1_ctrl.log | head -3"; }

echo "--- 5. restore certified spec-4 serve ---"
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
SPL2=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
echo "restored spec_lines=$SPL2 (want >=1)"
[ "$SPL2" -ge 1 ] || { echo "ABORT_RESTORE_NOT_SPEC4"; exit 1; }

echo "--- 6. certified spec-4 quick battery (must stay coherent) ---"
battery cert16s4b

echo "--- 7. cert v60g (must be 0) ---"
V6B=$(docker exec lsv-test sh -c "grep -c 'v60g NO-DRAFT' /root/serve_full.log || echo 0")
echo "cert v60g count=$V6B"

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C1 FIXVAL DONE $(date +%F' '%T) ==="
echo "P22C1_FIXVAL_DONE"
} > "$L" 2>&1
tail -90 "$L"
echo P22C1_FIXVAL_SCRIPT_DONE
