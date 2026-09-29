#!/bin/bash
# p22c1_eager.sh — non-spec surface defect: graph-replay discriminator.
#
# Evidence chain so far: ctrl16ns (fp16, spec-stripped) emits one real
# token then uniform logits (logprob -12.42 ~= ln(1/250k) on EVERY top
# candidate) from decode step 2 onward = zero/constant residual at the
# LM head; cert16s4 coherent; traj op-level BOUNDED. Certified spec-4
# NEVER exercises the non-spec decode path (mixed batches reclassify
# non-spec decodes as prefills, gdn_attn.py ~266) — run 6 was its first
# exercise ever. Full_cudagraph metadata plumbing exists on both paths,
# so static reading can't separate "graph replay output wiring broken"
# from "eager non-spec forward broken". This leg boots the ctrl config
# with --enforce-eager (compilation-config stripped):
#   coherent  -> mechanism = cudagraph replay of dense non-spec decode
#                (bs 1..N graphs from the v89 85-size set);
#   degenerate-> mechanism = the eager non-spec forward itself.
# Battery: MT sweep with logprobs + the v3 four no-think probes + one
# think-on dual capture. Restore + re-arm after.
set -u
L=/root/build/lce1/p22c1_eager.log
touch /root/build/lane_watchdog.paused
{
echo "=== P22C1 EAGER start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
echo "container image=$IMG"
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }

echo "--- 1. build eager serve script ---"
docker exec lsv-test sh -c "sed '/--speculative-config/d; /--compilation-config/d; s|serve_p22c1_ctrl.log|serve_p22c1_eager.log|; s|serve_p22c1_ctrl.sh|serve_p22c1_eager.sh|' /root/serve_p22c1_ctrl.sh > /root/serve_p22c1_eager.sh; sed -i 's|exec vllm serve|exec vllm serve --enforce-eager|' /root/serve_p22c1_eager.sh; chmod +x /root/serve_p22c1_eager.sh; grep -c 'enforce-eager' /root/serve_p22c1_eager.sh; grep -c 'compilation-config' /root/serve_p22c1_eager.sh || true"

echo "--- 2. tear down certified serve ---"
docker exec lsv-test sh -c "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'from vllm' 2>/dev/null; true"
PD=0
for i in $(seq 1 18); do
  sleep 5
  if ! curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health; then PD=1; echo "port 8000 down after ${i}x5s"; break; fi
done
[ "$PD" -eq 1 ] || { echo "ABORT_SERVE_STILL_UP"; exit 1; }
sleep 4
docker exec lsv-test sh -c ": > /root/serve_p22c1_eager.log; cd /root && nohup bash serve_p22c1_eager.sh > /dev/null 2>&1 & echo eager-launched"

echo "--- 3. health wait (<=480s) ---"
HK=0
for i in $(seq 1 48); do
  sleep 10
  curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && { HK=1; echo "eager health OK after ${i}0s"; break; }
done
[ "$HK" -eq 1 ] || { echo "ABORT_HEALTH"; docker exec lsv-test sh -c "grep -iE 'error|traceback' /root/serve_p22c1_eager.log | tail -5"; exit 1; }
sleep 15
SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_p22c1_eager.log || echo 0")
echo "eager spec_lines=$SPL (want 0)"
CG=$(docker exec lsv-test sh -c "grep -ic 'cudagraph' /root/serve_p22c1_eager.log || echo 0")
echo "eager cudagraph mentions=$CG"

echo "--- 4. battery ---"
python3 - <<'PYEOF'
import json, urllib.request
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
        print(f"[eager16ns] {label} content_len={len(vis)} reasoning_len={len(rsn)} finish={ch.get('finish_reason')}")
        print(f"[eager16ns] {label} TEXT {((vis + ' ||| ' + rsn) if rsn else vis)[:300]!r}")
        if lp and ch.get("logprobs"):
            for i, s in enumerate(ch["logprobs"].get("content", [])[:6]):
                top = ",".join(f"{e['token']!r}:{e['logprob']:.2f}"
                               for e in s.get("top_logprobs", [])[:3])
                print(f"[eager16ns] {label} pos{i} {s['token']!r} top: {top}")
        return vis
    except Exception as e:
        print(f"[eager16ns] {label}_FAIL {type(e).__name__}: {e}")
        return ""
for mt in (8, 64):
    ask("What is 17*23? Answer with the number only.", mt, f"MT{mt}", lp=True)
a = ask("What is 17*23? Answer with the number only.", 600, "MATH600")
b = ask("List the numbers from 1 to 5, comma separated, nothing else.", 300, "LIST300")
c = ask("Write one paragraph (3 sentences) explaining why the sky is blue.", 250, "PROSE250")
d = ask("Write a short story (5 sentences) about a robot learning to garden.", 400, "STORY400")
print(f"[eager16ns] VERDICT math391={'391' in a} list5={'5' in b} prose_len={len(c)} story_len={len(d)}")
ask("What is 17*23? Answer with the number only.", 600, "THINK600", think=True)
PYEOF

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
SPL=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log || echo 0")
echo "restored spec_lines=$SPL (want >=1)"
[ "$SPL" -ge 1 ] || { echo "ABORT_RESTORE_NOT_SPEC4"; exit 1; }

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "=== P22C1 EAGER DONE $(date +%F' '%T) ==="
echo "P22C1_EAGER_DONE"
} > "$L" 2>&1
tail -70 "$L"
echo P22C1_EAGER_SCRIPT_DONE
