#!/bin/bash
# p22a_probe600_v125.sh — corrected numerics probes for the P22A fp8 legs.
#
# WHY: p22a_run_v125.sh's numerics() reads only message.content with
# max_tokens 40/40/120 — qwen3 burns that budget in reasoning_content,
# so every leg prints len=0 (known quirk; repair2 proved mt=600 +
# reasoning concat returns 391 on a healthy serve). The runner is a
# live bash process — editing it mid-run is the watchdog stale-parse
# trap — so we fire corrected probes from OUTSIDE, synchronized to the
# runner's own numerics marker for each fp8 leg ("-- <tag> numerics").
# The probe load (~60-90 s, single stream) overlaps only the leg's
# first genspeed round; r1/r2 stay clean.
#
# v2 FIXES (run-6 postmortem): v1's loop `while [ $i -lt 160 ]` with
# `i=$((i+15))` gave ~160 s total, not the intended 40 min — it died
# at 20:32:37 and MISSED the e5m2ns window (serve torn down ~20:34).
# Now i counts SECONDS against a 2400 s budget. Bookkeeping switched
# from declare -A (whose expansion printed empty at exit) to plain
# FIRED_<TAG> vars.
set -u
LOG=/root/build/lce1/p22a_v125.log
OUT=/root/build/lce1
FIRED_E4M3NS=0
FIRED_E5M2NS=0
echo "=== probe600 watcher v2 start $(date +%F' '%T) ==="
i=0
while [ $i -lt 2400 ]; do  # 40 min, i counts seconds
  if [ "$FIRED_E4M3NS" -eq 0 ] && grep -q "^-- e4m3ns numerics" "$LOG" 2>/dev/null; then
    FIRED_E4M3NS=1
    echo "--- probe600 e4m3ns firing $(date +%T) ---"
    if curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health; then
      python3 - e4m3ns <<'PYEOF' 2>&1 | tee -a "$OUT/p22a_probe600_e4m3ns.txt"
import json, sys, time, urllib.request
tag = sys.argv[1]
M = "qwen3.8-27b-fp8"
def ask(prompt, mt, label):
    body = json.dumps({"model": M, "max_tokens": mt, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
        m = r["choices"][0]["message"]
        vis = m.get("content") or ""
        print(f"[{tag}] {label} http-ok visible_len={len(vis)} contains391={'391' in vis}")
        print(f"[{tag}] {label} FULLTEXT {vis[:600]!r}")
        return vis
    except Exception as e:
        print(f"[{tag}] {label}_FAIL {type(e).__name__}: {e}")
        return ""
t0 = time.time()
a = ask("What is 17*23? Answer with the number only.", 600, "MATH600")
b = ask("List the numbers from 1 to 5, comma separated, nothing else.", 300, "LIST300")
c = ask("Write one paragraph (3 sentences) explaining why the sky is blue.", 250, "PROSE250")
print(f"[{tag}] VERDICT math391={'391' in a} list={'5' in b} prose_len={len(c)} wall={time.time()-t0:.0f}s")
PYEOF
    else
      echo "e4m3ns health-gone, skip"
    fi
  fi
  if [ "$FIRED_E5M2NS" -eq 0 ] && grep -q "^-- e5m2ns numerics" "$LOG" 2>/dev/null; then
    FIRED_E5M2NS=1
    echo "--- probe600 e5m2ns firing $(date +%T) ---"
    if curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health; then
      python3 - e5m2ns <<'PYEOF' 2>&1 | tee -a "$OUT/p22a_probe600_e5m2ns.txt"
import json, sys, time, urllib.request
tag = sys.argv[1]
M = "qwen3.8-27b-fp8"
def ask(prompt, mt, label):
    body = json.dumps({"model": M, "max_tokens": mt, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
        m = r["choices"][0]["message"]
        vis = m.get("content") or ""
        print(f"[{tag}] {label} http-ok visible_len={len(vis)} contains391={'391' in vis}")
        print(f"[{tag}] {label} FULLTEXT {vis[:600]!r}")
        return vis
    except Exception as e:
        print(f"[{tag}] {label}_FAIL {type(e).__name__}: {e}")
        return ""
t0 = time.time()
a = ask("What is 17*23? Answer with the number only.", 600, "MATH600")
b = ask("List the numbers from 1 to 5, comma separated, nothing else.", 300, "LIST300")
c = ask("Write one paragraph (3 sentences) explaining why the sky is blue.", 250, "PROSE250")
print(f"[{tag}] VERDICT math391={'391' in a} list={'5' in b} prose_len={len(c)} wall={time.time()-t0:.0f}s")
PYEOF
    else
      echo "e5m2ns health-gone, skip"
    fi
  fi
  [ "$FIRED_E4M3NS" -eq 1 ] && [ "$FIRED_E5M2NS" -eq 1 ] && break
  sleep 15; i=$((i+15))
done
echo "=== probe600 watcher v2 end $(date +%F' '%T) fired_e4m3ns=$FIRED_E4M3NS fired_e5m2ns=$FIRED_E5M2NS ==="
