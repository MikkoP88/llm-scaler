#!/bin/bash
# w2_leg_battery.sh — v127 W2 per-leg keep/rollback battery.
# Runs against the LIVE lane after a W2 knob boot. Load-contaminated by
# design (user fleet active); the gate here is STABILITY + QUALITY, with
# speed spots recorded for the quiet re-run set. Verdict markers: W2B_*
# Usage: w2_leg_battery.sh <TAG>   (e.g. w2_leg_battery.sh W2A_GMU)
set -u
TAG="${1:?usage: w2_leg_battery.sh <TAG>}"
OUT=/root/build/lce1/w2battery_${TAG}.txt
M=http://localhost:8000
MODEL=qwen3.8-27b-fp8
: > "$OUT"
q() { curl -s -m 5 $M/metrics | grep -E "^vllm:num_requests_(running|waiting)\{" | tr -s " " | cut -d " " -f2 | paste -sd/ -; }

echo "== $TAG battery start $(date '+%T') queue=$(q) ==" >> "$OUT"

echo "== SANITY direct + litellm ==" >> "$OUT"
python3 - <<'PYEOF' >> "$OUT" 2>&1
import json, urllib.request
def post(url, body, hdrs):
    req = urllib.request.Request(url, json.dumps(body).encode(), hdrs)
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode())
    return d
# direct :8000
d = post("http://localhost:8000/v1/chat/completions",
         {"model": "qwen3.8-27b-fp8", "max_tokens": 64, "temperature": 0,
          "messages": [{"role": "user", "content": "Reply with exactly: OK"}]},
         {"Content-Type": "application/json", "Authorization": "Bearer sk-dummy"})
c = d["choices"][0]
print("SANITY_DIRECT code=200 finish=%s text=%r" % (c["finish_reason"], c["message"]["content"][:60]))
# through litellm :4000, CC /v1/messages shape
d = post("http://localhost:4000/v1/messages",
         {"model": "qwen3.8-27b-fp8-sonnet", "max_tokens": 64,
          "messages": [{"role": "user", "content": "Reply with exactly: OK"}]},
         {"Content-Type": "application/json", "Authorization": "Bearer sk-dummy",
          "x-api-key": "sk-dummy", "anthropic-version": "2023-06-01"})
txt = "".join(b.get("text", "") for b in d.get("content", []))
print("SANITY_LITELLM stop_reason=%s text=%r" % (d.get("stop_reason"), txt[:60]))
PYEOF

echo "== SERIALIZED 24 (v88 wedge acceptance) ==" >> "$OUT"
OK=0; FAIL=0; H=200
for i in $(seq 1 24); do
  out=$(curl -s --max-time 20 -X POST $M/v1/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Count from 1 to 5.\",\"max_tokens\":24,\"temperature\":0}" \
    -o /dev/null -w "%{http_code}")
  [ "$out" = "200" ] && OK=$((OK+1)) || FAIL=$((FAIL+1))
  H=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 $M/health)
  [ "$H" != "200" ] && { echo "SERIALIZED DEAD cycle $i ok=$OK fail=$FAIL" >> "$OUT"; break; }
done
[ "$H" = "200" ] && echo "SERIALIZED SURVIVED ok=$OK fail=$FAIL" >> "$OUT"

echo "== SOLO GENSPEED 1x1024 x3 (quiet bank = 73.71) ==" >> "$OUT"
cd /root/build && python3 -u bench_genspeed.py 1 1024 3 >> "$OUT" 2>&1

echo "== ASYNC GENSPEED 4x1024 (quiet bank = 124.97) ==" >> "$OUT"
cd /root/build && python3 -u bench_genspeed.py 4 1024 1 >> "$OUT" 2>&1

echo "== XGRAMMAR SPOT (single call: t2 x3 + alt x2) ==" >> "$OUT"
cd /root/build && python3 -u t2_xgrammar_probe.py >> "$OUT" 2>&1

echo "== ENVELOPE: waiting_by_reason + kv (C1 watch: is the cap now KV or still SSM?) ==" >> "$OUT"
curl -s -m 5 $M/metrics | grep -E "^vllm:num_requests_waiting_by_reason\{|^vllm:gpu_cache_usage_perc\{" >> "$OUT"

echo "W2B_${TAG}_DONE $(date '+%T')" >> "$OUT"
tail -1 "$OUT"
