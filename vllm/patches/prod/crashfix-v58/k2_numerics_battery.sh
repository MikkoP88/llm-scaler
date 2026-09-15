#!/bin/bash
# k2_numerics_battery.sh — run INSIDE host, hits localhost:8000.
# 1) greedy determinism x2 (Paris probe, sha compare)
# 2) greedy variant probe (2+2 style)
# 3) sampled cargame coherence (temp 0.9)
set -u
PY=/opt/venv/bin/python3
probe() {
  curl -s -m 120 http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' -d "$1"
}

echo "== determinism probe x2 (greedy Paris)"
for i in 1 2; do
  probe '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":100,"temperature":0}' \
    | $PY -c "
import json,sys,hashlib
r=json.load(sys.stdin)
m=r['choices'][0]['message']
txt=(m.get('reasoning_content') or '')+'||'+(m.get('content') or '')
print('run$i sha:',hashlib.sha256(txt.encode()).hexdigest()[:16],'len:',len(txt))
print('run$i tail:',repr(txt[-80:]))
"
done

echo "== greedy variant probe"
probe '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"Calculate 17*23 step by step, then give the final answer."}],"max_tokens":220,"temperature":0}' \
  | $PY -c "
import json,sys
r=json.load(sys.stdin); m=r['choices'][0]['message']
txt=(m.get('reasoning_content') or '')+' || CONTENT: '+(m.get('content') or '')
print(txt[-400:])
"

echo "== sampled cargame coherence (temp 0.9)"
probe '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"Write a html car game"}],"max_tokens":700,"temperature":0.9,"top_k":20,"top_p":0.95}' \
  | $PY -c "
import json,sys
r=json.load(sys.stdin); m=r['choices'][0]['message']
c=(m.get('content') or '')
print('content_len:',len(c))
print('starts:',repr(c[:200]))
print('has_html:',('<html' in c.lower() or '<canvas' in c.lower() or '<div' in c.lower()))
print('usage:',r.get('usage',{}).get('completion_tokens'))
"
echo BATTERY_DONE
