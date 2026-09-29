#!/bin/bash
# cc_battery_fix_v1224.sh — root-cause + fix the P14 CC battery 400s.
#
# Ship log evidence: cc_t1..t3 http=400 EMPTY, cc_thinking NO_THINKING.
# Captured body: {"error":{"message":"Invalid JSON payload: unexpected
# character: line 1 column 133 (char 132)"}}. litellm log shows the parse
# error stacked with "No api key passed in." on one request (400) and a
# clean 401 on another. v1222's green battery (probe_think_cc*.py) sent
# "authorization": "Bearer " + KEY — ours sent none.
#
# This script: (1) reproduces the exact t1 body construction from
# ship_v1224.sh and validates the JSON, (2) prints the offending bytes,
# (3) re-runs the FULL CC battery WITH the master key and clean JSON.
set -u
KEY=sk-dummy
B=http://127.0.0.1:4000
M=qwen3.8-27b-fp8-opus

echo "--- 1. reproduce ship-script t1 body construction ---"
PROMPT='Use the Bash tool to run: echo ship-v1224-ok'
BODY="{\"model\":\"$M\",\"max_tokens\":600,\"messages\":[{\"role\":\"user\",\"content\":$(
  python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$PROMPT")]}"
echo "len=${#BODY}"
python3 - "$BODY" <<'PYEOF'
import json, sys
raw = sys.argv[1]
try:
    d = json.loads(raw)
    print("JSON_OK content=%r" % d["messages"][0]["content"])
except json.JSONDecodeError as e:
    print("JSON_BAD: %s" % e)
    i = e.pos
    print("around pos %d: %r" % (i, raw[max(0,i-25):i+25]))
PYEOF

echo "--- 2. what the ship script ACTUALLY built (trace its own lines) ---"
sed -n '86,99p' /root/build/ship_v1224.sh | cat -A | sed -n '1,14p' | head -14

echo "--- 3. full CC battery WITH auth + validated JSON ---"
for t in t1 t2 t3; do
  case $t in
    t1) PROMPT='Use the Bash tool to run: echo ship-v1224-ok';;
    t2) PROMPT='Read the file /etc/hostname and report its contents using the Read tool';;
    t3) PROMPT='What is 17*23? Answer with just the number.';;
  esac
  python3 - "$PROMPT" > /tmp/cc_body_$t.json <<'PYEOF'
import json, sys
print(json.dumps({"model": "qwen3.8-27b-fp8-opus", "max_tokens": 600,
                  "messages": [{"role": "user", "content": sys.argv[1]}]}))
PYEOF
  code=$(curl -s -m 300 $B/v1/messages -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $KEY" -d @/tmp/cc_body_$t.json \
    -o /tmp/ccfix_$t.json -w '%{http_code}')
  TUSE=$(python3 -c "import json; d=json.load(open('/tmp/ccfix_$t.json')); print('tool_use' if any(b.get('type')=='tool_use' for b in d.get('content',[])) else ('text' if d.get('content') else 'EMPTY'))" 2>/dev/null || echo PARSE_ERR)
  echo "ccfix_$t http=$code blocks=$TUSE"
done
python3 - > /tmp/cc_body_th.json <<'PYEOF'
import json
print(json.dumps({"model": "qwen3.8-27b-fp8-opus", "max_tokens": 400,
                  "thinking": {"type": "enabled", "budget_tokens": 2000},
                  "messages": [{"role": "user", "content": "Think step by step: what is the capital of Finland?"}]}))
PYEOF
code=$(curl -s -m 300 $B/v1/messages -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KEY" -d @/tmp/cc_body_th.json -o /tmp/ccfix_th.json -w '%{http_code}')
TH=$(python3 -c "import json; d=json.load(open('/tmp/ccfix_th.json')); types=[b.get('type') for b in d.get('content',[])]; print('thinking' if 'thinking' in types else ('ERR:'+str(d.get('error',{}).get('message',''))[:120] if d.get('error') else 'NO_THINKING'))" 2>/dev/null)
echo "ccfix_thinking http=$code result=$TH"
echo "--- 4. verdict inputs ---"
grep -c '"type":"error"' /tmp/ccfix_t1.json /tmp/ccfix_t3.json 2>/dev/null | head -2
echo CC_BATTERY_FIX_DONE
