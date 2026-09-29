#!/bin/bash
# cc_battery_fix_v1224.sh — re-run the v1.2.24 ship CC battery legs that
# http-400'd at 17:0x. Root cause (client-side, lane never at fault): the
# ship/cont scripts' curl -d shell-assembly ends `...}]}\""` — the escaped
# quote appends a literal `"` AFTER the valid JSON document ("unexpected
# content after document: char 134/135/136" = prompt length + 1 exactly;
# the literal-JSON thinking body returned 200 in the same round). This is
# the same defect class the v1223 ship hit; the durable fix (v1223 lesson,
# now applied fully): build the ENTIRE request body with python3
# json.dumps into a file and curl -d @file — no shell string assembly.
# Model/lane untouched: the lane is UP on llm-scaler-exp:v1.2.24.
set -u
L=/root/build/lce1/v1224_ship.log
FAIL=0
B=http://127.0.0.1:4000
{
echo "=== CC_BATTERY_FIX_V1224 start $(date +%F' '%T) ==="
for t in t1 t2 t3; do
  case $t in
    t1) PROMPT='Use the Bash tool to run: echo ship-v1224-ok';;
    t2) PROMPT='Read the file /etc/hostname and report its contents using the Read tool';;
    t3) PROMPT='What is 17*23? Answer with just the number.';;
  esac
  python3 - "$PROMPT" > /tmp/cc_body_$t.json <<'PYEOF'
import json, sys
body = {"model": "qwen3.8-27b-fp8-opus", "max_tokens": 600,
        "messages": [{"role": "user", "content": sys.argv[1]}]}
print(json.dumps(body))
PYEOF
  python3 -c "import json; json.load(open('/tmp/cc_body_$t.json')); print('body_valid')" || { echo "cc_$t BODY_INVALID"; FAIL=1; continue; }
  code=$(curl -s -m 300 $B/v1/messages -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-dummy' \
    -d @/tmp/cc_body_$t.json \
    -o /tmp/cc_$t.json -w '%{http_code}')
  TUSE=$(python3 -c "import json; d=json.load(open('/tmp/cc_$t.json')); print('tool_use' if any(b.get('type')=='tool_use' for b in d.get('content',[])) else ('text' if d.get('content') else 'EMPTY'))" 2>/dev/null)
  echo "cc_$t http=$code blocks=$TUSE"
  [ "$code" = "200" ] || FAIL=1
done
# thinking leg re-check (passed pre-fix; re-verified for a complete green record)
python3 - > /tmp/cc_body_th.json <<'PYEOF'
import json
body = {"model": "qwen3.8-27b-fp8-opus", "max_tokens": 400,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "messages": [{"role": "user", "content": "Think step by step: what is the capital of Finland?"}]}
print(json.dumps(body))
PYEOF
code=$(curl -s -m 300 $B/v1/messages -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dummy' \
  -d @/tmp/cc_body_th.json \
  -o /tmp/cc_th.json -w '%{http_code}')
TH=$(python3 -c 'import json; d=json.load(open("/tmp/cc_th.json")); types=[b.get("type") for b in d.get("content",[])]; print("thinking" if "thinking" in types else "NO_THINKING")' 2>/dev/null)
echo "cc_thinking http=$code blocks=$TH"
{ [ "$code" = "200" ] && [ "$TH" = "thinking" ]; } || FAIL=1

if [ "$FAIL" = "0" ]; then
  echo "=== CC_BATTERY_FIX_V1224 ALL GREEN $(date +%T) ==="
else
  echo "=== CC_BATTERY_FIX_V1224 STILL FAILING $(date +%T) ==="
fi
} >> "$L" 2>&1
tail -12 "$L"
[ "$FAIL" = "0" ] || exit 1
echo CC_BATTERY_FIX_V1224_DONE
