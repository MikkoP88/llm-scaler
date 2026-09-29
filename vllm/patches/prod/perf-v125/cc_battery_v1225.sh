#!/bin/bash
# cc_battery_v1225.sh — standalone CC battery through litellm :4000 (the
# ship's step-6 re-run). Bodies are built ENTIRELY by python (json.dumps)
# and passed via -d @file — the standing v1223 lesson: shell-assembled
# JSON tails are a recurring corruption source (this round: `)}]}\""`
# appended a stray quote after the document; v1223: missing close brace).
# Acceptance: t1/t2/t3 http=200; thinking leg returns a thinking block.
set -u
B=http://127.0.0.1:4000
FAIL=0

LIT=$(curl -s -o /dev/null -m 5 -w '%{http_code}' $B/health/liveness)
echo "litellm_liveness=$LIT"
[ "$LIT" = "200" ] || { echo "CC ABORT: litellm down"; exit 1; }

for t in t1 t2 t3; do
  case $t in
    t1) PROMPT='Use the Bash tool to run: echo ship-v1225-ok';;
    t2) PROMPT='Read the file /etc/hostname and report its contents using the Read tool';;
    t3) PROMPT='What is 17*23? Answer with just the number.';;
  esac
  python3 -c 'import json,sys; print(json.dumps({"model":"qwen3.8-27b-fp8-opus","max_tokens":600,"messages":[{"role":"user","content":sys.argv[1]}]}))' "$PROMPT" > /tmp/cc_${t}_body.json
  python3 -c "import json; json.load(open('/tmp/cc_${t}_body.json'))" || { echo "CC ABORT: $t body not valid JSON"; exit 1; }
  code=$(curl -s -m 300 $B/v1/messages -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-dummy' \
    -d @/tmp/cc_${t}_body.json -o /tmp/cc_$t.json -w '%{http_code}')
  TUSE=$(python3 -c "import json; d=json.load(open('/tmp/cc_$t.json')); print('tool_use' if any(b.get('type')=='tool_use' for b in d.get('content',[])) else ('text' if d.get('content') else 'EMPTY'))" 2>/dev/null)
  echo "cc_$t http=$code blocks=$TUSE"
  [ "$code" = "200" ] || FAIL=1
done

python3 -c 'import json; print(json.dumps({"model":"qwen3.8-27b-fp8-opus","max_tokens":400,"thinking":{"type":"enabled","budget_tokens":2000},"messages":[{"role":"user","content":"Think step by step: what is the capital of Finland?"}]}))' > /tmp/cc_th_body.json
TH=$(curl -s -m 300 $B/v1/messages -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dummy' \
  -d @/tmp/cc_th_body.json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); types=[b.get("type") for b in d.get("content",[])]; print("thinking" if "thinking" in types else "NO_THINKING")' 2>/dev/null)
echo "cc_thinking=$TH"
[ "$TH" = "thinking" ] || FAIL=1

if [ "$FAIL" != "0" ]; then
  echo "CC_BATTERY_V1225: FAIL"; exit 1
fi
echo "CC_BATTERY_V1225: ALL GREEN"
