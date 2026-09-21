#!/bin/bash
# e2e_tool_test.sh — v59 §1 E2E tool-call validation via litellm :4000 (anthropic /v1/messages)
# Usage: e2e_tool_test.sh [model]   default: qwen3.8-27b-fp8-sonnet
M="${1:-qwen3.8-27b-fp8-sonnet}"
LT="http://10.100.8.6:4000"
AUTH="Authorization: Bearer sk-dummy"
CT="Content-Type: application/json"
OUT=/root/build/e2e_tool_$(date +%H%M%S)
mkdir -p "$OUT"

TOOLS='[{"name":"get_weather","description":"Get current weather for a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'
MSG='[{"role":"user","content":"What is the weather in Tokyo right now? Call the get_weather tool with city Tokyo."}]'

echo "=== T1 $M forced tool_choice ==="
curl -s -m 180 "$LT/v1/messages" -H "$AUTH" -H "$CT" -d '{
  "model": "'"$M"'",
  "max_tokens": 2048,
  "tools": '"$TOOLS"',
  "tool_choice": {"type": "tool", "name": "get_weather"},
  "messages": '"$MSG"'
}' > "$OUT/forced.json"
python3 -c '
import json,sys
d=json.load(open("'"$OUT"'/forced.json"))
print("type:",d.get("type"),"stop_reason:",d.get("stop_reason"))
for b in d.get("content",[]):
    if b.get("type")=="tool_use": print("TOOL_USE:",b.get("name"),json.dumps(b.get("input")))
    elif b.get("type")=="text": print("TEXT:",b.get("text","")[:120])
    elif b.get("type")=="thinking": print("THINKING len:",len(b.get("thinking","")))
if d.get("type")=="error": print("ERROR:",json.dumps(d)[:400])
' 2>&1 || echo "T1 PARSE_FAIL: $(head -c 300 "$OUT/forced.json")"

echo "=== T2 $M auto tools (previous STALL path) ==="
curl -s -m 180 "$LT/v1/messages" -H "$AUTH" -H "$CT" -d '{
  "model": "'"$M"'",
  "max_tokens": 2048,
  "tools": '"$TOOLS"',
  "messages": '"$MSG"'
}' > "$OUT/auto.json"
python3 -c '
import json,sys
d=json.load(open("'"$OUT"'/auto.json"))
print("type:",d.get("type"),"stop_reason:",d.get("stop_reason"))
for b in d.get("content",[]):
    if b.get("type")=="tool_use": print("TOOL_USE:",b.get("name"),json.dumps(b.get("input")))
    elif b.get("type")=="text": print("TEXT:",b.get("text","")[:120])
    elif b.get("type")=="thinking": print("THINKING len:",len(b.get("thinking","")))
if d.get("type")=="error": print("ERROR:",json.dumps(d)[:400])
' 2>&1 || echo "T2 PARSE_FAIL: $(head -c 300 "$OUT/auto.json")"

echo "=== T3 $M streaming auto tools ==="
curl -s -m 180 -N "$LT/v1/messages" -H "$AUTH" -H "$CT" -d '{
  "model": "'"$M"'",
  "max_tokens": 2048,
  "stream": true,
  "tools": '"$TOOLS"',
  "messages": '"$MSG"'
}' > "$OUT/stream.txt"
echo "stream_bytes=$(wc -c < "$OUT/stream.txt")"
grep -c 'content_block_stop' "$OUT/stream.txt" | xargs echo "content_block_stop_events="
grep -c 'tool_use' "$OUT/stream.txt" | xargs echo "tool_use_mentions="
grep 'message_stop' "$OUT/stream.txt" | head -1
grep -o '"name":"get_weather"' "$OUT/stream.txt" | head -1
echo "E2E_DONE_OUTDIR=$OUT"
