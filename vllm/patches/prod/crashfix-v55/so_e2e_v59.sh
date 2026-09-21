#!/bin/bash
# so_e2e.sh — v59 §1 structured-output (xgrammar) E2E + auto-tools regression
echo "=== S1 structured output json_schema direct vLLM (xgrammar path) ==="
T0=$(date +%s)
curl -s -m 120 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer sk-dummy" \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":512,"response_format":{"type":"json_schema","json_schema":{"name":"weather","strict":true,"schema":{"type":"object","properties":{"city":{"type":"string"},"temperature_c":{"type":"number"}},"required":["city","temperature_c"],"additionalProperties":false}}},"messages":[{"role":"user","content":"Tokyo weather. Fill the JSON."}]}' \
  > /root/build/so_test.json
T1=$(date +%s)
echo "latency=$((T1-T0))s"
python3 - <<'PYEOF'
import json
d = json.load(open('/root/build/so_test.json'))
if d.get('choices'):
    c = d['choices'][0]['message']['content']
    print('CONTENT:', (c or '')[:200])
    j = json.loads(c)
    print('VALID_JSON keys=', sorted(j.keys()))
else:
    print('NO_CHOICES:', json.dumps(d)[:300])
PYEOF
echo "=== S2 auto-tools regression via litellm ==="
curl -s -m 120 http://10.100.8.6:4000/v1/messages \
  -H "Authorization: Bearer sk-dummy" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-fp8-sonnet","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"messages":[{"role":"user","content":"Weather in Paris? Use get_weather."}]}' \
  > /root/build/s2_test.json
python3 - <<'PYEOF'
import json
d = json.load(open('/root/build/s2_test.json'))
print('type:', d.get('type'), 'stop:', d.get('stop_reason'))
for b in d.get('content', []):
    if b.get('type') == 'tool_use':
        print('TOOL_USE:', b.get('name'), json.dumps(b.get('input')))
PYEOF
echo SO_E2E_DONE
