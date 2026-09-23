import json, urllib.request, urllib.error

BASE = "http://127.0.0.1:4000"
H = {"Content-Type": "application/json", "x-api-key": "sk-dummy", "anthropic-version": "2023-06-01"}

base = {"model": "qwen3.8-27b-fp8-sonnet", "max_tokens": 300,
        "messages": [{"role": "user", "content": "Think briefly: 17*23? Final number only."}]}
TOOLS = [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]

# E2: 400 body of thinking param
try:
    req = urllib.request.Request(BASE + "/v1/messages", json.dumps({**base, "thinking": {"type": "enabled", "budget_tokens": 5000}}).encode(), H)
    urllib.request.urlopen(req, timeout=60)
except urllib.error.HTTPError as e:
    print("E2_400_BODY", e.read().decode()[:400])

# H: CC shape WITHOUT thinking param, streaming, full betas
hbody = {**base, "max_tokens": 32768, "system": "You are Claude Code, a coding assistant. Be terse.",
         "temperature": 1.0, "tools": TOOLS, "stream": True, "metadata": {"user_id": "test"}}
hh = dict(H); hh["anthropic-beta"] = "claude-code-20250219,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14"
req = urllib.request.Request(BASE + "/v1/messages", json.dumps(hbody).encode(), hh)
counts = {}
with urllib.request.urlopen(req, timeout=240) as r:
    for raw in r:
        line = raw.decode(errors="replace").strip()
        if line.startswith("data:"):
            try: j = json.loads(line.split(":",1)[1])
            except Exception: continue
            if j.get("type")=="content_block_start":
                t="start_"+str(j.get("content_block",{}).get("type")); counts[t]=counts.get(t,0)+1
            if j.get("type")=="content_block_delta":
                t="delta_"+str(j.get("delta",{}).get("type")); counts[t]=counts.get(t,0)+1
print("H_CC_SHAPE_STREAM", json.dumps(counts))

# H2: same but max_tokens 300 (isolate 32768)
h2 = dict(hbody); h2["max_tokens"] = 300
req = urllib.request.Request(BASE + "/v1/messages", json.dumps(h2).encode(), hh)
counts = {}
with urllib.request.urlopen(req, timeout=240) as r:
    for raw in r:
        line = raw.decode(errors="replace").strip()
        if line.startswith("data:"):
            try: j = json.loads(line.split(":",1)[1])
            except Exception: continue
            if j.get("type")=="content_block_start":
                t="start_"+str(j.get("content_block",{}).get("type")); counts[t]=counts.get(t,0)+1
            if j.get("type")=="content_block_delta":
                t="delta_"+str(j.get("delta",{}).get("type")); counts[t]=counts.get(t,0)+1
print("H2_CC_SHAPE_MT300", json.dumps(counts))
