import json, urllib.request, urllib.error

BASE = "http://127.0.0.1:4000"
H = {"Content-Type": "application/json", "x-api-key": "sk-dummy", "anthropic-version": "2023-06-01",
     "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14"}
TOOLS = [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]

def call(body, stream=False):
    b = dict(body); b["stream"] = stream
    req = urllib.request.Request(BASE + "/v1/messages", json.dumps(b).encode(), H)
    if not stream:
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.loads(r.read())
        return [x.get("type") for x in resp.get("content", [])]
    counts = {}
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if line.startswith("data:"):
                try: j = json.loads(line.split(":",1)[1])
                except Exception: continue
                if j.get("type")=="content_block_start":
                    t="start_"+str(j.get("content_block",{}).get("type")); counts[t]=counts.get(t,0)+1
                if j.get("type")=="content_block_delta":
                    t="delta_"+str(j.get("delta",{}).get("type")); counts[t]=counts.get(t,0)+1
    return counts

base = {"model": "qwen3.8-27b-fp8-sonnet", "max_tokens": 300,
        "messages": [{"role": "user", "content": "Think briefly: 17*23? Final number only."}]}

print("E2_THINKPARAM ", end="")
try:
    print(call({**base, "thinking": {"type": "enabled", "budget_tokens": 5000}}))
except urllib.error.HTTPError as e:
    print("ERR400", e.read().decode()[:200])
print("E2B_THINKPARAM_STREAM ", end="")
try:
    print(call({**base, "thinking": {"type": "enabled", "budget_tokens": 5000}}, stream=True))
except urllib.error.HTTPError as e:
    print("ERR400", e.read().decode()[:200])
print("A_plain       ", call(dict(base)))
print("A2_plain_stream", call(dict(base), stream=True))
print("CC_shape      ", call({**base, "max_tokens": 32768, "system": "You are Claude Code. Be terse.", "temperature": 1.0, "tools": TOOLS, "metadata": {"user_id": "t"}}, stream=True))
print("T4_notthink   ", call({"model": "qwen3.8-27b-fp8-nonthinking", "max_tokens": 100, "messages": [{"role": "user", "content": "Say OK only."}]}))
