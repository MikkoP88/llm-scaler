#!/bin/bash
# stage4_soak_battery.sh — v60 CC-shaped soak (~15 min, 3 concurrent
# long-context tool-call workers straight to :8000) + E2E battery:
# litellm tool call (sonnet), structured output (XGrammar-2), RESPROBE
# ledger, spec-decode throughput sanity, wedge counters, health.
L=/root/build/lce1/v60_soak.log
{
echo "=== V60 SOAK+BATTERY $(date +%F' '%T) ==="
R0=$(dmesg | grep -ac 'Engine reset' || true)
T0=$(date +%s)

python3 - <<'PYEOF'
import json, threading, time, urllib.request

END = time.time() + 900  # 15 min
BASE = "http://localhost:8000"

def long_prompt(n_chunks):
    # CC-shaped: big stable system prefix + history blobs + tool defs
    sys = ("You are a senior systems engineer. " * 40)
    blob = ("CONTEXT LOG: the lane serves qwen3.8-27b on Xe2 TP2 with e4m3 KV. "
            "Watchdog relaunches take 140s. Prefix caching is on. " * 60)
    hist = "\n".join(f"turn {i}: assistant analyzed telemetry batch {i} and scheduled a probe. {blob}"
                     for i in range(n_chunks))
    return [{"role": "system", "content": sys},
            {"role": "user", "content": hist + "\nNow call get_weather for Tokyo."}]

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

res = {"done": 0, "tool_ok": 0, "err": 0}
lock = threading.Lock()

def worker(wid):
    i = 0
    while time.time() < END:
        i += 1
        body = json.dumps({
            "model": "qwen3.8-27b-fp8", "max_tokens": 768,
            "tools": TOOLS, "tool_choice": "auto",
            "messages": long_prompt((i % 3) + 3),  # ~12k-30k token prompts
        }).encode()
        req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            m = d["choices"][0]["message"]
            tc = m.get("tool_calls")
            with lock:
                res["done"] += 1
                if tc and tc[0]["function"]["name"] == "get_weather":
                    res["tool_ok"] += 1
        except Exception as e:
            with lock:
                res["err"] += 1
                print(f"[w{wid}] ERR {type(e).__name__} {e}", flush=True)
        time.sleep(2)

ts = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(3)]
[t.start() for t in ts]
[t.join() for t in ts]
print("SOAK_RESULTS", json.dumps(res))
PYEOF

T1=$(date +%s)
echo "soak_wall=$((T1-T0))s"
R1=$(dmesg | grep -ac 'Engine reset' || true)
echo "engine_resets_delta=$((R1-R0)) (before=$R0 after=$R1)"
echo "--- v55_3_crash.log (must be empty) ---"
docker exec lsv-test sh -c 'cat /root/v55_3_crash.log 2>/dev/null' || echo none

echo "=== BATTERY T1: litellm tool call (sonnet) ==="
curl -s -m 120 http://10.100.8.6:4000/v1/messages \
  -H "Authorization: Bearer sk-dummy" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-fp8-sonnet","max_tokens":1024,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"messages":[{"role":"user","content":"Weather in Paris? Use get_weather."}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); tbs=[b for b in d.get("content",[]) if b.get("type")=="tool_use"]; print("T1", d.get("stop_reason"), [b.get("name") for b in tbs], [json.dumps(b.get("input")) for b in tbs])'

echo "=== BATTERY T2: structured output (XGrammar-2 0.2.7) ==="
curl -s -m 180 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer sk-dummy" \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2048,"chat_template_kwargs":{"enable_thinking":false},"response_format":{"type":"json_schema","json_schema":{"name":"weather","strict":true,"schema":{"type":"object","properties":{"city":{"type":"string"},"temperature_c":{"type":"number"}},"required":["city","temperature_c"],"additionalProperties":false}}},"messages":[{"role":"user","content":"Tokyo weather (guess values). Fill the JSON now."}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["choices"][0]["message"]["content"]; j=json.loads(c); print("T2 VALID_JSON", sorted(j.keys()), "finish", d["choices"][0]["finish_reason"])'

echo "=== BATTERY T3: spec-decode throughput sanity ==="
python3 - <<'PYEOF'
import json, time, urllib.request
body = json.dumps({"model": "qwen3.8-27b-fp8", "max_tokens": 512,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": "Count from 1 to 200 with short commentary."}]}).encode()
req = urllib.request.Request("http://localhost:8000/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    d = json.load(r)
dt = time.time() - t0
u = d.get("usage", {})
ct = u.get("completion_tokens", 0)
print(f"T3 completion_tokens={ct} wall={dt:.1f}s tok/s={ct/dt:.1f} (spec MTP active if >> 15)")
PYEOF

echo "=== BATTERY T4: RESPROBE ledger ==="
docker exec lsv-test sh -c 'grep -c "RESPROBE START" /root/serve_full.log; grep -c "RESPROBE DONE" /root/serve_full.log'
echo "=== BATTERY T5: v60c zombie-guard census (strikes must contain, engine must live) ==="
docker exec lsv-test sh -c 'echo strikeouts=$(grep -c "v60c STRIKE-OUT" /root/serve_full.log); echo zeroem=$(grep -c "v60c ZERO-EMISSION" /root/serve_full.log); echo sycl_asserts=$(grep -c "index out of bounds" /root/serve_full.log)'
curl -s -o /dev/null -w 'final_health=%{http_code}\n' -m 4 http://localhost:8000/health
echo "=== V60 SOAK+BATTERY DONE $(date +%F' '%T) ==="
} > "$L" 2>&1
tail -45 "$L"
echo STAGE4_DONE
