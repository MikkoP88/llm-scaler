#!/usr/bin/env python3
"""T2 xgrammar-2 probe: re-run the exact stage4-T2 structured-output
request N times against :8000 and print raw outcomes (status, keys,
finish_reason, content head). Also a second distinct schema to exercise
a fresh grammar compile. Evidence for the 15:13:26 backend_xgrammar.py:158
'Failed to advance FSM ... for tokens 0' failure right after the v60 soak.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"


def post(body, tag):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-dummy"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            code, d = r.status, json.load(r)
    except urllib.error.HTTPError as e:
        code = e.code
        d = json.load(e)
    except Exception as e:
        print(f"[{tag}] EXC {type(e).__name__}: {e}")
        return
    wall = time.time() - t0
    if "choices" in d:
        m = d["choices"][0]
        c = (m.get("message") or {}).get("content") or ""
        print(f"[{tag}] HTTP={code} wall={wall:.1f}s keys=choices "
              f"finish={m.get('finish_reason')} content={c[:120]!r}")
        try:
            j = json.loads(c)
            print(f"[{tag}]   PARSED_JSON keys={sorted(j.keys())}")
        except Exception as e:
            print(f"[{tag}]   JSON_PARSE_FAIL {type(e).__name__}: {e}")
    else:
        print(f"[{tag}] HTTP={code} wall={wall:.1f}s NO_CHOICES "
              f"body={json.dumps(d)[:400]}")


T2_BODY = {
    "model": "qwen3.8-27b-fp8", "max_tokens": 2048,
    "chat_template_kwargs": {"enable_thinking": False},
    "response_format": {"type": "json_schema", "json_schema": {
        "name": "weather", "strict": True,
        "schema": {"type": "object",
                   "properties": {"city": {"type": "string"},
                                  "temperature_c": {"type": "number"}},
                   "required": ["city", "temperature_c"],
                   "additionalProperties": False}}},
    "messages": [{"role": "user",
                  "content": "Tokyo weather (guess values). Fill the JSON now."}],
}

ALT_BODY = {
    "model": "qwen3.8-27b-fp8", "max_tokens": 1024,
    "chat_template_kwargs": {"enable_thinking": False},
    "response_format": {"type": "json_schema", "json_schema": {
        "name": "person", "strict": True,
        "schema": {"type": "object",
                   "properties": {"name": {"type": "string"},
                                  "age": {"type": "integer"}},
                   "required": ["name", "age"],
                   "additionalProperties": False}}},
    "messages": [{"role": "user",
                  "content": "Invent a person. Fill the JSON now."}],
}

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for i in range(N):
    post(T2_BODY, f"t2-run{i+1}")
    time.sleep(2)
for i in range(2):
    post(ALT_BODY, f"alt-run{i+1}")
    time.sleep(2)
print("PROBE_DONE")
