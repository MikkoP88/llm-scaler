#!/usr/bin/env python3
"""xg_active_check_v128.py — is json_schema guidance ACTIVE on this lane?

Ambiguity being resolved: sanity t2 returned a 7-integer-digit value under
a ^-?[0-9]{1,4}(\.[0-9]{1,4})?$ pattern — either guidance was inactive
(freeform luck) or pattern enforcement is broken. A freeform model asked
for a temperature will NEVER emit the sentinel prefix, so:

  schema field "code": {"type":"string","pattern":"^ZZ[0-9]{2}$"}
  prompt asks for a numeric reading.

  guided-active  -> content matches ^ZZ[0-9]{2}$ on every run
  guided-inactive-> content is anything else (bare number, longer digits)

Runs 6x, reports per-run match. Also re-tests the original bounded
numeric pattern 6x to see whether 7-digit escapes repeat.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer sk-dummy"}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

SENTINEL = {"type": "object",
            "properties": {"code": {"type": "string",
                                    "pattern": r"^ZZ[0-9]{2}$"}},
            "required": ["code"], "additionalProperties": False}
BOUNDED = {"type": "object",
           "properties": {"city": {"type": "string"},
                          "t": {"type": "string",
                                "pattern": r"^-?[0-9]{1,4}(\.[0-9]{1,4})?$"}},
           "required": ["city", "t"], "additionalProperties": False}


def post(schema, name):
    body = {"model": "qwen3.8-27b-fp8", "max_tokens": 256, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": name, "strict": True, "schema": schema}},
            "messages": [{"role": "user", "content":
                          "Tokyo weather (guess values). Fill the JSON now."}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)


sent_ok = 0
for i in range(N):
    d = post(SENTINEL, "senti")
    c = d["choices"][0]["message"]["content"]
    m = None
    try:
        m = re.search(r"ZZ[0-9]{2}", json.loads(c).get("code", ""))
    except Exception:
        pass
    sent_ok += bool(m)
    print(f"[sentinel run{i+1}] finish={d['choices'][0]['finish_reason']} "
          f"match={bool(m)} content={c[:60]!r}", flush=True)
    time.sleep(0.5)

bnd_ok = 0
for i in range(N):
    d = post(BOUNDED, "w")
    c = d["choices"][0]["message"]["content"]
    v, ok = None, False
    try:
        v = json.loads(c).get("t")
        ok = bool(re.fullmatch(r"-?[0-9]{1,4}(\.[0-9]{1,4})?", str(v)))
    except Exception:
        pass
    bnd_ok += ok
    print(f"[bounded run{i+1}] finish={d['choices'][0]['finish_reason']} "
          f"pattern_ok={ok} val={v!r}", flush=True)
    time.sleep(0.5)

print(f"SUMMARY sentinel={sent_ok}/{N} bounded={bnd_ok}/{N}")
print("GUIDANCE_ACTIVE" if sent_ok == N else "GUIDANCE_INACTIVE_OR_BROKEN")
