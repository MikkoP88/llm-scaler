#!/usr/bin/env python3
"""sanity_foldv_v128.py — post-fold quick admission sanity.

Three checks on the folded-checkpoint lane (certified posture boot):
  t1  thinking mode: finish=stop, non-empty reasoning field
  t2  XGrammar-2 guided JSON (bounded numeric recipe schema):
      parses, value matches the bounded pattern
  log scan lives in the calling shell (serve_full.log)
"""
import json
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer sk-dummy"}
BOUNDED = {"type": "object",
           "properties": {"city": {"type": "string"},
                          "t": {"type": "string",
                                "pattern": r"^-?[0-9]{1,4}(\.[0-9]{1,4})?$"}},
           "required": ["city", "t"], "additionalProperties": False}


def post(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)


d1 = post({"model": "qwen3.8-27b-fp8", "max_tokens": 512, "temperature": 0.6,
           "messages": [{"role": "user", "content": "2+2? one word."}]})
m1 = d1["choices"][0]["message"]
print("t1 finish=%s reasoning_len=%d content=%r" % (
    d1["choices"][0]["finish_reason"], len(m1.get("reasoning") or ""),
    m1["content"][:40]))

d2 = post({"model": "qwen3.8-27b-fp8", "max_tokens": 256, "temperature": 0.6,
           "chat_template_kwargs": {"enable_thinking": False},
           "response_format": {"type": "json_schema", "json_schema": {
               "name": "w", "strict": True, "schema": BOUNDED}},
           "messages": [{"role": "user",
                         "content": "Tokyo weather, fill JSON now."}]})
c2 = d2["choices"][0]["message"]["content"]
try:
    j2 = json.loads(c2)
    print("t2 xgrammar finish=%s parsed=True val=%r" % (
        d2["choices"][0]["finish_reason"], j2.get("t")))
except Exception:
    print("t2 xgrammar finish=%s parsed=False content=%r" % (
        d2["choices"][0]["finish_reason"], c2[:60]))
print("SANITY_FOLDV_DONE")
