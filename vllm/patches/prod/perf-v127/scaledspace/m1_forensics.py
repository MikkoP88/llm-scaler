#!/usr/bin/env python3
"""m1_forensics.py — full-generation capture from the M1 scaled-space leg.

Maps the degradation onset: short decodes clean (math 0/80 wrong at <=10
tokens), tool syntax corrupts ~15 tokens in (<function=run then !!!!).
Captures FULL bodies so the compounding curve is on record:

  F1  T1 thinkOff max_tokens 300   -> full content, token-position of first '!'
  F2  counting 1..60 thinkOff 400  -> watch digit integrity vs step count
  F3  T1 thinkON  max_tokens 2048  -> where did 2048 tokens go (content empty?)
"""
import json
import urllib.request

src = open("/root/p23f_probe.py").read().split("def call(")[0]
ns = {}
exec(src, ns)  # noqa: S102 - probe constants only
T1, MODEL, BASE = ns["T1"], ns["MODEL"], ns["BASE"]


def call(body: dict) -> dict:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=600))


def show(tag: str, body: dict) -> dict:
    r = call(body)
    ch = r["choices"][0]
    m = ch["message"]
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or ""
    tc = m.get("tool_calls")
    ct = r.get("usage", {}).get("completion_tokens")
    pos = next((i for i, c in enumerate(content) if c == "!"), -1)
    print(f"=== {tag}: finish={ch.get('finish_reason')} ctokens={ct} "
          f"tool_name={tc[0]['function']['name'] if tc else None} "
          f"first_bang_pos={pos}", flush=True)
    print(f"--- content ({len(content)} ch): {content[:600]!r}", flush=True)
    if reasoning:
        print(f"--- reasoning ({len(reasoning)} ch): {reasoning[:300]!r}", flush=True)
    return r


b = dict(T1)
b["chat_template_kwargs"] = {"enable_thinking": False}
show("F1 T1-thinkOff-300", b)

b = {
    "model": MODEL,
    "messages": [{"role": "user", "content":
                  "Write the numbers 1 to 60, one number per line, nothing else."}],
    "max_tokens": 400,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
show("F2 count-1..60", b)

b = dict(T1)
b["max_tokens"] = 2048
show("F3 T1-thinkOn-2048", b)
print("M1_FORENSICS_DONE", flush=True)
