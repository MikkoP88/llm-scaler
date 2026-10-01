#!/usr/bin/env python3
"""m1_tool_discriminator.py — adjudicate the M1 mode-A tool SALAD.

Mode A reported 30/30 tool requests SALAD with finish=length, name=None,
and THINKING-style text — the signature of a token-budget artifact (T1/T2
send no enable_thinking kwarg, so the high-thinking template runs first
and 300-400 tokens cannot finish reasoning + emit the call), NOT of state
corruption. This script discriminates on the SAME lane:

  leg A  T1 with max_tokens 2048 (thinking still on)  -> call must arrive
  leg B  T1 thinking OFF, original 300                -> call must arrive
  leg C  T2 with max_tokens 2048                      -> call must arrive
  leg D  T2 thinking OFF, original 400                -> call must arrive

All four clean  -> numerics fine, budget artifact confirmed.
Garbage names/args after thinking completes -> REAL M1 damage.
"""
import json
import urllib.request

src = open("/root/p23f_probe.py").read().split("def call(")[0]
ns = {}
exec(src, ns)  # noqa: S102 - probe constants only (BASE/MODEL/T1/T2/FLAT/NESTED)
T1, T2, BASE = ns["T1"], ns["T2"], ns["BASE"]


def run(tag: str, body: dict) -> None:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=300))
    m = r["choices"][0]["message"]
    tc = m.get("tool_calls")
    fin = r["choices"][0].get("finish_reason")
    name = tc[0]["function"]["name"] if tc else None
    args = tc[0]["function"]["arguments"][:80] if tc else ""
    txt = (m.get("reasoning_content") or m.get("content") or "")[:40].replace("\n", " ")
    print(f"{tag}: finish={fin} name={name} args={args!r} think_head={txt!r}", flush=True)


b = dict(T1)
b["max_tokens"] = 2048
run("T1-bud2048      ", b)

b = dict(T1)
b["chat_template_kwargs"] = {"enable_thinking": False}
run("T1-thinkOff-300 ", b)

b = dict(T2)
b["max_tokens"] = 2048
run("T2-bud2048      ", b)

b = dict(T2)
b["chat_template_kwargs"] = {"enable_thinking": False}
run("T2-thinkOff-400 ", b)
print("M1_DISCRIMINATOR_DONE", flush=True)
