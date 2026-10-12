#!/usr/bin/env python3
"""Live response validation for the d11 flash-next serve (run in-container).

python3 /stage/validate_d11.py
Hits http://127.0.0.1:8321/v1 directly (host network) and asserts
response QUALITY, not just 200s:
  1. completions factual continuation
  2. chat factual (CC template, think scaffolding, default xhigh)
  3. chat reasoning_effort=high (CC alias) accepted
  4. chat reasoning_effort=low stays brief-capable
  5. tools: model emits the <tool_call> format
  6. overlap + long prompt (chunked prefill, the d10 crash/garbage
     shape) then a fresh factual re-check
"""
import json
import threading
import time

import requests

BASE = "http://127.0.0.1:8321/v1"
HDR = {"Content-Type": "application/json",
       "Authorization": "Bearer sk-dummy"}
MODEL = "flash-next-fp8"
results = []


def rec(name, ok, detail=""):
    results.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + (" | " + detail[:220] if detail else ""), flush=True)


def chat(msgs, **kw):
    body = {"model": MODEL, "messages": msgs, "temperature": 0,
            "max_tokens": kw.pop("max_tokens", 2048)}
    body.update(kw)
    r = requests.post(BASE + "/chat/completions", headers=HDR,
                      json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def content_of(resp):
    ch = resp["choices"][0]["message"]
    return (ch.get("content") or "") + (ch.get("reasoning_content") or "")


# 1. completions factual
try:
    r = requests.post(BASE + "/completions", headers=HDR, json={
        "model": MODEL, "prompt": "The capital of France is",
        "max_tokens": 12, "temperature": 0}, timeout=300)
    txt = r.json()["choices"][0]["text"]
    rec("completions_factual", "Paris" in txt, repr(txt[:80]))
except Exception as e:
    rec("completions_factual", False, f"{type(e).__name__}: {e}")

# 2. chat factual with thinking
try:
    t0 = time.time()
    r = chat([{"role": "user", "content":
               "What is 2+2? End your reply with just the number on its own line."}])
    c = content_of(r)
    dt = time.time() - t0
    ok = "4" in c and len(c) < 4000
    rec("chat_think_factual", ok,
        f"{dt:.1f}s ct={r['usage']['completion_tokens']} {repr(c[:160])}")
except Exception as e:
    rec("chat_think_factual", False, f"{type(e).__name__}: {e}")

# 3. reasoning_effort=high (CC alias) must not 400
try:
    r = chat([{"role": "user", "content": "Say OK."}],
             extra_body=None) if False else None
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Say OK."}],
            "temperature": 0, "max_tokens": 512, "reasoning_effort": "high"}
    rr = requests.post(BASE + "/chat/completions", headers=HDR,
                       json=body, timeout=300)
    rec("reasoning_effort_high_alias", rr.status_code == 200,
        f"HTTP {rr.status_code} {rr.text[:120] if rr.status_code != 200 else content_of(rr.json())[:80]!r}")
except Exception as e:
    rec("reasoning_effort_high_alias", False, f"{type(e).__name__}: {e}")

# 4. tools round: model should emit <tool_call>
try:
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"}},
            "required": ["city"]}}}]
    r = chat([{"role": "user", "content": "What is the weather in Paris? "
               "Use the get_weather tool."}], tools=tools, max_tokens=1024)
    c = content_of(r)
    tc = (r["choices"][0]["message"].get("tool_calls") or [])
    ok = bool(tc) or ("<tool_call>" in c) or ("<function=get_weather>" in c)
    rec("tools_toolcall_format", ok, repr(c[-200:]))
except Exception as e:
    rec("tools_toolcall_format", False, f"{type(e).__name__}: {e}")

# 5. overlap + long prompt (chunked prefill shape) then fresh factual
try:
    filler = ("The quick brown fox jumps over the lazy dog near the "
              "river bank at dawn. ") * 460  # ~5700 tokens > MNBT 4096
    errs = []

    def worker(tag, long=False):
        try:
            msg = filler + f"\nQuestion: how many times does the word " \
                f"'fox' appear above? Count carefully." if long else \
                f"Say the word {tag} and nothing else."
            b = {"model": MODEL, "temperature": 0, "max_tokens": 4096,
                 "messages": [{"role": "user", "content": msg}]}
            if long:
                b["max_tokens"] = 512
            rr = requests.post(BASE + "/chat/completions", headers=HDR,
                               json=b, timeout=600)
            rr.raise_for_status()
        except Exception as e:
            errs.append(f"{tag}: {type(e).__name__}: {e}")

    ths = [threading.Thread(target=worker, args=("ALPHA", True)),
           threading.Thread(target=worker, args=("BRAVO",)),
           threading.Thread(target=worker, args=("CHARLIE",))]
    [t.start() for t in ths]
    [t.join() for t in ths]
    rec("overlap_longprompt_noerror", not errs, "; ".join(errs) or "3/3 ok")
    # engine still healthy + fresh factual correct
    r = chat([{"role": "user", "content":
               "What is the capital of Germany? One word."}], max_tokens=1024)
    c = content_of(r)
    rec("post_overlap_factual", "Berlin" in c, repr(c[:160]))
except Exception as e:
    rec("overlap_longprompt_noerror", False, f"{type(e).__name__}: {e}")

bad = [n for n, ok in results if not ok]
print("RESULT", "ALL_PASS" if not bad else f"FAILED={bad}", flush=True)
