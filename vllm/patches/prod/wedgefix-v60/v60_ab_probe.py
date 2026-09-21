#!/usr/bin/env python3
"""v60_ab_probe.py — identical tok/s probe for the barrier A/B.

Three shapes, thinking off, greedy-ish (temp 0.6), direct :8000:
  S1 short decode   : tiny prompt, 512 out
  L2 ~2k context    : 2k-token prompt, 512 out  (v33-era comparison point)
  C4 concurrency    : 4 x 2k prompts, 512 out, aggregate
Prints one TOKS line per shape per repeat for easy grep.
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:8000"
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def make_req(prompt_tokens_hint, max_tokens):
    if prompt_tokens_hint <= 64:
        content = "Count from 1 to 300 with short commentary."
    else:
        blob = ("The lane serves qwen3.8-27b on Xe2 TP2 with e4m3 KV and "
                "MTP x4 speculative decoding. Watchdog relaunches take 140s. ")
        n = max(1, prompt_tokens_hint // 30)
        content = (blob * n)[: prompt_tokens_hint * 4]
    return json.dumps({
        "model": "qwen3.8-27b-fp8", "max_tokens": max_tokens, "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": content}],
    }).encode()


def one(prompt_hint, max_tokens):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=make_req(prompt_hint, max_tokens),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    dt = time.time() - t0
    ct = d.get("usage", {}).get("completion_tokens", 0)
    pt = d.get("usage", {}).get("prompt_tokens", 0)
    return ct, dt, pt


def conc4(prompt_hint, max_tokens):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        rs = list(ex.map(lambda _: one(prompt_hint, max_tokens), range(4)))
    dt = time.time() - t0
    tot = sum(r[0] for r in rs)
    return tot, dt


for rep in range(REPEATS):
    ct, dt, pt = one(32, 512)
    print(f"TOKS shape=S1_short rep={rep} prompt_tok={pt} out={ct} wall={dt:.1f}s tok_s={ct/dt:.1f}", flush=True)
    ct, dt, pt = one(2000, 512)
    print(f"TOKS shape=L2_ctx2k rep={rep} prompt_tok={pt} out={ct} wall={dt:.1f}s tok_s={ct/dt:.1f}", flush=True)
    tot, dt = conc4(2000, 512)
    print(f"TOKS shape=C4_conc4 rep={rep} out_total={tot} wall={dt:.1f}s agg_tok_s={tot/dt:.1f}", flush=True)
print("PROBE_DONE", flush=True)
