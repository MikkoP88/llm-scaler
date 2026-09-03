#!/usr/bin/env python3
"""v38 long-gen stability: cross the block-boundary hot zone (511/512/1024)
with 700-token greedy generations x4 + f8ref x3. The ESIMD race was worst
exactly at these lengths (969/998 @511); the rerouted vxk FA2 path must be
stable through them."""
import json, urllib.request, hashlib

URL = "http://localhost:8000/v1/completions"
P1 = "Explain the role of Hadamard rotations in KV cache quantization."


def gen(prompt, mt):
    body = {"model": "qwen3.8-27b-fp8", "prompt": prompt,
            "max_tokens": mt, "temperature": 0}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["text"]


def h(t):
    return hashlib.sha256(t.encode()).hexdigest()[:12]


print("=== L: 700-token generations x4 (crosses 512/1024 KV boundaries) ===",
      flush=True)
ref = None
for i in range(4):
    txt = gen(P1, 700)
    hh = h(txt)
    if ref is None:
        ref = txt
        print(f"L{i}: hash={hh} len={len(txt)} (ref)", flush=True)
    else:
        fd = next((k for k in range(min(len(ref), len(txt)))
                   if ref[k] != txt[k]), -1)
        same = "SAME" if hh == h(ref) else f"firstdiff={fd}"
        print(f"L{i}: hash={hh} len={len(txt)} vs_L0={same}", flush=True)

PROMPTS = [P1,
           "Write a Python function that computes the Nth Fibonacci number iteratively.",
           "Summarize the plot of Moby-Dick in five sentences."]
print("=== C: f8ref x3 ===", flush=True)
for rep in range(3):
    print(f"C rep{rep}: {[h(gen(p, 160)) for p in PROMPTS]}", flush=True)
