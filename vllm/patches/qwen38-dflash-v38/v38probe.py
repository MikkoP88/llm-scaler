#!/usr/bin/env python3
"""v38 probe: localize the #18 fp8 bimodality mechanism on the pinned arm
(VLLM_XPU_FA2_EXACT_SPLITS=32 live).

Ladder:
  A) P1 x6 back-to-back — decode determinism given identical state
     (runs 2..6 are prefix-cache hits: same KV bits, same batch=1).
  B) P1 interleaved with short unique distractors — cached-decode
     sensitivity to scheduler/allocation state (no eviction: KV is
     near-empty, distractors are 12 tokens).
  C) full f8ref 3-prompt x2 — in-session bimodality reproduction.
Every run records first-divergence CHAR index vs A0 (with context) —
the divergence onset localizes the mechanism (near a specific generated
position => boundary effect; random => kernel-level noise).
"""
import json, urllib.request, hashlib, time

URL = "http://localhost:8000/v1/completions"
P1 = "Explain the role of Hadamard rotations in KV cache quantization."
DIST = "List three prime numbers greater than {n}."


def gen(prompt, mt):
    body = {"model": "qwen3.8-27b-fp8", "prompt": prompt,
            "max_tokens": mt, "temperature": 0}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["choices"][0]["text"]


def h(t):
    return hashlib.sha256(t.encode()).hexdigest()[:12]


def firstdiff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, a[max(0, i - 30):i + 30], b[max(0, i - 30):i + 30]
    if len(a) != len(b):
        return n, "<len-diverge>", "<len-diverge>"
    return -1, "", ""


print("=== A: P1 x6 back-to-back (runs 2+ = prefix-cache hits) ===", flush=True)
runs = []
for i in range(6):
    t0 = time.time()
    txt = gen(P1, 160)
    runs.append(txt)
    d, ca, cb = (-1, "", "") if i == 0 else firstdiff(runs[0], txt)
    print(f"A{i}: hash={h(txt)} firstdiff_vs_A0={d} ({time.time()-t0:.1f}s)",
          flush=True)
    if d >= 0:
        print(f"    A0: ...{ca!r}", flush=True)
        print(f"    X : ...{cb!r}", flush=True)

print("=== B: P1 x4 interleaved with unique 12-token distractors ===", flush=True)
for i in range(4):
    gen(DIST.format(n=100 + i * 7), 12)
    txt = gen(P1, 160)
    d, ca, cb = firstdiff(runs[0], txt)
    print(f"B{i}: hash={h(txt)} firstdiff_vs_A0={d}", flush=True)
    if d >= 0:
        print(f"    A0: ...{ca!r}", flush=True)
        print(f"    X : ...{cb!r}", flush=True)

print("=== C: f8ref reproduction (3 prompts x 2 reps) ===", flush=True)
PROMPTS = [P1,
           "Write a Python function that computes the Nth Fibonacci number iteratively.",
           "Summarize the plot of Moby-Dick in five sentences."]
for rep in range(2):
    hs = [h(gen(p, 160)) for p in PROMPTS]
    print(f"C rep{rep}: {hs}", flush=True)
