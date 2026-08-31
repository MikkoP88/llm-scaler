#!/usr/bin/env python3
"""conc_n_probe.py N ROUNDS — N concurrent identical temp-0 requests, R rounds.

Padding-leakage discriminator: captured piece = smallest size >= rows.
  k1: rows = 2*bs  (bs=2 -> 4 = exact piece-4; bs=3 -> 6 -> PADDED piece-8)
  k4: rows = 5*bs  (bs=1 -> 5 -> PADDED piece-8; bs=2 -> 10 -> eager fallback)
Prediction: exact-fit replays are clean; padded replays are corrupt.
"""
import json
import sys
import threading
import urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
URL = "http://localhost:8000/v1/completions"
BODY = json.dumps({
    "model": "qwen3.8-27b-fp8",
    "prompt": "The capital of France is",
    "max_tokens": 24, "temperature": 0,
}).encode()


def one(out, i):
    req = urllib.request.Request(URL, data=BODY,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out[i] = json.load(r)["choices"][0]["text"][:60]


for r in range(ROUNDS):
    out = [None] * N
    ts = [threading.Thread(target=one, args=(out, i)) for i in range(N)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    uniq = len(set(out))
    print(f"round {r+1}: distinct={uniq}/{N}")
    for i, t in enumerate(out):
        print(f"  [{i}] {t!r}")
