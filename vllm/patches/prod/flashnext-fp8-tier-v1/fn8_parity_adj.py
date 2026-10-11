#!/usr/bin/env python3
"""fn8 TP1 parity adjudication — numeric-flip vs corruption (2026-10-11).

p2_run.json TP1: fib50 PASS + fib20 PASS (long exact chains through the
store) but gcd 21->33; baseline g1 smoke passed gcd.  Discriminator:
a temp-0 arithmetic battery on the SAME live engine.  Systematic weight
corruption (slot-map/eviction/scale bug) breaks coherence and most exact
answers; a GEMM-tiling numeric flip misses only borderline multi-digit
reasoning.  Expect >=4/6 exact incl. a DIFFERENT gcd pair.

Run: docker exec fn8-p2 python3 /stage/fn8_parity_adj.py
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8321/v1/completions"

LEGS = [
    ("gcd_other",  "The greatest common divisor of 252 and 105 is", "21"),
    ("gcd_orig",   "The greatest common divisor of 1071 and 462 is", "21"),
    ("gcd_third",  "The greatest common divisor of 48 and 180 is", "12"),
    ("fib12",      "The 12th Fibonacci number is", "144"),
    ("mul",        "17 times 23 is", "391"),
    ("pow",        "The value of 2 to the power of 10 is", "1024"),
]

ok = 0
for name, prompt, expect in LEGS:
    body = json.dumps({
        "model": "flash-next-fp8",
        "prompt": prompt,
        "max_tokens": 48,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["text"]
    dt = time.time() - t0
    hit = expect in text or expect in text.replace(",", "")
    ok += hit
    print(f"ADJ {name} {'HIT ' if hit else 'MISS'} expect={expect} "
          f"{dt:.2f}s :: {text[:110]!r}")

print(f"ADJ_SUMMARY {ok}/{len(LEGS)}")
verdict = "NUMERIC_FLIP_CLASS" if ok >= 4 else "CORRUPTION_SUSPECT"
print("ADJ_VERDICT", verdict)
