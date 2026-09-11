#!/usr/bin/env python3
"""bench_genspeed.py — generation-throughput A/B probe (v1.2.7 vs v1.2.8).

Pure decode speed on a WARM engine: N concurrent streams, short prompt
(28 tok), greedy, fixed max_tokens. Reports per-stream tps and aggregate
tps. Run 3 rounds, keep each round's aggregate; the middle behavior is
what matters (round 1 still settles caches).

Usage: bench_genspeed.py [nstreams] [max_tokens] [rounds]
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"

PROMPTS = [
    "Write a python function that merges two sorted lists.",
    "Explain the difference between latency and throughput.",
    "Write a bash one-liner that counts lines in *.py files.",
    "Describe the water cycle in one paragraph.",
    "Write a haiku about GPU kernels.",
    "List five uses for a hash map.",
]


def one(i: int, max_tokens: int, out: list) -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            j = json.loads(r.read())
        ct = (j.get("usage") or {}).get("completion_tokens", -1)
    except Exception as e:  # noqa: BLE001
        ct = -1
        print(f"  stream{i}: ERROR {type(e).__name__}: {e}", flush=True)
    dt = time.time() - t0
    out.append((ct, dt))


def main() -> int:
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    mt = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    for rd in range(rounds):
        res: list = []
        ts = [threading.Thread(target=one, args=(i, mt, res))
              for i in range(ns)]
        t0 = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.time() - t0
        tot = sum(ct for ct, _ in res)
        per = " ".join(f"{ct/max(dt,1e-9):.1f}" for ct, dt in res)
        print(f"GENSPEED r{rd}: {ns}x{mt} total={tot} tok wall={wall:.1f}s "
              f"aggregate={tot/wall:.2f} tok/s per-stream=[{per}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
