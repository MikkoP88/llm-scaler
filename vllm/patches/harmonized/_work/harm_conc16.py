#!/usr/bin/env python3
"""harm_conc16 — 16 concurrent identical temp-0 requests (P1, max_tokens 64).
Coherence gate: all 16 hashes equal (deterministic under concurrency)."""
import hashlib
import json
import threading
import urllib.request

URL = "http://localhost:8000/v1/completions"
P1 = "Explain the role of Hadamard rotations in KV cache quantization."

results = [None] * 16
errors = []


def worker(i):
    try:
        body = {"model": "qwen3.8-27b-fp8", "prompt": P1,
                "max_tokens": 64, "temperature": 0}
        req = urllib.request.Request(
            URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=300))
        results[i] = hashlib.sha256(
            r["choices"][0]["text"].encode()).hexdigest()[:12]
    except Exception as e:  # noqa: BLE001
        errors.append(f"w{i}: {e}")


threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
for t in threads:
    t.start()
for t in threads:
    t.join()

if errors:
    print("CONC16 ERRORS:", *errors, sep="\n  ")
ok = [r for r in results if r]
distinct = sorted(set(ok))
print(f"CONC16: {len(ok)}/16 ok, {len(distinct)} distinct "
      f"-> {'COHERENT' if len(distinct) == 1 and not errors else 'DIVERGENT'}")
for d in distinct:
    print(f"  distinct: {d} x{ok.count(d)}")
