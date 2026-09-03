#!/usr/bin/env python3
"""harm_probe10 — P1 x10, temp 0, identical params to the v38/v37 certified
probe batteries (prompt + max_tokens=160 + temperature=0 only). Prints
sha256[:12] per run + distinct count. Cross-image comparable: prod 4bit
ref = 0ce080630035, fp8 fixed = 91d489262cb5."""
import hashlib
import json
import time
import urllib.request

URL = "http://localhost:8000/v1/completions"
P1 = "Explain the role of Hadamard rotations in KV cache quantization."


def gen(prompt, mt):
    body = {"model": "qwen3.8-27b-fp8", "prompt": prompt,
            "max_tokens": mt, "temperature": 0}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["choices"][0]["text"]


hashes = []
for i in range(10):
    t0 = time.time()
    txt = gen(P1, 160)
    h = hashlib.sha256(txt.encode()).hexdigest()[:12]
    hashes.append(h)
    print(f"P{i+1:02d}: {h}  ({time.time()-t0:.1f}s)", flush=True)

distinct = sorted(set(hashes))
print(f"PROBE10: {len(hashes) - len(distinct)}/10 repeat, "
      f"{len(distinct)} distinct -> {'STABLE' if len(distinct) == 1 else 'UNSTABLE'}")
for d in distinct:
    print(f"  distinct: {d} x{hashes.count(d)}")
