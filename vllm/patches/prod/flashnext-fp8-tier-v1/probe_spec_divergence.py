#!/usr/bin/env python3
"""Deterministic temp-0 probe for spec-vs-nospec divergence (in-container).

python3 /stage/probe_spec_divergence.py <legname>

At temperature 0, a CORRECT spec implementation must produce the SAME
token stream as no-spec (verify accepts only the target argmax; any
mismatch is rejected and replaced by the target token).  So the FIRST
divergent token between legs localizes the verify/state-continuation
bug.  2026-10-12: count-to-300 showed spec cycling ("Need ensure no
extra text?") vs nospec clean on identical prompts.

Banks content+reasoning+usage to /stage/probe_<leg>.json.
"""
import json
import sys
import time

import requests

leg = sys.argv[1] if len(sys.argv) > 1 else "leg"
BASE = "http://127.0.0.1:8321/v1"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer sk-d"}

PROBES = [
    ("count50", "Count from 1 to 50, one number per line, then write DONE.", 400),
    ("essay", "Write a 150-word essay about rivers.", 700),
]

out = {}
for name, q, mt in PROBES:
    t0 = time.time()
    r = requests.post(BASE + "/chat/completions", headers=HDR, json={
        "model": "flash-next-fp8", "messages": [{"role": "user", "content": q}],
        "temperature": 0, "max_tokens": mt}, timeout=900)
    try:
        j = r.json()
        ch = j["choices"][0]
        msg = ch["message"]
        out[name] = {
            "http": r.status_code,
            "finish": ch.get("finish_reason"),
            "ct": j["usage"]["completion_tokens"],
            "pt": j["usage"]["prompt_tokens"],
            "sec": round(time.time() - t0, 1),
            "reasoning": msg.get("reasoning_content") or "",
            "content": msg.get("content") or "",
        }
        print(f"{leg}/{name}: HTTP{r.status_code} finish={ch.get('finish_reason')} "
              f"ct={j['usage']['completion_tokens']} {time.time()-t0:.0f}s")
        print(f"  tail={((msg.get('content') or '') + (msg.get('reasoning_content') or ''))[-140:]!r}")
    except Exception as e:
        out[name] = {"http": r.status_code, "error": f"{type(e).__name__}: {e}",
                     "body": r.text[:300]}
        print(f"{leg}/{name}: HTTP{r.status_code} ERROR {e}")

with open(f"/stage/probe_{leg}.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"BANKED /stage/probe_{leg}.json")
