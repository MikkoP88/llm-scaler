#!/usr/bin/env python3
"""spot_v130.py — perf-v130 P56 inertness spots against the direct lane.

Modes: solo (1 request) / serial24 (24 sequential requests). Asserts
HTTP 200 + non-empty content per request; prints per-request JSON rows
and a caps line SPOT_<mode> ok=N/N wall=... PASS|FAIL. Stdlib only
(host python3 has no torch; urllib suffices).
"""
import json
import sys
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
HDR = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-dummy",
}
BODY = json.dumps({
    "model": "qwen3.8-27b-fp8",
    "messages": [
        {"role": "user", "content": "Reply with exactly the word OK."}
    ],
    "max_tokens": 64,
    "temperature": 0.6,
}).encode()


def one(i):
    t0 = time.time()
    req = urllib.request.Request(URL, data=BODY, headers=HDR)
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read().decode())
    dt = time.time() - t0
    ch = out["choices"][0]
    txt = (ch["message"].get("content") or "").strip()
    fin = ch.get("finish_reason")
    usage = out.get("usage", {})
    ok = len(txt) > 0
    print(json.dumps({
        "i": i,
        "ok": bool(ok),
        "finish": fin,
        "ms": round(dt * 1000),
        "ptok": usage.get("prompt_tokens"),
        "ctok": usage.get("completion_tokens"),
        "head": txt[:40],
    }), flush=True)
    return bool(ok)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "solo"
    n = 1 if mode == "solo" else 24
    t0 = time.time()
    oks = sum(one(i) for i in range(n))
    wall = time.time() - t0
    verdict = "PASS" if oks == n else "FAIL"
    print(
        "SPOT_%s ok=%d/%d wall=%.1fs %s" % (mode, oks, n, wall, verdict)
    )
    return 0 if oks == n else 1


if __name__ == "__main__":
    sys.exit(main())
