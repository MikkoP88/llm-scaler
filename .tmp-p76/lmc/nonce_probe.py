#!/usr/bin/env python3
"""nonce_probe.py — degeneration isolation with unique-nonce prompts.
Each prompt embeds a fresh nonce => no prefix-cache or LMCache reuse possible;
any corruption is decode-time, not retrieval-time."""
import json
import re
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-27b-fp8"
BAD_RE = re.compile(r"([!@#$%^&*])\1{9,}")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def chat(p: str, max_tokens: int = 150):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": p}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"], time.perf_counter() - t0


def main() -> int:
    bad = 0
    for i in range(N):
        nonce = f"{i}-{int(time.time() * 1000) % 1000000}"
        p = (f"[ref {nonce}] A train travels {60 + i} km in 45 minutes. "
             f"What is its speed in km/h? Show the calculation briefly.")
        try:
            txt, dt = chat(p)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] FAIL {type(e).__name__}: {e}", flush=True)
            bad += 1
            continue
        m = BAD_RE.search(txt)
        if m:
            bad += 1
            j = m.start()
            print(f"[{i}] DEGENERATE wall={dt:.2f}s at "
                  f"{txt[max(0, j - 30):j]!r} + {txt[j:j + 15]!r}", flush=True)
        else:
            print(f"[{i}] clean wall={dt:.2f}s {txt.strip()[:60]!r}", flush=True)
    print(f"NONCE_DONE bad={bad}/{N}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
