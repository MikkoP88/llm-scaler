#!/usr/bin/env python3
"""rerun_corrupt.py — quantify mid-answer degeneration on the standing-best
engine. Repeats the prompt that corrupted + variants; flags !!!-style runs."""
import hashlib
import json
import re
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-27b-fp8"

PROMPTS = [
    ("math1", "A train travels 60 km in 45 minutes. What is its speed in km/h? Show the calculation briefly."),
    ("math2", "If a shirt costs 40 euros and is discounted by 25 percent, what is the final price? Show the steps."),
    ("math3", "Solve for x: 3x + 7 = 22. Show your steps briefly."),
    ("logic1", "All cats are animals. Tom is a cat. What follows? Answer in one sentence."),
    ("math4", "Convert 90 minutes to hours and compute how far a car at 80 km/h travels. Brief."),
]

BAD_RE = re.compile(r"([!@#$%^&*])\1{9,}")  # 10+ same punctuation in a row


def chat(p: str, max_tokens: int = 200):
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
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    bad = tot = 0
    for r in range(rounds):
        for tag, p in PROMPTS:
            tot += 1
            try:
                txt, dt = chat(p)
            except Exception as e:  # noqa: BLE001
                print(f"[r{r} {tag}] FAIL {type(e).__name__}: {e}", flush=True)
                bad += 1
                continue
            m = BAD_RE.search(txt)
            md5 = hashlib.md5(txt.encode()).hexdigest()[:12]
            if m:
                bad += 1
                i = m.start()
                print(f"[r{r} {tag}] DEGENERATE md5={md5} wall={dt:.2f}s "
                      f"at ...{txt[max(0, i - 40):i]!r} + {txt[i:i + 20]!r}...",
                      flush=True)
            else:
                print(f"[r{r} {tag}] clean md5={md5} wall={dt:.2f}s "
                      f"{txt.strip()[:80]!r}", flush=True)
    print(f"RERUN_DONE bad={bad}/{tot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
