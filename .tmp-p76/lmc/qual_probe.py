#!/usr/bin/env python3
"""qual_probe.py — output-quality probe for the standing-best engine
(kv fp8_e4m3 + mamba float16 + LMCache local, ratio 4).
Sends diverse temp-0 prompts, prints outputs for review, checks
determinism (same prompt twice -> md5 match), flags empty/short/nan-ish."""
import hashlib
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-27b-fp8"

PROMPTS = [
    ("factual", "Name the capital of Australia and its approximate population, one sentence."),
    ("reasoning", "A train travels 60 km in 45 minutes. What is its speed in km/h? Show the calculation briefly."),
    ("code", "Write a Python function is_prime(n) with early exits, no comments."),
    ("recall", "Summarize in one sentence what the Pythagorean theorem states."),
    ("longform", "Write exactly three sentences about the water cycle."),
    ("multiling", "Antworte mit einem Satz: Was ist die Hauptstadt von Finnland?"),
]


def chat(p: str, max_tokens: int = 200):
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": p}],
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    txt = d["choices"][0]["message"]["content"]
    return txt, time.perf_counter() - t0, d.get("usage", {})


def main() -> int:
    print("=== QUAL PROBE (standing best: fp16 mamba + LMCache ratio 4) ===",
          flush=True)
    bad = 0
    for tag, p in PROMPTS:
        try:
            txt, dt, usage = chat(p)
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] FAIL {type(e).__name__}: {e}", flush=True)
            bad += 1
            continue
        s = txt.strip()
        flags = []
        if not s:
            flags.append("EMPTY")
        if len(s) < 5:
            flags.append("SHORT")
        low = s.lower()
        if low in ("nan", "none", "null") or low.startswith("nan"):
            flags.append("NAN")
        print(f"\n[{tag}] wall={dt:.2f}s ctok={usage.get('completion_tokens')} "
              f"{'FLAGS=' + ','.join(flags) if flags else 'OK'}", flush=True)
        print("  " + s.replace("\n", "\n  ")[:600], flush=True)
        if flags:
            bad += 1

    # determinism: same prompt twice at temp 0 (fp16 numerics + spec decode)
    p = PROMPTS[1][1]
    a, _, _ = chat(p)
    b, _, _ = chat(p)
    ma = hashlib.md5(a.encode()).hexdigest()
    mb = hashlib.md5(b.encode()).hexdigest()
    det = "MATCH" if ma == mb else "DIVERGE"
    print(f"\nDETERMINISM: {ma[:12]} vs {mb[:12]} -> {det}", flush=True)
    print(f"QUAL_PROBE_DONE bad={bad} det={det}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
