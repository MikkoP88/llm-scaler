#!/usr/bin/env python3
"""p29s_cadence.py — v126 P29S: per-token cadence probe for the
ship-posture speed-matrix adjudication.

The 09-30 matrix (bench_genspeed.py, short-task prompts, max_tokens=512,
thinking kwarg ABSENT) read solo 26.8-31.0 tok/s vs the Run-7n baseline
75.06/74.89 — while a 4-contended stream hit 63 tok/s, which solo must
dominate. Suspect: a fixed per-request overhead (TTFT) or a decode-cadence
change, not a uniform decode slowdown.

This probe streams ONE solo request per leg and records:
  - TTFT (first content token)
  - inter-token gap histogram (p50/p90/max) after warmup
  - decode-only tok/s (tokens after the 10th / time from 10th to last)
Legs:
  A. exact Run-7n methodology — "Count from 1 to 1000.", 256 max_tokens,
     enable_thinking=False  (baseline-comparable)
  B. today's methodology    — bench_genspeed prompt 0, 512 max_tokens,
     NO thinking kwarg     (reproduces the 31 tok/s number if real)
  C. count prompt, thinking OFF, 512 max_tokens (cadence at longer walk)
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-27b-fp8"


def stream_leg(tag: str, prompt: str, max_tokens: int,
               thinking_off: bool) -> None:
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking_off:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    stamps = []
    ntok = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                j = json.loads(payload)
            except Exception:
                continue
            choices = j.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                ntok += 1
                stamps.append(time.time())
                if ttft is None:
                    ttft = time.time() - t0
    if not stamps:
        print(f"[{tag}] NO_TOKENS")
        return
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    gaps_sorted = sorted(gaps)
    p50 = gaps_sorted[len(gaps) // 2]
    p90 = gaps_sorted[int(len(gaps) * 0.9)]
    dec_t0 = stamps[10] if len(stamps) > 12 else stamps[0]
    dec_n = ntok - (11 if len(stamps) > 12 else 1)
    dec_r = dec_n / max(stamps[-1] - dec_t0, 1e-9)
    slow = sum(1 for g in gaps if g > 0.25)
    print(
        f"[{tag}] ntok={ntok} ttft={ttft:.2f}s total={stamps[-1]-t0:.2f}s "
        f"gap_p50={p50*1000:.0f}ms p90={p90*1000:.0f}ms "
        f"max={max(gaps)*1000:.0f}ms gaps>250ms={slow} "
        f"decode_only={dec_r:.1f} tok/s",
        flush=True)


def main() -> int:
    print("=== P29S cadence probe ===", flush=True)
    stream_leg("A run7n-style count256 thinkOFF",
               "Count from 1 to 1000.", 256, True)
    stream_leg("B matrix-style prompt0-512 thinkDEFAULT",
               "Write a python function that merges two sorted lists.",
               512, False)
    stream_leg("C count512 thinkOFF",
               "Count from 1 to 1000.", 512, True)
    print("P29S_CADENCE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
