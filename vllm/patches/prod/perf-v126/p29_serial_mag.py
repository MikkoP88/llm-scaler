#!/usr/bin/env python3
"""p29_serial_mag.py — v126 P29H v9 helper: SERIAL-ONLY magnitude probe.

The full P23F battery kills the fp16+poolpath lane in its CONCURRENT
phase (wedge under bridge + nsd>1 load — separate defect, under
adjudication), before the serial phase ever runs. Serial requests are
nsd=1 spec decodes -> the fp16 lane routes them through the NATIVE
instrumented branch (E2B keeps the certified nsd==1 gate), which is
exactly where the v9 running-max scalars live.

This probe runs strictly sequential requests with generous max_tokens
(state accumulates over the walk) so the FLUSH prints
(pmx/premx/bamx) and the ring dump carry the REAL fp16-pool state
magnitude — the ground-truth number for the e4m3/e5m2 range verdict.
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"
N = 24
MAXTOK = 512

PROMPTS = [
    "Count from 1 to 40 slowly, one number per line.",
    "Write a 300-word story about a lighthouse keeper.",
    "List the first 30 prime numbers.",
    "Explain how a transformer attention head works, in detail.",
    "Recite the alphabet backwards, then forwards, twice.",
    "Describe the water cycle in exhaustive detail.",
    "Name 35 countries and their capitals.",
    "Write detailed instructions for brewing coffee, step by step.",
    "Summarize the plot of Moby-Dick chapter by chapter.",
    "List 40 animals grouped by habitat.",
    "Explain recursion with five worked examples.",
    "Write a long technical description of a CPU pipeline.",
    "Enumerate the periodic table groups and their properties.",
    "Give a detailed travel itinerary for a week in Kyoto.",
    "Explain the Krebs cycle step by step.",
    "List 40 common Linux commands with one-line descriptions.",
    "Write a dialogue between a pilot and a control tower.",
    "Describe the history of the Roman Republic in detail.",
    "Explain photosynthesis at three levels of depth.",
    "List the Fibonacci numbers up to F(30) with the recurrence.",
    "Write a long poem about winter.",
    "Explain database normalization through third normal form.",
    "Describe every step of making sourdough bread.",
    "Review the differences between TCP and UDP in depth.",
]


def one(i: int) -> str:
    body = {
        "model": MODEL,
        "max_tokens": MAXTOK,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    txt = (out["choices"][0]["message"].get("content") or "").strip()
    return "req%02d %5.1fs len=%4d head=%r" % (
        i, time.time() - t0, len(txt), txt[:40])


def main() -> int:
    print("=== P29 serial magnitude probe (N=%d maxtok=%d) ===" % (N, MAXTOK))
    ok = 0
    for i in range(N):
        try:
            line = one(i)
            ok += 1
        except Exception as e:  # noqa: BLE001
            line = "req%02d FAIL:%r" % (i, e)
        print(line, flush=True)
    print("SERIAL_MAG_DONE ok=%d/%d" % (ok, N))
    return 0


if __name__ == "__main__":
    sys.exit(main())
