#!/usr/bin/env python3
"""P85 agentic soak — mirrors the live crash-exposure traffic shape.

3 concurrent agentic workers: deep histories (45k seed, growing to ~80k,
then held deep by trimming), full-history resend every turn, sequential
deep spec decode, unbounded rounds until the duration elapses.

This is the shape that killed the lane 6x on 2026-10-07 (exposure-bound
guc-ccs class, 13-41 min). Gate: zero resets for >= 3x worst exposure.

Usage: python3 p85_soak.py [minutes]   (default 120)
Output: per-turn lines to stdout (tee to a file), SOAK_DONE marker.
"""
import json
import sys
import threading
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen3.8-27b-fp8"
WORKERS = 3
SEED_PARAS = 45          # ~1k tokens each -> ~45k seed history
GROW_PARAS = 2           # filler injected every GROW_EVERY turns
GROW_EVERY = 6
MAX_PARAS = 78           # hold ~78k deep; trim oldest when exceeded
TRIM_TO = 66
MAX_TOKENS = 400

FILLER = (
    "System diagnostics continue across the deployment pipeline. "
    "The engineering ledger records throughput envelopes, acceptance "
    "telemetry, boundary carry events, and scheduler metadata drift "
    "across long horizons. Each paragraph extends the conversation "
    "context so the decode path operates at production depth. "
    "Analysis proceeds methodically through cache slots, state "
    "boundaries, and rollout columns while the engine sustains "
    "speculative multi-token prediction under concurrent load. "
)

PROMPTS = [
    "Summarize the current state of the deployment in two sentences.",
    "List the top risks you see and rate each low/medium/high.",
    "Propose the next concrete step and justify it briefly.",
    "What evidence would confirm or refute your last claim?",
    "Give a one-paragraph status update for the log.",
]


def turn(history: str, out: list) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": history}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    dt = time.time() - t0
    u = resp.get("usage", {})
    out.append((dt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))
    return resp["choices"][0]["message"]["content"]


def worker(wid: int, minutes: float, stop_at: float):
    paras = [FILLER] * SEED_PARAS
    n = 0
    while time.time() < stop_at:
        n += 1
        if n % GROW_EVERY == 0:
            paras.extend([FILLER] * GROW_PARAS)
            if len(paras) > MAX_PARAS:
                paras = paras[-TRIM_TO:]
        ask = PROMPTS[n % len(PROMPTS)]
        history = "\n\n".join(paras) + "\n\n" + ask
        out = []
        try:
            reply = turn(history, out)
        except Exception as e:  # noqa: BLE001 — soak must survive and log
            print(f"W{wid} t{n} ERROR {type(e).__name__}: {e}", flush=True)
            time.sleep(5)
            continue
        dt, pt, ct = out[0]
        paras.append(f"Assistant previously said: {reply[:400]}")
        tps = ct / dt if dt > 0 else 0.0
        print(f"W{wid} t{n} {dt:6.1f}s ptok={pt} ctok={ct} "
              f"tps={tps:5.1f} paras={len(paras)}", flush=True)


def main() -> int:
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    stop_at = time.time() + minutes * 60
    print(f"SOAK_START workers={WORKERS} minutes={minutes}", flush=True)
    threads = [threading.Thread(target=worker, args=(i, minutes, stop_at),
                                daemon=True) for i in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("SOAK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
