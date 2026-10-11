#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 P0 GPU smoke client (against fn8-g1 :8319).

Waits for health, then banks greedy completions + timings as the P0
reference set for the later P2 parity gate.  Writes /stage/bank/g1_smoke.json.
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8319"
BANK = "/stage/bank/g1_smoke.json"

PROMPTS = [
    ("fib50", "The 50th Fibonacci number is", 12586269025),
    ("gcd", "The greatest common divisor of 1071 and 462 is", 21),
    ("fib20", "The 20th Fibonacci number is", 6765),
    ("exec1", "In Python, print(sys.argv[1:]) run as script.py a b c outputs",
     "['a', 'b', 'c']"),
]


def req(path, payload=None, timeout=14400):
    url = BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_health(deadline_s=14400):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    print("HEALTH_200 after", round(time.time() - t0, 1), "s")
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(10)
    raise SystemExit("health deadline exceeded")


def main():
    ready = wait_health()
    bank = {"ready_s": round(ready, 1), "ts": time.time(), "legs": []}
    ok = 0
    for name, prompt, expect in PROMPTS:
        t0 = time.time()
        out = req("/v1/completions", {
            "model": "flash-next-fp8",
            "prompt": prompt,
            "max_tokens": 48,
            "temperature": 0.0,
        })
        dt = time.time() - t0
        text = out["choices"][0]["text"]
        # fib50 banked "12,586,269,025" (comma-grouped) on leg-1 — the model
        # was RIGHT, the substring check wasn't.  Also compare comma-stripped
        # so number formatting can never fail a leg.
        passed = str(expect) in text or str(expect) in text.replace(",", "")
        ok += passed
        leg = {"name": name, "prompt": prompt, "expect": str(expect),
               "text": text, "passed": passed, "latency_s": round(dt, 2),
               "usage": out.get("usage", {})}
        bank["legs"].append(leg)
        print(f"LEG {name} {'PASS' if passed else 'FAIL'} {dt:.2f}s :: {text[:120]!r}", flush=True)
    bank["pass"] = f"{ok}/{len(PROMPTS)}"
    with open(BANK, "w") as f:
        json.dump(bank, f, indent=1)
    print("BANKED", BANK, "pass", bank["pass"])
    if ok != len(PROMPTS):
        raise SystemExit(1)
    print("FN8_G1_SMOKE_OK")


if __name__ == "__main__":
    main()
