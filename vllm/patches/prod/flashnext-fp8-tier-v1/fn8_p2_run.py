#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 P2 window runner (against the tiered serve :8321).

Flow: wait health -> PARITY legs (P0 semantic gate, same 4 prompts) ->
PERF legs (corpus-shaped gens; P0 reference = 944.6 s/leg at p=127 c=32)
-> tier-telemetry readout (hits/misses/dummy share/H2D ms, the store's
own counters) -> verdict + bank JSON.  Env: FN8_P2_PORT (8321),
FN8_P2_TIER_DUMP (must match the serve's V1X_TIER_DUMP),
FN8_P2_BANK (/stage/bank/p2_run.json), FN8_P2_SKIP_PERF=1 to end after
parity (smoke mode).

Tier contract note: non-resident experts contribute EXACT ZERO (dummy
slot), so outputs are semantically — not token — identical to full
compute at high seed coverage.  A failed parity leg is a finding about
S/seed quality, not a bug class.
"""
import json
import os
import time
import urllib.request

BASE = f"http://127.0.0.1:{os.environ.get('FN8_P2_PORT', '8321')}"
TIER_DUMP = os.environ.get("FN8_P2_TIER_DUMP", "/stage/bank/tier_tp1.json")
BANK = os.environ.get("FN8_P2_BANK", "/stage/bank/p2_run.json")
P0_REF_LEG_S = 944.6  # corpus leg 1 (py_fib p=127 c=32), P0 pageable shape

PARITY = [
    ("fib50", "The 50th Fibonacci number is", 12586269025),
    ("gcd", "The greatest common divisor of 1071 and 462 is", 21),
    ("fib20", "The 20th Fibonacci number is", 6765),
    ("exec1", "In Python, print(sys.argv[1:]) run as script.py a b c outputs",
     "['a', 'b', 'c']"),
]

PERF = [
    ("perf_fib", "Write a Python function computing the nth Fibonacci "
     "number with memoization, then explain its complexity.", 127, 32),
    ("perf_sort", "Write a Python three-way quicksort with commentary "
     "on pivot selection.", 119, 32),
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


def read_tier():
    try:
        with open(TIER_DUMP) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 — readout must never kill the run
        return {"error": str(e)}


def tier_summary(tier):
    stores = tier.get("stores", {})
    if not stores:
        return {"stores": 0}
    agg = {k: 0 for k in ("dispatches", "dummy_dispatches", "hits", "misses",
                          "evictions", "h2d_copies", "h2d_bytes")}
    for rec in stores.values():
        for k in agg:
            agg[k] += rec.get(k, 0)
    live = agg["dispatches"] - agg["dummy_dispatches"]
    agg["stores"] = len(stores)
    agg["hit_rate"] = round(agg["hits"] / max(agg["hits"] + agg["misses"], 1), 4)
    agg["live_dispatches"] = live
    resident = [rec.get("resident", 0) for rec in stores.values()]
    agg["resident_min_max"] = [min(resident), max(resident)]
    return agg


def main():
    ready = wait_health()
    bank = {"ready_s": round(ready, 1), "ts": time.time(),
            "parity": [], "perf": [], "tier": None}

    # -- parity (P0 semantic gate) ----------------------------------------
    ok = 0
    for name, prompt, expect in PARITY:
        t0 = time.time()
        out = req("/v1/completions", {
            "model": "flash-next-fp8", "prompt": prompt,
            "max_tokens": 48, "temperature": 0.0,
        })
        dt = time.time() - t0
        text = out["choices"][0]["text"]
        passed = str(expect) in text or str(expect) in text.replace(",", "")
        ok += passed
        leg = {"name": name, "text": text, "passed": passed,
               "latency_s": round(dt, 2)}
        bank["parity"].append(leg)
        print(f"PARITY {name} {'PASS' if passed else 'FAIL'} {dt:.2f}s :: "
              f"{text[:100]!r}", flush=True)
    bank["parity_pass"] = f"{ok}/{len(PARITY)}"
    print("PARITY_SUMMARY", bank["parity_pass"])

    # -- perf (corpus-shaped; P0 reference comparison) ----------------------
    if os.environ.get("FN8_P2_SKIP_PERF") != "1":
        for name, prompt, ptok, ctok in PERF:
            t0 = time.time()
            out = req("/v1/completions", {
                "model": "flash-next-fp8", "prompt": prompt,
                "max_tokens": ctok, "temperature": 0.0,
            })
            dt = time.time() - t0
            ctot = out.get("usage", {}).get("completion_tokens", ctok)
            leg = {"name": name, "latency_s": round(dt, 2),
                   "completion_tokens": ctot,
                   "tok_per_s": round(ctok / dt, 3),
                   "speedup_vs_p0_ref": round(P0_REF_LEG_S / dt, 2)}
            bank["perf"].append(leg)
            print(f"PERF {name} {dt:.2f}s {ctok}ct "
                  f"{leg['tok_per_s']} tok/s x{leg['speedup_vs_p0_ref']} "
                  f"vs P0-ref", flush=True)

    # -- tier telemetry readout ---------------------------------------------
    tier = read_tier()
    bank["tier"] = tier
    summ = tier_summary(tier)
    bank["tier_summary"] = summ
    print("TIER", json.dumps(summ, sort_keys=True))
    if summ.get("hit_rate") is not None and summ["hit_rate"] < 0.75:
        print("TIER_WARNING hit_rate < 0.75 — raise FN8_TIER_SLOTS or "
              "improve seed")

    with open(BANK, "w") as f:
        json.dump(bank, f, indent=1)
    print("BANKED", BANK, "parity", bank["parity_pass"])
    if ok != len(PARITY):
        raise SystemExit(1)
    print("FN8_P2_RUN_OK")


if __name__ == "__main__":
    main()
