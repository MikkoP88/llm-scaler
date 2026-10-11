#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 P5 window runner (against the tiered serve :8321).

Two-corpus hit-rate legs with BEFORE/MID/AFTER tier-telemetry deltas —
the store counters are lifetime-cumulative, so every leg reads the dump
at its boundaries and works on deltas (racy mid-flush reads retried).

Corpora:
  LEG A  code-class prompts (the seed profile's domain — sanity lane:
          persistence must keep-or-improve here, never regress).
  LEG B  distant-domain prose prompts (history/biology/literature) —
          the seed does NOT cover its routing; this is where adaptive
          swaps + persistence must earn the gate.

Gate math (banks compared across boots by the operator):
  boot-15 control (FN8_P5_CONTROL=1): legs AB -> hB_ctl (+ hA_ctl)
  boot-16 adapt ON:                   leg B  -> hB_ad16 (intra-leg climb,
                                       persist .r files accumulate)
  boot-17 adapt ON (restart):         legs AB -> hB_ad17 (pre-warmed from
                                       the MERGED seed = base + .r files)
  GATE: hB_ad17 - hB_ctl >= +0.05 AND restart-reproduce evidence:
  hB_ad17 first-half ~= its second-half (warm from step one, unlike ctl).

Env: FN8_P5_PORT (8321), FN8_P5_TIER_DUMP (must match the serve's
V1X_TIER_DUMP), FN8_P5_BANK, FN8_P5_LEGS ("A"|"B"|"AB", default "B"),
FN8_P5_SETTLE (s to let trailing telemetry flushes land, 20),
FN8_P5_PERSIST (serve's V1X_TIER_PERSIST, for the .r file report).
"""
import glob
import json
import os
import time
import urllib.request

BASE = f"http://127.0.0.1:{os.environ.get('FN8_P5_PORT', '8321')}"
TIER_DUMP = os.environ.get("FN8_P5_TIER_DUMP", "/stage/bank/tier_p5.json")
BANK = os.environ.get("FN8_P5_BANK", "/stage/bank/p5_run.json")
LEGS = os.environ.get("FN8_P5_LEGS", "B").upper()
SETTLE = float(os.environ.get("FN8_P5_SETTLE", "20"))
PERSIST = os.environ.get("FN8_P5_PERSIST", "/stage/bank/routing_hist_p5.json")

CT = 192  # completion tokens: ~193 dispatches/prompt/layer -> windows fire

LEG_A = [  # code-class (seed domain)
    ("a1_fib", "Write a Python function computing the nth Fibonacci "
     "number with memoization, then explain its complexity."),
    ("a2_sort", "Write a Python three-way quicksort with commentary "
     "on pivot selection."),
    ("a3_lru", "Write a Python LRU cache class with get and put, both "
     "O(1), using an ordered dict; explain the eviction path."),
    ("a4_bsearch", "Implement binary search in Python with edge-case "
     "commentary (empty list, duplicates, overflow-free midpoint)."),
    ("a5_sql", "Write a SQL query joining orders to customers, grouping "
     "by month, filtering to totals over 1000; explain each clause."),
    ("a6_bash", "Write a bash pipeline that counts unique client IPs per "
     "hour from an nginx access log; explain each stage."),
    ("a7_dijkstra", "Implement Dijkstra's shortest path in Python with a "
     "heap; walk through one small example graph."),
    ("a8_regex", "Write a regex validating ISO-8601 dates, then explain "
     "each component and one deliberate limitation."),
]

LEG_B = [  # distant-domain prose (NOT in the seed profile)
    ("b1_1929", "Explain the principal causes of the 1929 stock market "
     "crash and which policy mistakes deepened the depression."),
    ("b2_moby", "Summarize the plot of Moby-Dick and analyze what the "
     "white whale symbolizes for Ahab and for the narrator."),
    ("b3_train", "A train leaves at 60 mph; another leaves 2 hours later "
     "at 90 mph on the same track. Solve when it catches up, step by step."),
    ("b4_photo", "Describe photosynthesis for a curious ten-year-old, "
     "from sunlight to sugar, naming each stage in order."),
    ("b5_mitosis", "Compare mitosis and meiosis: stages, outcomes, and "
     "why sexual reproduction needs the second division."),
    ("b6_sky", "Explain why the sky is blue at noon and red at sunset, "
     "including the role of wavelength and scattering angle."),
    ("b7_silk", "Write a short essay on the Silk Road's economic effects "
     "on the cities that lined it, with three concrete examples."),
    ("b8_ww1", "What were the main causes of the First World War? Argue "
     "the two most decisive in a short essay."),
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


def read_tier(retries=5):
    """Lenient dump read: the serve flushes every V1X_TIER_FLUSH
    dispatches; a read racing an os.replace lands a clean old frame or
    a torn one — retry the torn ones."""
    for i in range(retries):
        try:
            with open(TIER_DUMP) as f:
                return json.load(f)
        except json.JSONDecodeError:
            time.sleep(2)
        except Exception as e:  # noqa: BLE001 — readout must never kill
            return {"error": str(e)}
    return {"error": "torn dump after retries"}


def agg(tier, keys):
    stores = tier.get("stores", {})
    out = {k: 0 for k in keys}
    if not stores:
        return None
    for rec in stores.values():
        for k in keys:
            out[k] += rec.get(k, 0)
    out["stores"] = len(stores)
    return out


KEYS = ("dispatches", "dummy_dispatches", "hits", "misses", "evictions",
        "h2d_copies", "pf_fills", "adapt_windows", "adapt_swaps",
        "persist_flushes")


def hit_rate(t0, t1):
    dh = t1["hits"] - t0["hits"]
    dm = t1["misses"] - t0["misses"]
    return round(dh / max(dh + dm, 1), 4), dh, dm


def run_leg(letter, prompts, bank):
    t_start = agg(read_tier(), KEYS)
    assert t_start, "tier dump unreadable at leg start"
    half = len(prompts) // 2
    t_mid = None
    for i, (name, prompt) in enumerate(prompts):
        t0 = time.time()
        out = req("/v1/completions", {
            "model": "flash-next-fp8", "prompt": prompt,
            "max_tokens": CT, "temperature": 0.0,
        })
        dt = time.time() - t0
        ctot = out.get("usage", {}).get("completion_tokens", CT)
        text = out["choices"][0]["text"][:60].replace("\n", " ")
        print(f"LEG{letter} {name} {dt:.1f}s {ctot}ct :: {text!r}",
              flush=True)
        if i + 1 == half:
            time.sleep(SETTLE)
            t_mid = agg(read_tier(), KEYS)
    time.sleep(SETTLE)
    t_end = agg(read_tier(), KEYS)
    leg = {"letter": letter, "start": t_start, "mid": t_mid, "end": t_end}
    leg["hit_rate_full"], leg["d_hits"], leg["d_misses"] = hit_rate(
        t_start, t_end)
    if t_mid:
        leg["hit_rate_first_half"], _, _ = hit_rate(t_start, t_mid)
        leg["hit_rate_second_half"], _, _ = hit_rate(t_mid, t_end)
    d_ad = t_end["adapt_windows"] - t_start["adapt_windows"]
    leg["adapt_windows"] = d_ad
    leg["adapt_swaps"] = t_end["adapt_swaps"] - t_start["adapt_swaps"]
    leg["persist_flushes"] = (
        t_end["persist_flushes"] - t_start["persist_flushes"])
    bank["legs"].append(leg)
    print(f"LEG{letter}_SUMMARY hit={leg['hit_rate_full']} "
          f"1h={leg.get('hit_rate_first_half')} "
          f"2h={leg.get('hit_rate_second_half')} "
          f"windows={d_ad} swaps={leg['adapt_swaps']}", flush=True)
    return leg


def persist_report():
    """Accumulated rank files — the restart-reproduce evidence."""
    out = []
    for path in sorted(glob.glob(PERSIST + ".r*")):
        if path.endswith(".tmp") or ".tmp" in os.path.basename(path):
            continue
        try:
            with open(path) as f:
                counts = json.load(f)["counts"]
            out.append({"path": path,
                        "total_touches": sum(sum(r) for r in counts),
                        "layers": len(counts)})
        except Exception as e:  # noqa: BLE001
            out.append({"path": path, "error": str(e)})
    return out


def main():
    ready = wait_health()
    bank = {"ready_s": round(ready, 1), "ts": time.time(),
            "legs": [], "persist": None}
    for letter in LEGS:
        prompts = LEG_A if letter == "A" else LEG_B
        run_leg(letter, prompts, bank)
    bank["persist"] = persist_report()
    if bank["persist"]:
        print("PERSIST", json.dumps(bank["persist"]))
    with open(BANK, "w") as f:
        json.dump(bank, f, indent=1)
    print("BANKED", BANK)
    print("FN8_P5_RUN_OK")


if __name__ == "__main__":
    main()
