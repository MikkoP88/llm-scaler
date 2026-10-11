#!/usr/bin/env python3
"""fn8 bank JSON summarizer — docker exec fn8-p2 python3 /stage/fn8_sum.py <path>"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/stage/bank/p2_run_tp2.json"
d = json.load(open(path))
for k in ("ready_s", "health", "stores_built", "build_fails",
          "prewarm", "speedup_ref"):
    if k in d:
        print(k, "=", d[k])
for leg in d.get("parity", []):
    print("parity %-8s %-5s %.1fs" % (
        leg["name"], "PASS" if leg["passed"] else "FAIL",
        leg.get("latency_s", -1)))
for leg in d.get("perf", []):
    print("perf %-10s %6.3f tok/s  lat %5.2fs  x%-7s" % (
        leg["name"], leg["tok_per_s"], leg["latency_s"],
        leg.get("speedup_vs_p0_ref", "?")))
t = d.get("tier")
if t:
    st = t["stores"]
    n = len(st)
    agg = {k: sum(s.get(k, 0) for s in st.values()) for k in (
        "dispatches", "dummy_dispatches", "hits", "misses", "evictions",
        "h2d_copies", "pf_fills", "pf_promotes", "pf_waits")}
    agg["h2d_gb"] = sum(s.get("h2d_bytes", 0) for s in st.values()) / 1e9
    hitr = [100.0 * s["hits"] / max(1, s["hits"] + s["misses"])
            for s in st.values()]
    print("tier: %d stores slots=%d seed=%s" % (n, t.get("slots", -1),
                                                t.get("seed", "?")))
    print("tier_agg", json.dumps(agg))
    print("hit_rate min/med/max %.1f/%.1f/%.1f%%" % (
        min(hitr), sorted(hitr)[len(hitr) // 2], max(hitr)))
    d2h = [s.get("d2h_ms", 0) for s in st.values()]
    ens = [s.get("ensure_ms", 0) for s in st.values()]
    h2d = [s.get("h2d_ms", 0) for s in st.values()]
    print("ms totals: d2h %.1f ensure %.1f h2d %.1f" % (
        sum(d2h), sum(ens), sum(h2d)))
    per = sorted((s["layer"], s["hits"], s["misses"]) for s in st.values())
    print("first/last layer rows:", per[0], per[-1])
