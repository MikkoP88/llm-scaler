#!/usr/bin/env python3
"""d4_d2_measurement.py — v127 WS-D in-window measurement over WS-B telemetry.

D4  (GDN 4096-token granularity recompute share): from recorder deltas,
  computed-vs-cached token flow per request. The GDN state snaps to 4096-
  token boundaries, so each turn re-computes up to ~4k already-seen tokens
  (mean ~2048 for long CC sessions). Escalation bar (plan D4): a kernel
  work package only if recompute share >= 15 % of computed tokens.
  Aggregate counters cannot split recompute from eviction misses — the
  note reports the bounded estimate with confounds named.

D2  (measurement half): waiting-by-reason traces + per-interval TTFT
  bucket arrivals + KV/preemption series — the "before" numbers for the
  W2-B (mnbt 16384) / D2 (v63 budget) legs. v66 firing itself is not in
  the recorder schema; serve-log V66 lines are grepped separately.

Input : /root/build/telemetry/metrics_*.jsonl  (10 s cadence, cumulative)
Output: JSON to stdout-path arg (default d4_d2_measurement_20261001.json)
"""
import glob
import json
import math
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "d4_d2_measurement_20261001.json"

rows = []
for path in sorted(glob.glob("/root/build/telemetry/metrics_*.jsonl")):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def dlt(key_series, i):
    v_now = num(key_series[i])
    v_prev = num(key_series[i - 1]) if i > 0 else 0.0
    d = v_now - v_prev
    return d if d > 0 else 0.0  # counters reset on engine restart

series = {k: [num(r.get(k)) for r in rows] for k in (
    "prompt_tokens_total", "generation_tokens_total", "num_preemptions_total",
    "kv_cache_usage_perc", "num_requests_running", "num_requests_waiting",
    "prefix_cache_queries_total", "prefix_cache_hits_total")}
comp = [num((r.get("prompt_tokens_by_source_total") or {}).get("local_compute")) for r in rows]
hit = [num((r.get("prompt_tokens_by_source_total") or {}).get("local_cache_hit")) for r in rows]
wait_cap = [num((r.get("num_requests_waiting_by_reason") or {}).get("capacity")) for r in rows]
wait_def = [num((r.get("num_requests_waiting_by_reason") or {}).get("deferred")) for r in rows]
ttft_n = [sum(num(v) for v in (r.get("ttft_bucket") or {}).values()) for r in rows]
ttft_buckets = sorted({b for r in rows for b in (r.get("ttft_bucket") or {})},
                      key=float)

# ---- per-interval deltas --------------------------------------------------
iv = []
for i in range(len(rows)):
    iv.append({
        "ts": rows[i]["ts"],
        "comp": dlt(comp, i),
        "hit": dlt(hit, i),
        "gen": dlt(series["generation_tokens_total"], i),
        "req": dlt(ttft_n, i),          # requests that reached first token
        "wait": num(rows[i]["num_requests_waiting"]),
        "wait_cap": wait_cap[i],
        "run": num(rows[i]["num_requests_running"]),
        "kv": num(rows[i]["kv_cache_usage_perc"]),
        "preempt": dlt(series["num_requests_total"] if False else
                       series["num_preemptions_total"], i),
    })

tot = {k: sum(x[k] for x in iv) for k in ("comp", "hit", "gen", "req", "preempt")}

# ---- D4 core numbers ------------------------------------------------------
absorption = tot["hit"] / (tot["hit"] + tot["comp"]) * 100 if (tot["hit"] + tot["comp"]) else 0.0
comp_per_req = tot["comp"] / tot["req"] if tot["req"] else 0.0
hit_per_req = tot["hit"] / tot["req"] if tot["req"] else 0.0
gen_per_req = tot["gen"] / tot["req"] if tot["req"] else 0.0
# New-suffix floor per turn: generated tokens must re-enter the next prompt;
# add ~300 static overhead (tool calls / user turn framing). GDN granularity
# recompute per turn averages ~2048 for sessions >> 4096 (uniform mod 4096).
new_floor = gen_per_req + 300.0
gdn_mean = 4096.0 / 2.0
recompute_est = max(0.0, comp_per_req - new_floor)
share_low = recompute_est / comp_per_req * 100 if comp_per_req else 0.0     # excess-over-new floor
share_gdn = gdn_mean / comp_per_req * 100 if comp_per_req else 0.0         # theory cross-check

# storm sub-window (waiting > 0) vs quiet — recompute behaves differently
storm = [x for x in iv if x["wait"] > 0]
quiet = [x for x in iv if x["wait"] == 0]
def per_req(xs):
    rq = sum(x["req"] for x in xs)
    return (sum(x["comp"] for x in xs) / rq) if rq else 0.0

# ---- D2 core numbers ------------------------------------------------------
storm_iv = len(storm)
wait_nonzero_pct = storm_iv / len(iv) * 100 if iv else 0.0
wait_cap_nonzero = sum(1 for x in iv if x["wait_cap"] > 0)
max_wait = max((x["wait"] for x in iv), default=0.0)
max_wait_cap = max(wait_cap)
kv_p50 = sorted(x["kv"] for x in iv)[len(iv) // 2] if iv else 0.0
kv_max = max((x["kv"] for x in iv), default=0.0)
max_run = max((x["run"] for x in iv), default=0.0)

# per-interval TTFT arrivals by bucket (deltas of cumulative bucket counters)
ttft_new = {b: 0.0 for b in ttft_buckets}
prev = {b: 0.0 for b in ttft_buckets}
for r in rows:
    cur = {b: num(v) for b, v in (r.get("ttft_bucket") or {}).items()}
    for b in ttft_buckets:
        d = cur.get(b, 0.0) - prev.get(b, 0.0)
        if d > 0:
            ttft_new[b] += d
    prev = cur
# buckets are CUMULATIVE counters (le=bound): each delta = new requests
# with TTFT <= bound. Normalize by the +Inf delta (total new requests) so
# each value is a true cumulative share; per-band mass = difference.
ttft_total_new = ttft_new.get("+Inf", 0.0) or sum(ttft_new.values())
ttft_pct = {b: round(v / ttft_total_new * 100, 2) for b, v in ttft_new.items()} if ttft_total_new else {}

# wedge onset: last healthy ts (interval before comp/hit counters froze)
last_write = rows[-1]["ts"] if rows else None

out = {
    "tag": "d4_d2_measurement",
    "generated_from": "/root/build/telemetry/metrics_*.jsonl",
    "window": {"first_ts": rows[0]["ts"] if rows else None,
               "last_ts": last_write, "intervals": len(iv),
               "cadence_s": 10},
    "d4_gdn_granularity": {
        "absorption_pct": round(absorption, 2),
        "computed_total": tot["comp"], "cached_total": tot["hit"],
        "requests_total": tot["req"],
        "computed_per_request": round(comp_per_req, 1),
        "cached_per_request": round(hit_per_req, 1),
        "generated_per_request": round(gen_per_req, 1),
        "new_suffix_floor_per_request": round(new_floor, 1),
        "recompute_excess_per_request": round(recompute_est, 1),
        "recompute_share_of_computed_pct_excess_bound": round(share_low, 1),
        "recompute_share_of_computed_pct_gdn_theory_2048": round(share_gdn, 1),
        "computed_per_request_storm_vs_quiet": {
            "storm": round(per_req(storm), 1), "quiet": round(per_req(quiet), 1),
            "storm_intervals": storm_iv, "quiet_intervals": len(quiet)},
        "escalation_bar_pct": 15.0,
        "verdict": ("BAR EXCEEDED (excess-over-new bound alone >= 15%) — "
                    "escalate, but FIRST a per-request split leg "
                    "(prompt_len, hit, computed, mod-4096 regression) to "
                    "attribute excess between GDN-snap recompute and "
                    "eviction re-prefills before committing weeks-scale "
                    "kernel effort" if share_low >= 15.0 else
                    "NO ESCALATION — below the 15% bar"),
        "confounds": ["eviction re-prefills inflate computed (not GDN-specific)",
                      "aggregate counters cannot split GDN-snap recompute from "
                      "eviction misses; per-request instrumentation needed for "
                      "an exact split"],
    },
    "d2_waiting_traces": {
        "intervals_with_waiting_pct": round(wait_nonzero_pct, 1),
        "intervals_capacity_wait": wait_cap_nonzero,
        "max_waiting": max_wait, "max_waiting_capacity": max_wait_cap,
        "max_running": max_run,
        "kv_usage_p50": round(kv_p50, 3), "kv_usage_max": round(kv_max, 3),
        "preemptions_total_window": tot["preempt"],
        "ttft_new_arrivals_pct_by_bucket": ttft_pct,
        "note": "v66 firing not in recorder schema — serve-log V66 grep is the "
                "firing-frequency evidence (run separately)",
    },
    "wedge_onset": {"last_telemetry_ts": last_write,
                    "serve_log_death": "2026-09-30T22:13:33 EngineDeadError "
                                       "(execute_model RPC timeout -> device "
                                       "wedge class)"},
}
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({
    "D4": out["d4_gdn_granularity"], "D2": out["d2_waiting_traces"]},
    indent=1))
print("WROTE", OUT)
