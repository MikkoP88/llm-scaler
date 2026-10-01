#!/usr/bin/env python3
"""storm_summary_v127.py — aggregate a waiting-storm window from the WS-B
recorder JSONL (/root/build/telemetry/metrics_YYYYMMDD.jsonl).

Prints a compact certification-record block for the PHASES entry: queue
depth envelope, TTFT distribution (from cumulative buckets), preemption
count, KV vs capacity-wait contrast (the RC1/RC4 live proof), and cache
efficiency. Usage: storm_summary_v127.py [jsonl] [start_iso] [end_iso]
(defaults: today's file, full range).
"""
import json, sys, datetime as dt

path = sys.argv[1] if len(sys.argv) > 1 else "/root/build/telemetry/metrics_%s.jsonl" % dt.date.today().strftime("%Y%m%d")
lo = sys.argv[2] if len(sys.argv) > 2 else "0000"
hi = sys.argv[3] if len(sys.argv) > 3 else "9999"

rows = []
with open(path) as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("ts", "")
        if lo <= t[11:16 if len(t) > 16 else len(t)] <= hi or lo <= t[:len(lo)] <= hi:
            rows.append(r)

if not rows:
    print("STORM_SUMMARY_EMPTY")
    sys.exit(1)

w = [r.get("num_requests_waiting", 0) for r in rows]
run = [r.get("num_requests_running", 0) for r in rows]
kv = [r.get("kv_cache_usage_perc", 0) for r in rows]
cap = [r.get("num_requests_waiting_by_reason", {}).get("capacity", 0) for r in rows]
pre = [r.get("num_preemptions_total", 0) for r in rows]

first, last = rows[0], rows[-1]
def delta(k):
    return last.get(k, 0) - first.get(k, 0)

ttft0, ttft1 = first.get("ttft_bucket", {}), last.get("ttft_bucket", {})
total = ttft1.get("+Inf", 0) - ttft0.get("+Inf", 0)
if total > 0:
    keys = [k for k in ttft1 if k != "+Inf"]
    dist = []
    for k in keys:
        d = ttft1[k] - ttft0.get(k, 0)
        if d:
            dist.append("%s:%d" % (k, d))
    ttft_line = "ttft_delta=%d buckets[%s]" % (total, " ".join(dist))
else:
    ttft_line = "ttft_delta=0"

pc = last.get("prefix_cache_queries_total", 0) - first.get("prefix_cache_queries_total", 0)
ph = last.get("prefix_cache_hits_total", 0) - first.get("prefix_cache_hits_total", 0)
pt = last.get("prompt_tokens_total", 0) - first.get("prompt_tokens_total", 0)
pct = last.get("prompt_tokens_cached_total", 0) - first.get("prompt_tokens_cached_total", 0)

print("=== STORM_SUMMARY_V127 %s..%s (%d samples, %ds) ===" % (first.get("ts"), last.get("ts"), len(rows), len(rows) * 10))
print("waiting: max=%d p50=%.0f cap_reason_max=%d" % (max(w), sorted(w)[len(w)//2], max(cap)))
print("running: max=%d p50=%.0f" % (max(run), sorted(run)[len(run)//2]))
print("kv_cache_usage: min=%.2f max=%.2f p50=%.2f  <-- LOW kv while capacity-waits = SSM-pool concurrency cap (C1 proof)" % (min(kv), max(kv), sorted(kv)[len(kv)//2]))
print("preemptions_delta=%d" % (max(pre) - min(pre)))
print(ttft_line)
print("cache: block_q=%.0f block_hit=%.1f%% tok_cached=%.0f/%.0f=%.1f%%" % (pc, (100.0*ph/pc if pc else 0), pct, pt, (100.0*pct/pt if pt else 0)))
print("STORM_SUMMARY_DONE")
