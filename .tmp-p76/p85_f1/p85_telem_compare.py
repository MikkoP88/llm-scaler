#!/usr/bin/env python3
"""P85 telemetry compare — per-req decode rate, v1.2.31 era vs v1.2.32rc1 era.

Reads the standing metrics recorder JSONL for one day and computes, per
10-second sample interval, the generation-token rate divided by the
running-request count (only intervals with a stable running count > 0 and
a positive token delta; counter resets from restarts are skipped).

Windows are split by a boundary hour:minute argument (the rc1 boot time).
Prints per-era percentiles of tok/s/req for matched running-count bands so
the comparison is concurrency-controlled.

Usage: python3 p85_telem_compare.py <jsonl> <HH:MM-boundary> [jsonl2 ...]
"""
import json
import sys
from datetime import datetime


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> int:
    paths = sys.argv[1:-1]
    boundary = sys.argv[-1]
    bhour, bmin = (int(x) for x in boundary.split(":"))
    rows = []
    for p in paths:
        rows.extend(load(p))
    # de-dup by ts if multiple files overlap
    seen = set()
    uniq = []
    for r in rows:
        ts = r.get("ts") or r.get("time") or r.get("timestamp")
        if ts in seen:
            continue
        seen.add(ts)
        uniq.append(r)
    uniq.sort(key=lambda r: r.get("ts") or r.get("time") or r.get("timestamp"))

    samples = []
    for r in uniq:
        ts = r.get("ts") or r.get("time") or r.get("timestamp")
        run = r.get("num_requests_running")
        gen = r.get("generation_tokens_total")
        if run is None or gen is None:
            continue
        samples.append((ts, run, float(gen)))

    def is_rc(ts_str):
        try:
            t = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return False
        return (t.hour, t.minute) >= (bhour, bmin)

    eras = {"v1.2.31-era": [], "rc1-era": []}
    for (ts1, run1, gen1), (ts2, run2, gen2) in zip(samples, samples[1:]):
        if run1 <= 0 or run1 != run2:
            continue
        dtok = gen2 - gen1
        if dtok <= 0:  # restart reset or idle
            continue
        try:
            t1 = datetime.fromisoformat(ts1)
            t2 = datetime.fromisoformat(ts2)
            dt = (t2 - t1).total_seconds()
        except (ValueError, TypeError):
            dt = 10.0
        if dt <= 0 or dt > 30:
            continue
        rate = dtok / dt / run1
        era = "rc1-era" if is_rc(ts1) else "v1.2.31-era"
        eras[era].append((rate, run1))

    print("samples: v1.2.31-era=%d rc1-era=%d (boundary %s)"
          % (len(eras["v1.2.31-era"]), len(eras["rc1-era"]), boundary))
    for band in (1, 2, 3, 4, 5, 6, 8):
        for era in ("v1.2.31-era", "rc1-era"):
            vals = sorted(r for r, run in eras[era] if run == band)
            if len(vals) < 5:
                continue
            p50 = vals[len(vals) // 2]
            print("running=%d %-12s n=%4d p50=%.1f p10=%.1f p90=%.1f tok/s/req"
                  % (band, era, len(vals), p50,
                     vals[int(len(vals) * 0.1)], vals[int(len(vals) * 0.9)]))
    # aggregate any-concurrency medians
    for era in ("v1.2.31-era", "rc1-era"):
        vals = sorted(r for r, _ in eras[era])
        if vals:
            print("%s ALL n=%d p50=%.1f" % (era, len(vals), vals[len(vals) // 2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
