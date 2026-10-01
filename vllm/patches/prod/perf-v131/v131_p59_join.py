#!/usr/bin/env python3
"""v131_p59_join.py — perf-v131 P59 match/eviction forensics (v2).

v2 fixes vs v1: (1) wave split on admission gaps is useless — the
scheduler refreshes get_computed_blocks for queued requests every
step, so MATCH lines are continuous. Labels come instead from the
driver's exact prompt_tokens (engine usage == request.num_tokens),
which is turn+cohort unique (sizes grow ~150 tok/turn). (2) Per-rid
FIRST MATCH line isolates the true admission boundary (before any
same-wave allocation could evict the request's own cache). (3) EVICT
lines are bucketed into driver turn windows from completion ts.

Reads: argv[1] combined trace lines (MATCH+EVICT), argv[2] driver
jsonl. Prints per (turn, cohort): distinct rids, first-admission hit
distribution + max-hit-per-rid, freeq; EVICT totals per turn window.
Caps: V131_P59_JOIN_DONE.
"""
import json
import re
import sys
from collections import defaultdict

MATCH = re.compile(
    r"V131_MATCH rid=(\S+) ptok=(\d+) hit=(\d+) kmiss=(-?\d+)"
    r" freeq=(\d+) pre=(\d+)"
)
EVICT = re.compile(r"V131_EVICT ev_total=(\d+) freeq=(\d+) new=(\d+)")
TS = re.compile(r"INFO \d\d-\d\d (\d\d):(\d\d):(\d\d)")


def tsec(line):
    m = TS.search(line)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def hms(t):
    return "%02d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)


def main():
    trace_path = sys.argv[1] if len(sys.argv) > 1 else "v131_trace.txt"
    drv_path = sys.argv[2] if len(sys.argv) > 2 else \
        "/root/build/v131_stage/wsc_pressure_v131p59.jsonl"

    # ---- driver: ptok -> (turn, cohort); per-turn completion windows
    label, turn_win, okcnt = {}, defaultdict(lambda: [1 << 30, -1]), \
        defaultdict(int)
    with open(drv_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok" or not r.get("prompt_tokens"):
                continue
            t, ch, p = r["turn"], r.get("cohort", "?"), \
                r["prompt_tokens"]
            label[p] = (t, ch)
            okcnt[t] += 1
            m = re.match(r"(\d\d):(\d\d):(\d\d)", r.get("ts", ""))
            if m:
                s = int(m.group(1)) * 3600 + int(m.group(2)) * 60 \
                    + int(m.group(3))
                turn_win[t][0] = min(turn_win[t][0], s)
                turn_win[t][1] = max(turn_win[t][1], s)

    # ---- trace
    rows, evicts = [], []
    with open(trace_path) as f:
        for line in f:
            t = tsec(line)
            if t is None:
                continue
            m = MATCH.search(line)
            if m:
                rows.append({"t": t, "rid": m.group(1),
                             "ptok": int(m.group(2)),
                             "hit": int(m.group(3)),
                             "kmiss": int(m.group(4)),
                             "freeq": int(m.group(5)),
                             "pre": int(m.group(6))})
                continue
            m = EVICT.search(line)
            if m:
                evicts.append({"t": t, "ev": int(m.group(1)),
                               "freeq": int(m.group(2)),
                               "new": int(m.group(3))})

    rows.sort(key=lambda r: r["t"])
    print("MATCH lines=%d EVICT lines=%d ev_last=%s" % (
        len(rows), len(evicts),
        evicts[-1]["ev"] if evicts else "0"))
    print("driver ok/turn:", dict(sorted(okcnt.items())))

    # ---- per-rid first admission (true boundary) + max hit
    first, maxhit = {}, defaultdict(int)
    unlabeled = 0
    for r in rows:
        lab = label.get(r["ptok"])
        if lab is None:
            unlabeled += 1
            continue
        r["lab"] = lab
        maxhit[r["rid"]] = max(maxhit[r["rid"]], r["hit"])
        if r["rid"] not in first or r["t"] < first[r["rid"]]["t"]:
            first[r["rid"]] = r
    if unlabeled:
        print("note: %d MATCH rows had ptok outside driver labels "
              "(pre-boot traffic), excluded" % unlabeled)

    print("\n=== per (turn, cohort): FIRST-admission hits ===")
    stats = defaultdict(list)
    for rid, r in first.items():
        stats[r["lab"]].append(r)
    for (t, ch) in sorted(stats):
        rs = stats[(t, ch)]
        hits = sorted(x["hit"] for x in rs)
        fq = sorted(x["freeq"] for x in rs)
        mh = sorted(maxhit[x["rid"]] for x in rs)
        print("t%d %-6s n=%2d firsthit avg=%6.0f min=%5d max=%6d"
              " | per-rid MAXhit avg=%6.0f max=%6d | freeq"
              " min=%3d avg=%3.0f | span %s..%s"
              % (t, ch, len(rs), sum(hits) / len(hits), hits[0],
                 hits[-1], sum(mh) / len(mh), mh[-1], fq[0],
                 sum(fq) / len(fq), hms(min(x["t"] for x in rs)),
                 hms(max(x["t"] for x in rs))))

    print("\n=== EVICT lines per driver turn window "
          "(completion ts +/-60s) ===")
    for t in sorted(turn_win):
        lo, hi = turn_win[t][0] - 60, turn_win[t][1] + 60
        wev = [e for e in evicts if lo <= e["t"] <= hi]
        if wev:
            print("t%d [%s..%s] lines=%d ev %d->%d min_freeq=%d"
                  % (t, hms(lo), hms(hi), len(wev), wev[0]["ev"],
                     wev[-1]["ev"],
                     min(e["freeq"] for e in wev)))
        else:
            print("t%d [%s..%s] no evict lines" % (
                t, hms(lo), hms(hi)))
    print("\nV131_P59_JOIN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
