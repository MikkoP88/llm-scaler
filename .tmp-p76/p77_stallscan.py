#!/usr/bin/env python3
"""P77 stall scanner — tests the user's law: stalls coincide with v52f
PRE-COPY / DEF-PP log lines.

Parses the CURRENT boot of /root/serve_full.log (from the last
'Application startup complete'), computes inter-line timestamp gaps, and:
  1. reports the top gap windows (> GAP_S) with surrounding lines,
  2. classifies every v52f event (no-op vs real copy),
  3. compares the latency-to-next-log-line after each event class vs a
     baseline of random lines (the witness-vs-cost discriminator).

Pure text processing; runs anywhere with python3.
"""
import re
import sys
from datetime import datetime

LOG = sys.argv[1] if len(sys.argv) > 1 else "/root/serve_full.log"
GAP_S = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

TS = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse(path):
    rows = []
    startup_idx = -1
    with open(path, "r", errors="replace") as f:
        for i, raw in enumerate(f):
            m = TS.search(raw[:60])
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%m-%d %H:%M:%S")
            line = raw.rstrip("\n")
            # strip the common [file:lineno] module prefix noise for width
            rows.append((t, i, line))
            if "Application startup complete" in line:
                startup_idx = len(rows) - 1
    return rows[startup_idx + 1:] if startup_idx >= 0 else rows


def classify(line):
    if "v52f DEF-PP" in line:
        m = re.search(r"src=(\d+) dest=(\d+) bias=(\d+)", line)
        if not m:
            return None
        s, d, b = map(int, m.groups())
        if s == d and b == 0:
            return "DEFPP_NOOP"      # collect guard returns; copy skipped
        if s == d:
            return "DEFPP_INPLACE"   # bias>0: real in-place shift copy
        return "DEFPP_CROSS"         # real cross-block copy
    if "v52f PRE-COPY" in line:
        return "PRECOPY"             # real carry by construction (prev!=curr)
    if "v52f POST-COPY" in line:
        return "POSTCOPY"
    return None


def main():
    rows = parse(LOG)
    if len(rows) < 2:
        print("NOT_ENOUGH_LINES", len(rows))
        return
    t0, t1 = rows[0][0], rows[-1][0]
    span = (t1 - t0).total_seconds()
    print(f"BOOT_SPAN {t0.time()} -> {t1.time()} = {span:.0f}s, lines={len(rows)}")

    # --- classify events, record latency to next line -------------------
    events = []  # (idx, cls)
    for j, (t, i, line) in enumerate(rows):
        cls = classify(line)
        if cls:
            events.append((j, cls, t, line))

    from collections import Counter, defaultdict
    cnt = Counter(cls for _, cls, _, _ in events)
    print("EVENT_CLASSES:")
    for cls, n in sorted(cnt.items()):
        print(f"  {cls:14s} {n:6d}  ({n / max(span, 1):.2f}/s)")

    # latency from event line to the NEXT line (any) — 1s ts resolution
    stats = defaultdict(list)
    for j, cls, t, line in events:
        if j + 1 < len(rows):
            stats[cls].append((rows[j + 1][0] - t).total_seconds())
    # baseline: 5000 evenly spread lines that are NOT within 1 line of an event
    ev_idx = {j for j, _, _, _ in events}
    base = []
    step = max(1, len(rows) // 5000)
    for j in range(0, len(rows), step):
        if j in ev_idx or j - 1 in ev_idx or j + 1 >= len(rows):
            continue
        base.append((rows[j + 1][0] - rows[j][0]).total_seconds())
    print("NEXT_LINE_LATENCY_BY_CLASS (s: n / p50 / p90 / max):")
    print(f"  {'BASELINE':14s} {len(base):6d} / "
          f"{sorted(base)[len(base)//2] if base else 0} / "
          f"{sorted(base)[int(len(base)*0.9)] if base else 0} / "
          f"{max(base) if base else 0}")
    for cls in sorted(stats):
        v = sorted(stats[cls])
        print(f"  {cls:14s} {len(v):6d} / {v[len(v)//2]} / "
              f"{v[int(len(v)*0.9)]} / {v[-1]}")

    # --- top gap windows -------------------------------------------------
    gaps = []
    for j in range(1, len(rows)):
        g = (rows[j][0] - rows[j - 1][0]).total_seconds()
        if g >= GAP_S:
            gaps.append((g, j))
    gaps.sort(reverse=True)
    print(f"TOP_GAPS >= {GAP_S}s: {len(gaps)} windows, "
          f"total stalled {sum(g for g, _ in gaps):.0f}s")
    near = defaultdict(int)
    for g, j in gaps:
        # does a v52f event line sit within 3 lines before the gap end?
        for k in range(max(0, j - 3), min(len(rows), j + 3)):
            cls = classify(rows[k][2])
            if cls:
                near[cls] += 1
                break
        else:
            near["NONE"] += 1
    print("GAP_ADJACENCY (event class within +/-3 lines of the gap):")
    for cls, n in sorted(near.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:14s} {n:4d} / {len(gaps)}")

    for g, j in gaps[:12]:
        print(f"\n--- GAP {g:.0f}s before line {j} ---")
        for k in range(max(0, j - 2), min(len(rows), j + 3)):
            t, i, line = rows[k]
            mark = ">>" if k == j else "  "
            print(f"{mark} {t.time()} {line[:190]}")


if __name__ == "__main__":
    main()
