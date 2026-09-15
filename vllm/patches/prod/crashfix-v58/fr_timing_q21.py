#!/usr/bin/env python3
# fr_timing.py — A2 timing forensics on frozen Q21 fr rings.
# Questions:
#  (1) AR duration distribution by numel class; any mid-session near-stalls?
#  (2) step-period history from 'propose end' markers: degenerative growth
#      vs flat-then-cliff (sudden race)?
#  (3) last-N steps / last ARs before silence vs baseline.
import re
import sys


def parse(path):
    evs = []
    rx = re.compile(r'^(\d+\.\d+)\s+(AR begin|AR end|propose end|propose begin)(.*)$')
    for line in open(path, errors='replace'):
        m = rx.match(line)
        if not m:
            continue
        ts = float(m.group(1))
        kind = m.group(2)
        rest = m.group(3)
        mn = re.search(r'numel=(\d+)', rest)
        evs.append((ts, kind, int(mn.group(1)) if mn else 0))
    return evs


def pct(xs, p):
    if not xs:
        return float('nan')
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def analyze(path):
    evs = parse(path)
    span = evs[-1][0] - evs[0][0] if evs else 0
    print(f"===== {path}")
    print(f"events={len(evs)} span={span:.1f}s first={evs[0][0]:.1f} last={evs[-1][0]:.1f}")
    kinds = {}
    for _, k, _ in evs:
        kinds[k] = kinds.get(k, 0) + 1
    print(f"kinds={kinds}")

    # -- (1) AR durations by numel class
    cur = None
    unpaired = 0
    durs = []            # (ts, numel, dur)
    for ts, k, n in evs:
        if k == 'AR begin':
            if cur is not None:
                unpaired += 1
            cur = (ts, n)
        elif k == 'AR end':
            if cur is None:
                unpaired += 1
            else:
                durs.append((cur[0], cur[1], ts - cur[0]))
                cur = None
    if cur is not None:
        unpaired += 1
    print(f"paired ARs={len(durs)} unpaired={unpaired}")

    by_numel = {}
    for ts, n, d in durs:
        by_numel.setdefault(n, []).append(d)
    print("-- AR duration by numel (count / med / p99 / max ms):")
    for n in sorted(by_numel):
        ds = [d * 1e3 for d in by_numel[n]]
        print(f"   numel={n:>10}: n={len(ds):>6} med={pct(ds,.5):8.3f} p99={pct(ds,.99):9.3f} max={max(ds):10.3f}")

    slow = sorted(durs, key=lambda x: -x[2])[:10]
    print("-- 10 slowest ARs (ts, numel, dur ms):")
    for ts, n, d in slow:
        print(f"   t={ts:.1f} (+{ts-evs[0][0]:7.1f}s) numel={n:>9} dur={d*1e3:9.3f}")
    # near-stall census: >50ms
    stalls = [(ts, n, d) for ts, n, d in durs if d > 0.050]
    print(f"-- ARs >50ms: {len(stalls)}")
    for ts, n, d in stalls[:15]:
        print(f"   t=+{ts-evs[0][0]:7.1f}s numel={n:>9} dur={d*1e3:9.3f}")

    # -- (2) step periods from propose-end markers
    pe = [ts for ts, k, _ in evs if k == 'propose end']
    periods = [b - a for a, b in zip(pe, pe[1:])]
    if periods:
        med = pct(periods, .5)
        print(f"-- steps={len(pe)} period med={med*1e3:.1f}ms p90={pct(periods,.9)*1e3:.1f}ms p99={pct(periods,.99)*1e3:.1f}ms max={max(periods)*1e3:.1f}ms")
        # degenerative test: median period in 6 session slices
        print("   step-period median by session sixth (ms):")
        k6 = max(1, len(periods) // 6)
        for i in range(6):
            sl = periods[i * k6:(i + 1) * k6] if i < 5 else periods[5 * k6:]
            if sl:
                print(f"     slice{i+1}: med={pct(sl,.5)*1e3:8.1f} max={max(sl)*1e3:8.1f} n={len(sl)}")
        # (3) last 30 periods
        print("   last 30 step periods (ms):")
        tail = [p * 1e3 for p in periods[-30:]]
        for i in range(0, len(tail), 10):
            print("     " + " ".join(f"{v:7.1f}" for v in tail[i:i + 10]))

    # (3b) last 40 raw events before silence
    print("-- last 12 events:")
    for ts, k, n in evs[-12:]:
        print(f"   {ts:.4f} {k} {n}")

    # cadence of final 5120-burst vs typical: inter-begin gaps in last 2s
    t_end = evs[-1][0]
    burst = [ts for ts, k, n in evs if k == 'AR begin' and ts > t_end - 2.0]
    gaps = [b - a for a, b in zip(burst, burst[1:])]
    if gaps:
        print(f"-- final-2s AR-begin cadence: n={len(burst)} med_gap={pct(gaps,.5)*1e3:.2f}ms max_gap={max(gaps)*1e3:.2f}ms")
    print()


if __name__ == '__main__':
    for p in sys.argv[1:]:
        analyze(p)
