#!/usr/bin/env python3
"""v65 P1a analysis: per-step attn/gdn sums from v65_lt_layers.txt."""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/build/v65_lt_layers.txt"
attn = []  # (kmax, ms) per call, TP0 only (workers duplicate)
gdn_steps = []  # cum boundaries

cur = {}
for line in open(path, errors="replace"):
    if "Worker_TP1" in line:
        continue  # TP0 only; workers agree
    m = re.search(r"V65_ATTN ms=([\d.]+) q=(\d+) kmax=(\d+) cum_ms=(\d+) n=(\d+)", line)
    if m:
        ms, q, kmax, cum, n = float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        attn.append((ms, q, kmax, cum, n))
        continue
    g = re.search(r"V65_GDN layer_ms=([\d.]+) nt=(\d+) cum_ms=(\d+) n=(\d+)", line)
    if g:
        gdn_steps.append((float(g.group(1)), int(g.group(2)), int(g.group(3)), int(g.group(4))))

# group attn calls into steps by n
steps = {}
for ms, q, kmax, cum, n in attn:
    step_idx = (n - 1) // 16
    steps.setdefault(step_idx, {"ms": 0.0, "q": q, "kmax": kmax, "n0": n})
    steps[step_idx]["ms"] += ms
    steps[step_idx]["kmax"] = max(steps[step_idx]["kmax"], kmax)

# gdn cumulative diffs -> per-step gdn ms (48 calls/step)
gdn_per_step = []
prev_cum = 0
buf = 0.0
for ms, nt, cum, n in gdn_steps:
    buf += ms
    if n % 48 == 0:
        gdn_per_step.append((nt, cum, buf))
        buf = 0.0

print("step | q_tok | kmax | attn_ms | gdn_ms | resid_ms(est)")
# residual needs step wall; approximate later via known total
for i in sorted(steps):
    s = steps[i]
    g = gdn_per_step[i][2] if i < len(gdn_per_step) else 0.0
    print("%5d|%7d|%6d|%9.1f|%8.1f|" % (i, s["q"], s["kmax"], s["ms"], g))

tot_attn = sum(s["ms"] for s in steps.values())
print("TOTAL attn_ms=%.1f steps=%d" % (tot_attn, len(steps)))
if gdn_steps:
    print("TOTAL gdn_cum_ms=%d" % gdn_steps[-1][2])

# linear fit attn_ms vs kmax (quadratic term dominates growth)
import numpy as np

ks = np.array([steps[i]["kmax"] for i in sorted(steps)], dtype=float)
a = np.array([steps[i]["ms"] for i in sorted(steps)], dtype=float)
if len(ks) > 2:
    slope, intercept = np.polyfit(ks, a, 1)
    print("attn_fit slope=%.3e ms/prefix-token intercept=%.1f ms" % (slope, intercept))
    print("projected attn at kmax=0 (chunk-local+fixed): %.1f ms/step" % intercept)
    print("quadratic-prefix share of attn: %.1f s of %.1f s" % (
        (a.sum() - intercept * len(a)) / 1000.0, a.sum() / 1000.0))
