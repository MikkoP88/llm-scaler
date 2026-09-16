#!/usr/bin/env python3
"""p2_profdiff.py — compare two py-spy speedscope profiles (26.14 vs 26.18).
Reports cumulative weight per key frame and normalized share."""
import collections
import json
import sys


def load(path):
    d = json.load(open(path))
    frames = d["shared"]["frames"]
    cum = collections.Counter()
    leaf = collections.Counter()
    total = 0.0
    for prof in d["profiles"]:  # one profile per thread — aggregate all
        for s, w in zip(prof["samples"], prof["weights"]):
            if not s:
                continue
            total += w
            leaf[frames[s[-1]]["name"]] += w
            for fi in set(s):
                cum[frames[fi]["name"]] += w
    return total, cum, leaf


KEYS = [
    "forward", "sample_tokens", "propose_draft_token_ids", "run",
    "_update_states_after_model_execute", "get_top_tokens",
    "_propose_impl", "memory_fence", "all_gather", "sched_yield",
    "verify", "sample_tokens_from_probs", "_inner_forward",
    "rms_norm", "forward_native", "forward_static", "acquire_read",
]

a = sys.argv[1]
b = sys.argv[2]
ta, ca, la = load(a)
tb, cb, lb = load(b)
print(f"TOTAL {a}={ta:.1f}s {b}={tb:.1f}s  (ratio {tb/max(ta,1e-9):.2f})")
print(f"{'frame':42s} {'2614s':>7s} {'share':>6s} {'2618s':>7s} {'share':>6s}")
for k in KEYS:
    va, vb = ca.get(k, 0.0), cb.get(k, 0.0)
    print(f"{k:42s} {va:7.2f} {va/max(ta,1e-9)*100:5.1f}% "
          f"{vb:7.2f} {vb/max(tb,1e-9)*100:5.1f}%")
print("LEAF top8 2614:")
for n, c in la.most_common(8):
    print(f"  {n} {c:.2f}")
print("LEAF top8 2618:")
for n, c in lb.most_common(8):
    print(f"  {n} {c:.2f}")
# frames present only in one side with big weight
only_b = [(n, c) for n, c in cb.most_common(30) if ca.get(n, 0) < 0.05 * c]
print("2618-HEAVY (2618 >> 2614):")
for n, c in only_b[:8]:
    print(f"  {n} {c:.2f} vs {ca.get(n, 0):.2f}")
only_a = [(n, c) for n, c in ca.most_common(30) if cb.get(n, 0) < 0.05 * c]
print("2614-HEAVY (2614 >> 2618):")
for n, c in only_a[:8]:
    print(f"  {n} {c:.2f} vs {cb.get(n, 0):.2f}")
