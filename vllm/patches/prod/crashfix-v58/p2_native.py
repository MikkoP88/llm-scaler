#!/usr/bin/env python3
"""p2_native.py — analyze a py-spy speedscope WITH native frames.
Prints: top native leaves overall; deepest-frame weights under selected
python hotspots (what native call each hotspot is blocked in)."""
import collections
import json
import sys

path = sys.argv[1]
d = json.load(open(path))
frames = d["shared"]["frames"]
samples = []
for prof in d["profiles"]:  # aggregate all thread profiles
    samples.extend(zip(prof["samples"], prof["weights"]))

def is_native(fi):
    f = frames[fi]
    # py-spy: python frames carry a real .py file; native frames carry a
    # binary path or empty file
    fl = f.get("file") or ""
    return not fl.endswith(".py")

leaf = collections.Counter()
total = 0.0
hot_under = collections.defaultdict(collections.Counter)
HOT = ("_update_states_after_model_execute", "get_top_tokens",
       "propose_draft_token_ids", "sample_tokens")
for s, w in samples:
    if not s:
        continue
    total += w
    leaf[frames[s[-1]]["name"]] += w
    names = [frames[fi]["name"] for fi in s]
    seen = set()
    for h in HOT:
        if h in names and h not in seen:
            seen.add(h)
            # deepest frame in the whole stack = what it is blocked in
            hot_under[h][names[-1]] += w

print(f"TOTAL {total:.1f}s samples={len(samples)}")
print("TOP-15 LEAVES (any):")
for n, c in leaf.most_common(15):
    print(f"  {c:7.2f}s  {n}")
for h in HOT:
    print(f"DEEPEST-FRAME under {h}:")
    for n, c in hot_under[h].most_common(10):
        print(f"  {c:7.2f}s  {n}")
