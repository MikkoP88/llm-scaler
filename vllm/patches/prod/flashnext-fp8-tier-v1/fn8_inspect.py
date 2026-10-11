#!/usr/bin/env python3
"""fn8 checkpoint index inspection (flashnext-fp8-tier-v1 P0).
Prints weight-name groups, PLE ngram shard naming, mtp/visual keys, and the
safetensors header (dtype/shape) of every tensor whose name contains
'ngram' — settles the shard-index parse + scale-layout branches.
"""
import json
import struct
import sys
from collections import Counter

ROOT = "/models/flash-next-fp8"
INDEX = ROOT + "/model.safetensors.index.json"

with open(INDEX) as f:
    idx = json.load(f)
wm = idx["weight_map"]
print(f"total entries: {len(wm)}")
print(f"metadata: {idx.get('metadata')}")
print(f"shards: {len(set(wm.values()))}")

ple = sorted(k for k in wm if "ple" in k or "ngram" in k)
mtp = sorted(k for k in wm if "mtp" in k)
vis = sorted(k for k in wm if "visual" in k)
print(f"\n=== PLE/ngram keys ({len(ple)}) ===")
for k in ple[:8]:
    print(f"  {k}  -> {wm[k]}")
if len(ple) > 8:
    print(f"  ... {len(ple)-8} more; last 3:")
    for k in ple[-3:]:
        print(f"  {k}  -> {wm[k]}")
print(f"\n=== mtp keys ({len(mtp)}) first/last ===")
for k in mtp[:4] + mtp[-2:]:
    print(f"  {k}  -> {wm[k]}")
print(f"\n=== visual keys ({len(vis)}) ===")
for k in vis[:4]:
    print(f"  {k}  -> {wm[k]}")

# structural fingerprint of layer keys
c = Counter()
for k in wm:
    parts = k.split(".")
    # blank the numbers
    c[".".join(p if not p.isdigit() else "#" for p in parts)] += 1
print(f"\n=== distinct name shapes (top 40) ===")
for name, n in c.most_common(40):
    print(f"  {n:6d}  {name}")

# ---- safetensors headers of every distinct ngram-related tensor ----
def st_header(path, wanted):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    return {k: v for k, v in hdr.items() if any(w in k for w in wanted)}

targets = {"ngram", "hyper_connection", "input_mix", "block_inject", "hc_norm",
           "norm_key", "norm_query", "norm_conv", "key_proj", "value_proj"}
seen_shards = {}
for k in ple:
    shard = wm[k]
    if shard not in seen_shards:
        seen_shards[shard] = st_header(f"{ROOT}/{shard}", targets)
    for name, meta in seen_shards[shard].items():
        if name == k:
            print(f"  HDR {name}: dtype={meta.get('dtype')} shape={meta.get('shape')} splits={meta.get('data_offsets')}")
print(f"\nshards scanned for headers: {len(seen_shards)}")

# scale tensors anywhere
print("\n=== any *scale* keys mentioning ngram across all shards scanned ===")
for shard, hdr in seen_shards.items():
    for name, meta in hdr.items():
        if "scale" in name:
            print(f"  {shard}: {name} dtype={meta.get('dtype')} shape={meta.get('shape')}")
