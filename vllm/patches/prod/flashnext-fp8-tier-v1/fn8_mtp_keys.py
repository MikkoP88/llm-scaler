#!/usr/bin/env python3
"""fn8 P4 prep 2: every checkpoint key that is NOT a target decoder layer and
NOT a routed-expert tensor — the exact name set the MTP draft loader sees."""
import json

BASE = "/models/flash-next-fp8"
idx = json.load(open(f"{BASE}/model.safetensors.index.json"))
keys = sorted(idx["weight_map"])
skip = ("model.layers.", "mtp.", ".mlp.experts.")
sel = [k for k in keys if not any(s in k for s in skip)]
for k in sel:
    print(k)
print(f"-- {len(sel)} non-layer non-mtp keys, {len(keys)} total")
