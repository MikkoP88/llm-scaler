#!/usr/bin/env python3
"""fn8 P4 prep: dump shapes of every NON-EXPERT mtp.* weight + the target's
top-level hyper_connection_mixer/norm weights, from the checkpoint headers."""
import json
import re

from safetensors import safe_open

BASE = "/models/flash-next-fp8"
idx = json.load(open(f"{BASE}/model.safetensors.index.json"))
wm = idx["weight_map"]
want = [k for k in wm if k.startswith("mtp.") and ".mlp.experts." not in k]
want += [k for k in wm if "hyper_connection_mixer" in k or re.match(
    r"model\.(norm|embed_tokens|lm_head)\.", k)]
files = sorted({wm[k] for k in want})
shapes = {}
for fn in files:
    with safe_open(f"{BASE}/{fn}", framework="pt") as f:
        for k in f.keys():
            if k in want and k not in shapes:
                shapes[k] = tuple(f.get_slice(k).get_shape())
for k in sorted(shapes):
    print(k, shapes[k])
