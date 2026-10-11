#!/usr/bin/env python3
"""probe: real shapes/dtypes of representative tensors (per-tensor vs block fp8).

README says "fine-grained fp8 quantization with block size of 128" — verify
what the scale tensors actually look like (per-tensor [1]-ish vs one element
per 128x128 block), plus dtypes of the mixed-quant BF16 families.
"""
import json

from safetensors import safe_open

P = "/models/flash-next-fp8/"
wm = json.load(open(P + "model.safetensors.index.json"))["weight_map"]

targets = [
    # routed expert weight + its scale (the only quantized family)
    "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
    "model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale_inv",
    "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
    # mixed-quant BF16 families
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.2.self_attn.q_proj.weight",
    "model.language_model.layers.0.mlp.gate.weight",
    "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
    "model.language_model.layers.0.mlp.shared_expert.weight_scale_inv",
    # PLE table shard + scale
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
]

for t in targets:
    shard = wm.get(t)
    if shard is None:
        fam = t.rsplit(".", 1)[0]
        cands = sorted(k for k in wm if k.startswith(fam + "."))
        print("NOEXACT", t, "->", cands[:4])
        if not cands:
            continue
        t = cands[0]
        shard = wm[t]
    with safe_open(P + shard, framework="np") as f:
        sl = f.get_slice(t)
        print(t, "shape", sl.get_shape(), "dtype", sl.get_dtype())

# scale-shape census over ALL expert scales in one cheap shard
shard = wm["model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale_inv"]
shapes = {}
with safe_open(P + shard, framework="np") as f:
    for k in f.keys():
        if "weight_scale_inv" in k:
            s = tuple(f.get_slice(k).get_shape())
            shapes[s] = shapes.get(s, 0) + 1
print("SHARD_SCALE_SHAPES", shard, shapes)
print("FN8_SHAPES_OK")
