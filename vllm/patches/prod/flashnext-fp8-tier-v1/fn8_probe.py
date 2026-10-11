#!/usr/bin/env python3
"""probe: exact GDN/self_attn/mlp names in the checkpoint index"""
import json

wm = json.load(open("/models/flash-next-fp8/model.safetensors.index.json"))["weight_map"]
for prefix in (
    "model.language_model.layers.0.linear_attn.",
    "model.language_model.layers.2.self_attn.",
    "model.language_model.layers.0.mlp.",
    "model.language_model.layers.0.mlp.experts.0.",
    "model.language_model.layers.0.mlp.shared_expert.",
):
    hits = sorted(k.removeprefix(prefix) for k in wm if k.startswith(prefix))
    print(prefix, "->")
    for h in hits:
        print("   ", h)
    print()
