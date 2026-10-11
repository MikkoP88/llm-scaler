#!/usr/bin/env python3
"""fn8 P1 histogram-chain unit test (CPU, container fn8-t2, patch applied)."""
import json
import os

os.environ["V1X_ROUTING_DUMP"] = "/tmp/rh.json"

# banked CPU dry-run idiom (PHASES Round 2): force XPU platform before the
# vllm import chain that binds kernels/linear
import vllm.platforms as vp
from vllm.platforms.xpu import XPUPlatform

XPUPlatform.current_device = lambda: "cpu"
vp.current_platform = XPUPlatform()

import numpy as np  # noqa: E402

import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec  # noqa: E402

assert rec._V1X_HIST is not None, "V1X_ROUTING_DUMP not picked up"
h = rec._V1X_HIST

rng = np.random.RandomState(0)
N, L, K, E = 64, 48, 10, 512
hv = np.zeros((N, L, K), dtype=np.int16)
for l in range(L):
    hot = [(l * 7 + i) % E for i in range(20)]
    hotset = set(hot)
    cold = [e for e in range(E) if e not in hotset]
    for n in range(N):
        for k in range(K):
            hv[n, l, k] = (
                hot[rng.randint(20)] if rng.rand() < 0.8 else cold[rng.randint(len(cold))]
            )

rec._v1x_tap_routing_histogram(hv)
rec._v1x_tap_routing_histogram(hv)
h.flush()

d = json.load(open("/tmp/rh.json"))
total = sum(map(sum, d["counts"]))
print("scatters", d["scatters"], "tapped", d["tapped_tokens"],
      "sum", total, "shape", len(d["counts"]), len(d["counts"][0]))
assert d["scatters"] == 2
assert d["tapped_tokens"] == 2 * N
assert total == 2 * N * L * K
# hot-set sanity: each layer's 20 hot experts carry ~80%
for l in range(L):
    hot = {(l * 7 + i) % E: i for i in range(20)}
    row = d["counts"][l]
    hot_mass = sum(row[e] for e in hot)
    assert 0.7 < hot_mass / sum(row) < 0.9, (l, hot_mass / sum(row))
print("HIST_CHAIN_OK")
