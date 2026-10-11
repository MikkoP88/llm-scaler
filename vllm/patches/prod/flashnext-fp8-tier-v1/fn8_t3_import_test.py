#!/usr/bin/env python3
"""fn8-t3 validation: import patched modules, exercise inert + pure paths."""
import json
import os
import sys

# Part 1: inert import (FN8_TIERED unset) — module must import cleanly.
import vllm.model_executor.layers.fused_moe.fused_moe_modular_method as m  # noqa: E402
import vllm.model_executor.offloader.uva as u  # noqa: E402
print("IMPORT_OK", m.__file__.rsplit("/", 1)[-1], u.__file__.rsplit("/", 1)[-1])
assert m._FN8_TIERED is False, "FN8_TIERED leaked into env?"
assert u._FN8_TIERED is False, "FN8_TIERED leaked into env?"
assert u._FN8_TIER_OFFLOAD_PARAMS == {
    "w13_weight", "w2_weight",
    "w13_weight_scale_inv", "w2_weight_scale_inv",
}, u._FN8_TIER_OFFLOAD_PARAMS
print("INERT_ENV_OK")

# Part 2: seed hot-list parser (pure) against a synthetic histogram.
seed_path = "/tmp/fn8_seed.json"
hist = {"counts": [
    [5, 0, 9, 1],   # layer 0: order 2,0,3 (skip zero-count 1)
    [0, 0, 0, 0],   # layer 1: empty
    [7, 7, 1, 0],   # layer 2: tie 0,1 -> stable sort keeps 0 first
]}
with open(seed_path, "w") as f:
    json.dump(hist, f)
hot = m._fn8_seed_hot_lists(seed_path)
assert hot[0] == [2, 0, 3], hot[0]
assert hot[1] == [], hot[1]
assert hot[2][0] == 0 and sorted(hot[2][:2]) == [0, 1] and hot[2][2] == 2, hot[2]
hot_missing = m._fn8_seed_hot_lists("/tmp/definitely-absent.json")
assert hot_missing == {}, hot_missing
print("SEED_PARSER_OK", {k: v[:3] for k, v in hot.items()})

# Part 3: telemetry flush law (atomic tmp+rename, never-kill).
# NOTE: V1X_TIER_DUMP must be set in the environment BEFORE import
# (module reads it at import time); docker exec -e supplies it here.
assert m._FN8_TIER_DUMP == "/tmp/fn8_tier_dump.json", m._FN8_TIER_DUMP
tele = m._Fn8TierTelemetry()
tele.register(3, {"layer": 3, "dispatches": 1})
tele.flush()
with open("/tmp/fn8_tier_dump.json") as f:
    payload = json.load(f)
assert payload["stores"]["3"]["dispatches"] == 1
assert not os.path.exists("/tmp/fn8_tier_dump.json.tmp")
print("TELEMETRY_OK", sorted(payload), payload["slots"])

# Part 4: class-shape sanity on the store (no instantiation — needs GPU).
store_cls = m._Fn8TieredStore
for attr in ("__init__", "_load_slot", "_sync_slot_map",
             "ensure_resident", "dispatch"):
    assert callable(getattr(store_cls, attr)), attr
assert callable(m._fn8_store_for)
print("STORE_SHAPE_OK")
print("FN8_T3_ALL_OK")
sys.exit(0)
