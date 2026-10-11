#!/usr/bin/env python3
"""fn8-t3 validation part 4: fn8_p2_run.tier_summary against the store
test's dumped telemetry (layer 7: 3 dispatches, 6 misses, 1 hit, 2 evict)."""
import importlib.util
import json

spec = importlib.util.spec_from_file_location("p2run", "/tmp/fn8_p2_run.py")
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)

with open("/tmp/fn8_tier_dump2.json") as f:
    tier = json.load(f)
summ = p2.tier_summary(tier)
assert summ["stores"] == 1
assert summ["dispatches"] == 3 and summ["misses"] == 6 and summ["hits"] == 1
assert summ["evictions"] == 2 and summ["hit_rate"] == round(1 / 7, 4)
assert summ["live_dispatches"] == 3  # no dummy dispatches in the CPU test
print("TIER_SUMMARY_OK", json.dumps(summ, sort_keys=True))
