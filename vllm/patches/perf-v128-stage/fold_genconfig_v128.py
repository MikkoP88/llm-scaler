#!/usr/bin/env python3
"""fold_genconfig_v128.py — v128 default-sampling hygiene fold.

The certified checkpoint /models/qwen3.8-27b-fp8/generation_config.json
ships temperature 0.9 (swift export, Sep 13) — vendor recommendation for
Qwen3-27B is 0.6 (thinking) / 0.7 (non-thinking); 0.9 is neither and is
the regime where the guided-JSON digit-attractor entry rate is maximal.
Single-file, single-key fold: temperature 0.9 -> 0.6 (top_p 0.95,
top_k 20 already match the vendor rec; everything else untouched).

Scope law: explicit request params ALWAYS win (protocol.py
to_sampling_params fills only OMITTED fields) — CC fleet sends explicit
temperatures and is unaffected; only param-omitting clients change.
Engine reads this file at START only — the fold arms on the next natural
restore/reboot, no lane disruption. NOT a degeneration fix (measured:
0.6 still shows ~50% digit-class on unbounded schemas) — this is
default-quality alignment only.

Safety: backup beside the original; JSON-validated reread; idempotent.
"""
import json
import shutil
import time

P = "/models/qwen3.8-27b-fp8/generation_config.json"
BAK = P + ".v128prefold.bak"

with open(P) as f:
    cfg = json.load(f)

print("pre-fold:", json.dumps({k: cfg.get(k) for k in
                                ("temperature", "top_p", "top_k", "min_p")}))

if cfg.get("temperature") == 0.6:
    print("already folded, no-op")
    raise SystemExit(0)

assert cfg.get("temperature") == 0.9, "unexpected temperature=%r" % cfg.get("temperature")

shutil.copy2(P, BAK)
cfg["temperature"] = 0.6
with open(P, "w") as f:
    json.dump(cfg, f, indent=4)
    f.write("\n")

with open(P) as f:
    check = json.load(f)
assert check["temperature"] == 0.6
assert check["top_p"] == cfg["top_p"] and check["top_k"] == cfg["top_k"]
keys_unchanged = [k for k in check if k != "temperature"
                  and check[k] != json.load(open(BAK)).get(k)]
assert not keys_unchanged, "collateral keys changed: %s" % keys_unchanged

print("post-fold:", json.dumps({k: check.get(k) for k in
                                 ("temperature", "top_p", "top_k", "min_p")}))
print("backup:", BAK)
print("FOLD_OK", time.strftime("%Y-%m-%dT%H:%M:%S"))
