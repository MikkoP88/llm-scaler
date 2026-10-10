#!/usr/bin/env python3
"""fable_temp.py -- add/change temperature in the local fable tier
(qwen3.8-27b-fp8-fable) litellm_params via line surgery (no yaml
round-trip: the embedded litellm_system_prompt block stays byte-stable).

Usage: fable_temp.py <config-in> <config-out> <value>
"""
import sys

import yaml

CONFIG, OUT, VAL = sys.argv[1], sys.argv[2], sys.argv[3]

lines = open(CONFIG, encoding="utf-8").read().split("\n")

start = next(i for i, l in enumerate(lines)
             if l.strip() == "- model_name: qwen3.8-27b-fp8-fable")
end = next(i for i in range(start + 1, len(lines))
           if lines[i].startswith("  - model_name:"))
lp = next(i for i in range(start, end) if lines[i] == "    litellm_params:")

# drop any existing temperature lines in this block's params
block = [l for i, l in enumerate(lines[lp + 1:end], lp + 1)
         if not l.startswith("      temperature:")]

out = (lines[:lp + 1]
       + ["      temperature: " + VAL]
       + block
       + lines[end:])
open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")

d = yaml.safe_load(open(OUT, encoding="utf-8"))
f = [m for m in d["model_list"] if m["model_name"] == "qwen3.8-27b-fp8-fable"][0]
p = f["litellm_params"]
d_in = yaml.safe_load(open(CONFIG, encoding="utf-8"))
f_in = [m for m in d_in["model_list"]
        if m["model_name"] == "qwen3.8-27b-fp8-fable"][0]
assert p["temperature"] == float(VAL), "temperature not set"
assert len(d["model_list"]) == 14, "model count changed"
assert p["litellm_system_prompt"] == f_in["litellm_params"]["litellm_system_prompt"], \
    "prompt block changed"
print("TEMP", p["temperature"], "PROMPT_CHARS", len(p["litellm_system_prompt"]))
print("FABLE_TEMP_OK")
