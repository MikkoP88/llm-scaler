#!/usr/bin/env python3
"""fable_full_import.py — replace the qwen3.8-27b-fp8-fable
litellm_system_prompt body with the VERBATIM content of the fetched
prompt file (full import, user directive 2026-10-08).

Usage: fable_full_import.py <config-in> <prompt-file> <config-out>
Validates the round-trip (yaml-loaded prompt == file content) and the
rest of the config structure before writing.
"""
import hashlib
import sys

import yaml

CONFIG, PROMPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

lines = open(CONFIG, encoding="utf-8").read().split("\n")
prompt = open(PROMPT, encoding="utf-8").read().rstrip("\n")

start = next(i for i, l in enumerate(lines)
             if l.strip() == "- model_name: qwen3.8-27b-fp8-fable")
end = next(i for i in range(start + 1, len(lines))
           if lines[i].startswith("  - model_name:"))
key = next(i for i in range(start, end)
           if lines[i].strip() == "litellm_system_prompt: |")
body_start = key + 1
body_end = body_start
while body_end < end and (lines[body_end].startswith("        ")
                          or lines[body_end].strip() == ""):
    body_end += 1

new_body = [(("        " + l) if l.strip() else "") for l in prompt.split("\n")]
out = lines[:body_start] + new_body + lines[body_end:]
open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")

d = yaml.safe_load(open(OUT, encoding="utf-8"))
f = [m for m in d["model_list"] if m["model_name"] == "qwen3.8-27b-fp8-fable"][0]
sp = f["litellm_params"]["litellm_system_prompt"]
assert sp.strip() == prompt.strip(), "round-trip mismatch"
assert len(d["model_list"]) == 14, "model count changed"
print("OLD_BODY_END_LINE", body_end, "->", "NEW_BODY_LINES", len(new_body))
print("PROMPT_CHARS", len(prompt), "YAML_PROMPT_CHARS", len(sp))
print("MD5_PROMPT", hashlib.md5(sp.encode()).hexdigest())
print("HEAD:", repr(sp[:60]))
print("TAIL:", repr(sp[-60:]))
print("FABLE_FULL_IMPORT_OK")
