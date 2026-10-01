#!/usr/bin/env python3
"""xg_membership_test_v128.py — engine-side grammar membership law.

Lived evidence (restore lane, backend=auto, any_whitespace=True):
  ^A[0-9]{3}Z$ / ^A[0-9]{16}Z$ / ^-?[0-9]{1,4}$  enforce exactly
  ^-?[0-9]{1,4}(\\.[0-9]{1,4})?$ emitted '3141592' (7 int digits)
  ^[0-9]{1,4}(\\.[0-9]{1,4})?$   same corruption

This script removes the model from the loop: compile the schema exactly
as vllm's xgrammar backend does (GrammarCompiler.compile_json_schema,
strict_mode=True, both any_whitespace modes), then test STRING
MEMBERSHIP via GrammarMatcher.accept_string + is_completed.
Accepted            => the grammar licenses that string (conversion fact)
Rejected/incomplete => the string is grammar-ILLEGAL (any lived
                       violation must then be a mask-application leak)

Run INSIDE the container: /opt/venv/bin/python3 /tmp/xg_membership_test_v128.py
"""
import json

import xgrammar as xg
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/models/target", trust_remote_code=True)
TI = xg.TokenizerInfo.from_huggingface(tok)

PATTERNS = [
    ("A0_full", r"^-?[0-9]{1,4}(\.[0-9]{1,4})?$"),
    ("A1_noopt", r"^-?[0-9]{1,4}$"),
    ("A2_nominus", r"^[0-9]{1,4}(\.[0-9]{1,4})?$"),
    ("A3_wrapped", r"^A[0-9]{1,4}Z$"),
]
STRINGS = [
    ('bad7digit', '{"t": "3141592"}'),
    ('good4', '{"t": "2024"}'),
    ('good_dec', '{"t": "22.5"}'),
    ('good_wrap', '{"t": "A123Z"}'),
    ('ws_inside', '{"t": "22.5"\n}'),
]


def membership(cg, s):
    m = xg.GrammarMatcher(cg)
    if not m.accept_string(s):
        return "REJECTED", None
    if m.is_completed():
        return "ACCEPTED", None
    return "INCOMPLETE", None


rows = []
for label, pat in PATTERNS:
    schema = {"type": "object",
              "properties": {"t": {"type": "string", "pattern": pat}},
              "required": ["t"], "additionalProperties": False}
    compiler = xg.GrammarCompiler(tokenizer_info=TI)
    for aw in (True, False):
        cg = compiler.compile_json_schema(json.dumps(schema),
                                          any_whitespace=aw, strict_mode=True)
        for slabel, s in STRINGS:
            verdict, _ = membership(cg, s)
            rows.append({"pattern": label, "any_whitespace": aw,
                         "string": slabel, "verdict": verdict})
            print("%-10s aw=%-5s %-9s -> %s" % (
                label, aw, slabel, verdict), flush=True)

print()
print("bad7digit verdicts (the law):")
for r in rows:
    if r["string"] == "bad7digit":
        print("  %-10s aw=%-5s %s" % (r["pattern"], r["any_whitespace"],
                                      r["verdict"]))
with open("/tmp/xg_membership_v128.json", "w") as fh:
    json.dump(rows, fh, indent=1)
print("WROTE /tmp/xg_membership_v128.json")
