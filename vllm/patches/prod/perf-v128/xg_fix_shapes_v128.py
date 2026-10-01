#!/usr/bin/env python3
"""xg_fix_shapes_v128.py — recipe-v2 pattern shapes vs the conversion trap.

Established (xg_membership_test_v128): X[0-9]{1,4}(\\.[0-9]{1,4})?
compiles to a grammar licensing 7 int digits (bound destroyed) in BOTH
any_whitespace modes; without the trailing optional group the bound
enforces exactly. This matrix tests FIX SHAPES for a bounded
decimal-number string that must survive conversion:

  B1 alternation of full branches (no optional group anywhere)
  B3 alternation INSIDE a group after the int run
  B4 {0,1} quantifier instead of ?
  B5 leading {0,4} + optional group (does only the FIRST bound die?)
  B7 optional ATOM (\\.? single char) after bounded run
  B6 unbounded control (harness sanity: must ACCEPT bad)
  B8 B1 with 2-4 int digits and 1-4 decimals (final recipe candidate)

Verdict per shape = membership of: 8digit, 7digit, 6digit, 5digit,
good4, good_dec, good_neg_dec. A correct shape rejects ALL >4-int-digit
strings and accepts good4/good_dec (and neg variants where intended).

Run INSIDE the container: /opt/venv/bin/python3 /tmp/xg_fix_shapes_v128.py
"""
import json

import xgrammar as xg
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/models/target", trust_remote_code=True)
TI = xg.TokenizerInfo.from_huggingface(tok)

SHAPES = [
    ("B1_alt", r"^-?[0-9]{1,4}$|^-?[0-9]{1,4}\.[0-9]{1,4}$"),
    ("B3_alt_in_group", r"^-?([0-9]{1,4}|[0-9]{1,4}\.[0-9]{1,4})$"),
    ("B4_brace01", r"^-?[0-9]{1,4}(\.[0-9]{1,4}){0,1}$"),
    ("B5_lead04", r"^-?[0-9]{0,4}(\.[0-9]{1,4})?$"),
    ("B7_opt_atom", r"^-?[0-9]{1,4}\.?[0-9]{0,4}$"),
    ("B6_unbounded", r"^-?[0-9]+(\.[0-9]+)?$"),
    ("B8_alt_recipe", r"^-?[0-9]{2,4}$|^-?[0-9]{2,4}\.[0-9]{1,4}$"),
]
STRINGS = [
    ("bad8", '{"t": "31415926"}'),
    ("bad7", '{"t": "3141592"}'),
    ("bad6", '{"t": "314159"}'),
    ("bad5", '{"t": "31415"}'),
    ("good4", '{"t": "2024"}'),
    ("good_dec", '{"t": "22.5"}'),
    ("good_neg", '{"t": "-3.75"}'),
    ("good2", '{"t": "72"}'),
]


def membership(cg, s):
    m = xg.GrammarMatcher(cg)
    if not m.accept_string(s):
        return "REJECT"
    return "ACCEPT" if m.is_completed() else "INCOMPLETE"


rows = []
for label, pat in SHAPES:
    schema = {"type": "object",
              "properties": {"t": {"type": "string", "pattern": pat}},
              "required": ["t"], "additionalProperties": False}
    compiler = xg.GrammarCompiler(tokenizer_info=TI)
    for aw in (True, False):
        try:
            cg = compiler.compile_json_schema(json.dumps(schema),
                                              any_whitespace=aw,
                                              strict_mode=True)
        except Exception as e:
            print("%-15s aw=%-5s COMPILE_FAIL %s" % (label, aw, e), flush=True)
            rows.append({"shape": label, "aw": aw, "verdict": "COMPILE_FAIL"})
            continue
        line = []
        for slabel, s in STRINGS:
            v = membership(cg, s)
            line.append("%s=%s" % (slabel, v))
            rows.append({"shape": label, "aw": aw, "string": slabel,
                         "verdict": v})
        print("%-15s aw=%-5s %s" % (label, aw, " ".join(line)), flush=True)

print()
ok_shapes = []
for label, _ in SHAPES:
    sub = [r for r in rows if r["shape"] == label]
    if not sub or any(r.get("verdict") == "COMPILE_FAIL" for r in sub):
        continue
    bad_ok = any(r["string"].startswith("bad") and r["verdict"] == "ACCEPT"
                 for r in sub)
    good_rej = any(r["string"].startswith("good") and r["verdict"] != "ACCEPT"
                   for r in sub)
    if not bad_ok and not good_rej:
        ok_shapes.append(label)
print("SOUND SHAPES (reject all bad, accept all good):", ok_shapes)
with open("/tmp/xg_fix_shapes_v128.json", "w") as fh:
    json.dump(rows, fh, indent=1)
print("WROTE /tmp/xg_fix_shapes_v128.json")
