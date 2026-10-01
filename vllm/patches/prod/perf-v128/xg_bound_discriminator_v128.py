#!/usr/bin/env python3
"""xg_bound_discriminator_v128.py — why did {1,4} emit 7 digits?

Context: sanity + xg_active_check on the RESTORE lane (backend=auto,
any_whitespace=True, spec MTPx4) produced finish=stop values like
'2024052' under pattern ^-?[0-9]{1,4}(\\.[0-9]{1,4})?$. Guidance is
provably ACTIVE on this lane (sentinel ^ZZ[0-9]{2}$ conformant 5/6, one
in-grammar ws-runaway). Two live hypotheses:

  H-CONV  xgrammar's regex->grammar conversion approximates bounded
          quantifiers (e.g. {1,4} ~ unbounded): violations are
          SYSTEMATIC, structurally clean, only-ever the bounded class
          (long digit runs where digits are legal).
  H-SPEC  speculative MTPx4 tokens bypass/leak the grammar mask
          intermittently: violations are SPORADIC and can appear
          anywhere (wrong chars, missing sentinel tail, broken JSON).

Discriminator: pattern ^A[0-9]{3}Z$ with a prompt that BEGS for a
7-digit constant (pi-derived), so the model's pull maximally violates
the 3-digit bound. 12 runs. Also a control arm with spec-preserving
schema (^A[0-9]{16}Z$, bound ABOVE the pull) to separate "can't count
digits" from "can't enforce at all".

Readouts:
  all runs conform          -> bound enforced; earlier '2024052' needs
                              a different explanation (rerun arm)
  long digit runs, A/Z kept -> H-CONV (digit-count approximated)
  missing A/Z / stray chars -> H-SPEC (mask leak)
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer sk-dummy"}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

TIGHT = {"type": "object",
         "properties": {"code": {"type": "string",
                                 "pattern": r"^A[0-9]{3}Z$"}},
         "required": ["code"], "additionalProperties": False}
WIDE = {"type": "object",
        "properties": {"code": {"type": "string",
                                "pattern": r"^A[0-9]{16}Z$"}},
        "required": ["code"], "additionalProperties": False}

PROMPT = ("The secret code is the first seven digits of the constant "
          "3141592 (exactly seven digits: 3141592). Emit it in the code "
          "field. Use the value 3141592.")


def post(schema, name):
    body = {"model": "qwen3.8-27b-fp8", "max_tokens": 128, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": name, "strict": True, "schema": schema}},
            "messages": [{"role": "user", "content": PROMPT}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)


def classify(content):
    v = None
    try:
        v = json.loads(content).get("code")
    except Exception:
        return "unparseable", None
    if v is None:
        return "missing", None
    if re.fullmatch(r"A[0-9]{3}Z", v):
        return "tight_conform", v
    if re.fullmatch(r"A[0-9]+Z", v):
        return "WRONG_DIGIT_COUNT(A..Z kept)", v
    if re.match(r"^A", v) and v.endswith("Z"):
        return "WRONG_CHARS(A/Z kept)", v
    return "STRUCT_BROKEN", v


res = {"tight_conform": 0}
vals = []
for i in range(N):
    d = post(TIGHT, "tight")
    c = d["choices"][0]["message"]["content"]
    tag, v = classify(c)
    res[tag] = res.get(tag, 0) + 1
    vals.append(v)
    print(f"[tight run{i+1}] finish={d['choices'][0]['finish_reason']} "
          f"{tag} val={v!r}", flush=True)
    time.sleep(0.4)

wide_ok = 0
for i in range(4):
    d = post(WIDE, "wide")
    c = d["choices"][0]["message"]["content"]
    try:
        v = json.loads(c).get("code")
        ok = bool(re.fullmatch(r"A[0-9]{16}Z", v))
    except Exception:
        v, ok = c[:40], False
    wide_ok += ok
    print(f"[wide run{i+1}] finish={d['choices'][0]['finish_reason']} "
          f"conform={ok} val={v!r}", flush=True)
    time.sleep(0.4)

print("SUMMARY tight:", json.dumps(res), "wide_conform=%d/4" % wide_ok)
if res.get("tight_conform") == N and wide_ok == 4:
    print("VERDICT BOUND_ENFORCED (rerun anomaly — earlier 7-digit needs re-examination)")
elif res.get("WRONG_DIGIT_COUNT(A..Z kept)", 0) > 0 and not res.get("STRUCT_BROKEN"):
    print("VERDICT H-CONV: digit-count bound not enforced, structure kept")
elif res.get("STRUCT_BROKEN") or res.get("WRONG_CHARS(A/Z kept)"):
    print("VERDICT H-SPEC: mask leak (illegal structure emitted)")
else:
    print("VERDICT MIXED — inspect per-run tags")
