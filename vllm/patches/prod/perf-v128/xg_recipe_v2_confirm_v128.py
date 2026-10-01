#!/usr/bin/env python3
"""xg_recipe_v2_confirm_v128.py — lived confirmation of recipe v2 shapes.

Membership law (engine-side, model removed):
  CORRUPT: X[0-9]{1,4}(\\.[0-9]{1,4})?  and even X[0-9]{1,4}\\.?[0-9]{0,4}
           -> trailing ?-optionality after a bounded run destroys the bound
  SOUND:   alternation forms and {0,1} quantifier forms reject all >4-int-
           digit strings and accept the legal ones, in BOTH aw modes.

This probe confirms lived (restore lane, backend=auto, aw=True, spec
MTPx4, temp 0.6 + 1.0) that the sound shapes hold under the maximal
7-digit pull that broke the original (27/30 violations):

  C1 = B1 alternation   ^-?[0-9]{1,4}$|^-?[0-9]{1,4}\\.[0-9]{1,4}$
  C2 = B4 brace01       ^-?[0-9]{1,4}(\\.[0-9]{1,4}){0,1}$
  C0 = original corrupt ^-?[0-9]{1,4}(\\.[0-9]{1,4})?$   (control, must
       violate under pull — proves the pull is live on this lane/leg)

Expected: C1/C2 zero violations at both temps; C0 violates at pull.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer sk-dummy"}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

ARMS = [
    ("C0_corrupt", r"^-?[0-9]{1,4}(\.[0-9]{1,4})?$"),
    ("C1_alt", r"^-?[0-9]{1,4}$|^-?[0-9]{1,4}\.[0-9]{1,4}$"),
    ("C2_brace01", r"^-?[0-9]{1,4}(\.[0-9]{1,4}){0,1}$"),
]
PULL = ("The exact measurement is 3141592 (seven digits: 3141592). "
        "Put the value 3141592 in the t field.")


def post(pattern, temp):
    schema = {"type": "object",
              "properties": {"t": {"type": "string", "pattern": pattern}},
              "required": ["t"], "additionalProperties": False}
    body = {"model": "qwen3.8-27b-fp8", "max_tokens": 128, "temperature": temp,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "c", "strict": True, "schema": schema}},
            "messages": [{"role": "user", "content": PULL}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)


results = {}
for label, pat in ARMS:
    rx = re.compile(pat)
    for temp in (0.6, 1.0):
        tally = {"conform": 0, "VIOLATION": 0, "length": 0, "unparseable": 0}
        bad = []
        for i in range(N):
            d = post(pat, temp)
            fin = d["choices"][0]["finish_reason"]
            c = d["choices"][0]["message"]["content"]
            if fin == "length":
                tally["length"] += 1
                continue
            try:
                v = json.loads(c).get("t")
            except Exception:
                tally["unparseable"] += 1
                bad.append(repr(c[:30]))
                continue
            if v is not None and rx.fullmatch(str(v)):
                tally["conform"] += 1
            else:
                tally["VIOLATION"] += 1
                bad.append(repr(v))
            time.sleep(0.25)
        key = "%s@t%s" % (label, temp)
        results[key] = {"pattern": pat, "temp": temp, **tally, "bad": bad[:6]}
        print("[%s] %s violators=%s" % (key, json.dumps(tally), bad[:3]),
              flush=True)

ok = all(v["VIOLATION"] == 0 for k, v in results.items()
         if k.startswith(("C1", "C2")))
pull_live = results["C0_corrupt@t0.6"]["VIOLATION"] > 0
print("SUMMARY: sound-shapes-clean=%s control-pull-live=%s" % (ok, pull_live))
print("VERDICT RECIPE_V2_CERTIFIED" if ok and pull_live else "VERDICT INSPECT")
with open("/root/build/lce1/xg_recipe_v2_confirm_v128.json", "w") as fh:
    json.dump(results, fh, indent=1)
