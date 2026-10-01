#!/usr/bin/env python3
"""xg_repro_violation_v128.py — reproduce the {1,4} 7-digit anomaly.

Established so far on the restore lane:
  ^A[0-9]{3}Z$  and  ^A[0-9]{16}Z$  enforce EXACTLY (12/12 + 4/4) under
  maximal digit pull -> xgrammar bounds are enforced.
  Yet ^-?[0-9]{1,4}(\\.[0-9]{1,4})?$ produced '2024052' (7 int digits,
  finish=stop) in 2/6 runs with a weak-pull weather prompt.

Candidate mechanisms:
  M1 schema-shape: the optional decimal group (\\.[0-9]{1,4})? or the
     leading -? corrupts the conversion for the int-part bound.
  M2 flaky spec-mask leak: rare, load/timing-dependent.

Arms x N (strong 7-digit pull "use value 3141592" AND the original weak
weather prompt):
  A0 full recipe pattern   ^-?[0-9]{1,4}(\\.[0-9]{1,4})?$
  A1 no optional group     ^-?[0-9]{1,4}$
  A2 no minus              ^[0-9]{1,4}(\\.[0-9]{1,4})?$
  A3 sentinel-wrapped      ^A[0-9]{1,4}Z$   (known-good shape, control)

Violation = value parses but fails its own pattern with finish=stop.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer sk-dummy"}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

ARMS = [
    ("A0_full", r"^-?[0-9]{1,4}(\.[0-9]{1,4})?$"),
    ("A1_noopt", r"^-?[0-9]{1,4}$"),
    ("A2_nominus", r"^[0-9]{1,4}(\.[0-9]{1,4})?$"),
    ("A3_wrapped", r"^A[0-9]{1,4}Z$"),
]
PULL = ("The exact measurement is 3141592 (seven digits: 3141592). "
        "Put the value 3141592 in the t field.")
WEAK = "Tokyo weather (guess values). Fill the JSON now."


def post(pattern, prompt, name):
    schema = {"type": "object",
              "properties": {"t": {"type": "string", "pattern": pattern}},
              "required": ["t"], "additionalProperties": False}
    body = {"model": "qwen3.8-27b-fp8", "max_tokens": 128, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": name, "strict": True, "schema": schema}},
            "messages": [{"role": "user", "content": prompt}]}
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
    tally = {"conform": 0, "VIOLATION": 0, "unparseable": 0, "length": 0}
    bad_vals = []
    for prompt_tag, prompt in (("pull", PULL), ("weak", WEAK)):
        for i in range(N // 2):
            d = post(pat, prompt, label)
            fin = d["choices"][0]["finish_reason"]
            c = d["choices"][0]["message"]["content"]
            if fin == "length":
                tally["length"] += 1
                continue
            try:
                v = json.loads(c).get("t")
            except Exception:
                tally["unparseable"] += 1
                bad_vals.append(repr(c[:40]))
                continue
            if v is not None and rx.fullmatch(str(v)):
                tally["conform"] += 1
            else:
                tally["VIOLATION"] += 1
                bad_vals.append("%s:%r" % (prompt_tag, v))
            time.sleep(0.3)
    results[label] = {"pattern": pat, **tally, "bad": bad_vals[:8]}
    print("[%s] %s" % (label, json.dumps(tally)), flush=True)
    if bad_vals:
        print("   violators:", bad_vals[:8], flush=True)

viol = {k: v["VIOLATION"] for k, v in results.items()}
print("SUMMARY violations:", json.dumps(viol))
if not any(viol.values()):
    print("VERDICT NO_REPRO (anomaly did not recur — likely flaky; see note)")
elif viol.get("A0_full") and not viol.get("A3_wrapped"):
    print("VERDICT M1: optional-group/minus shape corrupts int bound")
else:
    print("VERDICT see matrix")
with open("/root/build/lce1/xg_repro_violation_v128.json", "w") as fh:
    json.dump(results, fh, indent=1)
