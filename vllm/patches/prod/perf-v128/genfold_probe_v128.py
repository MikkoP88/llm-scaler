#!/usr/bin/env python3
"""genfold_probe_v128.py — v128 generation_config fold certification probe.

Seed-pinned A/B discriminator proving which default temperature the
engine applies to requests that OMIT the parameter (the only fleet
surface a checkpoint generation_config fold can touch — explicit
request params always win).

Arms (3 repeats each, seed pinned per repeat so equal-param arms are
deterministic solo requests):
  A = temperature OMITTED        -> resolves via generation_config.json
  B = temperature 0.9 explicit   -> the swift-export default (pre-fold value)
  C = temperature 0.6 explicit   -> the vendor-rec fold target

Verdicts:
  PRE-FOLD  (file says 0.9):  A==B  and  A!=C   (omitted -> 0.9)
  POST-FOLD (file says 0.6):  A==C  and  A!=B   (omitted -> 0.6)

Equality = exact content match on the generated completion (same seed,
same params, solo request -> identical sampling draws). Under live
concurrent traffic the batch shape differs between back-to-back
requests, so logits wobble microscopically and near-ties at flat
temperatures can flip late tokens: measured on the live lane, omitted
vs explicit-0.9 matched exactly on 2/3 reps (148- and 179-char
identical completions) with one late-token divergence. The verdict is
therefore MARGIN-based over REPS=5 repeats (majority exact matches for
the claimed arm, ~zero for the other), with a first-30-chars prefix
metric as the wobble-robust secondary.

Usage: genfold_probe_v128.py pre|post [out_json]
"""
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8000"
MODE = sys.argv[1] if len(sys.argv) > 1 else "pre"
OUT = (sys.argv[2] if len(sys.argv) > 2 else
       "/root/build/lce1/genfold_probe_v128_%s.json" % MODE)
REPS = 5
SEED = 1234
MAXTOK = 64
MIN_MATCH = 3          # majority bar for the claimed arm
MAX_OTHER = 1          # allowance for the other arm

# deliberately open-ended / creative -> temperature-sensitive continuation
PROMPT = ("Write one vivid sentence describing a harbor at dawn. "
          "Do not explain, just the sentence.")


def gen(temperature, seed, rep):
    body = {
        "model": "qwen3.8-27b-fp8",
        "max_tokens": MAXTOK,
        "seed": seed + rep,            # pinned per repeat, stable across arms
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": PROMPT}],
    }
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-dummy"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    m = (d.get("choices") or [{}])[0]
    return {"finish": m.get("finish_reason"),
            "content": ((m.get("message") or {}).get("content") or "")}


def arm(temperature, label):
    rows = []
    for rep in range(REPS):
        r = gen(temperature, SEED, rep)
        rows.append(r)
        print(f"[{label} rep{rep}] finish={r['finish']} "
              f"len={len(r['content'])} head={r['content'][:48]!r}", flush=True)
        time.sleep(0.5)
    return rows


def eq(a, b):
    return [x["content"] == y["content"] for x, y in zip(a, b)]


def pref(a, b, n=30):
    return [x["content"][:n] == y["content"][:n] for x, y in zip(a, b)]


A = arm(None, "A-omitted")
B = arm(0.9, "B-temp0.9")
C = arm(0.6, "C-temp0.6")

a_eq_b = sum(eq(A, B))
a_eq_c = sum(eq(A, C))
a_pf_b = sum(pref(A, B))
a_pf_c = sum(pref(A, C))

if MODE == "pre":
    verdict = ("PRE_BASELINE_OK" if (a_eq_b >= MIN_MATCH and a_eq_c <= MAX_OTHER)
               else "PRE_BASELINE_UNEXPECTED")
else:
    verdict = ("POST_FOLD_CERTIFIED" if (a_eq_c >= MIN_MATCH and a_eq_b <= MAX_OTHER)
               else "POST_FOLD_FAILED")

out = {
    "tag": "genfold_probe_v128", "mode": MODE, "seed_base": SEED,
    "reps": REPS,
    "pairwise_exact_counts": {"A==B (omitted vs 0.9)": a_eq_b,
                              "A==C (omitted vs 0.6)": a_eq_c},
    "pairwise_prefix30_counts": {"A~B": a_pf_b, "A~C": a_pf_c},
    "bars": {"min_match_claimed": MIN_MATCH, "max_match_other": MAX_OTHER},
    "contents": {k: [r["content"] for r in v] for k, v in
                 (("A", A), ("B", B), ("C", C))},
    "verdict": verdict,
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)
print("GENFOLD_%s_DONE" % MODE.upper())
print(json.dumps({k: v for k, v in out.items() if k != "contents"}, indent=1))
print("WROTE", OUT)
