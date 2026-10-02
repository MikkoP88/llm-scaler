#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# llm-scaler perf-v133 P69 leg A2 readout: pressure rows + perreq engine numbers.
import json
import sys
from collections import defaultdict

STAGE = "/root/build/v133_stage"

# --- pressure rows (client-side) ------------------------------------
rows = []
with open(f"{STAGE}/wsc_pressure_v133a.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
ok = [r for r in rows if r["status"] == "ok"]
finishes = defaultdict(int)
for r in rows:
    finishes[r.get("finish")] += 1
print(f"pressure rows={len(rows)} ok={len(ok)} finishes={dict(finishes)}")
print(f"TOTAL completion_tokens={sum(r['completion_tokens'] for r in rows)}")
print(f"TOTAL prompt_tokens={sum(r['prompt_tokens'] for r in rows)}")
by = defaultdict(lambda: [0, 0, 0.0, 0])
for r in rows:
    b = by[r["cohort"]]
    b[0] += 1
    b[1] += r["completion_tokens"]
    b[2] += r["wall"]
    b[3] += r["prompt_tokens"]
for k in sorted(by):
    v = by[k]
    print(
        f"  {k}: n={v[0]} compl={v[1]} prompt={v[3]} "
        f"avg_wall={v[2] / v[0]:.0f}s"
    )

# --- perreq rows (engine-side, v128 telemetry) ----------------------
try:
    pr = []
    with open(f"{STAGE}/wsc_perreq_v133a.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pr.append(json.loads(line))
except FileNotFoundError:
    print("perreq file MISSING")
    sys.exit(0)
print(f"perreq rows={len(pr)}")
if pr:
    print("sample keys:", sorted(pr[0].keys()))
