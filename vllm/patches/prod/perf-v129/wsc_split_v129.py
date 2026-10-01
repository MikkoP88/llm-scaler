#!/usr/bin/env python3
"""wsc_split_v129.py — perf-v129 P52 WS-C cohort analysis.

Joins the pressure driver rows (wsc_pressure_v129.jsonl: cohort/turn/
prompt_tokens per turn) with the engine-side perreq rows
(/root/v128_perreq.jsonl: prompt/cached/computed/generated per finished
request) by exact prompt-size match within completion order, then
answers the WS-C lever question:

  1. HEAD RETENTION — on SHARED-cohort turn>=2 rows that MISSED, does
     cached floor at the ~shared-head size (LRU already pins hot
     shared prefixes -> pinning lever DEAD) or drop to ~0 (lever
     EXISTS)?
  2. SMALL residency — does the 4k cohort stay ~fully cached?
  3. Waste — recompute share of computed on turn>=2 (P48 comparand:
     18.8% excess, 90.1% eviction-class, ~640 tok/affected-req).

Caps verdicts: WSC_SPLIT_DONE, HEAD_RETAINED|HEAD_EVICTED|NO_MISS_ROWS,
SMALL_RESIDENT|SMALL_PRESSURED, WASTE_PCT=<x> MISS_TOK_PER_AFFECTED=<x>.

Usage: wsc_split_v129.py [driver_jsonl] [perreq_jsonl]
"""
import json
import sys
from collections import defaultdict

DRV = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/build/lce1/wsc_pressure_v129.jsonl"
PRQ = sys.argv[2] if len(sys.argv) > 2 else \
    "/root/build/lce1/wsc_perreq_v129.jsonl"

HEAD_TARGET_TOKENS = 8000          # driver target; actual ~0.75x
HEAD_FLOOR = HEAD_TARGET_TOKENS * 0.6   # retention bar (>=4.8k cached)


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


drv = [r for r in load(DRV) if r.get("status") == "ok"
       and r.get("prompt_tokens")]
prq = load(PRQ)

# --- join: engine row -> driver row by exact prompt size, consumed in
# order (both streams are completion-ordered; sizes repeat across
# cohorts but rarely back-to-back unmatched) ---
avail = list(prq)
joined = []
for d in sorted(drv, key=lambda r: r.get("ts", "")):
    p = d["prompt_tokens"]
    best = None
    for i, q in enumerate(avail):
        if q.get("prompt") == p:
            best = i
            break
    if best is None:  # tolerance band +-0.7% (none observed; fallback)
        for i, q in enumerate(avail):
            if abs(q.get("prompt", 0) - p) <= max(8, p * 0.007):
                best = i
                break
    if best is not None:
        joined.append({**d, "cached": avail[best]["cached"],
                       "computed": avail[best]["computed"],
                       "hit_ratio": avail[best]["hit_ratio"],
                       "ttft": avail[best].get("ttft")})
        avail.pop(best)

print("joined %d/%d driver rows (unmatched engine rows: %d)"
      % (len(joined), len(drv), len(avail)), flush=True)

# --- per-conv deltas from driver prompts (exact per-turn new suffix) ---
byprompt = defaultdict(list)
for j in joined:
    byprompt[j["id"]].append(j)
for cid, rows in byprompt.items():
    rows.sort(key=lambda r: r["turn"])
    prev = None
    for r in rows:
        r["delta"] = (r["prompt_tokens"] - prev) if prev is not None \
            else r["prompt_tokens"]
        prev = r["prompt_tokens"]

# --- cohort x turn table ---
print("\n%-7s %-5s %4s %9s %9s %9s %9s %8s"
      % ("cohort", "turn", "n", "prompt", "cached", "computed",
         "hit", "ttft_p50"))
perrow = []
for cohort in ("SHARED", "PRIV", "SMALL"):
    for turn in range(1, 9):
        rows = [j for j in joined if j["cohort"] == cohort
                and j["turn"] == turn]
        if not rows:
            continue
        n = len(rows)
        mp = sum(r["prompt_tokens"] for r in rows) / n
        mc = sum(r["cached"] for r in rows) / n
        mco = sum(r["computed"] for r in rows) / n
        mh = sum(r["hit_ratio"] for r in rows) / n
        tts = sorted(r["ttft"] or 0 for r in rows)
        print("%-7s %-5d %4d %9.0f %9.0f %9.0f %9.3f %8.2f"
              % (cohort, turn, n, mp, mc, mco, mh,
                 tts[len(tts) // 2] if tts else 0))
        perrow.extend(rows)

# --- miss analysis (turn>=2): miss = computed - delta ---
miss_rows = [r for r in joined if r["turn"] >= 2
             and r["computed"] > r["delta"] * 1.5]
waste = sum(max(r["computed"] - r["delta"], 0) for r in joined
            if r["turn"] >= 2)
comp2 = sum(r["computed"] for r in joined if r["turn"] >= 2)
shared_miss = [r for r in miss_rows if r["cohort"] == "SHARED"]
small = [r for r in joined if r["cohort"] == "SMALL" and r["turn"] >= 2]

print("\nMISS rows (turn>=2, computed>1.5x delta): %d  "
      "(SHARED %d / PRIV %d / SMALL %d)"
      % (len(miss_rows), len(shared_miss),
         len([r for r in miss_rows if r["cohort"] == "PRIV"]),
         len([r for r in miss_rows if r["cohort"] == "SMALL"])))
for r in sorted(miss_rows, key=lambda x: -x["computed"])[:12]:
    print("  %s t%d prompt=%d cached=%d computed=%d delta=%d miss=%d"
          % (r["id"], r["turn"], r["prompt_tokens"], r["cached"],
             r["computed"], r["delta"], r["computed"] - r["delta"]))

waste_pct = 100.0 * waste / comp2 if comp2 else 0.0
miss_per = (sum(r["computed"] - r["delta"] for r in miss_rows)
            / len(miss_rows)) if miss_rows else 0.0
print("\nWASTE_PCT=%.1f (turn>=2 recompute share of computed; "
      "P48: 18.8%%)" % waste_pct)
print("MISS_TOK_PER_AFFECTED=%.0f (P48: ~640)" % miss_per)

# --- verdict 1: head retention ---
if shared_miss:
    floors = sorted(r["cached"] for r in shared_miss)
    floor_med = floors[len(floors) // 2]
    verdict1 = "HEAD_RETAINED" if floor_med >= HEAD_FLOOR else \
        "HEAD_EVICTED"
    print("SHARED miss cached floor: min=%d med=%d (head target ~%d, "
          "bar %d) -> %s" % (floors[0], floor_med,
                             HEAD_TARGET_TOKENS, HEAD_FLOOR, verdict1))
else:
    verdict1 = "NO_MISS_ROWS"
    print("no SHARED miss rows -> %s (pressure insufficient or LRU "
          "held everything)" % verdict1)

# --- verdict 2: small residency ---
if small:
    hr = sum(r["hit_ratio"] for r in small) / len(small)
    verdict2 = "SMALL_RESIDENT" if hr >= 0.95 else "SMALL_PRESSURED"
    print("SMALL turn>=2 mean hit=%.3f -> %s" % (hr, verdict2))
else:
    verdict2 = "SMALL_NO_ROWS"

out = {"joined": len(joined), "driver_rows": len(drv),
       "engine_rows": len(prq), "waste_pct": round(waste_pct, 2),
       "miss_rows": len(miss_rows), "miss_tok_per_affected":
           round(miss_per, 1), "head_verdict": verdict1,
       "small_verdict": verdict2,
       "rows": [{k: r[k] for k in ("id", "cohort", "turn",
                                   "prompt_tokens", "cached",
                                   "computed", "delta", "hit_ratio",
                                   "ttft")}
                for r in sorted(joined, key=lambda x: (x["cohort"],
                                                       x["id"],
                                                       x["turn"]))]}
with open("/root/build/lce1/wsc_split_v129.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nWSC_SPLIT_DONE verdict1=%s verdict2=%s" % (verdict1, verdict2))
