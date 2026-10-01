#!/usr/bin/env python3
"""d4_split_v128.py — v128 WS-D D4 per-request split analysis (post-leg).

Reads the perreq JSONL harvested from the V1227_PERREQ=1 leg
(/root/v128_perreq.jsonl in-container -> host copy) and attributes the
recompute excess (P37 aggregate: 24.7% of computed, bar 15%) between:

  GDN-snap recompute — the hybrid cache manager snaps prefix hits to the
    GDN state boundary, so a FULL-HIT request still fires the remainder
    span. For a full-hit row (raw KV hit = prompt - new_suffix), the
    snap is EXACTLY:   snap = (prompt - generated) - cached
    and must equal     (prompt - generated) mod GRAN
    — a per-row law check that also confirms GRAN empirically (tested
    at 1024/2048/4096/8192).

  eviction re-prefill — low-hit rows: miss = prompt - generated - cached
    is the evicted/new span re-fired wholesale.

Suffix estimator: `generated` of the request itself (a turn's next-turn
new suffix is distributed as this turn's output; fleet-level exact).

Usage: d4_split_v128.py <perreq.jsonl> [out.json]
"""
import json
import sys
from collections import Counter

SRC = sys.argv[1] if len(sys.argv) > 1 else "/root/build/v128_stage/v128_perreq.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/build/lce1/d4_split_v128.json"

rows = []
with open(SRC) as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

# finish-class hygiene: keep completed generations only (ABORTED rows
# carry partial `generated` and poison the suffix estimator). The
# telemetry emits FinishReason .name — normalize case.
fin = Counter((r.get("finish") or "").lower() for r in rows)
rows = [r for r in rows if (r.get("finish") or "").lower() in ("stop", "length")]
# multi-turn relevance: prompts >= 8192 tokens (2+ GDN boundaries)
big = [r for r in rows if r.get("prompt", 0) >= 8192]

FULL_HIT_TOL = 512   # miss span (tokens) below which a row counts full-hit
GRANS = (1024, 2048, 4096, 8192)

def classify(r):
    suffix = r["generated"]
    miss = max(r["prompt"] - suffix - r["cached"], 0)
    return miss

cohorts = {"full": [], "low": []}
for r in big:
    (cohorts["full" if classify(r) <= FULL_HIT_TOL else "low"]).append(r)

out = {
    "tag": "d4_split_v128",
    "source": SRC,
    "rows_total": len(rows), "rows_bigprompt": len(big),
    "finish_classes": dict(fin),
    "full_hit_tol": FULL_HIT_TOL,
    "cohorts": {
        "full_hit": {"n": len(cohorts["full"])},
        "low_hit": {"n": len(cohorts["low"])},
    },
}

# ---- GDN law check on the full-hit cohort --------------------------------
full = cohorts["full"]
snap_rows = []
for r in full:
    hit_span = r["prompt"] - r["generated"]          # raw KV hit (est)
    snap = hit_span - r["cached"]                    # fired-though-cached
    snap_rows.append((hit_span, snap))

gran_fit = {}
for g in GRANS:
    diffs = [abs((hs % g) - s) for hs, s in snap_rows]
    gran_fit[g] = {
        "mean_abs_diff": round(sum(diffs) / len(diffs), 1) if diffs else None,
        "p50_abs_diff": sorted(diffs)[len(diffs) // 2] if diffs else None,
        "max_abs_diff": max(diffs) if diffs else None,
    }
best_gran = min(gran_fit, key=lambda g: gran_fit[g]["mean_abs_diff"]
                if gran_fit[g]["mean_abs_diff"] is not None else 1e9)

gdn_total = sum(s for _, s in snap_rows)
# low-hit rows also snap, bounded by gran each — bounded contribution
gdn_low_bound = len(cohorts["low"]) * best_gran
out["gdn_law"] = {
    "snap_total_full_hit": gdn_total,
    "snap_mean_full_hit": round(gdn_total / len(snap_rows), 1) if snap_rows else None,
    "hit_span_mod_gran_fit": gran_fit,
    "best_granularity": best_gran,
    "note": "mean_abs_diff ~0 at the true granularity = per-row law holds",
}

# ---- eviction cohort ------------------------------------------------------
low = cohorts["low"]
evict_misses = [max(r["prompt"] - r["generated"] - r["cached"], 0) for r in low]
out["eviction"] = {
    "miss_total": sum(evict_misses),
    "miss_mean": round(sum(evict_misses) / len(low), 1) if low else None,
    "miss_p50": sorted(evict_misses)[len(low) // 2] if low else None,
    "prompt_mean": round(sum(r["prompt"] for r in low) / len(low), 1) if low else None,
}

# ---- fleet-level split ----------------------------------------------------
tot_computed = sum(r["computed"] for r in big)
tot_suffix = sum(r["generated"] for r in big)
tot_excess = max(tot_computed - tot_suffix, 0)
evict_total = sum(evict_misses)
out["fleet_split"] = {
    "computed_total": tot_computed,
    "suffix_total": tot_suffix,
    "excess_total": tot_excess,
    "excess_share_of_computed_pct": round(tot_excess / tot_computed * 100, 1)
                                     if tot_computed else None,
    "gdn_snap_share_pct": round(gdn_total / tot_excess * 100, 1) if tot_excess else None,
    "eviction_share_pct": round(evict_total / tot_excess * 100, 1) if tot_excess else None,
    "residual_share_pct": round(
        max(tot_excess - gdn_total - evict_total, 0) / tot_excess * 100, 1)
        if tot_excess else None,
    "verdict": (
        "GDN-SNAP DOMINANT -> a boundary-aligned state checkpoint kernel "
        "(or boundary-aware scheduling) is the weeks-scale work package to"
        " cost; eviction is second-order"
        if tot_excess and gdn_total > evict_total * 2 else
        "EVICTION DOMINANT -> the win is cache/pool policy (WS-C track), "
        "not a GDN kernel; do not commit kernel effort"
        if tot_excess and evict_total > gdn_total * 2 else
        "MIXED -> both mechanisms material; cost them separately"
        if tot_excess else "no data"),
}

with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
print("WROTE", OUT)
