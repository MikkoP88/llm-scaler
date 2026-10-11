#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 P1 — routing histogram -> expert profile (Strata STRP analog).

Input: routing_hist.json written by patch_01's V1X_ROUTING_DUMP tap
(per-(layer, expert) activation counts over a profiling corpus).

Output profile JSON:
  - ``ranked``: [[layer, expert, count], ...] global heat-desc (the Strata
    STRP v1 analog: ranked (layer, expert) pairs for initial cache fill).
  - ``coverage_global``: {k: cumulative activation share of the top-k
    ranked pairs} at the banked k marks.
  - ``coverage_per_layer_quota``: cumulative share when each layer gets the
    SAME number of slots (the R4.2g per-layer quota view — what a
    per-layer-slotted cache actually achieves, vs the unconstrained global
    ranking).
  - ``layer_gini`` / hottest-experts-per-layer for a quick eyeball.

P1 GATE (PHASES.md): >=75% cumulative coverage within <=8192 experts on
code-ish traffic.

--split even|odd reports one TP2 whole-expert half (expert_id % 2) — the P2
partition view: per-rank coverage = what each rank's own cache can hit.
"""
import argparse
import json
import sys

K_MARKS = (512, 1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576)


def load_hist(path):
    with open(path) as f:
        h = json.load(f)
    counts = h["counts"]  # [L][E]
    layer_tokens = h.get("layer_tokens")
    return counts, layer_tokens, h


def rank(counts, split):
    triples = []
    for l, row in enumerate(counts):
        for e, c in enumerate(row):
            if split == "even" and e % 2 != 0:
                continue
            if split == "odd" and e % 2 != 1:
                continue
            if c:
                triples.append((l, e, c))
    triples.sort(key=lambda t: (-t[2], t[0], t[1]))
    return triples


def coverage_at(triples, total, marks):
    out = {}
    cum = 0
    mi = 0
    marks = sorted(marks)
    for i, (_l, _e, c) in enumerate(triples, 1):
        cum += c
        while mi < len(marks) and i >= marks[mi]:
            out[marks[mi]] = cum / total if total else 0.0
            mi += 1
    while mi < len(marks):
        out[marks[mi]] = cum / total if total else 0.0
        mi += 1
    return out


def per_layer_quota_coverage(counts, slots_per_layer):
    """Each layer ranks its own experts; cache = q slots/layer (R4.2g view)."""
    total = sum(sum(r) for r in counts)
    cum = 0
    for row in counts:
        s = sorted(row, reverse=True)[:slots_per_layer]
        cum += sum(s)
    return cum / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hist", help="routing_hist.json from patch_01 tap")
    ap.add_argument("--out", default=None, help="profile JSON path")
    ap.add_argument(
        "--split", choices=["none", "even", "odd"], default="none",
        help="restrict to one whole-expert TP2 half (expert_id parity)",
    )
    args = ap.parse_args()

    counts, layer_tokens, raw = load_hist(args.hist)
    num_layers = len(counts)
    num_experts = len(counts[0]) if num_layers else 0

    triples = rank(counts, args.split)
    total = sum(c for _l, _e, c in triples)

    profile = {
        "v": 1,
        "split": args.split,
        "num_layers": num_layers,
        "num_experts_seen": num_experts,
        "scatters": raw.get("scatters"),
        "tapped_tokens": raw.get("tapped_tokens"),
        "total_activations": total,
        "distinct_active": len(triples),
        "ranked": [[l, e, c] for l, e, c in triples],
        "coverage_global": {
            str(k): v
            for k, v in coverage_at(triples, total, K_MARKS).items()
        },
        "coverage_per_layer_quota": {
            str(q): per_layer_quota_coverage(counts, q)
            for q in (4, 8, 16, 32, 64, 128, 256, 512)
        },
    }

    # per-layer heat summary (top-3 experts + concentration)
    per_layer = []
    for l, row in enumerate(counts):
        lt = sum(row)
        if not lt:
            per_layer.append({"layer": l, "tokens": 0})
            continue
        top3 = sorted(range(num_experts), key=lambda e: -row[e])[:3]
        per_layer.append({
            "layer": l,
            "tokens": lt,
            "top3": [[e, row[e], row[e] / lt] for e in top3],
        })
    profile["per_layer"] = per_layer

    # gate readout
    cov8k = profile["coverage_global"].get("8192", 0.0)
    profile["gate_75pct_within_8192"] = {
        "coverage": cov8k,
        "pass": bool(cov8k >= 0.75),
    }

    out = args.out or args.hist.replace("routing_hist", "profile")
    with open(out, "w") as f:
        json.dump(profile, f, indent=1)

    print(f"layers={num_layers} experts={num_experts} "
          f"activations={total} distinct={len(triples)}")
    print("coverage_global:")
    for k in K_MARKS:
        if str(k) in profile["coverage_global"]:
            print(f"  top-{k:>6}: {profile['coverage_global'][str(k)]:.4f}")
    print("coverage_per_layer_quota:")
    for q, v in profile["coverage_per_layer_quota"].items():
        print(f"  q={q:>4}/layer: {v:.4f}")
    print(f"GATE >=75% within 8192: "
          f"{'PASS' if profile['gate_75pct_within_8192']['pass'] else 'FAIL'} "
          f"({cov8k:.4f})")
    print("WROTE", out)
    return 0 if profile["gate_75pct_within_8192"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
