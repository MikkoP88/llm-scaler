#!/usr/bin/env python3
"""m0_harvest_v127.py — v127 M0-live harvest: merge per-rank runmax dumps,
report the TP2 cross-rank magnitude question, emit calibrate-ready stats.

The collector (m0_live_collect_v127.py) writes /root/m0live_runmax_<pid>.pt
INSIDE the lane container, one per TP worker, every 20 s (accumulated
elementwise max since boot). TP2 splits the GDN heads disjointly: rank 0's
feature (h, k) and rank 1's feature (h, k) are DIFFERENT physical heads,
but the M1 bridge / M2 kernel load ONE scales file per rank from a fixed
path — so a shared file must cover both head sets. Merging elementwise
across ranks is SAFE by construction (scale <= optimal per feature: at
worst under-uses the grid; never overflows) but costs precision up to the
cross-rank per-feature ratio — this script MEASURES that ratio so the
shared-vs-rank-keyed decision is data-driven, not guessed.

Steps:
  1. docker cp the dumps out of the container (they die with the
     container — harvest BEFORE any destroy).
  2. Merge: per-rank tensors kept separate; cross-rank elementwise max =
     the conservative shared-file stats.
  3. Report: F, coverage, raw-e4m3 over-ceiling fraction (the P29 mode),
     cross-rank ratio distribution, dead features.
  4. Write m0live_merged_stats.pt  ->  calibrate_offline.py --stats ...

Usage:
  host     python3 m0_harvest_v127.py            (docker cp pull + host torch)
  in-ctn   /opt/venv/bin/python3 m0_harvest_v127.py   (torch lives there)
Both write <STAGE>/m0live_merged_stats.pt; the in-container run writes
to /root/m0live_merged_stats.pt — docker cp it out afterwards.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

CONTAINER = sys.argv[1] if len(sys.argv) > 1 else "lsv-test"
IN_CONTAINER = bool(glob.glob("/root/m0live_runmax_*.pt")) and not os.path.exists(
    "/usr/bin/docker"
)
STAGE = "/root" if IN_CONTAINER else "/root/build/v127_stage/m0live_dumps"
CEILING = 448.0


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr, sep="\n")
        raise SystemExit(f"harvest: command failed: {cmd}")
    return r.stdout


def main() -> int:
    import torch

    os.makedirs(STAGE, exist_ok=True)
    if IN_CONTAINER:
        print("[harvest] in-container mode: reading /root/m0live_runmax_*.pt")
    else:
        listing = sh(
            f"docker exec {CONTAINER} sh -c 'ls /root/m0live_runmax_*.pt 2>/dev/null'"
        ).split()
        if not listing:
            raise SystemExit("harvest: no dumps in container — run traffic first")
        for path in listing:
            sh(f"docker cp {CONTAINER}:{path} {STAGE}/")
        print(f"[harvest] pulled {len(listing)} dumps from {CONTAINER}")

    ranks = []
    for path in sorted(glob.glob(f"{STAGE}/m0live_runmax_*.pt")):
        obj = torch.load(path, map_location="cpu")
        rm = obj["runmax"].float().flatten()
        ranks.append((path, rm, obj))
        print(
            f"[harvest] {os.path.basename(path)}: F={rm.numel()} "
            f"pools={obj.get('pools')} max={float(rm.max()):.4f} "
            f"nonzero={int((rm > 0).sum())}"
        )

    merged = ranks[0][1].clone()
    for _, rm, _ in ranks[1:]:
        merged = torch.maximum(merged, rm)

    live = merged[merged > 0]
    print(f"[harvest] merged: F={merged.numel()} nonzero={int((merged > 0).sum())}")
    if live.numel():
        q = torch.quantile(
            live, torch.tensor([0.01, 0.5, 0.9, 0.99, 1.0])
        )
        print(
            "[harvest] live runmax quantiles: "
            + " ".join(f"p{int(p*100)}={float(v):.4f}" for p, v in zip([1, 50, 90, 99, 100], q))
        )
        over = int((live > CEILING).sum())
        print(
            f"[harvest] raw-e4m3 over-ceiling(>{CEILING:.0f}): {over} "
            f"({100.0 * over / live.numel():.2f}% of live features) — the "
            "P29 format-impossibility share scaled-space removes"
        )

    if len(ranks) == 2:
        a, b = ranks[0][1], ranks[1][1]
        both = (a > 0) & (b > 0)
        if int(both.sum()):
            ratio = torch.maximum(a[both] / b[both], b[both] / a[both])
            qr = torch.quantile(ratio, torch.tensor([0.5, 0.9, 1.0]))
            print(
                "[harvest] cross-rank ratio (n=%d): p50=%.2f p90=%.2f "
                "max=%.2f  — shared-file precision loss is bounded by this "
                "ratio (safe direction only); decide shared vs rank-keyed "
                "scales from it" % (int(both.sum()), float(qr[0]), float(qr[1]), float(qr[2]))
            )
    else:
        print("[harvest] note: %d ranks found (expected 2 for TP2)" % len(ranks))

    torch.save(
        {"runmax": merged, "ranks": [r[1] for r in ranks], "source": STAGE},
        f"{STAGE}/m0live_merged_stats.pt",
    )
    print(f"[harvest] wrote {STAGE}/m0live_merged_stats.pt")
    print("[harvest] next: calibrate_offline.py --stats m0live_merged_stats.pt "
          "--no-clamp-up --out v127_scales_e4m3.pt")
    print("V127_M0HARVEST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
