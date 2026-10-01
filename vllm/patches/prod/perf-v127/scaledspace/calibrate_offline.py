#!/usr/bin/env python3
"""calibrate_offline.py — v127 scaled-space M0: per-feature scale calibration.

Computes per-feature (per-head-axis) scale factors that map GDN/SSM state
values into the fp8_e4m3 representable range:  stored = state * scale[f],
scale[f] = min(1.0, 0.98 * 448 / runmax[f]).  e4m3 ceiling 448 is the whole
reason raw fp8 state was FORMAT-IMPOSSIBLE (P29: legit states >= 1280); the
per-feature scale makes the stored value fit BY CONSTRUCTION.

Inputs (either):
  --stats <.pt>  a dict with 'runmax' (1-D per-feature max |value|) — the
                 artifact the P29M-style telemetry leg writes.
  --dump  <.pt>  raw state dump tensor [N, F] (runmax computed here).
  --demo         synthetic prior from the banked P29M constants
                 (|ba| runmax 16.0-20.2 across runs) — smoke-test only,
                 NEVER for production scales.

Output: scales file (default /root/v127_scales_e4m3.pt) with
  {'scale': fp32[F], 'inv_scale': fp32[F], 'runmax': fp32[F],
   'fmt': 'fp8_e4m3', 'ceiling': 448.0, 'guard': 0.98, 'source': <input>}

Report: max stored value, % features clipped at the guard, worst-case e4m3
half-step error bound (rel step 2^-3 -> half-step 2^-4 of runmax), and the
error vs a hypothetical NO-scale e4m3 cast (the P29 failure mode).

Run anywhere python3+torch exists (host or container; CPU is fine).
"""
from __future__ import annotations

import argparse
import sys

CEILING = 448.0  # e4m3 max normal on XPU
GUARD = 0.98     # keep 2% headroom for intra-run growth spikes
REL_HALF_STEP = 2.0 ** -4  # 3 mantissa bits -> half-ULP of the leading binade


def load_runmax(args) -> tuple:
    import torch

    if args.demo:
        # P29M banked prior: per-run |ba| runmax 16.0-20.2 over a small
        # feature axis; broaden with a plausible head spread. Smoke-test
        # only — production scales MUST come from a live telemetry leg.
        base = torch.tensor([16.0, 18.5, 20.2, 17.3, 19.8, 16.9, 20.0, 18.0])
        spread = torch.linspace(0.3, 1.0, base.numel())
        runmax = (base * (1.0 + 60.0 * spread)).float()  # up to ~1280+
        return runmax, "demo(P29M prior — smoke-test only)"
    if args.stats:
        obj = torch.load(args.stats, map_location="cpu")
        return obj["runmax"].float().cpu(), args.stats
    if args.dump:
        t = torch.load(args.dump, map_location="cpu").float().cpu()
        if t.dim() != 2:
            raise SystemExit(f"dump must be [N, F], got {tuple(t.shape)}")
        return t.abs().amax(dim=0), args.dump
    raise SystemExit("one of --stats/--dump/--demo is required")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats")
    ap.add_argument("--dump")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument(
        "--no-clamp-up",
        action="store_true",
        help="M2 policy: drop the max=1.0 scale clamp so small-runmax "
        "features scale UP onto the full e4m3 grid (required at the M3 "
        "bake — the clamp is the M1-bridge-era guard, M2_DESIGN §2)",
    )
    ap.add_argument("--out", default="/root/v127_scales_e4m3.pt")
    args = ap.parse_args()

    import torch

    runmax, source = load_runmax(args)
    if (runmax <= 0).any():
        dead = int((runmax <= 0).sum())
        runmax = torch.where(runmax > 0, runmax, torch.ones_like(runmax))
        print(f"[calibrate] {dead} dead features (runmax<=0) pinned to 1.0")

    if args.no_clamp_up:
        # M2 policy (M2_DESIGN §2): drop the M1-bridge-era max=1.0 clamp —
        # scaling small runmax UP onto the full e4m3 grid is the point; the
        # clamp cannot fix the P23F small-feature flush.
        scale = (GUARD * CEILING / runmax).float()
    else:
        scale = torch.clamp(GUARD * CEILING / runmax, max=1.0).float()
    inv_scale = (1.0 / scale).float()
    stored_max = float((runmax * scale).max())
    clipped = int(((runmax * scale) > CEILING).sum())

    # Absolute worst-case storage error per feature: the half-step of the
    # feature's own top stored binade (<= guard*ceiling * 2^-4) maps back
    # through inv_scale to <= runmax * 2^-4 in state units.
    err_bound_state = runmax * REL_HALF_STEP
    worst_abs = float(err_bound_state.max())

    # The P29 no-scale counterfactual: values above the ceiling cannot be
    # represented at all (saturation/NaN on XPU cast).
    over_raw = int((runmax > CEILING).sum())
    raw_pct = 100.0 * over_raw / runmax.numel()

    torch.save(
        {
            "scale": scale,
            "inv_scale": inv_scale,
            "runmax": runmax,
            "fmt": "fp8_e4m3",
            "ceiling": CEILING,
            "guard": GUARD,
            "source": source,
        },
        args.out,
    )

    print(f"[calibrate] source={source} features={runmax.numel()}")
    print(f"[calibrate] runmax: min={float(runmax.min()):.3f} "
          f"max={float(runmax.max()):.3f}")
    print(f"[calibrate] stored max {stored_max:.1f} / ceiling {CEILING:.0f} "
          f"(clipped features: {clipped})")
    print(f"[calibrate] worst per-element storage error bound "
          f"~{worst_abs:.4f} (state units; e4m3 half-step x runmax)")
    print(f"[calibrate] NO-SCALE counterfactual: {over_raw} features "
          f"({raw_pct:.1f}%) exceed the e4m3 ceiling raw — the P29 "
          "format-impossibility mode scaled-space removes")
    print(f"[calibrate] wrote {args.out}")
    print("V127_CALIBRATE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
