#!/usr/bin/env python3
"""patch_fp8mq_defaults.py — bake validated fp8mq kernel defaults.

Diagnostic outcome (fp8-mtp4-v1, 2026-09-09/10, host 2x Arc Pro B70):
- Tile retune REFUTED: BLOCK_KV=8/1/1 is 1.6x worse @64k and 10x worse
  @262k vs the shipped BLOCK_KV=32/warps=4/stages=2. Shipped tile config
  is the measured optimum — DO NOT change it.
- The fp8+mtp4 long-context collapse tracks VLLM_FP8MQ_SPLITS: 64 ->
  flat ~5.8 s/step at every ctx (and boot-time mid scratch of
  [capture_max_B=512, Hq, SPLITS, Q_BLOCK=8, D+1=257] fp32 = 4.3 GB/GPU);
  32 -> 330-460 ms/step floor (2.15 GB scratch); lower SPLITS shrink both
  the scratch and the floor (validated by the diag4 ladder before bake).

This patcher rewrites the env-var DEFAULTS in the installed
vllm/v1/attention/ops/triton_fp8_mq.py. Defaults here = the SHIPPED
values; pass the validated combo at build time (e.g. --splits 8).
Knobs stay live for override/A-B. Fails loudly if the expected current
text is not found (protects against a drifted base image).
"""
import argparse
import py_compile
import re
import sys
from pathlib import Path

DEFAULT_TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_fp8_mq.py"
)

# knob -> (env var, shipped default)
KNOBS = {
    "block_kv": ("VLLM_FP8MQ_BLOCK_KV", 32),
    "splits": ("VLLM_FP8MQ_SPLITS", 32),
    "warps": ("VLLM_FP8MQ_STAGE1_WARPS", 4),
    "stages": ("VLLM_FP8MQ_STAGE1_STAGES", 2),
}

PROVENANCE = """\
# llm-scaler fp8-mtp4-v1 (2026-09-09): default retuned after live A/B on
# llm-scaler-exp:v1.2.5 (fp8-mtp4-v1/REPORT.md). Tiles 32/4/2 are the
# measured optimum (BLOCK_KV=8/1/1 was 1.6-10x WORSE at long ctx). The
# collapse driver is SPLITS-scaled (mid scratch + per-step cost):
# 64 -> ~5.8 s/step flat; 32 -> 330-460 ms/step floor; validated ladder
# value baked below. Env knobs stay live for override.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(DEFAULT_TARGET), type=Path)
    ap.add_argument("--block-kv", type=int, default=32)
    ap.add_argument("--splits", type=int, default=32)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--stages", type=int, default=2)
    args = ap.parse_args()

    vals = {
        "block_kv": args.block_kv,
        "splits": args.splits,
        "warps": args.warps,
        "stages": args.stages,
    }
    for k, v in vals.items():
        if v < 1 or (k in ("block_kv", "splits") and (v & (v - 1)) != 0):
            print(f"FAIL: invalid value {v} for {k}")
            return 1

    path = args.target
    src = path.read_text()
    orig = src

    for k, (env, shipped) in KNOBS.items():
        v = vals[k]
        old = f'os.environ.get("{env}", "{shipped}") or {shipped}'
        new = f'os.environ.get("{env}", "{v}") or {v}'
        if src.count(old) != 1:
            print(f"FAIL: expected exactly 1 occurrence of {old!r}, "
                  f"found {src.count(old)} — base image drifted?")
            return 1
        src = src.replace(old, new, 1)

    anchor = "_V51_FP8MQ_BLOCK_KV = max(1,"
    if src.count(anchor) != 1:
        print(f"FAIL: provenance anchor {anchor!r} not found")
        return 1
    src = src.replace(anchor, PROVENANCE + anchor, 1)

    if src == orig:
        print("NOTE: no changes needed (values already match)")
        return 0

    path.write_text(src)
    py_compile.compile(str(path), doraise=True)

    for line in src.splitlines():
        if re.search(r"os\.environ\.get\(\"VLLM_FP8MQ", line):
            print("PATCHED:", line.strip())
    print(f"OK: {path} -> BLOCK_KV={args.block_kv} SPLITS={args.splits} "
          f"warps={args.warps} stages={args.stages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
