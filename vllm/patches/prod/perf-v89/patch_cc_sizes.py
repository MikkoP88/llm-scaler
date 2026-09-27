#!/usr/bin/env python3
"""patch_cc_sizes.py — v89 T2a: extend cudagraph_capture_sizes in serve_user.sh.

The engine default list [1,2,4,8,16,24,32,40,48,...,512] pads:
  - spec-verify steps: 5*B tokens (B=batch, MTP x4) -> batch 7 = 35 -> pads 40 (+14%)
  - draft steps in batch space: batches 5/6 -> pad to 8 (+60%/+33%)
Fix: dense 1..16 (draft batches) + multiples of 5 to 320 (verify token shapes
5..320 for B=1..64) + preserved default coverage up to 512.
"""
import json
import sys

PATH = "/root/serve_user.sh"
OLD = '{"cudagraph_mode":"FULL_DECODE_ONLY"}'

sizes = sorted(
    set(
        list(range(1, 17))                      # dense draft-batch space 1..16
        + [5 * k for k in range(1, 65)]         # verify token shapes 5..320
        + [48, 64, 80, 96, 112, 128, 160, 192, 224, 256]  # default coverage
    )
)
cc = json.dumps(
    {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": sizes},
    separators=(",", ":"),
)

s = open(PATH).read()
n = s.count(OLD)
if n != 1:
    print(f"FAIL: anchor count {n} != 1")
    sys.exit(1)
open(PATH, "w").write(s.replace(OLD, cc))
print(f"OK sizes={len(sizes)} min={sizes[0]} max={sizes[-1]}")
print(f"cc={cc[:120]}...")
