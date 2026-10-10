#!/usr/bin/env python3
"""toks_sum.py — tok/s summarizer for the P83 tok/s benchmark round (2026-10-06).

Reads a wsc_pressure driver jsonl (one record per request turn), filters
status == "ok" rows, sums prompt/completion tokens, and prints one
machine-greppable line. Optional second arg = storm wall seconds from the
leg's PRESSUREB_DONE line; when present, aggregate rates are appended.

Usage: python3 toks_sum.py <driver_jsonl> [wall_seconds]
Output:
  TOKS_SUM rows=<n> ok=<n> bad=<n> ptok=<sum> ctok=<sum>
  TOKS_RATES out_tok_s=<f> total_tok_s=<f> mean_ctok=<f>
"""
import json
import sys


def main() -> int:
    path = sys.argv[1]
    wall = float(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = ok = 0
    ptok = ctok = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows += 1
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("status") != "ok":
                continue
            ok += 1
            ptok += rec.get("prompt_tokens") or 0
            ctok += rec.get("completion_tokens") or 0
    print("TOKS_SUM rows=%d ok=%d bad=%d ptok=%d ctok=%d"
          % (rows, ok, rows - ok, ptok, ctok))
    if ok:
        print("TOKS_RATES mean_ctok=%.1f%s"
              % (ctok / float(ok),
                 "" if wall is None
                 else " wall=%.0fs out_tok_s=%.1f total_tok_s=%.1f"
                      % (wall, ctok / wall, (ptok + ctok) / wall)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
