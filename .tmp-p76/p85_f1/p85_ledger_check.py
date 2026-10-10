#!/usr/bin/env python3
"""P85 ledger analyzer — envelope scan of the v85 forensic step ledger.

Gate: every logged accepted value in [1, num_spec+1] (spec x4 -> [1,5]),
rep counts reported per line; any out-of-envelope value or a rep burst
immediately before an engine silence = M1/M2 signature caught host-side.

Usage: python3 p85_ledger_check.py <ledger-file> [num_spec=4]
"""
import sys


def main() -> int:
    path = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    hi = lim + 1
    n = 0
    bad = 0
    rep_lines = 0
    rep_total = 0
    last20 = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            parts = line.split()
            if len(parts) < 4:
                continue
            rep = int(parts[2].split("=")[1])
            if rep > 0:
                rep_lines += 1
                rep_total += rep
            # ledger line: "<step> n=<n> rep=<r> acc=<v1> <v2> ..."
            acc = []
            for k, tok in enumerate(parts):
                if tok.startswith("acc="):
                    if len(tok) > 4:
                        acc.append(int(tok[4:]))
                    acc.extend(int(v) for v in parts[k + 1:])
                    break
            for v in acc:
                if v < 1 or v > hi:
                    bad += 1
            last20.append(line.rstrip())
            if len(last20) > 20:
                last20.pop(0)
    print(f"lines={n} out_of_envelope={bad} (envelope [1,{hi}])")
    print(f"repair: lines_with_rep={rep_lines} total_rows_repaired={rep_total}")
    print("--- last 20 ledger lines ---")
    for l in last20:
        print(l)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
