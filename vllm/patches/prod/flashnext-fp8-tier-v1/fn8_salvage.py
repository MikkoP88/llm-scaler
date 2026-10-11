#!/usr/bin/env python3
"""fn8 tier-dump salvage — sum counters from a CORRUPT (rank-interleaved)
dump by regex over the raw text.  TP2 telemetry law: both ranks flush
through one shared tmp path, so collided frames stay parseable-prefix only.

Usage: docker exec fn8-p2 python3 /stage/fn8_salvage.py /stage/bank/tier_p3.json
"""
import re
import sys

raw = open(sys.argv[1]).read()
print("file_len", len(raw))
# per-store field occurrences (prefix may end mid-store; that partial row's
# fields still count — acceptable over-count only if a field repeats in the
# torn row, which the writer's field order makes a +/-1 effect at most)
for field in ("pf_fills", "pf_promotes", "pf_waits", "pf_drops",
              "hits", "misses", "evictions", "dispatches"):
    vals = [int(v) for v in re.findall(r'"%s": (\d+)' % field, raw)]
    if vals:
        print("%-12s n=%3d sum=%d" % (field, len(vals), sum(vals)))
# torn-store boundary: last complete store key seen
stores = re.findall(r'"(\d+)": \{"layer"', raw)
print("stores_seen", len(stores), "first", stores[0] if stores else "-",
      "last_complete", stores[-1] if stores else "-")
