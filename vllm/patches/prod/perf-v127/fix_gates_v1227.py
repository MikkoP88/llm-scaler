#!/usr/bin/env python3
"""fix_gates_v1227.py — repair gates_v1227_lane.sh on the host (2026-10-01).

mk_gates_v1227_lane.py's step-4 anchor matched only a PREFIX of the
v126_no_p29h_markers line (it ended at ...print s+0}'" without the line's
closing `)"`), so the inserted v127 block split that command mid-line and
stranded the orphaned `)"` at the end of the v127_no_bak_files line
(...grep -c bak_v127 || true")")"). Bash then parsed a mega command
substitution swallowing the whole v127 block; ck() received "0" plus the
swallowed newlines as its actual-value argument and the string compare
failed -> GATE-FAIL v126_no_p29h_markers expected=0 got=0 (visible
identical values, differing invisible bytes). All v127 gates themselves
ran green; the committed image was untouched (host-side script bug).

This restores the v126 line to its certified full form and removes the
orphaned paren, leaving the v127 block intact after the complete line.
"""
P = "/root/build/gates_v1227_lane.sh"

with open(P) as f:
    t = f.read()

BAD1 = ("ck v126_no_p29h_markers 0 \"$(docker exec lsv-gate sh -c "
        "\"grep -rc -e 'P29H' -e 'P29M' -e 'v126_p29h' "
        "$SP/model_executor/layers/mamba/ | awk -F: '{s+=\\$2} END{print s+0}'\"\n")
GOOD1 = ("ck v126_no_p29h_markers 0 \"$(docker exec lsv-gate sh -c "
         "\"grep -rc -e 'P29H' -e 'P29M' -e 'v126_p29h' "
         "$SP/model_executor/layers/mamba/ | awk -F: '{s+=\\$2} END{print s+0}'\")\"\n")
assert t.count(BAD1) == 1, "truncated v126 line count=%d" % t.count(BAD1)
t = t.replace(BAD1, GOOD1)

BAD2 = 'grep -c bak_v127 || true")")"'
GOOD2 = 'grep -c bak_v127 || true")"'
assert t.count(BAD2) == 1, "orphan paren count=%d" % t.count(BAD2)
t = t.replace(BAD2, GOOD2)

with open(P, "w") as f:
    f.write(t)

print("fixed", P)
print("v126 line complete:", t.count("print s+0}'\")\""))
print("v127 gates present:", t.count("ck v127_"))
print("double-paren orphans:", t.count(')")")"'))
