import collections
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/stage/bank/pyspy_d7_tp0.raw"
leaf = collections.Counter()
kw = collections.Counter()
tot = 0
KWS = [
    "ple", "moe", "tier", "eagle", "propose", "draft", "verify", "tolist",
    ".item()", "synchronize", "linear.py", "parameter.py", "_ops.py",
    "all_reduce", "all_gather", "gdn", "rms", "sampler", "spec", "mtp",
    "forward",
]
for line in open(path):
    line = line.rstrip("\n")
    if not line:
        continue
    m = re.match(r"^(.*) (\d+)$", line)
    if not m:
        continue
    stack, c = m.group(1), int(m.group(2))
    tot += c
    frames = stack.split(";")
    leaf[frames[-1]] += c
    low = stack.lower()
    for k in KWS:
        if k in low:
            kw[k] += c
print("TOTAL", tot)
print("== LEAF TOP 20")
for f, c in leaf.most_common(20):
    print(f"{c/tot*100:5.1f}% {c:6d} {f[:150]}")
print("== KEYWORD PRESENCE")
for k in KWS:
    print(f"{kw[k]/tot*100:5.1f}% {k}")
