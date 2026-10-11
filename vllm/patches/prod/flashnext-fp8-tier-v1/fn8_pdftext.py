#!/usr/bin/env python3
"""stdlib PDF text extractor (Flate streams, Tj/TJ literal strings)"""
import re
import sys
import zlib

data = open(sys.argv[1], "rb").read()
out = []
for m in re.finditer(rb"stream\r?\n", data):
    start = m.end()
    end = data.find(b"endstream", start)
    if end < 0:
        continue
    try:
        dec = zlib.decompress(data[start:end])
    except Exception:
        continue
    if b"BT" not in dec:
        continue
    txt = []
    for tm in re.finditer(rb"\((?:\\.|[^\\()])*\)", dec):
        s = tm.group(0)[1:-1]
        s = s.replace(b"\\(", b"(").replace(b"\\)", b")").replace(b"\\\\", b"\\")
        # approximate glyph spacing between TJ array items
        txt.append(s.decode("latin-1"))
    if txt:
        out.append("".join(txt))
open(sys.argv[2], "w").write("\n".join(out))
print("chunks", len(out), "chars", sum(len(o) for o in out))
