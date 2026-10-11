#!/usr/bin/env python3
"""stdlib PDF text extractor, round 2: Type0/CID fonts via ToUnicode CMaps.

Strategy: collect ALL Flate streams.  Streams containing `beginbfchar`/
`beginbfrange` are ToUnicode CMaps -> merged into one global code->unicode
map (2-byte codes assumed; 1-byte maps merge harmlessly into the same
space).  Every other BT-bearing stream is treated as page content: walk it
linearly, decoding <hex> and (literal) strings through the map, inserting
spaces at TJ-array gaps (negative kern numbers roughly = space).
"""
import re
import sys
import zlib

data = open(sys.argv[1], "rb").read()

streams = []
for m in re.finditer(rb"stream\r?\n", data):
    start = m.end()
    end = data.find(b"endstream", start)
    if end < 0:
        continue
    raw = data[start:end].rstrip(b"\r\n")
    try:
        dec = zlib.decompress(raw)
    except Exception:
        dec = raw  # some streams are uncompressed
    streams.append(dec)

cmap = {}
for s in streams:
    if b"beginbfchar" not in s and b"beginbfrange" not in s:
        continue
    for lo, hi, uni in re.findall(
            rb"beginbfrange\s+<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>",
            s):
        lo_i, hi_i = int(lo, 16), int(hi, 16)
        base = int(uni, 16)
        for i in range(hi_i - lo_i + 1):
            cmap[lo_i + i] = chr(base + i)
    for code, uni in re.findall(
            rb"beginbfchar\s+<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", s):
        # bfchar blocks list pairs repeatedly; grab all pairs in one pass
        pass
    # bfchar bodies: <code> <uni> pairs (multiple per block)
    for block in re.findall(rb"beginbfchar(.*?)endbfchar", s, re.S):
        for code, uni in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            u = int(uni, 16)
            if u > 0x10FFFF:
                continue
            cmap[int(code, 16)] = chr(u)

print("CMAP entries", len(cmap), file=sys.stderr)


def decode_hex(h):
    if len(h) % 2:
        h = "0" + h
    out = []
    for i in range(0, len(h), 4):  # 2-byte codes
        code = int(h[i:i + 4], 16)
        out.append(cmap.get(code, ""))
    return "".join(out)


pages = []
for s in streams:
    if b"BT" not in s or b"beginbf" in s:
        continue
    txt = []
    # walk tokens: hex strings, literal strings, TJ separators
    for m in re.finditer(rb"<([0-9A-Fa-f\s]+)>|\((?:\\.|[^\\()])*\)|(-?\d+\.?\d*)", s):
        tok = m.group(0)
        if tok.startswith(b"<"):
            txt.append(decode_hex(tok[1:-1].decode()))
        elif tok.startswith(b"("):
            body = tok[1:-1].replace(b"\\(", b"(").replace(b"\\)", b")").replace(b"\\\\", b"\\")
            txt.append(body.decode("latin-1"))
        else:
            num = float(tok)
            if num < -100:
                txt.append(" ")
    if any(t.strip() for t in txt):
        pages.append("".join(txt))

open(sys.argv[2], "w").write("\n\n".join(pages))
print("pages", len(pages), "chars", sum(len(p) for p in pages), file=sys.stderr)
