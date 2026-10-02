#!/usr/bin/env python3
"""p73_decode_blobs.py — CRASHFIX-76 Stage A2: decode ASCII85 blobs from the
banked devcoredump (ccs32 [HWSP].data 0x1000, [HWCTX].data 0x2000).

Writes lce1/crash132400_{hwsp,hwctx}.bin and prints:
  - expected vs decoded length
  - all nonzero dword runs (offset: value) — LRC regs + HWSP seqno slots
  - seqno anchors 2508/2509 (0x9cc/0x9cd) locations
"""
import sys, re

def a85_decode(s):
    out = bytearray()
    grp = []
    for ch in s:
        if ch == 'z' and not grp:
            out += b'\x00\x00\x00\x00'
            continue
        if ch == 'y' and not grp:      # btoa space-run (unlikely here)
            out += b'\x20\x20\x20\x20'
            continue
        if not (33 <= ord(ch) <= 117):
            continue                    # skip whitespace/newlines
        grp.append(ord(ch) - 33)
        if len(grp) == 5:
            v = 0
            for d in grp:
                v = v * 85 + d
            out += v.to_bytes(4, 'big')
            grp = []
    if grp:                             # final partial group (N-1 bytes)
        n = len(grp)
        v = 0
        for d in grp + [84] * (5 - n):
            v = v * 85 + d
        out += v.to_bytes(4, 'big')[: n - 1]
    return bytes(out)

def nonzero_dwords(b):
    runs = []
    cur = None
    for off in range(0, len(b), 4):
        v = int.from_bytes(b[off:off + 4], 'little')
        if v != 0:
            if cur is None:
                cur = [off, []]
            cur[1].append((off, v))
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    return runs

def main():
    dump = open('/root/build/lce1/WD_crash_132400_card2.devcoredump', 'rb').read().decode('ascii', 'replace')
    for tag, expect in (('HWSP', 0x1000), ('HWCTX', 0x2000)):
        m = re.search(r'\[%(t)s\]\.length: 0x[0-9a-f]+\n\t\[%(t)s\]\.data: (.*)' % {'t': tag}, dump)
        if not m:
            print(f"{tag}: NOT FOUND")
            continue
        raw = a85_decode(m.group(1))
        path = f"/root/build/lce1/crash132400_{tag.lower()}.bin"
        open(path, 'wb').write(raw)
        print(f"=== {tag}: decoded {len(raw)} B (expect {expect}) -> {path}")
        for start, words in nonzero_dwords(raw):
            vals = ' '.join(f"{off:04x}:{v:08x}" for off, v in words[:24])
            more = f" (+{len(words)-24} more)" if len(words) > 24 else ""
            print(f"  run 0x{start:04x}..0x{words[-1][0]+4:04x}: {vals}{more}")
        for anchor in (2508, 2509, 2507):
            hits = [off for off in range(0, len(raw) - 3, 4)
                    if int.from_bytes(raw[off:off + 4], 'little') == anchor]
            print(f"  seqno {anchor} (0x{anchor:x}) little-endian at: "
                  f"{' '.join(hex(h) for h in hits) or '-'}")

if __name__ == '__main__':
    main()
