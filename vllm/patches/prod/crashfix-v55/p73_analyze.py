#!/usr/bin/env python3
"""p73_analyze.py — CRASHFIX-76 Stage A1: devcoredump format discovery.

Offline, read-only against the banked dump:
  lce1/WD_crash_132400_card2.devcoredump (512,555 B, md5 9eaf8fa7...)

Discovers the xe (6.17) devcoredump layout so A2/A3 can parse it:
section markers, offsets of the forensic anchors (ACTHD, ccs32,
seqno, VM state), and where text ends / binary blobs begin.

Usage: python3 p73_analyze.py <dumpfile> [--hex N] [--around PAT:N]
"""
import sys

def main():
    path = sys.argv[1]
    data = open(path, 'rb').read()
    print(f"SIZE {len(data)}")

    # --- 1. head hexdump -------------------------------------------------
    n = 0x200
    print(f"--- HEAD hexdump 0x0..0x{n:x}")
    for off in range(0, min(n, len(data)), 16):
        chunk = data[off:off + 16]
        hexs = ' '.join(f'{b:02x}' for b in chunk)
        asc = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"{off:08x}  {hexs:<47}  {asc}")

    # --- 2. section markers ---------------------------------------------
    print("--- SECTION MARKERS (lines with leading **** or == )")
    import re
    text_lines = []
    start = 0
    for m in re.finditer(rb'(?m)^(.*)$', data):
        line = m.group(1)
        text_lines.append((m.start(), line))
    markers = [(o, l) for o, l in text_lines
               if l.startswith(b'****') or l.startswith(b'====')
               or (b'****' in l and len(l) < 120)]
    for o, l in markers[:200]:
        print(f"{o:08x}  {l[:110].decode('ascii', 'replace')}")

    # --- 3. anchor offsets ----------------------------------------------
    print("--- ANCHOR OFFSETS (first occurrence + count)")
    anchors = [b'Process:', b'Reason:', b'guc_id', b'ccs32', b'seqno',
               b'ACTHD', b'RING_BBADDR', b'RING_HEAD', b'RING_TAIL',
               b'INSTDONE', b'HWCTX', b'VM ', b'PDE', b'PTE', b'GuC Log',
               b'GuC CT', b'GuC Log', b'CONTEXT', b'Job:', b'finished',
               b'LRC', b'BBADDR', b'IPEHR', b'execlist', b'VIRT_ADDR',
               b'ggtt', b'default_512']
    for a in anchors:
        idxs = []
        i = data.find(a)
        while i != -1 and len(idxs) < 5:
            idxs.append(i)
            i = data.find(a, i + 1)
        cnt = data.count(a)
        print(f"{a.decode():12s} count={cnt:<5d} at {' '.join(hex(x) for x in idxs) if idxs else '-'}")

    # --- 4. text/binary boundary scan ------------------------------------
    print("--- NON-ASCII SCAN (4KB blocks: fraction of non-printable bytes)")
    blk = 4096
    runs = []
    cur_bin = None
    for off in range(0, len(data), blk):
        chunk = data[off:off + blk]
        np = sum(1 for b in chunk if b < 9 or (13 < b < 32) or b > 126)
        frac = np / len(chunk)
        is_bin = frac > 0.30
        if is_bin and cur_bin is None:
            cur_bin = off
        elif not is_bin and cur_bin is not None:
            runs.append((cur_bin, off))
            cur_bin = None
    if cur_bin is not None:
        runs.append((cur_bin, len(data)))
    for s, e in runs:
        print(f"BINARY RUN 0x{s:08x}..0x{e:08x}  ({e - s} B)")

    # --- 5. context lines around a pattern --------------------------------
    args = sys.argv[2:]
    if len(args) >= 2 and args[0] == '--around':
        pat, k = args[1].encode(), int(args[2]) if len(args) > 2 else 8
        hits = []
        i = data.find(pat)
        while i != -1 and len(hits) < k:
            hits.append(i)
            i = data.find(pat, i + 1)
        print(f"--- AROUND {pat!r} ({len(hits)} shown)")
        for h in hits:
            lo = data.rfind(b'\n', 0, h) + 1
            hi = data.find(b'\n', h)
            print(f"@0x{h:08x}: {data[lo:hi][:200].decode('ascii', 'replace')}")

if __name__ == '__main__':
    main()
