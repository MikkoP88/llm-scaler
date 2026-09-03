#!/usr/bin/env python3
"""harm-vaudit: harmonization audit — enumerate EVERY local divergence of the
installed vllm tree from its wheel manifest (RECORD).

For each file under site-packages/vllm:
  - RECORD hash mismatch -> MODIFIED post-install
  - not in RECORD        -> ADDED locally
Also greps 'llm-scaler v' markers to classify which modifications carry
patch-stack markers (covered by committed patch scripts) vs orphan edits.

Output: grouped summary (modified / added / marker coverage).
"""
import base64
import csv
import glob
import hashlib
import os
import re
import sys

SP = "/opt/venv/lib/python3.12/site-packages"
PKG = os.path.join(SP, "vllm")
DISTS = glob.glob(os.path.join(SP, "vllm-*.dist-info"))
if not DISTS:
    print("NO_VLLM_DIST_INFO")
    sys.exit(1)
DI = sorted(DISTS)[-1]
print("DIST_INFO:", os.path.basename(DI))

record = {}
with open(os.path.join(DI, "RECORD")) as f:
    for row in csv.reader(f):
        if len(row) >= 2 and row[1]:
            record[row[0]] = row[1]  # path -> "sha256=..."


def rec_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return "sha256=" + base64.urlsafe_b64encode(
        h.digest()).rstrip(b"=").decode()


modified, added, marker_files = [], [], []
for root, dirs, files in os.walk(PKG):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for fn in files:
        if not fn.endswith((".py", ".hpp", ".cpp", ".cu", ".so")):
            continue
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, SP).replace("\\", "/")
        if rel not in record:
            added.append(rel)
            continue
        try:
            if rec_digest(p) != record[rel]:
                modified.append(rel)
        except OSError:
            pass

marker_re = re.compile(r"llm-scaler v\d")
for rel in modified:
    try:
        with open(os.path.join(SP, rel), errors="replace") as f:
            if marker_re.search(f.read()):
                marker_files.append(rel)
    except OSError:
        pass

print("\n=== MODIFIED (%d) ===" % len(modified))
for r in sorted(modified):
    print("  M", r)
print("\n=== ADDED (%d) ===" % len(added))
for r in sorted(added):
    print("  A", r)
print("\n=== MARKER COVERAGE: %d/%d modified files carry 'llm-scaler vN' "
      "markers ===" % (len(marker_files), len(modified)))
for r in sorted(marker_files):
    print("  K", r)
print("\nORPHANS (modified, no marker):")
for r in sorted(set(modified) - set(marker_files)):
    print("  O", r)
