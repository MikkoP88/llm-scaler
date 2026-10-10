#!/usr/bin/env python3
"""fable_strip.py -- build the compact fable system-prompt import.

Reads the raw leak import (/root/fable_prompt_raw.md, the verbatim
claude-fable-5.1.md, md5 4cadca32...) and writes
/root/fable_prompt_stripped.md with:
  - every whitespace-only line removed ("all empty spaces")
  - per-line leading/trailing whitespace removed
  - horizontal-rule-only separator lines removed (---/***/===/___)
    = "really not needed stuff" (pure visual separators, zero content)
Everything else is kept VERBATIM, in order: headers, prose, tables,
JSON tool schemas, code blocks (internal spacing untouched).

Prints before/after stats so the reduction is auditable.
"""
import re
import sys

RAW = "/root/fable_prompt_raw.md"
OUT = "/root/fable_prompt_stripped.md"

raw = open(RAW, encoding="utf-8").read()
lines = raw.split("\n")

blank = sum(1 for l in lines if not l.strip())
trailing = sum(1 for l in lines if l != l.rstrip())
leading = sum(1 for l in lines if l != l.lstrip())
hr = sum(1 for l in lines if re.fullmatch(r"([-*_])\1{2,}", l.strip()))
headers = sum(1 for l in lines if l.startswith("#"))
fences = sum(1 for l in lines if l.strip().startswith("\x60\x60\x60"))
tables = sum(1 for l in lines if l.lstrip().startswith("|"))
bom = raw.startswith("﻿")

kept = []
dropped_hr = 0
for l in lines:
    s = l.strip()
    if not s:
        continue
    if re.fullmatch(r"([-*_])\1{2,}", s):
        dropped_hr += 1
        continue
    kept.append(s)

stripped = "\n".join(kept) + "\n"
open(OUT, "w", encoding="utf-8", newline="\n").write(stripped)

print(f"RAW_BYTES      {len(raw.encode())}")
print(f"RAW_LINES      {len(lines)}")
print(f"BLANK_LINES    {blank}")
print(f"LEADING_WS     {leading}")
print(f"TRAILING_WS    {trailing}")
print(f"HR_LINES       {hr} (dropped {dropped_hr})")
print(f"HEADER_LINES   {headers}")
print(f"CODE_FENCES    {fences}")
print(f"TABLE_ROWS     {tables}")
print(f"BOM            {bom}")
print(f"OUT_BYTES      {len(stripped.encode())}")
print(f"OUT_LINES      {len(kept)}")
print(f"BYTE_SAVED     {len(raw.encode()) - len(stripped.encode())}")
print(f"BYTE_SAVED_PCT {100 * (len(raw.encode()) - len(stripped.encode())) / len(raw.encode()):.1f}")
print("TAIL_RAW:", repr(raw[-200:]))
print("TAIL_OUT:", repr(stripped[-200:]))
print("STRIP_OK")
