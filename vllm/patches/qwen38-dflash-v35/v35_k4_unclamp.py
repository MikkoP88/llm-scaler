#!/usr/bin/env python3
"""v35 boot patch: remove the v31 k>3 speculative-decode clamp (#12).

Per user decision 2026-09-02: k=4+ is user-selectable with NO code
protection; the corruption warning lives in documents/commit only
(KNOWN_ISSUES #12, qwen38-dflash README addendum 3). The clamp and its
VLLM_XPU_ALLOW_K4_CAPTURE bypass are both deleted from the installed
platforms/xpu.py. Idempotent; FAILS loudly on tree drift. Applied by
bootp.sh to every lane (no-op for nospec and k<=3).
"""
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py"
SENTINEL = "# llm-scaler v35: #12 clamp removed"
MARK_START = "        # llm-scaler v31 (#12 clamp): piecewise XPU capture corrupts"
MARK_END = "            _spec.num_speculative_tokens = 3\n"
REPL = (
    "        # llm-scaler v35: #12 clamp removed - k>3 user-selectable,\n"
    "        # docs-only warning (KNOWN_ISSUES #12, README addendum 3).\n"
)

src = open(P).read()
if SENTINEL in src:
    print("V35_K4UNCLAMP OK: already patched")
    sys.exit(0)
if MARK_START not in src or MARK_END not in src:
    print("V35_K4UNCLAMP FAIL: v31 clamp block not found (tree drift)")
    sys.exit(1)
i = src.index(MARK_START)
j = src.index(MARK_END, i) + len(MARK_END)
if src[j] == "\n":  # swallow the blank separator line after the block
    j += 1
open(P, "w").write(src[:i] + REPL + src[j:])
chk = open(P).read()
assert SENTINEL in chk and MARK_START not in chk
print("V35_K4UNCLAMP OK: k>3 clamp removed, k=4+ user-selectable")
