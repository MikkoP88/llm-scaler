#!/usr/bin/env python3
"""fable_cc_cut.py -- Claude-Code-optimized fable system prompt.

Keeps ONLY the persona core (# claude_behavior, lines 1-222 of the raw
import) and drops every web-app-interface section that a Claude Code
session cannot use (CC injects its own harness prompt + tool set):
memory_filesystem, end_conversation_tool_info, artifact
storage/publishing/visualizer UX, connector/plugin suggestions,
past-chats/preferences, search_instructions, the entire # Tools catalog
(Gmail/Calendar/Drive/Claude_Docs/display widgets/web tools),
anthropic_api_in_artifacts, available_skills, network_configuration,
and the captured <system-reminder>/<user_memory_snapshot> tail.
Applies the same whitespace strip as v4 (no blank lines, no leading/
trailing space, no --- separators).
"""
import re

RAW = "/root/fable_prompt_raw.md"
OUT = "/root/fable_prompt_cc.md"

lines = open(RAW, encoding="utf-8").read().split("\n")

# Boundary verification: cut starts at '# memory_filesystem'
idx = next(i for i, l in enumerate(lines) if l.startswith("# memory_filesystem"))
assert idx == 222, f"unexpected boundary {idx}"

kept = []
for l in lines[:idx]:
    s = l.strip()
    if not s:
        continue
    if re.fullmatch(r"([-*_])\1{2,}", s):
        continue
    kept.append(s)

out = "\n".join(kept) + "\n"
open(OUT, "w", encoding="utf-8", newline="\n").write(out)

print(f"KEPT_LINES     {len(kept)}")
print(f"OUT_BYTES      {len(out.encode())}")
print(f"RAW_BYTES      {sum(len(l.encode()) + 1 for l in lines)}")
print("FIRST:", repr(out[:80]))
print("LAST:", repr(out[-80:]))
print("CC_CUT_OK")
