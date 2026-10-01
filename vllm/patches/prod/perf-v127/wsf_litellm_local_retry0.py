#!/usr/bin/env python3
"""wsf_litellm_local_retry0.py — v127 WS-F: per-entry num_retries: 0 for
LOCAL engine entries in /root/litellm_config.yaml.

Why: the global `num_retries: 1` (right for cheap cloud retries) is an
RC6 amplifier on the local lane — when a client times out / aborts a huge
prefill, litellm RETRIES it once, double-queueing a 100k-token prefill on
the engine during the exact storms that caused the timeout
(FIX_AND_TEST_PLAN WS-F). Per-entry num_retries overrides the global, so
local entries get 0 while cloud entries keep the global 1.

Idempotent: skips entries that already carry num_retries. NEVER prints
file content (api_key lines live in this file — security law).

Usage: python3 wsf_litellm_local_retry0.py   (on the litellm host)
"""
import re
import shutil
import sys

CFG = "/root/litellm_config.yaml"
BAK = CFG + ".bak_wsf_v127"
LOCAL_API_BASE = re.compile(r"api_base:\s*https?://(localhost|10\.20\.3\.65):8000")
REQ_TIMEOUT = re.compile(r"(\s*)request_timeout:\s*\d+.*\n")


def main() -> int:
    shutil.copy2(CFG, BAK)
    lines = open(CFG, encoding="utf-8").read().splitlines(keepends=True)
    out, in_local, edited, already = [], False, 0, 0
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("- model_name:"):
            in_local = False
        if LOCAL_API_BASE.search(stripped):
            in_local = True
        out.append(ln)
        m = REQ_TIMEOUT.match(ln)
        if m and in_local:
            entry_block = "".join(lines[i - 4: i + 6])
            if "num_retries" in entry_block:
                already += 1
            else:
                out.append(
                    m.group(1) + "num_retries: 0"
                    "   # WS-F v127: aborting client must not re-queue a 100k prefill (RC6)\n"
                )
                edited += 1
            in_local = False
    open(CFG, "w", encoding="utf-8").write("".join(out))
    print(f"WSF_LOCAL_RETRY0_DONE inserted={edited} already_present={already} backup={BAK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
