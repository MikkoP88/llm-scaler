#!/usr/bin/env python3
"""llm-scaler v66 FAIRFIX — waiting-admission starvation bypass (test lane).

Root cause (v1.2.18/v1.2.19 symptom: clients stuck in Waiting while decode
runs, 'Avg prompt throughput: 0.0'): the v63 contended budget (baked 1024)
and v64 interleave (K=2, budget 512) cap the per-step token budget whenever
decode-phase co-runners exist. After decode-token deduction the
mamba-block-aligned chunk floors to 0, so a WAITING request is never
admitted while any long decode session is running -> clients drop.

Fix: when the waiting queue has been continuously non-empty for more than
VLLM_V66_PREFILL_STARVE_S seconds (default 2.0, 0=off), bypass BOTH caps for
exactly one scheduler step: the head of the waiting queue gets a full-budget
chunk, then the caps resume. Solo prefills, pure-decode steps, and the
empty-queue big-prefill crawl are untouched (the bypass only depends on
waiting-queue age, not on running composition).

Anchors (v1.2.18 baked scheduler.py):
  - module header right after the v63 clamp (count must be 1)
  - eval site: 'token_budget = self.max_num_scheduled_tokens' (count must be 1)
  - v63 elif condition: 'and token_budget > _V63_CONTENDED_BUDGET' (count 1)
  - v64 if  condition: 'and token_budget > _V64_DECODE_BUDGET' (count 1)

Usage:
  python3 patch_sched_v66.py --apply | --revert | --check
"""
import sys
from pathlib import Path

V = Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py")
MARK = "llm-scaler v66 FAIRFIX"

HEADER_ANCHOR = (
    "if 0 < _V63_CONTENDED_BUDGET < 1024:\n"
    "    _V63_CONTENDED_BUDGET = 1024\n"
)
HEADER_ADD = (
    "\n"
    "# llm-scaler v66 FAIRFIX: waiting-queue admission starvation bypass.\n"
    "import os as _os66\n"
    '_V66_STARVE_S = float(_os66.environ.get("VLLM_V66_PREFILL_STARVE_S", "2.0"))\n'
    "\n"
    "\n"
    "def _v66_waiting_starved(sched) -> bool:\n"
    "    # True for exactly ONE scheduler step after the waiting queue has\n"
    "    # been continuously non-empty for > _V66_STARVE_S seconds. Stateful:\n"
    "    # returning True resets the age tracking (consumed by this step).\n"
    "    if _V66_STARVE_S <= 0:\n"
    "        return False\n"
    "    wq = getattr(sched, \"waiting\", None)\n"
    "    if not wq:\n"
    "        sched._v66_wait_since = None\n"
    "        return False\n"
    "    now = time.monotonic()\n"
    "    since = getattr(sched, \"_v66_wait_since\", None)\n"
    "    if since is None:\n"
    "        sched._v66_wait_since = now\n"
    "        return False\n"
    "    if now - since > _V66_STARVE_S:\n"
    "        sched._v66_wait_since = None  # this step carries the bypass chunk\n"
    "        sched._v66_bypass = getattr(sched, \"_v66_bypass\", 0) + 1\n"
    "        n = sched._v66_bypass\n"
    "        if n == 1 or n % 50 == 0:\n"
    "            logger.info(\n"
    "                \"V66_FAIRFIX_ACTIVE bypass=%d waiting=%d starve_s=%.1f\",\n"
    "                n,\n"
    "                len(wq),\n"
    "                _V66_STARVE_S,\n"
    "            )\n"
    "        return True\n"
    "    return False\n"
)

EVAL_OLD = "        token_budget = self.max_num_scheduled_tokens\n"
EVAL_NEW = (
    "        token_budget = self.max_num_scheduled_tokens\n"
    "        # llm-scaler v66 FAIRFIX: evaluate the waiting-starvation bypass\n"
    "        # ONCE per step (stateful check) before any budget cap below.\n"
    "        _v66_bypass_now = _v66_waiting_starved(self)\n"
)

NEEDLE_V63_OLD = (
    "            and token_budget > _V63_CONTENDED_BUDGET\n"
    "            and len(self.running) > 1\n"
    "            and any(\n"
)
NEEDLE_V63_NEW = (
    "            and token_budget > _V63_CONTENDED_BUDGET\n"
    "            and len(self.running) > 1\n"
    "            and not _v66_bypass_now  # llm-scaler v66 FAIRFIX\n"
    "            and any(\n"
)

NEEDLE_V64_OLD = (
    "            and token_budget > _V64_DECODE_BUDGET\n"
    "            and len(self.running) > 1\n"
    "            and any(\n"
)
NEEDLE_V64_NEW = (
    "            and token_budget > _V64_DECODE_BUDGET\n"
    "            and len(self.running) > 1\n"
    "            and not _v66_bypass_now  # llm-scaler v66 FAIRFIX\n"
    "            and any(\n"
)


def _abort(msg: str) -> None:
    print("ABORT:", msg)
    sys.exit(1)


def _count(txt: str, sub: str) -> int:
    return txt.count(sub)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("--apply", "--revert", "--check"):
        print(__doc__)
        sys.exit(2)
    src = V.read_text()
    backup = V.with_suffix(".py.pre_v66")

    marks = _count(src, MARK)
    header = _count(src, HEADER_ANCHOR)
    if mode == "--check":
        print(
            "CHECK %s: marks=%d header=%d eval=%d v63=%d v64=%d"
            % (
                V,
                marks,
                header,
                _count(src, EVAL_NEW),
                _count(src, NEEDLE_V63_NEW),
                _count(src, NEEDLE_V64_NEW),
            )
        )
        ok = (
            marks == 4
            and header == 1
            and _count(src, EVAL_NEW) == 1
            and _count(src, NEEDLE_V63_NEW) == 1
            and _count(src, NEEDLE_V64_NEW) == 1
        )
        print("-> %s" % ("APPLIED" if ok else "NOT-APPLIED"))
        sys.exit(0 if ok else 1)

    if mode == "--revert":
        if not backup.exists():
            _abort("no backup %s" % backup)
        V.write_text(backup.read_text())
        print("REVERTED from", backup)
        return

    # --apply
    if marks:
        _abort("already applied (marks=%d)" % marks)
    if header != 1:
        _abort("header anchor count=%d (want 1)" % header)
    for name, old in (
        ("eval", EVAL_OLD),
        ("v63", NEEDLE_V63_OLD),
        ("v64", NEEDLE_V64_OLD),
    ):
        c = _count(src, old)
        if c != 1:
            _abort("%s needle anchor count=%d (want 1)" % (name, c))
    backup.write_text(src)
    src = src.replace(HEADER_ANCHOR, HEADER_ANCHOR + HEADER_ADD, 1)
    src = src.replace(EVAL_OLD, EVAL_NEW, 1)
    src = src.replace(NEEDLE_V63_OLD, NEEDLE_V63_NEW, 1)
    src = src.replace(NEEDLE_V64_OLD, NEEDLE_V64_NEW, 1)
    V.write_text(src)
    new = V.read_text()
    ok = (
        _count(new, MARK) == 4
        and _count(new, HEADER_ADD) == 1
        and _count(new, EVAL_NEW) == 1
        and _count(new, NEEDLE_V63_NEW) == 1
        and _count(new, NEEDLE_V64_NEW) == 1
    )
    print("APPLIED marks=%d backup=%s" % (_count(new, MARK), backup))
    if not ok:
        _abort("post-apply verification failed")
    print("OK")


if __name__ == "__main__":
    main()
