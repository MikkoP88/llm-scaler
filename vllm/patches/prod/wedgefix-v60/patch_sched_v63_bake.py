#!/usr/bin/env python3
"""llm-scaler v63 TTFTFIX — adaptive chunked-prefill budget under decode
contention (BAKE variant, default VLLM_V63_CONTENDED_BUDGET=2048).

Validated on test lane 2026-09-22 (106k cold prefill + concurrent decode):
  stock 8192: decode-during 0.255 tok/s, TTFT 101.8 s, gaps 6.7-8.25 s
  budget 4096: 0.433 tok/s (+70%),  TTFT 106.2 s (+4%),  gaps ~3.5 s
  budget 2048: 0.998 tok/s (3.9x),  TTFT 118.0 s (+16%), gaps ~1.4 s  <- BAKED

Same needles as patch_sched_v63.py (test variant); only the baked default
differs (2048). Solo prefills keep the full budget; pure-decode steps are
unaffected (decode demand << cap); floor-clamped to 1024 (mamba block).
Spec + XGrammar-2 untouched (WEDGEFIX-E/G, STALFIX, v55.3 all preserved).

Modes:  --apply (default)  |  --check
Idempotent; refuses partial-mark states.
"""
import pathlib
import py_compile
import shutil
import sys

SCHED = "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"

N1_OLD = "logger = init_logger(__name__)\n"

N1_NEW = '''logger = init_logger(__name__)

# llm-scaler v63 TTFTFIX: adaptive chunked-prefill budget under decode
# contention. See the schedule() needle for semantics.
import os as _os63

try:
    _V63_CONTENDED_BUDGET = int(
        _os63.environ.get("VLLM_V63_CONTENDED_BUDGET", "2048")
    )
except ValueError:
    _V63_CONTENDED_BUDGET = 2048
# Floor at one mamba block (1024 tok): smaller budgets yield zero-token
# block-aligned chunks and would stall the prefill entirely.
if 0 < _V63_CONTENDED_BUDGET < 1024:
    _V63_CONTENDED_BUDGET = 1024
'''

N2_OLD = '''        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0
'''

N2_NEW = '''        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0
        elif (
            _V63_CONTENDED_BUDGET > 0
            and token_budget > _V63_CONTENDED_BUDGET
            and len(self.running) > 1
            and any(
                r.num_computed_tokens >= r.num_prompt_tokens
                for r in self.running
            )
        ):
            # llm-scaler v63 TTFTFIX: decode-phase co-runners present -> cap
            # the per-step budget so chunked prefill advances in shorter
            # steps and concurrent decode runs ~2-3x more often (CC stall
            # fix). Solo prefills and pure-decode steps are unaffected.
            token_budget = _V63_CONTENDED_BUDGET
            if not getattr(self, "_v63_announced", False):
                self._v63_announced = True
                logger.info(
                    "V63_TTFTFIX_ACTIVE contended_budget=%d running=%d",
                    _V63_CONTENDED_BUDGET,
                    len(self.running),
                )
'''

MARK = "llm-scaler v63 TTFTFIX"


def check(path):
    txt = path.read_text()
    n = txt.count(MARK)
    ok = (
        n == 2
        and "V63_TTFTFIX_ACTIVE" in txt
        and 'VLLM_V63_CONTENDED_BUDGET", "2048"' in txt
    )
    print(f"CHECK {path}: marks={n} active_str={'V63_TTFTFIX_ACTIVE' in txt} "
          f"default2048={'VLLM_V63_CONTENDED_BUDGET\", \"2048\"' in txt} -> "
          f"{'OK' if ok else 'NOT-APPLIED'}")
    return ok


def apply(path):
    txt = path.read_text()
    if txt.count(MARK) >= 2:
        print(f"ALREADY-APPLIED {path}")
        return check(path)
    if txt.count(MARK) == 1 or "V63_TTFTFIX_ACTIVE" in txt:
        print(f"PARTIAL-MARKS {path}: refusing to double-apply")
        sys.exit(2)
    for tag, old in (("N1", N1_OLD), ("N2", N2_OLD)):
        c = txt.count(old)
        if c != 1:
            print(f"NEEDLE-{tag} count={c} (expected 1) — ABORT, file untouched")
            sys.exit(3)
    txt = txt.replace(N1_OLD, N1_NEW, 1)
    txt = txt.replace(N2_OLD, N2_NEW, 1)
    bak = path.with_suffix(".py.pre_v63")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"BACKUP {bak}")
    path.write_text(txt)
    py_compile.compile(str(path), doraise=True)
    print("PY_COMPILE OK")
    return check(path)


def main():
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "--APPLY"
    p = pathlib.Path(SCHED)
    if mode == "--CHECK":
        sys.exit(0 if check(p) else 1)
    ok = apply(p)
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
