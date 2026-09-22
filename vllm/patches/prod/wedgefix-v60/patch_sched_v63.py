#!/usr/bin/env python3
"""llm-scaler v63 TTFTFIX — adaptive chunked-prefill budget under decode
contention (TEST-LANE PATCHER).

Root fix target: "the stalls are TTFT-under-mamba-recurrence with zero prefix
reuse — 129 s of serialized chunked prefill that simultaneously starves every
other session."

Mechanism (v63 Phase-0 RCA): a ~136k-token CC prompt prefills in 7168-token
chunks (max_num_batched_tokens 8192, mamba block-aligned to 1024); GDN
attention over the growing prefix stretches chunk steps from 4 s to 9 s; every
co-running decode request advances exactly once per scheduler step, so
concurrent sessions starve at 0.1–0.5 tok/s for the whole prefill.

Fix: when >= 1 RUNNING request is in decode phase
(num_computed_tokens >= num_prompt_tokens) and more than one request shares
the engine, cap the per-step token budget:

    VLLM_V63_CONTENDED_BUDGET   (default 4096; 0 disables)

Floor-clamped to 1024 (one mamba block — smaller budgets would produce
zero-token block-aligned chunks and stall the prefill). Solo prefills keep the
full budget (max TTFT speed); pure-decode steps are unaffected because decode
token demand is far below the cap; min() clamps downstream handle everything
else. Expected effect: chunk steps roughly halve at 4096 (late chunks shrink
proportionally to the cap), so co-running decode advances ~2-3x more often
during big prefills.

Modes:  APPLY (default)  |  CHECK (verify marks only)
Idempotent; backup <file>.pre_v63 created on first apply.
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
        _os63.environ.get("VLLM_V63_CONTENDED_BUDGET", "4096")
    )
except ValueError:
    _V63_CONTENDED_BUDGET = 4096
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
    ok = n == 2 and "V63_TTFTFIX_ACTIVE" in txt and "VLLM_V63_CONTENDED_BUDGET" in txt
    print(f"CHECK {path}: marks={n} active_str={'V63_TTFTFIX_ACTIVE' in txt} -> "
          f"{'OK' if ok else 'NOT-APPLIED'}")
    return ok


def apply(path):
    txt = path.read_text()
    if txt.count(MARK) >= 2:
        print(f"ALREADY-APPLIED {path}")
        return check(path)
    if txt.count(MARK) == 1 or "V63_TTFTFIX_ACTIVE" in txt:
        print(f"PARTIAL-MARKS {path}: refusing to double-apply; restore from "
              f".pre_v63 backup first")
        sys.exit(2)
    for tag, old in (("N1", N1_OLD), ("N2", N2_OLD)):
        c = txt.count(old)
        if c != 1:
            print(f"NEEDLE-{tag} count={c} (expected 1) — ABORT, file untouched")
            sys.exit(3)
    if N1_OLD in N2_OLD or N2_OLD in N1_OLD:
        pass
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
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "APPLY"
    p = pathlib.Path(SCHED)
    if mode == "CHECK":
        sys.exit(0 if check(p) else 1)
    ok = apply(p)
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
