#!/usr/bin/env python3
"""llm-scaler v64 TTFTFIX-2 — decode-step interleave under contention
(TEST-LANE PATCHER; stacks on the baked v63 TTFTFIX in v1.2.17).

Problem: even with the v63 contended budget (2048), a co-running decode
session advances only once per chunk step (~1.4 s late-chunk) — starved
throughput ~1.0 tok/s. A decode-ONLY step (no prefill work) runs the
FULL_DECODE_ONLY graph in ~0.2 s. Dedicating every K-th contended step
to decode multiplies starved-session throughput ~3x for ~8% prefill
time (K=2).

Mechanics: on contended steps (same predicate as v63: >=1 RUNNING
request in decode phase, >1 running), a step counter picks every K-th
step and caps token_budget to VLLM_V64_DECODE_BUDGET (default 512,
below one mamba block 1024). Decode requests (~5 tok each, <=64 seqs)
fit; the chunked prefill's block-aligned chunk floors to 0 tokens and
takes the certified `num_new_tokens <= 0 -> continue` skip (running-
loop skip reason #4 — the request simply yields this step; no
misaligned mamba state is ever built). Solo prefills and pure-decode
steps are unaffected.

Env knobs (read once at import):
  VLLM_V64_DECODE_INTERLEAVE  every K-th contended step is decode-only
                              (default 2; 0/1 disables)
  VLLM_V64_DECODE_BUDGET      token cap on interleave steps (default 512;
                              clamped to [64, 1023])

Modes:  APPLY (default)  |  CHECK
Idempotent; refuses partial-mark states. Backup .pre_v64.
"""
import pathlib
import py_compile
import shutil
import sys

SCHED = "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"

N1_OLD = '''if 0 < _V63_CONTENDED_BUDGET < 1024:
    _V63_CONTENDED_BUDGET = 1024
'''

N1_NEW = '''if 0 < _V63_CONTENDED_BUDGET < 1024:
    _V63_CONTENDED_BUDGET = 1024

# llm-scaler v64 TTFTFIX-2: decode-step interleave under contention.
try:
    _V64_DECODE_INTERLEAVE = int(
        _os63.environ.get("VLLM_V64_DECODE_INTERLEAVE", "2")
    )
except ValueError:
    _V64_DECODE_INTERLEAVE = 2
try:
    _V64_DECODE_BUDGET = int(
        _os63.environ.get("VLLM_V64_DECODE_BUDGET", "512")
    )
except ValueError:
    _V64_DECODE_BUDGET = 512
# Interleave cap must stay below one mamba block (1024): the chunked
# prefill has to floor to a zero-token chunk and SKIP the step, never
# run a misaligned partial block.
_V64_DECODE_BUDGET = max(64, min(_V64_DECODE_BUDGET, 1023))
if _V64_DECODE_INTERLEAVE < 2:
    _V64_DECODE_INTERLEAVE = 0
'''

N2_OLD = '''            if not getattr(self, "_v63_announced", False):
                self._v63_announced = True
                logger.info(
                    "V63_TTFTFIX_ACTIVE contended_budget=%d running=%d",
                    _V63_CONTENDED_BUDGET,
                    len(self.running),
                )
'''

N2_NEW = '''            if not getattr(self, "_v63_announced", False):
                self._v63_announced = True
                logger.info(
                    "V63_TTFTFIX_ACTIVE contended_budget=%d running=%d",
                    _V63_CONTENDED_BUDGET,
                    len(self.running),
                )

        # llm-scaler v64 TTFTFIX-2: on every K-th contended step, cap the
        # budget below one mamba block so chunked prefill yields the step
        # (block-aligned chunk floors to 0 -> certified skip path) and
        # co-running decode gets a dedicated ~0.2 s graph step.
        if (
            _V64_DECODE_INTERLEAVE >= 2
            and token_budget > _V64_DECODE_BUDGET
            and len(self.running) > 1
            and any(
                r.num_computed_tokens >= r.num_prompt_tokens
                for r in self.running
            )
        ):
            self._v64_step = getattr(self, "_v64_step", 0) + 1
            if self._v64_step % _V64_DECODE_INTERLEAVE == 0:
                token_budget = _V64_DECODE_BUDGET
                if not getattr(self, "_v64_announced", False):
                    self._v64_announced = True
                    logger.info(
                        "V64_INTERLEAVE_ACTIVE k=%d decode_budget=%d",
                        _V64_DECODE_INTERLEAVE,
                        _V64_DECODE_BUDGET,
                    )
'''

MARK = "llm-scaler v64 TTFTFIX-2"


def check(path):
    txt = path.read_text()
    n = txt.count(MARK)
    ok = n == 2 and "V64_INTERLEAVE_ACTIVE" in txt and "VLLM_V64_DECODE_INTERLEAVE" in txt
    print(f"CHECK {path}: marks={n} active_str={'V64_INTERLEAVE_ACTIVE' in txt} -> "
          f"{'OK' if ok else 'NOT-APPLIED'}")
    return ok


def apply(path):
    txt = path.read_text()
    if txt.count(MARK) >= 2:
        print(f"ALREADY-APPLIED {path}")
        return check(path)
    if txt.count(MARK) == 1 or "V64_INTERLEAVE_ACTIVE" in txt:
        print(f"PARTIAL-MARKS {path}: refusing to double-apply")
        sys.exit(2)
    for tag, old in (("N1", N1_OLD), ("N2", N2_OLD)):
        c = txt.count(old)
        if c != 1:
            print(f"NEEDLE-{tag} count={c} (expected 1) — ABORT, file untouched")
            sys.exit(3)
    txt = txt.replace(N1_OLD, N1_NEW, 1)
    txt = txt.replace(N2_OLD, N2_NEW, 1)
    bak = path.with_suffix(".py.pre_v64")
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
