#!/usr/bin/env python3
"""llm-scaler v64 TTFTFIX-2 — decode-step interleave under contention
(BAKE VARIANT; stacks on the baked v63 TTFTFIX in v1.2.17).

Baked defaults: VLLM_V64_DECODE_INTERLEAVE=2, VLLM_V64_DECODE_BUDGET=512.

See patch_sched_v64.py for the full mechanism description. Test-lane A/B
(seed 107/108, 106k cold prefill + concurrent decode):
  stock 8192: 0.255 tok/s during, big TTFT 101.8 s
  v63 2048:   1.007 tok/s, 117.1 s
  v64 K=2:    1.794 tok/s (7.0x stock), 125.2 s
  v64 K=3:    1.411 tok/s, 121.7 s
K=2 baked: starvation minimization is the named directive.

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

N3_OLD = '''try:
    _V63_CONTENDED_BUDGET = int(
        _os63.environ.get("VLLM_V63_CONTENDED_BUDGET", "2048")
    )
except ValueError:
    _V63_CONTENDED_BUDGET = 2048
'''

N3_NEW = '''try:
    # llm-scaler v64: contended default 1024 (max starved-decode tok/s
    # directive; budget-1024 + K=2 interleave measured fairest point).
    _V63_CONTENDED_BUDGET = int(
        _os63.environ.get("VLLM_V63_CONTENDED_BUDGET", "1024")
    )
except ValueError:
    _V63_CONTENDED_BUDGET = 1024
'''

MARK = "llm-scaler v64 TTFTFIX-2"


def check(path):
    txt = path.read_text()
    n = txt.count(MARK)
    baked = ('VLLM_V64_DECODE_INTERLEAVE", "2"' in txt
             and 'VLLM_V64_DECODE_BUDGET", "512"' in txt)
    v63_1024 = 'VLLM_V63_CONTENDED_BUDGET", "1024"' in txt
    v63_2048 = 'VLLM_V63_CONTENDED_BUDGET", "2048"' in txt
    ok = (n == 2 and "V64_INTERLEAVE_ACTIVE" in txt
          and "VLLM_V64_DECODE_INTERLEAVE" in txt and baked
          and v63_1024 and not v63_2048)
    print(f"CHECK {path}: marks={n} active_str={'V64_INTERLEAVE_ACTIVE' in txt} "
          f"baked_defaults={baked} v63_1024={v63_1024} v63_2048_gone="
          f"{not v63_2048} -> {'OK' if ok else 'NOT-APPLIED'}")
    return ok


def apply(path):
    txt = path.read_text()
    if txt.count(MARK) >= 2:
        print(f"ALREADY-APPLIED {path}")
        return check(path)
    if txt.count(MARK) == 1 or "V64_INTERLEAVE_ACTIVE" in txt:
        print(f"PARTIAL-MARKS {path}: refusing to double-apply")
        sys.exit(2)
    for tag, old in (("N1", N1_OLD), ("N2", N2_OLD), ("N3", N3_OLD)):
        c = txt.count(old)
        if c != 1:
            print(f"NEEDLE-{tag} count={c} (expected 1) — ABORT, file untouched")
            sys.exit(3)
    txt = txt.replace(N1_OLD, N1_NEW, 1)
    txt = txt.replace(N2_OLD, N2_NEW, 1)
    txt = txt.replace(N3_OLD, N3_NEW, 1)
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
