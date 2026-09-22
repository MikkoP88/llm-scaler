#!/usr/bin/env python3
"""patch_v65_probe.py — TEST-LANE-ONLY step cadence probe (never bake).

Appends a module-level wrap of Scheduler.schedule to vllm/v1/core/sched/scheduler.py.
Gap between successive schedule() entries ~= full engine step wall time
(schedule + execute_model + output processing). Gated by
VLLM_V65_STEP_LOG=1 (default off -> zero behavior change).

Usage: python3 patch_v65_probe.py --apply | --check | --revert
"""
import sys

PATH = "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"
MARK = "# llm-scaler v65 STEP-PROBE (test-only)"

TAIL = '''

# llm-scaler v65 STEP-PROBE (test-only)
if _os63.environ.get("VLLM_V65_STEP_LOG", "0") == "1":
    import time as _time65
    _pc65 = _time65.perf_counter
    _orig_sched65 = Scheduler.schedule

    def _sched65_probe(self, *a, **k):
        _t0 = _pc65()
        _gap = _t0 - getattr(self, "_v65_last_entry", _t0)
        self._v65_last_entry = _t0
        _out = _orig_sched65(self, *a, **k)
        _ntok = getattr(_out, "num_scheduled_tokens", -1)
        if _gap > 0.0005:
            logger.info(
                "V65_STEP gap=%.3f new_tok=%d running=%d waiting=%d",
                _gap,
                _ntok,
                len(self.running),
                len(self.waiting),
            )
        return _out

    Scheduler.schedule = _sched65_probe
'''


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    n = src.count(MARK)
    if mode == "--check":
        print(f"CHECK {PATH}: marks={n} -> {'OK' if n == 1 else 'NOT-APPLIED'}")
        return 0 if n == 1 else 1
    if mode == "--apply":
        if n:
            print(f"ALREADY-APPLIED {PATH} (marks={n})")
            return 0
        backup = PATH + ".pre_v65probe"
        with open(backup, "w", encoding="utf-8") as f:
            f.write(src)
        with open(PATH, "a", encoding="utf-8") as f:
            f.write(TAIL)
        import py_compile

        py_compile.compile(PATH, doraise=True)
        print(f"APPLIED {PATH} (backup {backup})")
        return 0
    if mode == "--revert":
        backup = PATH + ".pre_v65probe"
        try:
            with open(backup, "r", encoding="utf-8") as f:
                old = f.read()
            with open(PATH, "w", encoding="utf-8") as f:
                f.write(old)
            print(f"REVERTED from {backup}")
        except FileNotFoundError:
            print("NO BACKUP — nothing reverted")
        return 0
    print("unknown mode")
    return 2


if __name__ == "__main__":
    sys.exit(main())
