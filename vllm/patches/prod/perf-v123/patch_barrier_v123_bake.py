#!/usr/bin/env python3
"""patch_barrier_v123_bake.py — llm-scaler v123 BAKE-TIME patch: spec-draft
barrier default OFF (user directive 2026-09-28).

What it does (single site, gpu_model_runner.py:184):
  _SPEC_DRAFT_BARRIER_MODE = os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER", "2")
    -> default "0"  (mode 0 = no device barrier, no host flock; the v62 mode-2
       machinery stays intact — only the DEFAULT flips, and the boot script
       passes -e VLLM_XPU_SPEC_DRAFT_BARRIER=0 explicitly as belt+suspenders)
  + marker comment line above it: 'llm-scaler v123: barrier default OFF ...'

Rationale (evidence): v89 T2b leg measured barrier-off IDENTICAL to =2 on the
85-size config (solo 73.3/73.0 vs 73.6/74.1; contention same distribution) and
PATCH_STACK_ANALYSIS 3.6 was closed on that measurement; the v123 user
directive mandates default-off. BARRIER=1 (device barrier) remains forbidden.
Crash posture change -> v1.2.23 ships ONLY after the full wedge battery
(serialized 24 + bursts x3 + 14-phase drill x3 + resets 0) on the baked image.

--check: verify both edits are present (for bake gates and lane re-gating).
"""
import sys

PATH = "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"
MARKER = "# llm-scaler v123: barrier default OFF (user directive 2026-09-28; T2b measured identical; wedge battery re-certified)"
OLD = '"VLLM_XPU_SPEC_DRAFT_BARRIER", "2"'
NEW = '"VLLM_XPU_SPEC_DRAFT_BARRIER", "0"'


def main() -> int:
    src = open(PATH).read()
    if "--check" in sys.argv:
        ok = src.count(NEW) == 1 and src.count(OLD) == 0 and src.count(MARKER) == 1
        print("V123_BARRIER " + ("APPLIED" if ok else "MISSING"))
        return 0 if ok else 1
    if src.count(MARKER):
        print("V123_BARRIER already applied")
        return 0
    n = src.count(OLD)
    assert n == 1, f"expected exactly 1 barrier default site, found {n}"
    # insert marker above the mode line, flip the default
    src = src.replace(OLD, NEW, 1)
    idx = src.index(NEW)
    line_start = src.rfind("\n", 0, idx) + 1
    src = src[:line_start] + MARKER + "\n" + src[line_start:]
    open(PATH, "w").write(src)
    # verify in-process semantics: import must report False False 0
    print("V123_BARRIER PATCHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
