#!/usr/bin/env python3
"""patch_v125_opsall_fix.py (v125 P22-C ROOT FIX, Run 7h conviction).

ROOT CAUSE (D3 3-leg matrix, 2026-09-28 22:38-22:48): every
non-enforce-eager spec-off serve degenerated (decode hidden states
wholesale NaN from the FIRST decode step, prefills clean, next
prefill clean again) because custom_ops carries a STALE 'none' from
pass-1 resolution. Sequence: optimization_level=O2 -> pass 1 resolves
mode=VLLM_COMPILE -> the append rule adds custom_ops 'none' (op
substitution OFF); LATER, platforms/xpu.py check_and_update_config
(TP>1 unsafe gate, v31.1 #11 + v124 P16) sets TORCH_COMPILE_DISABLE=1
-> vllm.py flips mode to NONE but custom_ops stays 'none'. With op
substitution off, the native fallback of the registered custom-op set
computes NaN for single-token GDN decode forwards on this hybrid
model. enforce_eager wins at pass 1 (mode NONE -> ops 'all') which is
why eager boots were coherent; the spec-4 lane survives 'none' only
because verify steps are multi-token (different kernel path).

Leg matrix (all ns, cudagraph NONE, D3 instrument):
  A mode=VLLM_COMPILE ops 'none' -> DEGENERATE (v60g x20)
  B mode=NONE          ops 'all' -> coherent ('391', v60g 0)
  C mode=VLLM_COMPILE ops 'all' -> coherent ('391', v60g 0)
A vs C differ ONLY in custom_ops.

FIX: at the same point the gate forces compile off, restore the
intended compile-off posture — op substitution ON, exactly what
mode=NONE resolution picks at pass 1. Unconditional for TP>1 boots
(the spec lane moves from accidental-'none' to 'all' too; validated
by fixval Boot B + the P23 battery before bake).
"""
import py_compile
import shutil
import sys

XPU = "/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py"

OLD = """\
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            logger.warning(
                "inductor compilation disabled with TP=%d (v31.1 #11 + "
"""
NEW = """\
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            # llm-scaler v125 (perf-v125 root fix): forcing compile off
            # here leaves pass-1 custom_ops='none' (op substitution OFF)
            # stale — the native op fallback NaNs on single-token GDN
            # decode (spec-off decode wholesale NaN from step 1; prefill
            # and multi-token verify unaffected). Restore the intended
            # compile-off posture: ops substituted, exactly what
            # mode=NONE resolution picks at pass 1.
            if (
                "none" in compilation_config.custom_ops
                and "all" not in compilation_config.custom_ops
            ):
                compilation_config.custom_ops = [
                    "all" if c == "none" else c
                    for c in compilation_config.custom_ops
                ]
                logger.warning(
                    "v125 root fix: custom_ops 'none' -> 'all' "
                    "(native op fallback NaNs on spec-off GDN decode; "
                    "compile forced off for TP=%d)",
                    parallel_config.tensor_parallel_size,
                )
            logger.warning(
                "inductor compilation disabled with TP=%d (v31.1 #11 + "
"""


def main() -> int:
    try:
        shutil.copyfile(XPU + ".pre_v125_opsfix", XPU)
        print("V125_OPSFIX_RESTORED_PRE")
    except FileNotFoundError:
        pass
    with open(XPU, encoding="utf-8") as f:
        src = f.read()
    if "v125 root fix" in src:
        print("V125_OPSFIX_ALREADY")
        return 0
    n = src.count(OLD)
    if n != 1:
        print("V125_OPSFIX_ABORT anchor count=%d (want 1)" % n)
        return 1
    shutil.copyfile(XPU, XPU + ".pre_v125_opsfix")
    with open(XPU, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW, 1))
    py_compile.compile(XPU, doraise=True)
    print("V125_OPSFIX_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
