#!/usr/bin/env python3
"""llm-scaler v37: spec draft-barrier DEFAULT OFF.

The v2x-era VLLM_XPU_SPEC_DRAFT_BARRIER drain (oneCCL wedge mitigation,
KNOWN_ISSUES #11 era) defaults ON in gpu_model_runner.py. The v31.1
conviction removed the wedge source (compiled-piece path), and v33
measured barrier-off strictly better with unchanged hashes:
fp8+spec 2k +13.7%, 4bit+spec 2k +28.5%, 65k +3-4%, no wedge in
sustained testing. v34 posture lists barrier-off as the keeper state.

This patch flips the default to OFF (env VLLM_XPU_SPEC_DRAFT_BARRIER=1
explicitly restores the drain for diagnosis). Idempotent; FAILS on
tree drift.
"""
import py_compile

P = "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"
MARKER = "# llm-scaler v37: draft-barrier default OFF"
OLD = '''_SPEC_DRAFT_BARRIER = (
    os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER", "1") == "1"
)'''
NEW = '''# llm-scaler v37: draft-barrier default OFF (v33 keeper: fp8+spec 2k
# +13.7%, 4bit+spec 2k +28.5%, hashes unchanged, no wedge sustained;
# the v2x oneCCL mitigation is obsolete under the v31.1 posture).
# VLLM_XPU_SPEC_DRAFT_BARRIER=1 restores the drain for diagnosis.
_SPEC_DRAFT_BARRIER = (
    os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER", "0") == "1"
)'''

src = open(P).read()
if MARKER in src:
    print("V37_BARRIER OK: already patched")
    raise SystemExit(0)
if src.count(OLD) != 1:
    print("V37_BARRIER FAIL: anchor count %d (tree drift)" % src.count(OLD))
    raise SystemExit(1)
open(P, "w").write(src.replace(OLD, NEW))
py_compile.compile(P, doraise=True)
print("V37_BARRIER OK: draft-barrier default OFF")
