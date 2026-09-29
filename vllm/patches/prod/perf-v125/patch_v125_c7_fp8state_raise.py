#!/usr/bin/env python3
"""patch_v125_c7_fp8state_raise.py (v125 P25 bake guard).

P23D/P23F conviction (2026-09-29): every fp8 GDN/SSM-state posture is
QUALITY-BROKEN under concurrency on this stack —
  P23D: VLLM_XPU_GDN_FP8_NATIVE=2 (pool-straight spec fp8, P22B wheel)
        -> 20% post-prefix tool-call salad on BOTH formats; fp16
        control 0/30 on the same venv+wheel.
  P23F: =1 (ESIMD decode + v124 unique-bridge prefill) -> 8.75-13.75%
        WRONG answers fresh-boot, exploding to 25-42% post-prefix
        (e4m3 7/80 -> 10/24; e5m2 11/80 -> 6/24); every wrong answer
        CORRECT on serial re-ask = concurrency state-slot crosstalk;
        fp16 + fp8-KV control 0/104 and 60/60 tools clean.
  Run 7n: fp8-state is ALSO slower than fp16 (solo -1.5..-12.5%,
        agg -2.6..-13.9%) — double-disqualified.
Root cause = kernel round scope (fp8 DISPATCH_STATE_DTYPE state-slot
handling in _xpu_C.abi3.so).

Base-image fact (verified on v1.2.24-raw 0423c13f8c21, 2026-09-29):
the raw base _xpu_ops.py (811 lines) carries NO P22B-lineage switches
(VLLM_XPU_GDN_FP8_NATIVE: 0 occurrences) — those exist ONLY on the
v1.2.24 production lane-commit image, whose lane venv was P22-patched
before commit (=1 engaged silently there). The v1225 lineage boots
from raw, so the env is inert on it; this guard upgrades inert ->
REFUSED so the knob can never re-engage silently (kernel round or
accidental re-application) and carries the verdict in-code.

Guard is SELF-CONTAINED: appended at module EOF with its own aliased
import — no anchor line, no ordering dependency. Executed at import
time, before any serve can start.

Run inside the container:
  docker cp patch_v125_c7_fp8state_raise.py lsv-bake:/root/
  docker exec lsv-bake /opt/venv/bin/python3 /root/patch_v125_c7_fp8state_raise.py
Idempotent (V125_C7_ALREADY); --check prints APPLIED status.
"""
import py_compile
import sys

XO = "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
MARKER = "llm-scaler v125 C7"
ENV = "VLLM_XPU_GDN_FP8_NATIVE"

GUARD = '''

# llm-scaler v125 C7 (P23D/P23F conviction, 2026-09-29): fp8
# GDN/SSM-state postures are QUALITY-BROKEN under concurrency and are
# not supported on this image — refuse loudly at import instead of
# serving corrupted output. P23D: =2 pool-straight spec fp8 -> 20%
# post-prefix tool-call salad on BOTH formats (fp16 control 0/30).
# P23F: =1 ESIMD ns fp8 -> 8.75-13.75% WRONG answers fresh-boot
# exploding to 25-42% post-prefix; every serial re-ask correct (=
# concurrency state-slot crosstalk); fp16 + fp8-KV control 0/104.
# Run 7n: fp8-state slower than fp16 everywhere. Fix belongs to the
# kernel round (fp8 DISPATCH_STATE_DTYPE state-slot handling in
# _xpu_C.abi3.so); until then there is NO certified fp8-state posture.
import os as _v125_c7_os
_v125_c7_mode = _v125_c7_os.environ.get("VLLM_XPU_GDN_FP8_NATIVE", "")
if _v125_c7_mode in ("1", "2"):
    raise RuntimeError(
        "llm-scaler v125 C7: VLLM_XPU_GDN_FP8_NATIVE=%s is NOT supported "
        "on this image — fp8 GDN-state postures corrupt output under "
        "concurrency (P23D 20%% tool salad; P23F 8.75-41.7%% wrong answers) "
        "and are slower than fp16. Kernel-round fix required. Use the "
        "certified fp16 GDN pool (--mamba-ssm-cache-dtype float16, env "
        "unset)." % _v125_c7_mode
    )
'''


def main() -> int:
    with open(XO) as f:
        src = f.read()

    if MARKER in src:
        print("V125_C7_ALREADY")
        return 0

    if ENV in src:
        print(f"V125_C7_FAIL: {ENV} already referenced in base file — "
              f"unexpected lineage, refusing to append")
        return 1

    if not src.endswith("\n"):
        src += "\n"
    with open(XO, "w") as f:
        f.write(src + GUARD)
    py_compile.compile(XO, doraise=True)
    print("V125_C7_OK")


if __name__ == "__main__":
    if "--check" in sys.argv:
        with open(XO) as f:
            print("APPLIED" if MARKER in f.read() else "NOT_APPLIED")
        sys.exit(0)
    sys.exit(main())
