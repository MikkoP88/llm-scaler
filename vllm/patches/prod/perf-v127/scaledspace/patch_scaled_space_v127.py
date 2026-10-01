#!/usr/bin/env python3
"""patch_scaled_space_v127.py — v127 scaled-space fp8 SSM state, M1
(bridge-level scaled storage; numerics milestone of the weeks-scale kernel
program, FIX_AND_TEST_PLAN Layer 3).

Extends the v126 DUAL BRIDGE (_xpu_ops.py) with PER-FEATURE scale factors
so fp8_e4m3 state storage is representable BY CONSTRUCTION:

    stored_fp8[f] = clamp(state[f] * scale[f])   scale[f] = 0.98*448/runmax[f]
    gather:  state = stored * inv_scale[f]        (fp16 math, kernel unchanged)
    scatter: stored = state  * scale[f]

This removes the P29 format-impossibility failure mode (legit states >= 1280
vs the raw e4m3 ceiling 448) at the STORAGE layer. Speed is expected at par
with v126 fp8 storage (same gather/scatter roundtrips + one broadcast mul
each way) — M1 proves NUMERICS end-to-end in the engine. The speed prize is
M2 (native scaled-space kernel: dequant/requant in-register, no roundtrips).

Activation (worker processes cannot see env — marker-file law):
  /root/.v127_scaledspace        marker file (touch to arm)
  /root/v127_scales_e4m3.pt      scales from calibrate_offline.py
Absent either -> fully dormant == byte-identical v126 behavior.

Usage: /opt/venv/bin/python3 patch_scaled_space_v127.py   (idempotent)
Requires: v126 DUAL BRIDGE already applied (marker 'llm-scaler v126 DUAL
BRIDGE' present).
"""
from __future__ import annotations

import py_compile
import shutil
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
MARKER = "llm-scaler v127 SCALEDSPACE"
V126_MARKER = "llm-scaler v126 DUAL BRIDGE"

# ---------------------------------------------------------------------------
# Replacement 1: gather site (dequant on the way in)
# ---------------------------------------------------------------------------
OLD_GATHER = """        ssm_run = ssm_pool.index_select(0, _back_idx).to(torch.float16).contiguous()
"""

NEW_GATHER = """        ssm_run = ssm_pool.index_select(0, _back_idx).to(torch.float16).contiguous()
        # llm-scaler v127 SCALEDSPACE (M1): stored = state * scale[f].
        # Gather dequantizes with inv_scale; the companion scatter site
        # below requantizes with scale. Dormant unless the marker file AND
        # the scales file exist (env does not reach TP workers).
        if globals().get("_V127_WANT") is None:
            globals()["_V127_WANT"] = _v127_want_scaledspace()
        if globals()["_V127_WANT"]:
            _v127_load_scales(ssm_pool)
            ssm_run = ssm_run * globals()["_V127_INV"]
"""

# ---------------------------------------------------------------------------
# Replacement 2: scatter site (requant on the way out)
# ---------------------------------------------------------------------------
OLD_SCATTER = """        ssm_pool.view(torch.uint8).index_copy_(
            0, _back_idx, ssm_run.to(ssm_pool.dtype).view(torch.uint8)
        )
"""

NEW_SCATTER = """        if globals().get("_V127_WANT"):
            ssm_run = ssm_run * globals()["_V127_SC"]
        ssm_pool.view(torch.uint8).index_copy_(
            0, _back_idx, ssm_run.to(ssm_pool.dtype).view(torch.uint8)
        )
"""

# ---------------------------------------------------------------------------
# Appended helpers (module globals — FUNCTION-SCOPE LAW: the gate sites above
# reference only module globals / self attrs)
# ---------------------------------------------------------------------------
HELPERS = '''

# ---------------------------------------------------------------------------
# llm-scaler v127 SCALEDSPACE helpers (M1). See patch_scaled_space_v127.py.
# ---------------------------------------------------------------------------
def _v127_want_scaledspace():
    import os as _v127_os

    try:
        return _v127_os.path.exists("/root/.v127_scaledspace") and _v127_os.path.exists(
            "/root/v127_scales_e4m3.pt"
        )
    except Exception:  # noqa: BLE001 — dormant on any probe failure
        return False


def _v127_load_scales(pool):
    """Build the device-resident scale globals once (never under capture)."""
    g = globals()
    if g.get("_V127_SC") is not None:
        return
    if torch.xpu.is_current_stream_capturing():
        raise RuntimeError(
            "llm-scaler v127: scaled-space scales must load during eager "
            "prefill BEFORE graph capture — arming mid-capture is a config "
            "error (boot the leg with the marker in place from the start)."
        )
    if pool.dtype is not torch.float8_e4m3fn:
        raise RuntimeError(
            "llm-scaler v127 SCALEDSPACE M1 supports fp8_e4m3 pools only "
            f"(got {pool.dtype}); use --mamba-ssm-cache-dtype fp8_e4m3."
        )
    obj = torch.load("/root/v127_scales_e4m3.pt", map_location="cpu")
    if obj.get("fmt") != "fp8_e4m3":
        raise RuntimeError(
            f"llm-scaler v127: scales file fmt {obj.get('fmt')!r} != fp8_e4m3"
        )
    sc_flat = obj["scale"].float().reshape(-1)
    inv_flat = obj["inv_scale"].float().reshape(-1)
    trailing = 1
    for _d in pool.shape[1:]:
        trailing *= int(_d)
    if sc_flat.numel() == trailing:
        shape = (1, *pool.shape[1:])
        sc, inv = sc_flat.reshape(shape), inv_flat.reshape(shape)
    elif pool.dim() >= 3 and sc_flat.numel() == int(pool.shape[1]):
        sc, inv = sc_flat.reshape(1, -1, 1), inv_flat.reshape(1, -1, 1)
    else:
        raise RuntimeError(
            "llm-scaler v127: scales numel "
            f"{sc_flat.numel()} matches neither trailing {trailing} nor "
            f"head axis {pool.shape[1]} of pool {tuple(pool.shape)}"
        )
    # fp16-RANGE LAW (no-clamp-up policy, M2_DESIGN §2): scales for small-
    # runmax features reach 0.98*448/runmax ~ 4e5 >> fp16 max 65504 — the
    # SCATTER side must stay fp32 (safe: the product feeds only the
    # .to(fp8_e4m3) cast, never a kernel input). The GATHER side must stay
    # fp16 (ssm_run feeds the SYCL kernels): inv = runmax/439 <= ~11 for
    # live pools and only enters the fp16 DENORMAL tail below runmax
    # ~2.6e-5 — an absolutely-negligible dequant error on features that
    # small (pool max ~5e3).
    g["_V127_INV"] = inv.to(pool.device, torch.float16)
    g["_V127_SC"] = sc.to(pool.device, torch.float32)
    print(
        "v127 SCALEDSPACE engaged: fmt=fp8_e4m3 scales=%d pool=%s "
        "(stored = state * scale[f]; P29 ceiling failure mode removed)"
        % (sc_flat.numel(), tuple(pool.shape)),
        flush=True,
    )
'''


def main() -> int:
    with open(P, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print("v127 scaledspace: ALREADY_APPLIED (marker present)")
        print("V127_SCALEDSPACE_ALREADY")
        return 0
    if V126_MARKER not in src:
        print(
            "v127 scaledspace: ABORT — v126 DUAL BRIDGE marker not found in "
            f"{P}; apply patch_v126_dualbridge.py first."
        )
        return 2
    if src.count(OLD_GATHER) != 1:
        print(f"v127 scaledspace: ABORT — gather anchor count {src.count(OLD_GATHER)} != 1")
        return 3
    if src.count(OLD_SCATTER) != 1:
        print(f"v127 scaledspace: ABORT — scatter anchor count {src.count(OLD_SCATTER)} != 1")
        return 4

    shutil.copyfile(P, P + ".bak_v127")
    src = src.replace(OLD_GATHER, NEW_GATHER)
    src = src.replace(OLD_SCATTER, NEW_SCATTER)
    src = src.rstrip("\n") + "\n" + HELPERS
    with open(P, "w", encoding="utf-8") as f:
        f.write(src)
    py_compile.compile(P, doraise=True)

    with open(P, "r", encoding="utf-8") as f:
        chk = f.read()
    assert chk.count(MARKER) >= 1, "marker missing after apply"
    # 2 = call site in the gather block + the def in HELPERS (the M1LEG boot
    # of 06:39 falsely aborted its self-check at >=3 — the third count was
    # the error-message mention, which lives in this script not in _xpu_ops)
    assert chk.count("_v127_load_scales") >= 2, "helpers not wired"
    assert "_V127_INV" in chk and "_V127_SC" in chk, "scale globals missing"

    print("v127 scaledspace: applied (dormant until marker+scales present)")
    print("V127_SCALEDSPACE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
