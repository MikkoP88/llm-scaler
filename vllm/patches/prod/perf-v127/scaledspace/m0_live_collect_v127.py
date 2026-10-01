#!/usr/bin/env python3
"""m0_live_collect_v127.py — v127 scaled-space M0-LIVE: per-feature runmax
collector over the LIVE GDN SSM pool (production scales source).

P29M-skeleton (magnitude leg, PHASES 957-1011): the v126 dual-bridge gather
site notes the pool reference (PYTHON-ONLY note — no GPU op, no host sync,
cudagraph-capture-safe; refs also land during eager prefill every request),
and a DAEMON TIMER THREAD every 20 s computes

    runmax[hv, k] = max over rows |pool[row, hv, k]|      (fp32 before abs)

accumulated elementwise since boot and dumped to /root/m0live_runmax_<pid>.pt
(overwrite-in-place; TP workers write one file each — merge at harvest).
TICK lines land in serve_full.log exactly like P29M, so dumps precede any
death (wedge-proof by construction).

Activation (marker-file law — env does not reach TP workers):
  /root/.v127_m0live          touch to arm (absent -> fully dormant)

Usage (inside the lane container, BEFORE serve start):
  /opt/venv/bin/python3 m0_live_collect_v127.py      (idempotent)

Requires the v126 DUAL BRIDGE in _xpu_ops.py (shipped in v1.2.26+).

Harvest:
  merge runmax across /root/m0live_runmax_*.pt (elementwise max), then
  calibrate_offline.py --stats merged.pt --out /root/v127_scales_e4m3.pt
  (at M3 bake: --no-clamp-up per M2_DESIGN §2 — the min(1.0, ·) guard is
  the M1-bridge-era policy the scaled kernel must drop).

Caveats (documented, conservative direction only):
  - amax spans ALL pool rows including never-written slots (zeros after
    alloc) and freed rows (stale states) — stale data can only INFLATE
    runmax, which SHRINKS scale: a too-small scale under-uses the grid but
    can never overflow it. Safe direction for calibration.
"""
from __future__ import annotations

import py_compile
import shutil
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
MARKER = "llm-scaler v127 M0LIVE-COLLECTOR"
V126_MARKER = "llm-scaler v126 DUAL BRIDGE"

ANCHOR = """    ssm_pool = self.kv_cache[1]
"""

NOTE = """    ssm_pool = self.kv_cache[1]
    # llm-scaler v127 M0LIVE-COLLECTOR: note the pool ref for the telemetry
    # thread. PYTHON-ONLY (no GPU op, no host sync) — safe under cudagraph
    # capture; fires for BOTH fp16 and fp8 pools (the gather/scatter bridge
    # below is fp8-only, so the note must sit at dtype-neutral function
    # entry); dormant unless /root/.v127_m0live exists.
    _m0live_note(ssm_pool)
"""

HELPERS = '''

# ---------------------------------------------------------------------------
# llm-scaler v127 M0LIVE-COLLECTOR helpers. See m0_live_collect_v127.py.
# READOUT LAW: _m0live_note runs on the forward path and touches NO tensor
# op; every GPU read + torch.save lives on the daemon thread below.
# ---------------------------------------------------------------------------
def _m0live_want():
    import os as _os

    try:
        return _os.path.exists("/root/.v127_m0live")
    except Exception:  # noqa: BLE001 — dormant on any probe failure
        return False


def _m0live_note(pool):
    g = globals()
    want = g.get("_M0LIVE_WANT")
    if want is None:
        want = g["_M0LIVE_WANT"] = _m0live_want()
    if not want:
        return False
    try:
        key = (int(pool.data_ptr()), tuple(int(s) for s in pool.shape), str(pool.dtype))
        pools = g.setdefault("_M0LIVE_POOLS", {})
        if key not in pools:
            pools[key] = [pool, None]  # [ref, accumulated runmax]
        if g.get("_M0LIVE_THREAD") is None:
            import threading as _th

            t = _th.Thread(target=_m0live_tick, name="v127-m0live", daemon=True)
            g["_M0LIVE_THREAD"] = t
            t.start()
            print(
                "v127 M0LIVE collector engaged: pool=%s dtype=%s"
                % (tuple(pool.shape), pool.dtype),
                flush=True,
            )
    except Exception as _e:  # noqa: BLE001 — telemetry must never kill the lane
        print("M0LIVE_NOTE_ERR %r" % (_e,), flush=True)
        return False
    return True


def _m0live_tick():
    import os as _os
    import time as _time

    out_path = "/root/m0live_runmax_%d.pt" % _os.getpid()
    while True:
        _time.sleep(20.0)
        try:
            pools = globals().get("_M0LIVE_POOLS") or {}
            merged = None
            for key, rec in list(pools.items()):
                pool, acc = rec
                # cast to fp32 BEFORE abs(): abs() on float8 dtypes is not
                # universally supported (P29M law); fp16 pools are fine too.
                m = pool.detach().float().abs().amax(dim=0).flatten()
                acc = _torch_maximum(acc, m) if acc is not None else m
                rec[1] = acc
                merged = _torch_maximum(merged, acc) if merged is not None else acc.clone()
            if merged is not None:
                per_pool = {
                    "%s:%s" % (k[2], list(k[1])): v[1].cpu() for k, v in pools.items()
                }
                _torch_save(
                    {
                        "runmax": merged.cpu(),
                        "ts": _time.time(),
                        "pools": len(pools),
                        "per_pool": per_pool,
                    },
                    out_path,
                )
                print(
                    "M0LIVE_TICK pools=%d F=%d max=%.4f -> %s"
                    % (len(pools), merged.numel(), float(merged.max()), out_path),
                    flush=True,
                )
        except Exception as _e:  # noqa: BLE001 — never kill the lane from telemetry
            print("M0LIVE_TICK_ERR %r" % (_e,), flush=True)


def _torch_maximum(a, b):
    return torch.maximum(a, b)


def _torch_save(obj, path):
    torch.save(obj, path)
'''


def main() -> int:
    with open(P, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print("v127 m0live: ALREADY_APPLIED (marker present)")
        print("V127_M0LIVE_ALREADY")
        return 0
    if V126_MARKER not in src:
        print(
            "v127 m0live: ABORT — v126 DUAL BRIDGE marker not found in "
            f"{P}; this collector requires the shipped bridge."
        )
        return 2
    if src.count(ANCHOR) != 1:
        print(f"v127 m0live: ABORT — anchor count {src.count(ANCHOR)} != 1")
        return 3

    shutil.copyfile(P, P + ".bak_v127m0")
    src = src.replace(ANCHOR, NOTE)
    src = src.rstrip("\n") + "\n" + HELPERS
    with open(P, "w", encoding="utf-8") as f:
        f.write(src)
    py_compile.compile(P, doraise=True)

    with open(P, "r", encoding="utf-8") as f:
        chk = f.read()
    assert chk.count(MARKER) >= 1, "marker missing after apply"
    assert "_m0live_note" in chk and "_m0live_tick" in chk, "helpers not wired"
    assert "_M0LIVE_POOLS" in chk, "pool registry missing"

    print("v127 m0live: applied (dormant until /root/.v127_m0live present)")
    print("V127_M0LIVE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
