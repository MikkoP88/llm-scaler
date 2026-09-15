#!/usr/bin/env python3
"""llm-scaler AR single-stream funnel patcher (crashfix-v58, Fix M).

ROOT-CAUSE (v3, boots Q/Q2/Q3/Q4 2026-09-13/14 — see WEDGE_PLAN §13):
the 262144+mtp4 wedge is a CROSS-STREAM COLLECTIVE ORDERING DESYNC in
oneCCL kernel-mode collectives. Evidence chain:
  - Q4 census (arstage v2): every AR class 100% STAGED on persistent
    never-remapped VAs, 0 direct — and the wedge still struck the
    draft-propose ARs with the identical reset signature
    (da ccs guc_id=32). F12-as-scoped (user-buffer VA staleness)
    FALSIFIED.
  - Q4 census ALSO shows >= 4 stable stream handles issuing ARs of
    the same classes (e.g. numel 10240 under 4 streams: 74649 /
    53702 / 35904 / 28594 calls), matching the model runner's spec
    side-streams (async_output_copy / draft_token_ids_copy /
    num_valid_draft_tokens_copy / valid_sampled_token_count_copy)
    plus the main compute stream.
  - Host order is IDENTICAL on both ranks (Q2 lockstep forensics);
    DEVICE-side order across unsynchronized streams is NOT — and can
    differ BETWEEN ranks. oneCCL kernel-mode flag-poll assumes
    per-connection FIFO execution order; an order flip between ranks
    => each rank's poll kernel waits for a flag value the peer has
    already consumed => mutual spin => xe "LR job cleanup" engine
    reset => v55-fence wedge. Explains: spec-only bearer (drafter =
    second AR stream), nospec clean, victim class varies with phase
    (last interleaved pair), lockstep ranks, staging irrelevant,
    TMP_BUF knobs only delaying onset.

FIX (M): route EVERY _all_reduce_impl through ONE dedicated
per-device stream ("funnel"), event-ordered w.r.t. the caller's
stream in both directions:
    ev_in.record(caller); funnel.wait(ev_in)
    run collective on funnel stream
    ev_out.record(funnel); caller.wait(ev_out)
Device-side per-connection order becomes the host-call order on BOTH
ranks => desync impossible. No data copies, no kernel/math change
(numerics identical), in-place contract preserved. Cost: 2 events +
stream handoff per AR (µs-scale; ~90 ARs/decode step ~20 ms).

Cudagraph/compile safety: bypass (direct) while compiling or a
stream capture is active — capture-time ordering is graph-
topological; replay never enters Python. Bypass failures are
setup-only: run() executes EXACTLY once on exactly one stream.

Env knob: VLLM_XPU_AR_FUNNEL=0 disables (default on).

Apply in-container BEFORE serve start (anchor disjoint from
patch_f15b.py P6; apply order independent; do NOT combine with
patch_arstage staging in the same boot — single-variable discipline):
    python3 patch_arfunnel.py            # apply
    python3 patch_arfunnel.py --revert   # restore .afbak + remove module
Idempotent; preflight verifies anchors; backup <file>.afbak.
"""
import pathlib
import shutil
import sys

MARK = "llm-scaler arfunnel"
SP = pathlib.Path(
    sys.argv[sys.argv.index("--sp") + 1] if "--sp" in sys.argv else
    "/opt/venv/lib/python3.12/site-packages"
)

ARFUNNEL_MODULE = '''
# llm-scaler arfunnel: single-stream AR funnel (crashfix-v58 M).
# See patch_arfunnel.py header for the cross-stream ordering desync
# mechanism this kills. Diagnostic-grade logging, prod-safe paths.
import os
import threading

import torch

_OFF = os.environ.get("VLLM_XPU_AR_FUNNEL", "1") == "0"
_LOCK = threading.Lock()
_STREAMS: dict = {}
_LOGGED = False


def _capturing() -> bool:
    try:
        return bool(torch.xpu.is_current_stream_capturing())
    except Exception:
        return False


def funnel(dev, run):
    """Execute run() (a collective enqueue lambda) on the single
    dedicated AR stream for dev, ordered after the caller's stream and
    releasing the caller's stream only when the collective is enqueued.
    run() executes EXACTLY once, on exactly one stream, on every path."""
    global _LOGGED
    if _OFF or torch.compiler.is_compiling():
        return run()
    try:
        if _capturing():
            return run()
        with _LOCK:
            st = _STREAMS.get(dev)
            if st is None:
                st = torch.xpu.Stream(dev)
                _STREAMS[dev] = st
                if not _LOGGED:
                    _LOGGED = True
                    try:
                        import logging
                        logging.getLogger(__name__).info(
                            "llm-scaler arfunnel ACTIVE: dedicated AR "
                            "stream per device (crashfix-v58 M), dev=%s",
                            dev)
                    except Exception:
                        pass
        cur = torch.xpu.current_stream(dev)
        ev_in = torch.xpu.Event()
        ev_in.record(cur)
        st.wait_event(ev_in)
    except Exception:
        return run()
    with torch.xpu.stream(st):
        out = run()
        try:
            ev_out = torch.xpu.Event()
            ev_out.record(st)
        except Exception:
            ev_out = None
    try:
        if ev_out is not None:
            cur.wait_event(ev_out)
    except Exception:
        pass
    return out
'''


def patch_file(path: pathlib.Path, replaces: list, tag: str):
    src = path.read_text()
    if MARK in src:
        print(f"[arfunnel] {tag}: marker present, skipping")
        return
    for old, _ in replaces:
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                f"[arfunnel] {tag}: anchor x{n} (need 1):\n{old[:200]}")
    bak = path.with_suffix(path.suffix + ".afbak")
    if not bak.exists():
        shutil.copy2(path, bak)
    for old, new in replaces:
        src = src.replace(old, new)
    path.write_text(src)
    print(f"[arfunnel] {tag}: patched OK (backup {bak.name})")


def revert_file(path: pathlib.Path, tag: str):
    bak = path.with_suffix(path.suffix + ".afbak")
    if bak.exists():
        shutil.copy2(bak, path)
        bak.unlink()
        print(f"[arfunnel] {tag}: reverted")
    else:
        print(f"[arfunnel] {tag}: no backup, untouched")


def main():
    xc = SP / "vllm/distributed/device_communicators/xpu_communicator.py"
    mod = SP / "vllm/_arfunnel.py"

    if "--revert" in sys.argv:
        revert_file(xc, "xpu_comm")
        if mod.exists():
            mod.unlink()
            print("[arfunnel] module removed")
        print("ARFUNNEL_REVERT_OK")
        return

    xc_patches = [
        # Wrap the impl entry: run the collective on the dedicated
        # funnel stream (event-ordered vs caller). Anchor is disjoint
        # from patch_f15b.py P6 (outer all_reduce try/finally) and
        # textually identical to the arstage anchor — apply arfunnel
        # INSTEAD of arstage, never both.
        (
            "    def _all_reduce_impl(self, input_: torch.Tensor) -> torch.Tensor:\n"
            "        if os.environ.get(\"SKIP_ALL_REDUCE\") == \"1\":\n"
            "            return input_\n",
            "    def _all_reduce_impl(self, input_: torch.Tensor) -> torch.Tensor:\n"
            "        # llm-scaler arfunnel (crashfix-v58 M): single-stream AR\n"
            "        # funnel — kills the cross-stream collective ordering\n"
            "        # desync beneath the 262144+mtp4 wedge. See\n"
            "        # patch_arfunnel.py. Capture/compile-safe; run-once.\n"
            "        from vllm import _arfunnel as _arfunnel_mod\n"
            "        return _arfunnel_mod.funnel(\n"
            "            input_.get_device(),\n"
            "            lambda: self._all_reduce_impl_direct(input_))\n"
            "\n"
            "    def _all_reduce_impl_direct(self, input_: torch.Tensor) -> torch.Tensor:\n"
            "        if os.environ.get(\"SKIP_ALL_REDUCE\") == \"1\":\n"
            "            return input_\n",
        ),
    ]

    # preflight
    src = xc.read_text()
    if MARK in src:
        print("[arfunnel] xpu_comm: already patched")
    else:
        for old, _ in xc_patches:
            n = src.count(old)
            if n != 1:
                raise SystemExit(
                    f"[arfunnel] PREFLIGHT FAIL xpu_comm: anchor x{n}:\n{old[:200]}")
    print("[arfunnel] preflight: anchor unique OK")

    mod.write_text(ARFUNNEL_MODULE)
    print("[arfunnel] module vllm/_arfunnel.py installed")
    patch_file(xc, xc_patches, "xpu_comm")
    print("ARFUNNEL_APPLY_OK")


if __name__ == "__main__":
    main()
