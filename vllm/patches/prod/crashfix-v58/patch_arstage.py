#!/usr/bin/env python3
"""llm-scaler AR persistent-staging patcher (crashfix-v58, Fix L).

ROOT-CAUSE CONTEXT (WEDGE_PLAN F12/F14, boots Q/Q2/Q3 2026-09-13):
oneCCL kernel-mode all_reduce = flag-poll kernels exchanging peer data
through IPC-mapped USER buffers. AR inputs are ephemeral activation
tensors from the PyTorch XPU allocator; under expandable_segments a
freed VA can be unmapped/remapped, while oneCCL's open-IPC-handle cache
still references the OLD mapping => peer flag never observed => poll
kernel never retires => xe "LR job cleanup" engine reset => both ranks
wedge in v55 waits. At 0.9 util the steady-state margin is ~17 MiB.

BOOT Q3 VERDICT (staging v1 active, cap 262144): prefill-class ARs
SURVIVED (7.7M-class went through), but the wedge STILL struck the
draft-propose small class (numel 15360/5120 = 15-30 KB, deep under the
cap) with the identical reset signature (da ccs guc_id=32). Open
question this v2 answers: did those victims take the staged path
(=> F12-as-scoped falsified) or a passthrough (non-contiguity /
capture / exception => coverage hole)?

v2 = CENSUS + hole closure:
  - every call counts into (numel, dtype, contiguous, stream) keyed
    [staged, direct, reason] counters; last-8 direct/exception details
    kept; f15b dump prints the census (ARSTAGE_CENSUS line) and the
    f15b ring carries a periodic mark (every 512 calls, ~1/s at
    decode cadence) so the split is visible AT the freeze.
  - non-contiguous inputs are now COPY-staged (v1 passed them
    through; copy_ handles arbitrary strides, the staging dst is
    always contiguous).
  - stream census: staging-buffer reentrancy across streams would be
    a real race (single shared VA per (device,dtype)); the key records
    the current stream so a multi-stream workload is VISIBLE.
  - VLLM_XPU_ALLREDUCE_VIA_ALLGATHER guard: staging is skipped when
    the via path is active — its impl returns its OWN output tensor,
    so the copy-back would restore unreduced input (latent v1 bug,
    now guarded).

Caveat (known, accepted for the convicted eager lane): copy-back
assumes the impl reduces in-place on the staged tensor (default
dist.all_reduce path). The custom-op bounce path only runs while
compiling/capturing, where staging is already bypassed.

Cost: two on-stream copies <= 256 KiB each per staged AR (~tens of
us, decode step ~20 ms). Env knob: VLLM_XPU_AR_STAGE_MAX (bytes,
default 262144; 0 = disabled). Cudagraph-safe: never allocates/grows
while a stream capture is active (unseen size under capture falls
back to the direct path and counts it).

Apply in-container BEFORE serve start (order vs patch_f15b.py is
irrelevant — disjoint anchors):
    python3 patch_arstage.py            # apply
    python3 patch_arstage.py --revert   # restore .asbak + remove module
Idempotent; preflight verifies anchors; backup <file>.asbak.
"""
import pathlib
import shutil
import sys

MARK = "llm-scaler arstage"
SP = pathlib.Path(
    sys.argv[sys.argv.index("--sp") + 1] if "--sp" in sys.argv else
    "/opt/venv/lib/python3.12/site-packages"
)

ARSTAGE_MODULE = '''
# llm-scaler arstage v2: persistent staging + decision census (crashfix-v58 L).
# See patch_arstage.py header. v2: per-class staged/direct counters keyed
# (numel, dtype, contig, stream), non-contig copy-staged, via-allgather
# guard, periodic f15b ring marks, census in the f15b dump.
import os
import threading

import torch

_MAX_BYTES = int(os.environ.get("VLLM_XPU_AR_STAGE_MAX", "262144"))
_OFF = _MAX_BYTES <= 0
_VIA = os.environ.get("VLLM_XPU_ALLREDUCE_VIA_ALLGATHER", "0") == "1"
_LOCK = threading.Lock()
_BUFS: dict = {}
_STATS: dict = {}
_DIRECT_LAST: list = []
_LOGGED = False
_CALLS = 0
_MARK_EVERY = 512


def _capturing() -> bool:
    try:
        return bool(torch.xpu.is_current_stream_capturing())
    except Exception:
        return False


def _stream_key(dev) -> int:
    try:
        s = torch.xpu.current_stream(dev)
        return getattr(s, "xpu_stream", None) or getattr(s, "cuda_stream", None) or id(s)
    except Exception:
        return -1


def _summary() -> str:
    parts = []
    items = sorted(_STATS.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:8]
    for (n, dt, cg, st), (s, d, r) in items:
        parts.append("%dx%s%s s%s:%d/%d%s" % (n, dt, "C" if cg else "N", st, s, d, "(" + r + ")" if r else ""))
    return " ".join(parts) if parts else "nocalls"


def _count(key, staged: bool, reason: str = "") -> None:
    global _CALLS
    with _LOCK:
        v = _STATS.get(key)
        if v is None:
            v = _STATS[key] = [0, 0, ""]
        if staged:
            v[0] += 1
        else:
            v[1] += 1
            v[2] = reason
        _CALLS += 1
        do_mark = (_CALLS % _MARK_EVERY) == 0
        summ = _summary() if do_mark else ""
    if do_mark:
        try:
            from vllm import _f15b as _f
            _f.mark("arstage", summ)
        except Exception:
            pass


def stats_lines() -> str:
    with _LOCK:
        head = _summary()
        tail = "; ".join(_DIRECT_LAST[-8:])
    return head + " | direct_last: " + (tail if tail else "none")


def stage_in(input_: torch.Tensor) -> torch.Tensor:
    """Return a persistent-buffer view holding input_'s contents, or
    input_ itself when staging is off / via / unsafe / oversized for the
    call. On success the caller runs the collective on the returned tensor
    and MUST copy the result back into input_ before returning."""
    global _LOGGED
    try:
        if input_.device.type != "xpu":
            return input_
    except Exception:
        return input_
    # census-key extraction: a failure here COUNTS (reason=meta) — no
    # silent uncounted direct path is allowed in the Q4 verdict.
    n = 0
    esz = 0
    dt = "?"
    cg = False
    st = -2
    try:
        n = input_.numel()
        esz = input_.element_size()
        dt = str(input_.dtype)
        cg = bool(input_.is_contiguous())
        st = _stream_key(input_.get_device())
    except Exception:
        _count((n, dt, cg, st), False, "meta")
        return input_
    key = (n, dt, cg, st)
    if _OFF or _VIA or torch.compiler.is_compiling():
        _count(key, False, "off" if _OFF else ("via" if _VIA else "compiling"))
        return input_
    try:
        if n == 0 or n * esz > _MAX_BYTES:
            _count(key, False, "oversize")
            return input_
        capturing = _capturing()
        bkey = (input_.get_device(), input_.dtype)
        buf = None
        with _LOCK:
            buf = _BUFS.get(bkey)
            if buf is None or buf.numel() < n:
                if not capturing:
                    buf = torch.empty(_MAX_BYTES // esz, dtype=input_.dtype,
                                      device=input_.device)
                    _BUFS[bkey] = buf
                    if not _LOGGED:
                        _LOGGED = True
                        try:
                            import logging
                            logging.getLogger(__name__).info(
                                "llm-scaler arstage ACTIVE v2: persistent AR "
                                "staging cap=%d bytes key=%s (crashfix-v58 L)",
                                _MAX_BYTES, bkey)
                        except Exception:
                            pass
                else:
                    buf = None  # never allocate/grow under capture
        if buf is None:
            _count(key, False, "capture")
            return input_
        # v2: non-contiguous sources are copy-staged (copy_ handles
        # strides; dst is a contiguous view of the persistent buffer)
        dst = buf[:n].view(input_.shape)
        dst.copy_(input_)
        _count(key, True)
        return dst
    except Exception as ex:
        with _LOCK:
            _DIRECT_LAST.append("%s exc=%s" % (key, type(ex).__name__))
        _count(key, False, "exc")
        return input_
'''


def patch_file(path: pathlib.Path, replaces: list, tag: str):
    src = path.read_text()
    if MARK in src:
        print(f"[arstage] {tag}: marker present, skipping")
        return
    for old, _ in replaces:
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                f"[arstage] {tag}: anchor x{n} (need 1):\n{old[:200]}")
    bak = path.with_suffix(path.suffix + ".asbak")
    if not bak.exists():
        shutil.copy2(path, bak)
    for old, new in replaces:
        src = src.replace(old, new)
    path.write_text(src)
    print(f"[arstage] {tag}: patched OK (backup {bak.name})")


def revert_file(path: pathlib.Path, tag: str):
    bak = path.with_suffix(path.suffix + ".asbak")
    if bak.exists():
        shutil.copy2(bak, path)
        bak.unlink()
        print(f"[arstage] {tag}: reverted")
    else:
        print(f"[arstage] {tag}: no backup, untouched")


def main():
    xc = SP / "vllm/distributed/device_communicators/xpu_communicator.py"
    mod = SP / "vllm/_arstage.py"

    if "--revert" in sys.argv:
        revert_file(xc, "xpu_comm")
        if mod.exists():
            mod.unlink()
            print("[arstage] module removed")
        print("ARSTAGE_REVERT_OK")
        return

    xc_patches = [
        # Wrap the impl entry: stage small inputs on a persistent VA,
        # run the real impl on the staged tensor, copy back. Anchor is
        # disjoint from patch_f15b.py P6 (which wraps the OUTER
        # all_reduce try/finally) — apply order independent.
        (
            "    def _all_reduce_impl(self, input_: torch.Tensor) -> torch.Tensor:\n"
            "        if os.environ.get(\"SKIP_ALL_REDUCE\") == \"1\":\n"
            "            return input_\n",
            "    def _all_reduce_impl(self, input_: torch.Tensor) -> torch.Tensor:\n"
            "        # llm-scaler arstage v2 (crashfix-v58 L): persistent staging\n"
            "        # + per-class decision census for small ARs — kills the F12\n"
            "        # stale-IPC-handle window IF user-buffer VA recycling is the\n"
            "        # mechanism (Q3 verdict pending census). See patch_arstage.py.\n"
            "        from vllm import _arstage as _arstage_mod\n"
            "        _stg = _arstage_mod.stage_in(input_)\n"
            "        if _stg is input_:\n"
            "            return self._all_reduce_impl_direct(input_)\n"
            "        out = self._all_reduce_impl_direct(_stg)\n"
            "        input_.copy_(_stg)\n"
            "        return input_ if (out is None or out is _stg) else out\n"
            "\n"
            "    def _all_reduce_impl_direct(self, input_: torch.Tensor) -> torch.Tensor:\n"
            "        if os.environ.get(\"SKIP_ALL_REDUCE\") == \"1\":\n"
            "            return input_\n",
        ),
    ]

    # preflight
    src = xc.read_text()
    if MARK in src:
        print("[arstage] xpu_comm: already patched")
    else:
        for old, _ in xc_patches:
            n = src.count(old)
            if n != 1:
                raise SystemExit(
                    f"[arstage] PREFLIGHT FAIL xpu_comm: anchor x{n}:\n{old[:200]}")
    print("[arstage] preflight: anchor unique OK")

    mod.write_text(ARSTAGE_MODULE)
    print("[arstage] module vllm/_arstage.py installed (v2 census)")
    patch_file(xc, xc_patches, "xpu_comm")
    print("ARSTAGE_APPLY_OK")


if __name__ == "__main__":
    main()
