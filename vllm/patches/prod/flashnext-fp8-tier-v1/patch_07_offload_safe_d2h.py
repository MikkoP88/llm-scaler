#!/usr/bin/env python3
"""FN8_07 (DEC40): safe offloader D2H — turn the cold-boot bcs-reset load
wedge into detect / retry / fast-fail.

Root cause (3/3 boots 2026-10-12, host .65): every 125B offload boot trips
a `xe ... Engine reset: engine_class=bcs` on one card during shard loading.
The rank whose offloader D2H is in flight when the reset fires then hangs
FOREVER inside the driver at uva.py `p.data.to(device="cpu")` (py-spy
"active" C-spin; the copy's completion is lost); the peer rank completes and
parks at wait_until_ready.  Banked devcores: /root/bank_bcs_reset_1012{a,b}
.devcore (card2 t=262, card1 t=1432).

Fix in this file ONLY (loader untouched): replace the blocking D2H with
chunked non_blocking copies + POLLED events (Event.query() loop — never a
blocking driver wait):
  a) detect: a chunk whose event does not pass within
     FN8_UVA_D2H_TIMEOUT_S (default 60 s, chunks are ~0.1-0.3 s) is a lost
     completion, not a slow copy;
  b) recover: re-issue that chunk on a FRESH stream — post-reset engines
     accept new work, so the retry typically completes;
  c) terminate: if the fresh-stream chunk also stalls, os._exit(87) so the
     outer boot-retry wrapper (fn8_boot_retry.sh) relaunches the boot
     instead of wedging forever.  Bounded cost instead of infinite hang.

pin path: final buffer allocated pinned UP FRONT (no .pin_memory() second
copy), chunks land directly in it.  non-pin path: one pinned staging chunk
+ plain CPU memcpy into the pageable final (non_blocking D2H to pageable
memory silently degrades to a BLOCKING driver copy = the hang mode).

Telemetry: FN8_D2H {copies, chunks, timeouts, retries, exits, bytes, ms_max}
dumped at exit.  Gate: FN8_UVA_SAFE_D2H=1 (default stock behavior).
Event API probed lazily at first offload (device initialized by then);
unavailable API -> stock path + warning.

Idempotent; py_compile gate; backup .fn8_07.bak; EOL aware; anchor==1;
--restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_07"
REL = "model_executor/offloader/uva.py"
MARKER = "FN8_07"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# --- E1: safe-D2H helpers (module level, after the FN8_02B block) ---------------
OLD_E1 = '''_FN8_TIER_OFFLOAD_PARAMS = {
    seg for seg in _fn8_os.environ.get(
        "FN8_TIER_OFFLOAD_PARAMS",
        "w13_weight,w2_weight,w13_weight_scale_inv,w2_weight_scale_inv",
    ).split(",") if seg
}
'''
NEW_E1 = '''_FN8_TIER_OFFLOAD_PARAMS = {
    seg for seg in _fn8_os.environ.get(
        "FN8_TIER_OFFLOAD_PARAMS",
        "w13_weight,w2_weight,w13_weight_scale_inv,w2_weight_scale_inv",
    ).split(",") if seg
}


# FN8_07 (DEC40): see module docstring in patch_07_offload_safe_d2h.py.
# Chunked non_blocking D2H + POLLED events; lost completions (bcs engine
# reset) are detected and retried on a fresh stream, then fast-failed
# (exit 87) for the outer boot-retry wrapper.  Gate FN8_UVA_SAFE_D2H=1.
_FN8_SAFE_D2H = _fn8_os.environ.get("FN8_UVA_SAFE_D2H", "") == "1"
_FN8_D2H_CHUNK_ELEMS = (
    int(_fn8_os.environ.get("FN8_UVA_D2H_CHUNK_MB", "128")) * 1024 * 1024
)
_FN8_D2H_TIMEOUT_S = float(
    _fn8_os.environ.get("FN8_UVA_D2H_TIMEOUT_S", "60"))
_FN8_D2H_REC = {"copies": 0, "chunks": 0, "timeouts": 0, "retries": 0,
                "exits": 0, "bytes": 0, "ms_max": 0.0}
_fn8_d2h_api_checked = False
_fn8_d2h_api_ok_flag = False


def _fn8_d2h_api_ok() -> bool:
    """Lazy torch.xpu.Event probe (first offload call, device is up)."""
    global _fn8_d2h_api_checked, _fn8_d2h_api_ok_flag
    if not _fn8_d2h_api_checked:
        _fn8_d2h_api_checked = True
        try:
            _ev = torch.xpu.Event()
            _ = _ev.query()
            _fn8_d2h_api_ok_flag = True
        except Exception as _e:  # noqa: BLE001 (telemetry mandate)
            logger.warning(
                "FN8_07: torch.xpu.Event API unavailable (%s) — stock "
                "D2H path in use", _e)
    return _fn8_d2h_api_ok_flag


def _fn8_d2h_dump():
    import sys as _s
    print(f"FN8_D2H {_FN8_D2H_REC}", file=_s.stderr, flush=True)


import atexit as _fn8_atexit

_fn8_atexit.register(_fn8_d2h_dump)


def _fn8_chunk_poll(dst, src) -> None:
    """dst.copy_(src, non_blocking=True) + polled event — NEVER a
    blocking driver wait.  Stall -> one fresh-stream retry -> exit(87)."""
    import time as _fn8_t
    for _si, _st in enumerate((torch.xpu.current_stream(),
                               torch.xpu.Stream())):
        _ev = torch.xpu.Event()
        with torch.xpu.stream(_st):
            dst.copy_(src, non_blocking=True)
            _ev.record(_st)
        _t0 = _fn8_t.time()
        while not _ev.query():
            if _fn8_t.time() - _t0 > _FN8_D2H_TIMEOUT_S:
                _FN8_D2H_REC["timeouts"] += 1
                if _si == 0:
                    _FN8_D2H_REC["retries"] += 1
                    logger.warning(
                        "FN8_07: D2H chunk stalled %.0fs (bcs engine "
                        "reset?) — retrying chunk on a fresh stream",
                        _FN8_D2H_TIMEOUT_S)
                    break
                _FN8_D2H_REC["exits"] += 1
                logger.error("FN8_07: D2H chunk dead on fresh stream — "
                             "fast-fail exit(87) for boot-retry")
                _fn8_d2h_dump()
                _fn8_os._exit(87)
            _fn8_t.sleep(0.005)


def _fn8_safe_d2h(src: torch.Tensor, pin: bool) -> torch.Tensor:
    """Offloader D2H with polled chunk copies (FN8_07)."""
    import time as _fn8_t
    _t0 = _fn8_t.time()
    if not src.is_contiguous():
        src = src.contiguous()
    flat = src.reshape(-1)
    n = flat.numel()
    step = max(1, _FN8_D2H_CHUNK_ELEMS // max(1, src.element_size()))
    if pin:
        # final buffer pinned up front — chunks land in it directly
        out = torch.empty(src.shape, dtype=src.dtype,
                          device="cpu").pin_memory()
        out_flat = out.reshape(-1)
        for s0 in range(0, n, step):
            e0 = min(s0 + step, n)
            _fn8_chunk_poll(out_flat[s0:e0], flat[s0:e0])
            _FN8_D2H_REC["chunks"] += 1
    else:
        # non_blocking D2H to pageable memory degrades to a BLOCKING
        # driver copy (the hang mode) — stage through ONE pinned chunk,
        # then a plain CPU memcpy into the pageable final.
        out = torch.empty(src.shape, dtype=src.dtype, device="cpu")
        out_flat = out.reshape(-1)
        stage = torch.empty(min(step, max(1, n)), dtype=src.dtype,
                            device="cpu").pin_memory()
        for s0 in range(0, n, step):
            e0 = min(s0 + step, n)
            _fn8_chunk_poll(stage[: e0 - s0], flat[s0:e0])
            out_flat[s0:e0].copy_(stage[: e0 - s0])
            _FN8_D2H_REC["chunks"] += 1
    _FN8_D2H_REC["copies"] += 1
    _FN8_D2H_REC["bytes"] += n * src.element_size()
    _dt = (_fn8_t.time() - _t0) * 1000.0
    if _dt > _FN8_D2H_REC["ms_max"]:
        _FN8_D2H_REC["ms_max"] = _dt
    return out
'''

# --- E2: guarded copy site --------------------------------------------------------
OLD_E2 = '''            cpu_data = p.data.to(device="cpu")
            if self.pin_memory:
                cpu_data = cpu_data.pin_memory()
'''
NEW_E2 = '''            if _FN8_SAFE_D2H and _fn8_d2h_api_ok():
                # FN8_07: polled chunked D2H — detect lost completions
                # (bcs reset), fresh-stream retry, fast-fail exit(87).
                cpu_data = _fn8_safe_d2h(p.data, pin=self.pin_memory)
            else:
                cpu_data = p.data.to(device="cpu")
                if self.pin_memory:
                    cpu_data = cpu_data.pin_memory()
'''

EDITS = [
    ("E1_helpers", OLD_E1, NEW_E1),
    ("E2_copy_site", OLD_E2, NEW_E2),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_07.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup")
        sys.exit(4)
    sys.exit(0)

with open(path, "r", encoding="utf-8", newline="") as f:
    src = f.read()

if MARKER in src:
    print(f"{TAG} edit already present, skip (md5={md5(path)})")
    sys.exit(0)

eol = "\r\n" if "\r\n" in src else "\n"


def ed(lf_text):
    return lf_text.replace("\n", eol)


for name, old, _new in EDITS:
    n = src.count(ed(old))
    if n != 1:
        print(f"{TAG} FAIL: anchor '{name}' occurs {n}x (want 1)")
        sys.exit(2)

bak = path + ".fn8_07.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src
for name, old, new in EDITS:
    out = out.replace(ed(old), ed(new))

with tempfile.NamedTemporaryFile(
    "w", suffix=".py", delete=False, encoding="utf-8", newline=""
) as tf:
    tf.write(out)
    tmp = tf.name
try:
    py_compile.compile(tmp, doraise=True)
except py_compile.PyCompileError as e:
    print(f"{TAG} FAIL: compile error -> {e}")
    sys.exit(3)
finally:
    os.unlink(tmp)

with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(out)
print(
    f"{TAG} [{REL}] edited "
    f"(eol={'CRLF' if eol == chr(13) + chr(10) else 'LF'}), md5={md5(path)}"
)
print(f"{TAG} ALL_EDITS_OK")
