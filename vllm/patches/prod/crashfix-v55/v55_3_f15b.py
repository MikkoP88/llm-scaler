
# llm-scaler f15b: spec-step stall watchdog + completion-event ring. (v2: transition-based progress).
# Diagnostic-only (crashfix-v58 F15b). See patch_f15b.py header.
import os
import subprocess
import threading
import time
from collections import deque

DISABLED = os.environ.get("VLLM_F15B", "1") == "0"
STALL = float(os.environ.get("VLLM_F15B_STALL_S", "45"))
OUTDIR = os.environ.get("VLLM_F15B_DIR", "/root")
POLL = 2.0
MAX_DUMPS = 5

_ring = deque(maxlen=4096)  # v3: 256 only spanned ~3 s of prefill-class ARs (boot Q2 miss — the reset victim fell outside the window); 4096 spans minutes
_lock = threading.Lock()
_started = False
_last_progress = time.monotonic()
_wait_what = None
_dump_count = 0


def _start_watchdog():
    global _started
    with _lock:
        if _started:
            return
        _started = True
    t = threading.Thread(target=_watchdog, daemon=True, name="f15b-watchdog")
    t.start()


def _note_progress():
    global _last_progress
    _last_progress = time.monotonic()


def mark(name, detail=""):
    if DISABLED:
        return
    _ring.append((time.monotonic(), name, detail, None))
    _note_progress()
    _start_watchdog()


def evmark(name, detail=""):
    if DISABLED:
        return
    evt = None
    try:
        import torch
        if not torch.compiler.is_compiling():
            evt = torch.Event()
            evt.record()
    except Exception:
        evt = None
    _ring.append((time.monotonic(), name, detail, evt))
    _note_progress()
    _start_watchdog()


def enter_wait(what):
    global _wait_what
    if DISABLED:
        return
    _wait_what = what
    mark("v55wait", what)


def exit_wait():
    global _wait_what
    if DISABLED:
        return
    _wait_what = None


def _engine_pids():
    me = os.getpid()
    out = [me]
    try:
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                cb = open(f"/proc/{d}/cmdline", "rb").read()
            except Exception:
                continue
            if (b"EngineCore" in cb or b"spawn_main" in cb) and int(d) != me:
                out.append(int(d))
    except Exception:
        pass
    return out[:6]


def _dump(reason):
    global _dump_count
    _dump_count += 1
    path = f"{OUTDIR}/f15b_dump_{os.getpid()}.log"
    now = time.monotonic()
    lines = [
        f"=== F15B DUMP #{_dump_count} {time.strftime('%H:%M:%S')} "
        f"reason={reason} wait={_wait_what} pid={os.getpid()} ==="
    ]
    with _lock:
        items = list(_ring)
    for t, name, detail, evt in items[-96:]:
        st = ""
        if evt is not None:
            try:
                st = "DONE" if evt.query() else "PENDING"
            except Exception as ex:
                st = f"ERR({ex})"
        lines.append(f"  [{now - t:8.1f}s ago] {name} {detail} {st}")
    txt = "\n".join(lines) + "\n"
    try:
        open(path, "a").write(txt)
    except Exception:
        pass
    # llm-scaler f15b: arstage decision census into the dump (Q4 —
    # staged-vs-direct split per (numel,dtype,contig,stream) at freeze)
    try:
        from vllm import _arstage as _as
        open(path, "a").write("ARSTAGE_CENSUS " + _as.stats_lines() + "\n")
    except Exception:
        pass
    for pid in _engine_pids():
        try:
            r = subprocess.run(
                ["/opt/venv/bin/py-spy", "dump", "--pid", str(pid),
                 "--nonblocking"],
                capture_output=True, text=True, timeout=15)
            open(f"{OUTDIR}/f15b_pyspy_{pid}.log", "a").write(
                f"=== dump #{_dump_count} ===\n{r.stdout}\n{r.stderr}\n")
        except Exception as ex:
            open(path, "a").write(f"pyspy {pid} fail: {ex}\n")
    try:
        r = subprocess.run(["xpu-smi", "dump"], capture_output=True,
                           text=True, timeout=15)
        open(f"{OUTDIR}/f15b_xpusmi.log", "a").write(
            f"=== dump #{_dump_count} ===\n{r.stdout}\n")
    except Exception as ex:
        open(path, "a").write(f"xpu-smi fail: {ex}\n")
    try:
        r = subprocess.run(["dmesg"], capture_output=True, text=True,
                           timeout=15)
        open(f"{OUTDIR}/f15b_dmesg.log", "a").write(
            f"=== dump #{_dump_count} ===\n" +
            "\n".join(r.stdout.splitlines()[-30:]) + "\n")
    except Exception as ex:
        open(path, "a").write(f"dmesg fail: {ex}\n")


_done_prev: set = set()


def _watchdog():
    # v2: progress requires a DONE TRANSITION — an event newly completed
    # since the last tick — or a fresh mark. The v1 static check ("any
    # event in the window queries DONE") never fired during a stall with
    # completed events parked in the window (boot Q miss). Event objects
    # stay alive via the ring and _done_prev within a tick, so identity
    # across the set diff is stable.
    global _dump_count
    while True:
        time.sleep(POLL)
        if DISABLED:
            continue
        with _lock:
            items = list(_ring)
        progressed = (time.monotonic() - _last_progress) < POLL
        done_now = set()
        for _t, _n, _d, e in items[-24:]:
            if e is None:
                continue
            try:
                if e.query():
                    done_now.add(e)
            except Exception:
                pass
        if done_now - _done_prev:
            progressed = True
        _done_prev.clear()
        _done_prev.update(done_now)
        if progressed:
            _note_progress()
            continue
        stall = time.monotonic() - _last_progress
        if stall > STALL:
            if _dump_count < MAX_DUMPS:
                _dump(f"stall {stall:.0f}s")
            _note_progress()
