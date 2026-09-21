#!/usr/bin/env python3
"""patch_wedge_v62.py — WEDGEFIX-G: cheap HOST-side spec draft barrier.

Background (2026-09-22 bisect, boots 5-8):
  v61's replicated drafter removes all draft collectives but crashes
  (UR_RESULT_ERROR_DEVICE_LOST in target gdn_attention) under repeated
  concurrent prefills REGARDLESS of the v60 drain barrier, and costs
  ~-40% decode throughput (full-vocab lm_head + full MoE per rank).
  The stock sharded drafter with the v60 every-step torch.xpu.synchronize()
  drain is crash-free (boot 8) but the drain costs -9..-31%.

  The v25 wedge mechanism (KNOWN_ISSUES #11): ranks reach the drafter's
  eager oneCCL collectives skewed — one host still blocked in the
  align-mode .cpu() D2H drain, the peer already submitting collectives —
  and oneCCL's non-preemptible spin-wait kernels miss the rendezvous.
  The v24 matrix proved a DEVICE-side tiny-gather "barrier" still wedges
  (5/8): the tiny gather is itself a oneCCL spin op, so it inherits the
  same failure when the peer host is late.

  The drain works because it parks BOTH HOSTS at a defined point with
  empty queues. But a HOST-side rendezvous (no device sync at all) does
  the same job: both hosts arrive at the barrier before either submits
  draft collectives, device queues keep their (bounded, <=1 step)
  run-ahead, and the host-glue skew is absorbed by a CPU wait instead of
  a device spin. Cost: two flock round-trips + a us-scale CPU spin per
  spec step, vs a full device drain.

Mode map (VLLM_XPU_SPEC_DRAFT_BARRIER):
  0        off (v37/v61 posture)
  1        v60 full-device torch.xpu.synchronize() drain (diagnosis)
  2|host   v62 host-side flock barrier  <-- NEW

VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX gates both barrier forms
(default 0 = every spec step).

Usage: python3 patch_wedge_v62.py --check|--apply|--revert
Backups: *.v62bak   Marks: 'llm-scaler v62 (WEDGEFIX-G)'
"""
from __future__ import annotations

import os
import sys

SP = "/opt/venv/lib/python3.12/site-packages/vllm"
F_GMR = f"{SP}/v1/worker/gpu_model_runner.py"

# --- G0: host-barrier machinery, inserted after the MIN_CTX definition ----
G0_OLD = """\
_SPEC_DRAFT_BARRIER_MIN_CTX = int(
    os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX", "0")
)
"""

G0_NEW = """\
_SPEC_DRAFT_BARRIER_MIN_CTX = int(
    os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX", "0")
)
# llm-scaler v62 (WEDGEFIX-G): HOST-side spec draft barrier. The v60
# every-step torch.xpu.synchronize() drain is crash-safe but kills host
# run-ahead (-9..-31% decode). A device-side tiny-gather barrier was
# already proven insufficient (v24 matrix, 5/8 wedge) because the tiny
# gather is itself a non-preemptible oneCCL spin. Parking the two TP
# worker HOSTS at a file-lock rendezvous instead bounds collective
# submission skew without touching the device queues: both hosts pass
# the barrier before either submits draft collectives; queue divergence
# stays bounded by the <=1-step run-ahead the engine already enforces.
# VLLM_XPU_SPEC_DRAFT_BARRIER=2|host selects this mode.
import fcntl
import struct
import time as _v62_time

_SPEC_DRAFT_BARRIER_MODE = os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER", "0")
_SPEC_DRAFT_BARRIER_HOST = _SPEC_DRAFT_BARRIER_MODE in ("2", "host")
_V62_HOST_BARRIER_PARTIES = max(
    2, int(os.environ.get("VLLM_XPU_SPEC_HOST_BARRIER_PARTIES", "2"))
)
_V62_HOST_BARRIER_TIMEOUT_S = float(
    os.environ.get("VLLM_XPU_SPEC_HOST_BARRIER_TIMEOUT_S", "60")
)


class _V62HostSpecBarrier:
    \"\"\"Sense-reversing N-party barrier over /dev/shm state + flock.

    Two small files keyed by the EngineCore parent pid (shared by all TP
    workers of one engine, fresh path per engine generation):
      <base>.lock  — flock only, serializes state updates
      <base>.state — 16 bytes: (count, epoch) as two int64
    A party increments count under the lock; the party that reaches
    `parties` resets count and bumps epoch; the others spin (lock-free
    reads) until the epoch changes. Timeout releases the waiter with a
    one-line warning so a dead peer degrades to unsynchronized (the v37
    posture) instead of hanging the engine.
    \"\"\"

    def __init__(self, base: str, parties: int) -> None:
        self._base = base
        self._parties = parties
        self._lock_path = base + ".lock"
        self._state_path = base + ".state"
        open(self._lock_path, "a").close()
        with open(self._lock_path, "r+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            with open(self._state_path, "a+b") as sf:
                sf.seek(0, 2)  # end
                if sf.tell() < 16:
                    sf.seek(0)
                    sf.truncate()
                    sf.write(struct.pack("<qq", 0, 0))
                    sf.flush()

    def wait(self) -> None:
        with open(self._lock_path, "r+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            with open(self._state_path, "r+b") as sf:
                sf.seek(0)
                raw = sf.read(16)
                if len(raw) < 16:
                    raw = b"\\x00" * 16
                count, epoch = struct.unpack("<qq", raw)
                count += 1
                if count >= self._parties:
                    sf.seek(0)
                    sf.write(struct.pack("<qq", 0, epoch + 1))
                    sf.flush()
                    return  # passed the barrier (last arriver)
                sf.seek(0)
                sf.write(struct.pack("<qq", count, epoch))
                sf.flush()
                my_epoch = epoch
        deadline = _v62_time.monotonic() + _V62_HOST_BARRIER_TIMEOUT_S
        while True:
            with open(self._state_path, "rb") as sf:
                raw = sf.read(16)
            _, epoch = struct.unpack("<qq", raw if len(raw) == 16 else b"\\x00" * 16)
            if epoch != my_epoch:
                return
            if _v62_time.monotonic() > deadline:
                from vllm.logger import logger as _v62_log

                _v62_log.warning(
                    "llm-scaler v62 host spec barrier: peer did not arrive"
                    " within %.0fs (epoch=%d); proceeding unsynchronized",
                    _V62_HOST_BARRIER_TIMEOUT_S,
                    my_epoch,
                )
                return
            _v62_time.sleep(0.0002)


_V62_HOST_BARRIER: _V62HostSpecBarrier | None = None


def _v62_host_spec_barrier_wait() -> None:
    global _V62_HOST_BARRIER
    if _V62_HOST_BARRIER is None:
        base = f"/dev/shm/llm_scaler_spec_hb_{os.getppid()}"
        _V62_HOST_BARRIER = _V62HostSpecBarrier(base, _V62_HOST_BARRIER_PARTIES)
    _V62_HOST_BARRIER.wait()
"""

# --- G1: dispatch at the barrier site -------------------------------------
G1_OLD = """\
        if (
            _SPEC_DRAFT_BARRIER
            and common_attn_metadata.max_seq_len >= _SPEC_DRAFT_BARRIER_MIN_CTX
        ):
            torch.xpu.synchronize()
"""

G1_NEW = """\
        if common_attn_metadata.max_seq_len >= _SPEC_DRAFT_BARRIER_MIN_CTX:
            # llm-scaler v62 (WEDGEFIX-G): mode dispatch — 1 = v60 full
            # device drain, 2|host = host-side flock rendezvous (no
            # device sync; queues keep their <=1-step run-ahead).
            if _SPEC_DRAFT_BARRIER:
                torch.xpu.synchronize()
            elif _SPEC_DRAFT_BARRIER_HOST:
                _v62_host_spec_barrier_wait()
"""

PATCHES = [
    ("G0", F_GMR, "WEDGEFIX-G host barrier machinery (v62)",
     [(G0_OLD, G0_NEW, "llm-scaler v62 (WEDGEFIX-G): HOST-side spec draft barrier")]),
    ("G1", F_GMR, "WEDGEFIX-G barrier site dispatch (v62)",
     [(G1_OLD, G1_NEW, "llm-scaler v62 (WEDGEFIX-G): mode dispatch")]),
]


def die(msg: str) -> None:
    print(f"[v62] FAIL: {msg}")
    sys.exit(1)


def check_apply(revert: bool, report_only: bool = False) -> None:
    for tag, path, label, pairs in PATCHES:
        try:
            src = open(path, encoding="utf-8").read()
        except OSError as e:
            die(f"{label}: cannot read {path}: {e}")
        backup = path + ".v62bak"
        if revert:
            try:
                bak = open(backup, encoding="utf-8").read()
            except OSError:
                print(f"[v62] {tag} NO-BAK {path}")
                continue
            open(path, "w", encoding="utf-8").write(bak)
            print(f"[v62] {tag} REVERTED {path}")
            continue
        n_applied = 0
        n_bad = 0
        n_new = 0
        for old, new, mark in pairs:
            if mark in src:
                n_applied += 1
                print(f"[v62] {tag} ALREADY {path}")
                continue
            cnt = src.count(old)
            if cnt != 1:
                print(f"[v62] {tag} ANCHOR-BAD(cnt={cnt}) {path}")
                n_bad += 1
                continue
            src = src.replace(old, new, 1)
            n_new += 1
        if report_only:
            print(
                f"[v62] {tag} CHECK: already={n_applied} would-apply={n_new}"
                f" anchor-bad={n_bad} {path}"
            )
            continue
        if n_applied == len(pairs):
            continue
        if n_bad:
            die(f"{label}: anchor not found exactly once; file untouched")
        # One backup per file per run (G0/G1 share this file): first
        # needle in writes the pre-v62 state, later needles must not
        # clobber it, or --revert would only half-restore.
        if not os.path.exists(backup):
            open(backup, "w", encoding="utf-8").write(
                open(path, encoding="utf-8").read()
            )
        open(path, "w", encoding="utf-8").write(src)
        import py_compile

        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            open(path, "w", encoding="utf-8").write(
                open(backup, encoding="utf-8").read()
            )
            die(f"{label}: py_compile failed, restored backup: {e}")
        print(f"[v62] {tag} APPLIED {path}")
    print("[v62] pass complete")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("--check", "--apply", "--revert"):
        print(__doc__)
        die("usage: patch_wedge_v62.py --check|--apply|--revert")
    check_apply(
        revert=sys.argv[1] == "--revert",
        report_only=sys.argv[1] == "--check",
    )
