"""llm-scaler v28dbg flight recorder (KNOWN_ISSUES #11 graphs-x-spec residual).

Writes one unbuffered timestamped line per engine phase (graph replay,
collectives, drafter propose) to /tmp/fr_<pid>.log. When the engine
wedges, the LAST line names the phase that never returned; combined with
py-spy host stacks (where the submitting thread parks) this brackets the
hang without needing an external L0 tracer (onetrace/vtune absent from
the image; ZE loader debug trace unsupported by this loader build).

Zero-config: disabled entirely with VLLM_XPU_FR=0.
"""
import os
import time

import torch

_fh = None
_enabled = os.getenv("VLLM_XPU_FR", "1") not in ("", "0")


def _handle():
    global _fh
    if _fh is None:
        path = f"/tmp/fr_{os.getpid()}.log"
        _fh = open(path, "w", buffering=1)
    return _fh


def log(event: str) -> None:
    # Never trace into file I/O: this module is called from inside the
    # torch.compile region (all_reduce during profile_run/piece tracing)
    # and dynamo faults on open()/perf_counter (verified boot crash).
    if not _enabled or torch.compiler.is_compiling():
        return
    try:
        h = _handle()
        h.write(f"{time.perf_counter():.4f} {event}\n")
        h.flush()
    except Exception:
        pass
