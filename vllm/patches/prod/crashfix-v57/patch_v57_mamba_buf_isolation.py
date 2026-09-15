#!/usr/bin/env python3
"""llm-scaler v57 patcher — mamba deferred-copy buffer isolation.

ROOT CAUSE (2026-09-11 wedge class: stalls 1-4 + crash-2/5 lineage):
In async-scheduling mode ONE shared MambaCopyBuffers instance is used by
TWO batch_memcpy producers inside a single execute_model call:

  1. mamba_utils.run_deferred_postprocess (v32 deferred path, at
     execute_model ENTRY: offset=0, np fill, H2D, launch K_DEF)
  2. mamba_utils.preprocess_mamba (inside _update_states, milliseconds
     later in the SAME call: offset=0 RESET, np overwrite, H2D overwrite
     of the SAME gpu slots, launch K_PRE)

K_DEF is enqueued behind the previous step's still-running tail (the
async-scheduling overlap), so when K_DEF finally executes it can read a
MIX of K_PRE's metadata: src/dst/sizes triplets from two different
producers -> mamba state copied between wrong blocks with mismatched
sizes -> silent OOB writes (expandable_segments VA: no fault) ->
corrupted GDN state -> downstream kernel spin ->
num_accepted_tokens_event / async_copy_ready_event never signal ->
v55 ASYNC-EVENT-STALL fence -> engine death. Host/GPU timing race:
DEBUG logging slows the host between the two producers, letting K_DEF
start first (why DEBUG-instrumented boots resist reproduction). Fires
at 2048-block boundaries where both carries land in one step.

FIX: give the deferred producer its own MambaCopyBuffers instance.
Within-step aliasing becomes impossible; cross-step reuse of either
buffer stays safe because the execute_model entry fence
(num_accepted_tokens_event wait) guarantees the prior step's kernels
completed. Zero hot-path cost (second instance allocated once, tiny),
zero stream-order change, zero numerics change.

Idempotent: skips when _get_mamba_copy_bufs_deferred is present.
"""
import argparse
import py_compile
import sys
from pathlib import Path

GMR_ANCHOR = (
    "    def _get_mamba_copy_bufs(self) -> mamba_utils.MambaCopyBuffers:\n"
    "        if self._mamba_copy_bufs is None:\n"
    "            self._mamba_copy_bufs = mamba_utils.MambaCopyBuffers.create(\n"
    "                self.max_num_reqs,\n"
    "                self.kv_cache_config,\n"
    "                self.model.get_mamba_state_copy_func(),\n"
    "                self._make_buffer,\n"
    "            )\n"
    "        return self._mamba_copy_bufs\n"
)

GMR_NEW = GMR_ANCHOR + """
    def _get_mamba_copy_bufs_deferred(self) -> mamba_utils.MambaCopyBuffers:
        # llm-scaler v57: private instance for the v32 deferred mamba
        # postprocess. The shared instance is filled and launched TWICE
        # per async-scheduling execute_model (deferred entry +
        # preprocess_mamba in _update_states) with offset reset + H2D
        # overwrite between the launches; the first kernel is still
        # queued behind the previous step's tail and can read the second
        # producer's metadata (mismatched src/dst/sizes -> wrong-block
        # copies -> corrupted GDN state -> kernel spin -> v55
        # ASYNC-EVENT-STALL wedge). A private buffer set makes the
        # aliasing impossible. Cross-step reuse stays safe: the entry
        # fence guarantees the prior step's kernels completed.
        if getattr(self, "_mamba_copy_bufs_deferred", None) is None:
            self._mamba_copy_bufs_deferred = (
                mamba_utils.MambaCopyBuffers.create(
                    self.max_num_reqs,
                    self.kv_cache_config,
                    self.model.get_mamba_state_copy_func(),
                    self._make_buffer,
                )
            )
        return self._mamba_copy_bufs_deferred
"""

MU_OLD = "    copy_bufs = model_runner._get_mamba_copy_bufs()\n"
MU_NEW = (
    "    # llm-scaler v57: deferred producer uses its own buffer instance\n"
    "    # (within-step aliasing with preprocess_mamba = the wedge race).\n"
    "    copy_bufs = model_runner._get_mamba_copy_bufs_deferred()\n"
)


def _replace_once(src: str, old: str, new: str, what: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(
            f"ANCHOR-FAIL {what}: expected exactly 1 occurrence, found {n}")
    return src.replace(old, new, 1)


def _write(path: Path, src: str) -> None:
    path.write_text(src, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sp", default="/opt/venv/lib/python3.12/site-packages",
        help="site-packages root containing the vllm package")
    args = ap.parse_args()

    gmr = Path(args.sp) / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
    mu = Path(args.sp) / "vllm" / "v1" / "worker" / "mamba_utils.py"
    for p in (gmr, mu):
        if not p.is_file():
            raise SystemExit(f"target missing: {p}")

    src = gmr.read_text(encoding="utf-8")
    if "_get_mamba_copy_bufs_deferred" in src:
        print(f"{gmr.name}: already patched, skip")
    else:
        src = _replace_once(src, GMR_ANCHOR, GMR_NEW, "gmr deferred getter")
        _write(gmr, src)
        print(f"{gmr.name}: patched (deferred getter added)")

    src = mu.read_text(encoding="utf-8")
    if "_get_mamba_copy_bufs_deferred()" in src:
        print(f"{mu.name}: already patched, skip")
    else:
        src = _replace_once(src, MU_OLD, MU_NEW, "mamba deferred buf fetch")
        _write(mu, src)
        print(f"{mu.name}: patched (deferred producer isolated)")

    print("V57_MAMBA_BUF_ISOLATION_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
