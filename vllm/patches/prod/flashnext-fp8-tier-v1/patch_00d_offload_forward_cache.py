#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 patch_00d — empty XPU cache at each offloaded
module's forward boundary (offloader pageable functional_call wrapper).

Wedge evidence (fn8-g1 leg-5, 2026-10-11): with the weights loaded clean
(patch_00c kept the REPACK loop's per-module cache hygiene), the engine
stalled in the profile run — EngineCore main thread D-state with wchan
xe_vm_bind_ioctl, dmesg "xe ... Range is already in VRAM", xpu-smi and py-spy
both hang against the process.  The offloader's per-layer forward closures
copy ~2.9 GiB/layer H2D (device_state dict) with no allocator hygiene, so
expandable_segments hoards freed blocks across layers and segment growth
stalls in the driver's VM bind path — the standing P82 family, now on the
forward path.  patch_00c proved the medicine on the load path; this is the
same empty_cache per module boundary, applied where the copies actually
happen at serving time.

Idempotent (FN8_00D marker), backs up uva.py.q4dbak, --restore undoes.
"""
import sys

TARGET = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
          "offloader/uva.py")
MARKER = "FN8_00D"
ANCHOR = (
    "                module.forward = forward\n"
    "                return output\n"
)
INJECT = """                # FN8_00D (flashnext-fp8-tier-v1 P0): 00c law extended to
                # the forward path — each wrapped module's H2D copies hoard in
                # the XPU caching allocator (expandable_segments never returns
                # freed blocks); across layers the segment growth stalls in
                # xe_vm_bind_ioctl (P82 family; leg-5 profile wedge 2026-10-11:
                # D-state wchan xe_vm_bind_ioctl, xpu-smi/py-spy hung).  Empty
                # the cache per module boundary so every layer's copies start
                # from actually-free VRAM.
                try:
                    torch.xpu.empty_cache()
                except Exception:
                    pass
"""


def md5(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        bak = TARGET + ".q4dbak"
        import shutil
        shutil.copyfile(bak, TARGET)
        print("FN8_00D RESTORED", TARGET, md5(TARGET))
        return

    with open(TARGET) as f:
        src = f.read()

    if MARKER in src:
        print("FN8_00D already applied, nothing to do", md5(TARGET))
        return

    assert "import torch" in src, "torch import missing?!"
    n = src.count(ANCHOR)
    assert n == 1, f"anchor count {n} != 1 in {TARGET}"

    import shutil
    shutil.copyfile(TARGET, TARGET + ".q4dbak")
    before = md5(TARGET)
    src = src.replace(ANCHOR, INJECT + ANCHOR)
    with open(TARGET, "w") as f:
        f.write(src)
    print(f"FN8_00D backup written: {TARGET}.q4dbak (md5 {before})")
    print(f"FN8_00D PATCHED: {TARGET} md5 {before} -> {md5(TARGET)}")


if __name__ == "__main__":
    main()
