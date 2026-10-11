#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 patch_00c — empty XPU cache after each offloaded
module's repack round-trip (device_loading_context finally block).

Wedge evidence (fn8-g1 leg, 2026-10-11): process_weights_after_loading moves
each FusedMoE to GPU, repacks, restores to CPU.  The XPU caching allocator
(expandable_segments) never returns the freed per-layer blocks (~2.5 GiB each);
on top of ~20 GiB resident weights the 32 GiB card floor-hits around layer 5
(VRAM pinned at 31280 MiB, GPU read -> 0, restore copy parked in urEventWait
forever) — the standing P82 silent-submit-failure law.  Emptying the cache per
module keeps every repack within a ~5 GiB transient envelope.

Idempotent (FN8_00C marker), backs up utils.py.q4cbak, --restore undoes.
"""
import sys

TARGET = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
          "model_loader/utils.py")
MARKER = "FN8_00C"
ANCHOR = (
    "                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)\n"
    "                p._vllm_is_uva_offloaded = True\n"
)
INJECT = """        # FN8_00C (flashnext-fp8-tier-v1 P0): per-module repack round-trips
        # leave freed VRAM cached (expandable_segments never returns it);
        # ~2.5 GiB/layer of hoard on ~20 GiB resident floor-hits the card by
        # layer ~5 -> silent submit failure (P82 law) -> urEventWait wedge
        # in this finally block.  Empty the cache per module so every repack
        # starts from actually-free VRAM.
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
        bak = TARGET + ".q4cbak"
        import shutil
        shutil.copyfile(bak, TARGET)
        print("FN8_00C RESTORED", TARGET, md5(TARGET))
        return

    with open(TARGET) as f:
        src = f.read()

    if MARKER in src:
        print("FN8_00C already applied, nothing to do", md5(TARGET))
        return

    assert "import torch" in src, "torch import missing?!"
    n = src.count(ANCHOR)
    assert n == 1, f"anchor count {n} != 1 in {TARGET}"

    import shutil
    shutil.copyfile(TARGET, TARGET + ".q4cbak")
    before = md5(TARGET)
    src = src.replace(ANCHOR, ANCHOR + INJECT)
    with open(TARGET, "w") as f:
        f.write(src)
    print(f"FN8_00C backup written: {TARGET}.q4cbak (md5 {before})")
    print(f"FN8_00C PATCHED: {TARGET} md5 {before} -> {md5(TARGET)}")


if __name__ == "__main__":
    main()
