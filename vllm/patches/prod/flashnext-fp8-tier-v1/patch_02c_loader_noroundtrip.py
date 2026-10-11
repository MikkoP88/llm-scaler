#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 patch_02c — model_loader/utils.py: no round-trip.

BOOT-4 LAW (2026-10-11): with cpu-offloading, DefaultModelLoader.
process_weights_after_loading wraps every quant module in
device_loading_context, whose ENTRY moves all CPU params H2D and whose
EXIT restores them D2H — per FusedMoE module. For flash-next-fp8 that is
2.18 GiB x 48 layers x both directions = the whole 104.71 GiB expert
mass, re-paged to bake in ZERO changes:

  * Fp8MoEMethod.process_weights_after_loading (fp8.py:813) skips every
    requant branch for dynamic-activation block fp8 (no fnuz, no static
    input scales, no per-tensor strategy) and calls _setup_kernel ->
    convert_to_fp8_moe_kernel_format (oracle/fp8.py:425).
  * For Fp8MoeBackend.XPU with block_quant the convert is an IDENTITY:
    the branch only transposes when `not block_quant` ("block weights
    (and their per-block scales) [stay] untransposed"), and w13_scale /
    w2_scale are returned untouched for every backend.
  * patch_02a intercepts Fp8MoEMethod.apply, so nothing after the loader
    ever consumes the device-side copies — the tiered store slices from
    the host tensors.

Measured on boot 4: the pageable D2H restore crawled at ~50 MB/s
(xpu-smi: EU idle, copy engine idle, memory.read.bandwidth ~52 MB/s) —
layer 1 of 48 still restoring after 12+ min => hours projected. Not
moving the tensors is bit-identical to moving them.

Edit: in device_loading_context's entry loop, params whose name matches
the tiered whitelist (same exact-segment list as patch_02b) are skipped
when FN8_TIERED=1 — they stay host-resident, so original_device_states
never contains them and the exit restore skips them too. Everything
else (attention/linear quant modules, biases, router weights, shared
experts) keeps the round-trip. The FN8_00C empty_cache per module stays.

Env: FN8_TIERED=1 (master gate), FN8_TIER_OFFLOAD_PARAMS (same override
semantics as patch_02b).

Idempotent (FN8_02C marker), backs up .q4dbak, --restore undoes.
"""
import sys

TARGET = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
          "model_loader/utils.py")
MARKER = "FN8_02C"

ANCHOR_CTX = '''@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
'''

INJECT_CTX = '''# FN8_02C (flashnext-fp8-tier-v1): loader no-round-trip for tiered
# routed-expert tensors. Same master gate + exact-segment list as
# patch_02b; inert unless FN8_TIERED=1.
import os as _fn8_os

_FN8_TIERED = _fn8_os.environ.get("FN8_TIERED", "") == "1"
_FN8_TIER_OFFLOAD_PARAMS = {
    seg for seg in _fn8_os.environ.get(
        "FN8_TIER_OFFLOAD_PARAMS",
        "w13_weight,w2_weight,w13_weight_scale_inv,w2_weight_scale_inv",
    ).split(",") if seg
}


def _fn8_is_tiered_param(name: str) -> bool:
    # Exact-segment law (patch_02b): ".w13_weight." never matches
    # ".w13_weight_scale_inv.", and dotted foreign names never match.
    return any(
        f".{seg}." in f".{name}." for seg in _FN8_TIER_OFFLOAD_PARAMS
    )


@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
'''

ANCHOR_ENTRY = '''    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        if getattr(p, "_vllm_is_uva_offloaded", False):
            uva_offloaded_parameters.append(name)
'''

NEW_ENTRY = '''    for name, p in module.named_parameters():
        # FN8_02C: tiered routed-expert tensors skip the process-time
        # H2D/D2H round-trip. XPU block-fp8 convert is an identity
        # (oracle/fp8.py convert_to_fp8_moe_kernel_format leaves block
        # weights [E, N, K] and scales untouched) and patch_02a owns
        # apply, so re-paging 2.18 GiB/layer bakes in zero changes —
        # boot 4 measured the pageable restore at ~50 MB/s = hours.
        # Skipping keeps them host-resident (exit restore skips too,
        # via original_device_states).
        if _FN8_TIERED and _fn8_is_tiered_param(name):
            continue
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        if getattr(p, "_vllm_is_uva_offloaded", False):
            uva_offloaded_parameters.append(name)
'''


def md5(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        bak = TARGET + ".q4dbak"
        import shutil
        shutil.copyfile(bak, TARGET)
        print("FN8_02C RESTORED", TARGET, md5(TARGET))
        return

    with open(TARGET) as f:
        src = f.read()

    if MARKER in src:
        print("FN8_02C already applied, nothing to do", md5(TARGET))
        return

    for name, anchor in (
        ("context def", ANCHOR_CTX),
        ("entry loop", ANCHOR_ENTRY),
    ):
        n = src.count(anchor)
        assert n == 1, f"anchor {name} count {n} != 1"

    import shutil
    shutil.copyfile(TARGET, TARGET + ".q4dbak")
    before = md5(TARGET)

    src = src.replace(ANCHOR_CTX, INJECT_CTX)
    src = src.replace(ANCHOR_ENTRY, NEW_ENTRY)
    assert src.count(MARKER) >= 2, "marker missing after edit?!"

    with open(TARGET, "w") as f:
        f.write(src)
    print(f"FN8_02C backup written: {TARGET}.q4dbak (md5 {before})")
    print(f"FN8_02C PATCHED: {TARGET} md5 {before} -> {md5(TARGET)}")


if __name__ == "__main__":
    main()
