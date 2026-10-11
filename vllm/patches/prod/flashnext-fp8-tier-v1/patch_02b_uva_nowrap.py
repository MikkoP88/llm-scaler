#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 patch_02b — uva.py: no-wrap + expert whitelist.

Companion to patch_02a (TieredExpertsStore).  Two edits to the UVA
offloader's functional_call fallback path (the P0 shape: UVA disabled,
weights pageable-host):

1. Whitelist default: with FN8_TIERED=1 and no explicit
   cpu_offload_params, page ONLY the routed-expert tensors
   (w13/w2 [+ scale_inv segments if they are params]).  Dense weights,
   shared experts, gates, PLE/GDN stay device-resident.  The whitelist
   matcher is exact-segment (".w13_weight." does not match
   ".w13_weight_scale_inv."), and non-matching params skip WITHOUT
   consuming budget.  FN8_TIER_OFFLOAD_PARAMS overrides the default
   comma list; an explicit CLI --cpu-offload-params still wins (env only
   fills the empty set).

2. No-wrap: the per-step closure (state_dict `.to(device)` + functional_call
   per call = the measured 29.8 s/token wholesale copy) is NOT installed on
   any module whose (recursive) named_parameters include a whitelisted
   routed-expert tensor when FN8_TIERED=1.  GRANULARITY LAW (TP1 boot OOM
   2026-10-11): models/utils.py get_layers hands wrap_modules whole DECODER
   LAYERS (layer_fn generator) — the FusedMoE leaf is never visited, so the
   original direct-attribute hasattr(w13_weight) detection missed, the
   layer got the closure, and its recursive state_dict materialized the
   EP-path's CPU-registered 20Mx1280 fp16 ngram table (47.69 GiB) at the
   dummy run = OOM.  Detection now matches at the module granularity that
   actually wraps: whitelist containment over named_parameters.  Params
   still page at load (budget accounting intact); placement from the first
   apply on is owned by the tiered store, which copies expert rows into
   its slot pool on demand.  The FN8_00D empty-cache fix stays in place
   for every module that still wraps.

Guard law: patch_02a disables its store only loudly (build failure logs
+ raises through the kernel's device assert) — a host-weight GEMM crash
is the correct failure mode, never silent wrong output.

Env: FN8_TIERED=1 (same master gate as patch_02a),
FN8_TIER_OFFLOAD_PARAMS=<comma segments>.

Idempotent (FN8_02B marker), backs up .q4dbak, --restore undoes.
"""
import sys

TARGET = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
          "offloader/uva.py")
MARKER = "FN8_02B"

ANCHOR_LOGGER = "logger = init_logger(__name__)\n"

INJECT_LOGGER = '''logger = init_logger(__name__)

# FN8_02B (flashnext-fp8-tier-v1 P2): tiered-offload support.  Same
# master gate as patch_02a; inert unless FN8_TIERED=1.
import os as _fn8_os

_FN8_TIERED = _fn8_os.environ.get("FN8_TIERED", "") == "1"
_FN8_TIER_OFFLOAD_PARAMS = {
    seg for seg in _fn8_os.environ.get(
        "FN8_TIER_OFFLOAD_PARAMS",
        "w13_weight,w2_weight,w13_weight_scale_inv,w2_weight_scale_inv",
    ).split(",") if seg
}
'''

ANCHOR_INIT = "        self.cpu_offload_params = cpu_offload_params or set()\n"

NEW_INIT = '''        self.cpu_offload_params = cpu_offload_params or set()
        if _FN8_TIERED and not self.cpu_offload_params:
            # FN8_02B: page ONLY routed-expert tensors; the tiered store
            # (patch_02a) owns their placement from the first apply on.
            self.cpu_offload_params = set(_FN8_TIER_OFFLOAD_PARAMS)
'''

ANCHOR_WRAP = '''        if offloaded_parameters and not self.uva_offloading:
            original_forward = module.forward
'''

NEW_WRAP = '''        # FN8_02B: modules whose subtree contains a whitelisted routed-expert
        # tensor never get the per-step wholesale-copy closure under tiering
        # — the store does row-granular placement instead.  Granularity law:
        # wrap_modules receives whole decoder layers (get_layers layer_fn
        # generator), NOT FusedMoE leaves — detection must be containment
        # over recursive named_parameters, not direct attributes (the
        # hasattr form missed the layer and its state_dict materialized the
        # 47.69 GiB CPU-side ngram table at the dummy run = TP1 boot OOM).
        _fn8_owns_moe = (
            hasattr(module, "w13_weight")
            and hasattr(module, "w2_weight")
        )
        _fn8_no_wrap = (
            _FN8_TIERED
            and any(
                any(f".{seg}." in f".{n}." for seg in _FN8_TIER_OFFLOAD_PARAMS)
                for n, _ in module.named_parameters()
            )
        )
        if (_fn8_owns_moe and offloaded_parameters):
            module._fn8_tier_no_wrap = True
        if (offloaded_parameters and not self.uva_offloading
                and not _fn8_no_wrap):
            original_forward = module.forward
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
        print("FN8_02B RESTORED", TARGET, md5(TARGET))
        return

    with open(TARGET) as f:
        src = f.read()

    if MARKER in src:
        print("FN8_02B already applied, nothing to do", md5(TARGET))
        return

    for name, anchor in (
        ("logger", ANCHOR_LOGGER),
        ("init", ANCHOR_INIT),
        ("wrap", ANCHOR_WRAP),
    ):
        n = src.count(anchor)
        assert n == 1, f"anchor {name} count {n} != 1"

    import shutil
    shutil.copyfile(TARGET, TARGET + ".q4dbak")
    before = md5(TARGET)

    src = src.replace(ANCHOR_LOGGER, INJECT_LOGGER)
    src = src.replace(ANCHOR_INIT, NEW_INIT)
    src = src.replace(ANCHOR_WRAP, NEW_WRAP)
    assert src.count(MARKER) >= 3, "marker missing after edit?!"

    with open(TARGET, "w") as f:
        f.write(src)
    print(f"FN8_02B backup written: {TARGET}.q4dbak (md5 {before})")
    print(f"FN8_02B PATCHED: {TARGET} md5 {before} -> {md5(TARGET)}")


if __name__ == "__main__":
    main()
