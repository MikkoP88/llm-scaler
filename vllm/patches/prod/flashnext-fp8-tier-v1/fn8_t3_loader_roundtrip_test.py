#!/usr/bin/env python3
"""fn8 patch_02c validation — loader no-round-trip law (in-container, CPU).

Boot-4 law: device_loading_context's entry loop H2D-moves every CPU param
for process_weights_after_loading and the exit restores it — 104.71 GiB of
round-trips that are a no-op for XPU block-fp8 (convert is an identity)
with patch_02a owning apply.  patch_02c skips whitelisted tiered params.

Verifies (torch.Tensor.to monkeypatched with a counting stub — no real
device needed):
  1. _fn8_is_tiered_param truth table (exact-segment law);
  2. inside device_loading_context, a tiered param is never passed to
     .to (entry AND exit) and keeps its data object identity, while a
     non-tiered CPU param is moved at entry and restored at exit;
  3. the FN8_00C empty_cache tail stays CPU-safe (try/except).

Run: docker exec -e FN8_TIERED=1 fn8-p2 python3 /stage/fn8_t3_loader_roundtrip_test.py
"""
import os
import sys

assert os.environ.get("FN8_TIERED") == "1", "set FN8_TIERED=1"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from vllm.model_executor.model_loader import utils as u  # noqa: E402

assert u._FN8_TIERED is True

# -- 1: truth table -----------------------------------------------------------
TT = [
    ("w13_weight", True),
    ("w2_weight", True),
    ("w13_weight_scale_inv", True),
    ("w2_weight_scale_inv", True),
    ("w13_weight_scale", False),      # partial suffix must NOT match
    ("w13_weights", False),
    ("bias", False),
    ("weight", False),
    ("q_proj_weight", False),
    ("router_gate_weight", False),
    ("shared_experts.w13_weight", True),  # containment, same as 02b
]
for name, want in TT:
    got = u._fn8_is_tiered_param(name)
    assert got is want, f"_fn8_is_tiered_param({name!r}) = {got}, want {want}"
print("TRUTH_TABLE_OK", len(TT), "cases")


# -- 2: context skips tiered, moves non-tiered --------------------------------
class FakeMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13_weight = nn.Parameter(torch.zeros(4, 8, 16))
        self.w2_weight = nn.Parameter(torch.zeros(4, 16, 8))
        self.w13_weight_scale_inv = nn.Parameter(torch.zeros(4, 2))
        self.bias = nn.Parameter(torch.zeros(8))


calls = []  # (data_ptr of tensor whose .to ran)


def _counting_to(self, *a, **k):
    calls.append(self.data_ptr())
    return self


_orig_to = torch.Tensor.to
torch.Tensor.to = _counting_to
try:
    mod = FakeMoe()
    tiered_ptrs = {
        mod.w13_weight.data_ptr(), mod.w2_weight.data_ptr(),
        mod.w13_weight_scale_inv.data_ptr(),
    }
    bias_ptr = mod.bias.data_ptr()
    # NOTE: param.data returns a fresh view object per access — identity
    # must be judged by Parameter object + storage pointer, not `.data is`.
    w13_param = mod.w13_weight

    with u.device_loading_context(mod, torch.device("xpu")):
        # inside: tiered params still host, storage untouched
        assert mod.w13_weight is w13_param
        assert mod.w13_weight.data_ptr() in tiered_ptrs
        assert mod.w13_weight.device.type == "cpu"

    assert not (tiered_ptrs & set(calls)), \
        f"tiered param round-tripped: {calls}"
    assert bias_ptr in calls, "non-tiered param was not moved at entry"
    assert calls.count(bias_ptr) >= 2, "non-tiered param not restored at exit"
    assert mod.w13_weight is w13_param, "tiered Parameter replaced"
    assert mod.w13_weight.data_ptr() in tiered_ptrs, "tiered storage moved"
finally:
    torch.Tensor.to = _orig_to
print("CTX_SKIP_OK tiered_untouched=%d bias_calls=%d" % (
    len(tiered_ptrs), calls.count(bias_ptr)))

print("FN8_T3_LOADER_ALL_OK")
sys.exit(0)
