#!/usr/bin/env python3
"""fn8 patch_02b v2 validation — no-wrap containment law (in-container).

Reproduces the TP1 boot-OOM shape at the granularity wrap_modules actually
wraps: a DECODER-LAYER module (dense child + mlp.experts.{w13,w2,scales}
+ a CPU-registered ngram table inside the subtree) passed through
UVAOffloader._maybe_offload_to_cpu on the pageable path.

v2 laws under test:
  1. layer does NOT get the wholesale-copy closure (forward stays the
     class method — no instance-level forward attr), expert params page
     to CPU, dense stays resident, the ngram table is never touched,
     budget counts each expert param exactly once;
  2. FusedMoE-granularity module still sets _fn8_tier_no_wrap (belt);
  3. budget exhausted -> nothing offloads, nothing wraps;
  4. dense-only module: no whitelist match -> nothing happens.

Needs XPU (the offloader's device gate early-returns on CPU params).
"""
import os

os.environ["FN8_TIERED"] = "1"
os.environ["VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"] = "1"
os.environ["VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY"] = "1"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

assert torch.xpu.is_available(), "needs XPU for the offloader device gate"
DEV = torch.device("xpu")

from vllm.model_executor.offloader.uva import UVAOffloader  # noqa: E402

MOE_PARAMS = ("w13_weight", "w2_weight",
              "w13_weight_scale_inv", "w2_weight_scale_inv")


class FusedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13_weight = nn.Parameter(torch.zeros(4, 8, 16, device=DEV))
        self.w2_weight = nn.Parameter(torch.zeros(4, 16, 8, device=DEV))
        self.w13_weight_scale_inv = nn.Parameter(
            torch.zeros(4, 2, device=DEV))
        self.w2_weight_scale_inv = nn.Parameter(
            torch.zeros(4, 2, device=DEV))

    def forward(self, x):
        return x


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = FusedMoE()

    def forward(self, x):
        return x


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, device=DEV, dtype=torch.float16)

    def forward(self, x):
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Dense()
        self.mlp = Mlp()
        # EP-path shape: big table registered CPU-side INSIDE the layer
        # subtree (the 47.69 GiB ngram table, scaled down 1000x).
        self.ngram_table = torch.zeros(20_000, 1280, dtype=torch.float16)

    def forward(self, x):
        return x


def is_wrapped(mod):
    # the closure installs an instance-level forward attribute
    return "forward" in mod.__dict__


def main():
    # -- case 1: decoder-layer granularity (the TP1 boot shape) ----------
    layer = DecoderLayer()
    off = UVAOffloader(cpu_offload_max_bytes=1 << 30)
    out = off._maybe_offload_to_cpu(layer)
    assert out is layer
    assert not is_wrapped(layer), "v2 FAIL: layer got the wholesale closure"
    exp = layer.mlp.experts
    for pn in MOE_PARAMS:
        p = getattr(exp, pn)
        assert p.data.device.type == "cpu", f"{pn} did not page to CPU"
    assert layer.self_attn.q_proj.weight.device.type == "xpu", \
        "dense paged (whitelist leak)"
    assert (layer.ngram_table.device.type == "cpu"
            and layer.ngram_table.shape == (20000, 1280)), \
        "ngram table touched"
    assert not hasattr(layer, "_fn8_tier_no_wrap"), \
        "flag set on non-owner module"
    assert off.cpu_offload_bytes == sum(
        p.numel() * p.element_size() for p in exp.parameters()), \
        "budget accounting off"

    # -- case 2: FusedMoE granularity (belt check, keeps old behavior) ---
    moe = FusedMoE()
    off2 = UVAOffloader(cpu_offload_max_bytes=1 << 30)
    off2._maybe_offload_to_cpu(moe)
    assert not is_wrapped(moe) and getattr(moe, "_fn8_tier_no_wrap", False), \
        "FusedMoE-direct path regressed"

    # -- case 3: budget exhausted -> nothing offloads, nothing wraps -----
    layer3 = DecoderLayer()
    off3 = UVAOffloader(cpu_offload_max_bytes=1 << 30)
    off3.cpu_offload_bytes = off3.cpu_offload_max_bytes
    off3._maybe_offload_to_cpu(layer3)
    assert not is_wrapped(layer3)
    assert layer3.mlp.experts.w13_weight.device.type == "xpu"

    # -- case 4: dense-only module: no whitelist match --------------------
    d = Dense()
    off4 = UVAOffloader(cpu_offload_max_bytes=1 << 30)
    off4._maybe_offload_to_cpu(d)
    assert not is_wrapped(d) and d.q_proj.weight.device.type == "xpu"

    print("FN8_02B_V2_OK layer-unwrapped, experts paged, dense+table resident")


if __name__ == "__main__":
    main()
