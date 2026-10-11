#!/usr/bin/env python3
"""fn8-t3 validation part 5 (Round 11): fp8.py owns-MK interposition.

Boot-3 law: flash-next-fp8 never swaps to FusedMoEModularMethod (layer.py
maybe_init_modular_kernel early-returns on supports_internal_mk), so the
runner calls Fp8MoEMethod.apply — patch_02a v2 interposes the tiered-store
head THERE, bridging to the shared builder via a lazily-bound resolver.

Verifies, on CPU (torch.xpu monkeypatched, docker exec -e env like part 2):
  1. Fp8MoEMethod.apply's patched head routes through the bridge: builds a
     store from a mock method/layer and returns the kernel's result.
  2. The bridge and the modular path share ONE registry (same layer id ->
     same store object).
  3. Loud-gate law: a non-block-fp8 experts object disables the store AND
     sets the warned flag (no silent disable).
"""
import dataclasses
import sys
from types import SimpleNamespace

import torch

torch.xpu.synchronize = lambda *a, **k: None
torch.xpu.current_device = lambda: "cpu"

import vllm.model_executor.layers.fused_moe.fused_moe_modular_method as m  # noqa: E402
import vllm.model_executor.layers.quantization.fp8 as q  # noqa: E402

assert m._FN8_TIERED is True and m._FN8_TIER_SLOTS == 4

E, N1, K, N2 = 512, 64, 32, 16


@dataclasses.dataclass
class MockMoeConfig:
    # REAL FusedMoEConfig shape (boot-5 law): ep_rank/ep_size are
    # properties over moe_parallel_config, NOT fields; the pool trio is
    # num_experts/num_local_experts/num_logical_experts.
    num_experts: int = E
    num_local_experts: int = E
    num_logical_experts: int = E
    moe_parallel_config: object = None


class MockKernel:
    def __init__(self, experts):
        self.fused_experts = experts
        self.calls = []

    def apply(self, **kw):
        self.calls.append(kw)
        return "OUT_FP8"


class MockQuantDesc:
    # REAL FusedMoEQuantDesc: plain mutable `scale` field.
    def __init__(self, scale):
        self.scale = scale


class MockQuantConfig:
    # REAL FusedMoEQuantConfig shape (boot-6 law): w1_scale/w2_scale are
    # read-only properties over the _w1/_w2 descs; the desc `scale` field
    # is the plain mutable repoint target (qc._w1.scale = pool).
    def __init__(self, w1_scale, w2_scale):
        self._w1 = MockQuantDesc(w1_scale)
        self._w2 = MockQuantDesc(w2_scale)

    @property
    def w1_scale(self):
        return self._w1.scale

    @property
    def w2_scale(self):
        return self._w2.scale


experts = SimpleNamespace(
    is_block_fp8=True,
    moe_config=MockMoeConfig(),
    quant_config=MockQuantConfig(
        torch.arange(E * 4, dtype=torch.float32).reshape(E, 2, 2),
        torch.arange(E * 2, dtype=torch.float32).reshape(E, 2, 1),
    ),
)
kernel = MockKernel(experts)
# the fp8-owns-MK shape: the METHOD owns moe_kernel (no FusedMoEModularMethod)
method = SimpleNamespace(moe_kernel=kernel)
layer = SimpleNamespace(
    layer_name="model.layers.9.mlp.experts",
    global_num_experts=E,
    expert_map=None,
    activation="SILU_SENTINEL",
    apply_router_weight_on_input=False,
    w13_weight=torch.nn.Parameter(torch.randn(E, N1, K)),
    w2_weight=torch.nn.Parameter(torch.randn(E, N2, K // 2)),
)

# -- 1: the patched Fp8MoEMethod.apply routes through the bridge ------------
tk = torch.tensor([[5, 6]], dtype=torch.int32)
tw = torch.full((1, 2), 0.5, dtype=torch.float32)
x = torch.randn(1, K)
mock_self = SimpleNamespace(moe_kernel=kernel, is_monolithic=False)
out = q.Fp8MoEMethod.apply(mock_self, layer, x, tw, tk, "SHARED_FP8")
assert out == "OUT_FP8", out
kw = kernel.calls[-1]
assert kw["w1"].shape[0] == m._FN8_TIER_SLOTS + 1, "pool, not host mirror"
assert kw["expert_map"] is None
got = kw["topk_ids"].tolist()[0]
assert got == [st for st in (
    m._fn8_store_for(layer, method).slot_of.get(5),
    m._fn8_store_for(layer, method).slot_of.get(6))], got
assert got != [5, 6], "ids must be remapped to slots, not passed through"
print("FP8_APPLY_BRIDGE_OK topk", got)

# -- 2: one registry across both choke points -------------------------------
st_direct = m._fn8_store_for(layer, method)
st_via_fp8_bridge = q._fn8_store_for(layer, mock_self)
assert st_direct is st_via_fp8_bridge, "fp8 bridge must share the registry"
assert len(m._FN8_BUILT) == 1
print("SHARED_REGISTRY_OK built=%d" % len(m._FN8_BUILT))

# -- 3: loud gate — non-block-fp8 experts disables AND warns ----------------
assert m._FN8_WARNED_GATE is False, "no disable has happened yet"
bad_experts = SimpleNamespace(is_block_fp8=False)
bad_kernel = MockKernel(bad_experts)
bad_method = SimpleNamespace(moe_kernel=bad_kernel)
bad_layer = SimpleNamespace(
    layer_name="model.layers.10.mlp.experts",
    global_num_experts=E, expert_map=None, activation="SILU",
    apply_router_weight_on_input=False,
    w13_weight=torch.nn.Parameter(torch.randn(E, N1, K)),
    w2_weight=torch.nn.Parameter(torch.randn(E, N2, K // 2)),
)
assert m._fn8_store_for(bad_layer, bad_method) is None
assert m._FN8_WARNED_GATE is True, "gate disable must set the warned flag"
assert m._fn8_store_for(bad_layer, bad_method) is None, "stays disabled"
assert len(m._FN8_BUILT) == 1
print("LOUD_GATE_OK warned=%s disabled=%d" % (
    m._FN8_WARNED_GATE, len(m._FN8_DISABLED)))

print("FN8_T3_FP8BRIDGE_ALL_OK")
sys.exit(0)
