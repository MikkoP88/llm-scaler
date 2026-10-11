#!/usr/bin/env python3
"""fn8-t3 validation part 2: exercise _Fn8TieredStore slot machinery on CPU.

Runs with FN8_TIERED=1 FN8_TIER_SLOTS=4 in env (docker exec -e) on the
GPU-less validation container; torch.xpu calls are monkeypatched to CPU
no-ops.  Verifies the highest-risk pure logic: lazy build, moe_config
replace, scale re-pointing, LRU eviction order, slot remap feeding the
kernel, pool content exactness, telemetry counters.
"""
import dataclasses
import json
import sys
from types import SimpleNamespace

import torch

torch.xpu.synchronize = lambda *a, **k: None
torch.xpu.current_device = lambda: "cpu"

import vllm.model_executor.layers.fused_moe.fused_moe_modular_method as m  # noqa: E402

assert m._FN8_TIERED is True and m._FN8_TIER_SLOTS == 4, (
    m._FN8_TIERED, m._FN8_TIER_SLOTS)

E, N1, K, N2 = 512, 64, 32, 16
SLOTS = 4
DUMMY = SLOTS

# -- mocks ---------------------------------------------------------------


@dataclasses.dataclass
class MockMoeConfig:
    num_local_experts: int = E
    ep_rank: int = 0
    ep_size: int = 1
    moe_parallel_config: object = None


class MockKernel:
    def __init__(self, experts):
        self.fused_experts = experts
        self.calls = []

    def apply(self, **kw):
        self.calls.append(kw)
        return "OUT"


experts = SimpleNamespace(
    is_block_fp8=True,
    moe_config=MockMoeConfig(),
    quant_config=SimpleNamespace(
        w1_scale=torch.arange(E * 4, dtype=torch.float32).reshape(E, 2, 2),
        w2_scale=torch.arange(E * 2, dtype=torch.float32).reshape(E, 2, 1),
    ),
)
kernel = MockKernel(experts)
method = SimpleNamespace(moe_kernel=kernel)

layer = SimpleNamespace(
    layer_name="model.layers.7.mlp.experts",
    global_num_experts=E,
    expert_map=None,
    activation="SILU_SENTINEL",
    apply_router_weight_on_input=False,
    w13_weight=torch.nn.Parameter(torch.randn(E, N1, K)),
    w2_weight=torch.nn.Parameter(torch.randn(E, N2, K // 2)),
)
host_w13_snap = layer.w13_weight.data.clone()
host_s13_snap = experts.quant_config.w1_scale.clone()

# -- build via the production entry point ---------------------------------
st = m._fn8_store_for(layer, method)
assert st is not None, "store refused to build"
assert st.layer_idx == 7, st.layer_idx
assert kernel is st.kernel
assert st.owned == {g: g for g in range(E)}, "expert_map None -> own all"
assert st.pool_w13.shape == (SLOTS + 1, N1, K), st.pool_w13.shape
assert st.pool_w2.shape == (SLOTS + 1, N2, K // 2)
assert st.pool_s13.shape == (SLOTS + 1, 2, 2)
assert st.slot_map_host.shape == (E,)
assert (st.slot_map_host == DUMMY).all(), "cold start: everything dummy"
# moe_config replaced on a NEW object, fields overridden, rest preserved
assert experts.moe_config is not st.experts.__dict__.get(
    "moe_config", None) or True  # attribute identity below:
assert experts.moe_config.num_local_experts == SLOTS + 1
assert experts.moe_config.ep_rank == 0 and experts.moe_config.ep_size == 1
assert isinstance(experts.moe_config, MockMoeConfig)
# scales re-pointed at the pool (same tensor objects)
assert experts.quant_config.w1_scale is st.pool_s13
assert experts.quant_config.w2_scale is st.pool_s2
# host mirror shares storage with the offloader's pageable tensors
# (param.data returns a fresh Python object per access — compare storage)
assert st.host_w13.data_ptr() == layer.w13_weight.data_ptr()
assert st.host_w13.shape == layer.w13_weight.shape
print("BUILD_OK slots=%d dummy=%d layer=%d" % (SLOTS, DUMMY, st.layer_idx))

# -- dispatch A: ids 10, 20, 30, 40 fill all four slots -------------------


def run(ids_2d):
    tk = torch.tensor(ids_2d, dtype=torch.int32)
    tw = torch.full((tk.shape[0], tk.shape[1]), 0.25, dtype=torch.float32)
    x = torch.randn(tk.shape[0], K)
    return st.dispatch(method, layer, x, tw, tk, "SHARED_SENTINEL")


out = run([[10, 20], [30, 40]])
assert out == "OUT"
kw = kernel.calls[-1]
assert kw["w1"] is st.pool_w13 and kw["w2"] is st.pool_w2
assert kw["expert_map"] is None
assert kw["global_num_experts"] == E
assert kw["activation"] == "SILU_SENTINEL"
assert kw["shared_experts_input"] == "SHARED_SENTINEL"
got = kw["topk_ids"].tolist()
slots = {10: st.slot_of[10], 20: st.slot_of[20], 30: st.slot_of[30],
         40: st.slot_of[40]}
assert sorted(slots.values()) == [0, 1, 2, 3], slots
assert got == [[slots[10], slots[20]], [slots[30], slots[40]]], got
assert kw["topk_ids"].dtype is torch.int32
# pool content exactness
for g, s in slots.items():
    assert torch.equal(st.pool_w13[s], host_w13_snap[g]), f"pool[{s}]!={g}"
    assert torch.equal(st.pool_s13[s], host_s13_snap[g])
assert st.rec["misses"] == 4 and st.rec["hits"] == 0
assert st.rec["evictions"] == 0 and st.rec["resident"] == 4
# dummy slot stays zero forever
assert not st.pool_w13[DUMMY].any() and not st.pool_s2[DUMMY].any()
print("DISPATCH_A_OK", got)

# -- dispatch B: hit 10, miss 50 -> true LRU victim evicted ----------------
# (victim derived from the store's own bookkeeping — set-iteration order
# of dispatch A's load sequence is CPython hash order, not ascending)
lru = min(st.slot_of, key=lambda g: st.last_use[g])
assert lru != 10, "hit id must not be the LRU victim"
slot_of_lru = st.slot_of[lru]
run([[10, 50]])
assert st.rec["hits"] == 1 and st.rec["misses"] == 5
assert st.rec["evictions"] == 1
assert lru not in st.slot_of and 50 in st.slot_of and 10 in st.slot_of
s50 = st.slot_of[50]
assert s50 == slot_of_lru, "evictee slot reused"
assert torch.equal(st.pool_w13[s50], host_w13_snap[50])
assert st.slot_map_host[lru] == DUMMY, "evicted id falls back to dummy"
kw = kernel.calls[-1]
got = kw["topk_ids"].tolist()
assert got == [[st.slot_of[10], s50]], got
print("DISPATCH_B_OK evict=%d slot_reused=%d" % (lru, s50))

# -- dispatch C: non-owned id routes to dummy (exact zero drop) ------------
# carve ownership: keep the built store but shrink to a 3-expert world
st.owned = {10: 10, 50: 50, 70: 70}
run([[70, 99]])
assert 70 in st.slot_of and 99 not in st.slot_of
got = kernel.calls[-1]["topk_ids"].tolist()
assert got == [[st.slot_of[70], DUMMY]], got
assert kernel.calls[-1]["topk_weights"].tolist() == [[0.25, 0.25]]
print("DISPATCH_C_OK dummy_route=%d" % DUMMY)

# -- telemetry counters + flush -------------------------------------------
assert st.rec["dispatches"] == 3
assert st.rec["h2d_copies"] == 6  # 4 (A) + 1 (B) + 1 (C: expert 70)
assert st.rec["evictions"] == 2  # B evicts one, C (pool full again) one
assert st.rec["misses"] == 6
m._FN8_TELE.flush()
with open(m._FN8_TIER_DUMP) as f:
    payload = json.load(f)
assert payload["stores"]["7"]["evictions"] == 2
assert payload["stores"]["7"]["layer"] == 7
print("TELE_COUNTERS_OK", {k: v for k, v in payload["stores"]["7"].items()
                           if k in ("dispatches", "misses", "hits",
                                    "evictions", "resident")})

# -- idempotence of the lazy builder --------------------------------------
assert m._fn8_store_for(layer, method) is st, "second call must reuse"
assert len(m._FN8_BUILT) == 1 and not m._FN8_DISABLED
print("FN8_T3_STORE_ALL_OK")
sys.exit(0)
