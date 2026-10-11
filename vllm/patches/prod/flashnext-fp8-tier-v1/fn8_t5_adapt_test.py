#!/usr/bin/env python3
"""fn8 P5 CPU gate: patch_05 adaptive swaps + persistence (run INSIDE fn8-p2):

    docker exec -e FN8_TIERED=1 -e FN8_TIER_SLOTS=2 \
      -e V1X_TIER_ADAPT=1 -e V1X_TIER_ADAPT_EVERY=3 \
      -e V1X_TIER_ADAPT_SWAPS=1 -e V1X_TIER_ADAPT_DECAY=0.9 \
      -e V1X_TIER_ADAPT_MIN_SCORE=0.5 -e V1X_TIER_PERSIST=/tmp/fn8t5/hist.json \
      -e V1X_TIER_PERSIST_EVERY=1 -e V1X_TIER_PF_STREAM=0 \
      -e V1X_TIER_DUMP=/tmp/fn8t5/tele.json \
      fn8-p2 python3 /stage/fn8_t5_adapt_test.py

1. KNOBS   — every P5 env knob parsed at module level.
2. ADAPT   — 2-slot store, ids 1/2/3, window EVERY=3: the demand-evicted
             id 3 (EWMA 1.0 >= 0.5, non-resident) is swapped back in by
             _adapt_window via the blocking _pf_fill path; the window's
             LRU victim is derived from the store's own last_use ticks.
3. PERSIST — flush writes <PERSIST>.r0 (counts matrix, layer row 7 =
             touch tallies), touch_counts reset, rec counters bumped;
             telemetry flush parses and carries the new keys (per-rank
             tmp path, no shared .tmp).
4. MERGE   — seed merge law: base profile + every rank file, counts
             summed, hot order by summed counts (restart reproduces the
             adapted hot set); rank fallback law (0 in owned -> rank 0).
5. COMPILE — py_compile of the patched module.
"""
import dataclasses
import json
import os
import py_compile
import shutil
import sys
from types import SimpleNamespace

import torch

torch.xpu.synchronize = lambda *a, **k: None
torch.xpu.current_device = lambda: "cpu"

import vllm  # noqa: E402
import vllm.model_executor.layers.fused_moe.fused_moe_modular_method as m  # noqa: E402

# ---- 1. KNOBS ----------------------------------------------------------------
assert m._FN8_TIERED is True and m._FN8_TIER_SLOTS == 2
assert m._FN8_ADAPT is True, "V1X_TIER_ADAPT=1 not parsed"
assert m._FN8_ADAPT_EVERY == 3, m._FN8_ADAPT_EVERY
assert m._FN8_ADAPT_SWAPS == 1, m._FN8_ADAPT_SWAPS
assert abs(m._FN8_ADAPT_DECAY - 0.9) < 1e-9, m._FN8_ADAPT_DECAY
assert abs(m._FN8_ADAPT_MIN_SCORE - 0.5) < 1e-9, m._FN8_ADAPT_MIN_SCORE
assert m._FN8_PERSIST == "/tmp/fn8t5/hist.json", m._FN8_PERSIST
assert m._FN8_PERSIST_EVERY == 1, m._FN8_PERSIST_EVERY
print("KNOBS_OK adapt every=%d swaps=%d decay=%s min=%s persist=%s" % (
    m._FN8_ADAPT_EVERY, m._FN8_ADAPT_SWAPS, m._FN8_ADAPT_DECAY,
    m._FN8_ADAPT_MIN_SCORE, m._FN8_PERSIST))

shutil.rmtree("/tmp/fn8t5", ignore_errors=True)
os.makedirs("/tmp/fn8t5")

E, N1, K, N2 = 512, 64, 32, 16
SLOTS, DUMMY = 2, 2

# -- mocks (t3 harness) -------------------------------------------------------


@dataclasses.dataclass
class MockMoeConfig:
    # REAL FusedMoEConfig shape (t3 law): dataclass — the ctor calls
    # dataclasses.replace() on it.
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
        return "OUT"


class MockQuantDesc:
    def __init__(self, scale):
        self.scale = scale


class MockQuantConfig:
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
method = SimpleNamespace(moe_kernel=kernel)
layer = SimpleNamespace(
    layer_name="model.layers.7.mlp.experts",
    global_num_experts=E,
    expert_map=None,           # -> own all 512 -> rank fallback = 0
    activation="SILU_SENTINEL",
    apply_router_weight_on_input=False,
    w13_weight=torch.nn.Parameter(torch.randn(E, N1, K)),
    w2_weight=torch.nn.Parameter(torch.randn(E, N2, K // 2)),
)
host_w13_snap = layer.w13_weight.data.clone()

st = m._fn8_store_for(layer, method)
assert st is not None and st.layer_idx == 7
# P5 ctor state + rank derivation (no RANK env -> ownership fallback)
assert st.real_dispatches == 0 and st.live_scores == {} and st.touch_counts == {}
assert m._FN8_RANK == 0, m._FN8_RANK
for key in ("adapt_windows", "adapt_swaps", "persist_flushes"):
    assert key in st.rec, key
print("BUILD_OK rank=%d p5_state_init" % m._FN8_RANK)


def run(ids_2d):
    tk = torch.tensor(ids_2d, dtype=torch.int32)
    tw = torch.full((tk.shape[0], tk.shape[1]), 0.25, dtype=torch.float32)
    x = torch.randn(tk.shape[0], K)
    return st.dispatch(method, layer, x, tw, tk, "SHARED_SENTINEL")


# ---- 2. ADAPT: windowed swap of the demand-evicted hot id --------------------
# d1: 1,2 resident (ticks: 1<2).  d2: demand 3 evicts LRU=1.  d3: demand
# {1,2} -> 2 hits, 1 reloads evicting 3 (tick 5 < 2's fresh 6); id 3 now
# NON-RESIDENT with EWMA 1.0 >= 0.5 -> window 3 fires the adapt swap.
run([[1, 2]])
assert st.slot_of.keys() == {1, 2}, st.slot_of
run([[3, 3]])
assert st.slot_of.keys() == {2, 3}, st.slot_of
assert 1 not in st.last_use
assert abs(st.live_scores[3] - 1.0) < 1e-9, st.live_scores[3]
# d3 ends with the window FIRING INSIDE ensure_resident: the swap has
# already landed by the time run() returns -> resident {1, 3} (1 kept
# slot 0, 3 displaced 2 out of slot 1).
run([[1, 2]])
assert st.real_dispatches == 3
assert st.rec["hits"] == 1 and st.rec["misses"] == 4
assert st.slot_of.keys() == {1, 3}, st.slot_of

# window fired inside that third ensure_resident
assert st.rec["adapt_windows"] == 1, st.rec["adapt_windows"]
assert st.rec["adapt_swaps"] == 1, st.rec["adapt_swaps"]
assert 3 in st.slot_of, "adapt swap must re-resident the hot evicted id"
assert 2 not in st.slot_of, "swap victim = the window's LRU resident"
assert torch.equal(st.pool_w13[st.slot_of[3]], host_w13_snap[3])
assert st.rec["evictions"] == 3, st.rec["evictions"]  # d2, d3, swap
assert st.rec["pf_fills"] == 1, st.rec["pf_fills"]
# routed ids got the demand signal; non-routed ids never scored
assert abs(st.live_scores[1] - 1.9) < 1e-9, st.live_scores[1]  # 1.0*.9+1
assert abs(st.live_scores[2] - 1.9) < 1e-9, st.live_scores[2]
assert 4 not in st.live_scores
print("ADAPT_OK window=1 swap=1 resident=%s scores={1:%.2f,2:%.2f,3:%.2f}"
      % (sorted(st.slot_of), st.live_scores[1], st.live_scores[2],
         st.live_scores[3]))

# ---- 3. PERSIST: rank file + telemetry ---------------------------------------
assert st.rec["persist_flushes"] == 1, st.rec["persist_flushes"]
assert st.touch_counts == {}, "flush must consume the raw tallies"
with open("/tmp/fn8t5/hist.json.r0") as f:
    hist = json.load(f)
counts = hist["counts"]
assert len(counts) == 8 and len(counts[7]) == E, (len(counts), len(counts[7]))
row = counts[7]
assert row[1] == 2 and row[2] == 2 and row[3] == 1, row[:4]
assert sum(row) == 5 and all(c == 0 for c in row[4:])
assert all(sum(r) == 0 for r in counts[:7]), "single-layer store"

m._FN8_TELE.flush()
with open("/tmp/fn8t5/tele.json") as f:
    tele = json.load(f)
t7 = tele["stores"]["7"]
assert t7["adapt_windows"] == 1 and t7["adapt_swaps"] == 1
assert t7["persist_flushes"] == 1
print("PERSIST_OK r0 row7=[0,2,2,1,0...] tele_keys_ok")

# ---- 4. MERGE: restart reproduces the adapted hot set ------------------------
# unit A: base + foreign rank file, counts summed
os.makedirs("/tmp/fn8t5/m1")
with open("/tmp/fn8t5/m1/base.json", "w") as f:
    json.dump({"counts": [[0, 3, 0, 7]]}, f)
with open("/tmp/fn8t5/m1/hist2.json.r1", "w") as f:
    json.dump({"counts": [[10, 0, 2, 0]]}, f)
saved = m._FN8_PERSIST
m._FN8_PERSIST = "/tmp/fn8t5/m1/hist2.json"
merged = m._fn8_seed_hot_lists_merged("/tmp/fn8t5/m1/base.json")
assert merged == {0: [0, 3, 1, 2]}, merged  # sums 10,7,3,2 desc
# unit B: persisted touches reorder the boot hot set (restart law)
os.makedirs("/tmp/fn8t5/m2")
with open("/tmp/fn8t5/m2/base.json", "w") as f:
    json.dump({"counts": [[5, 0, 0, 0]]}, f)
with open("/tmp/fn8t5/m2/hist3.json.r0", "w") as f:
    json.dump({"counts": [[0, 1, 2, 0]]}, f)
m._FN8_PERSIST = "/tmp/fn8t5/m2/hist3.json"
merged2 = m._fn8_seed_hot_lists_merged("/tmp/fn8t5/m2/base.json")
assert merged2 == {0: [0, 2, 1]}, merged2  # 5,2,1
m._FN8_PERSIST = saved
# rank fallback law (RANK/LOCAL_RANK absent)
for var in ("RANK", "LOCAL_RANK", "VLLM_RANK_ID"):
    os.environ.pop(var, None)
assert m._fn8_persist_rank({0: 0, 1: 1}) == 0
assert m._fn8_persist_rank({256: 0}) == 1
os.environ["RANK"] = "1"
assert m._fn8_persist_rank({0: 0}) == 1  # env wins over ownership
os.environ.pop("RANK", None)
print("MERGE_OK base+rank summing, restart reorder, rank fallback")

# ---- 5. COMPILE ---------------------------------------------------------------
py_compile.compile(os.path.join(
    os.path.dirname(vllm.__file__),
    "model_executor", "layers", "fused_moe", "fused_moe_modular_method.py"),
    doraise=True)
print("FN8_T5_ADAPT_ALL_OK")
sys.exit(0)
