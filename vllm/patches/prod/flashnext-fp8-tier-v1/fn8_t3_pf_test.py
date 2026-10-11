#!/usr/bin/env python3
"""fn8-t3 validation part 6 (P3): patch_03 layer-ahead prefetch, on CPU.

Run BOTH ways (docker exec -e ... fn8-p2 python3 /stage/fn8_t3_pf_test.py):
  1. -e FN8_TIERED=1 -e FN8_TIER_SLOTS=6 \
     -e V1X_TIER_DUMP=/tmp/fn8_t3_pf_dump.json \
     -e V1X_TIER_PREFETCH=1 -e V1X_TIER_PF_LAYERS=4 \
     -e V1X_TIER_PF_MIN_ROWS=1
     -> full PF suite (side-stream fakes monkeypatched in; MIN_ROWS=1 so
     the tiny synthetic dispatches stage — the decode gate has its own
     scenario).
  2. -e FN8_TIERED=1 -e FN8_TIER_SLOTS=6 (NO V1X_TIER_PREFETCH)
     -> OFF-mode suite: default leaves every _pf_ dormant.

Verifies the highest-risk pure logic: replay staging + chunk-epoch tagging
(attn_metadata identity), layer-ahead fill from the SOURCE layer's fresh
replay, saturated-pool LRU displacement by prefetch (boot-8 law),
slot_map_host mirror staying dummy for un-promoted fills, ensure-side
promotion (needed -> event.synchronize + pf_waits; completed-unneeded ->
opportunistic), the authoritative miss path evicting residents but never
touching filling slots, budget cap, loud per-store disable, and the
pf_drops oldest-fill eviction when the pool is entirely in flight.
"""
import contextlib
import dataclasses
import json
import sys
from types import SimpleNamespace

import torch

# -- XPU fakes (CPU container): side stream + events are bookkeeping only --


class FakeEvent:
    def __init__(self):
        self._q = True     # opportunistically "complete" by default
        self.syncs = 0

    def record(self, stream=None):
        pass

    def query(self):
        return self._q

    def synchronize(self):
        self.syncs += 1


class FakeStream:
    def wait_event(self, ev):
        pass


SYNCS = [0]
torch.xpu.synchronize = lambda *a, **k: SYNCS.__setitem__(0, SYNCS[0] + 1)
torch.xpu.current_device = lambda: "cpu"
torch.xpu.Event = FakeEvent
torch.xpu.Stream = FakeStream
torch.xpu.stream = lambda s: contextlib.nullcontext()

import vllm.model_executor.layers.fused_moe.fused_moe_modular_method as m  # noqa: E402
import vllm.forward_context as fc  # noqa: E402

assert m._FN8_TIERED is True and m._FN8_TIER_SLOTS == 6

# forward_context tap: dispatch reads .attn_metadata off this holder
_MD = SimpleNamespace(attn_metadata=None)
fc.get_forward_context = lambda: _MD
if hasattr(m, "get_forward_context"):
    m.get_forward_context = lambda: _MD


# -- mocks (same REAL-shape laws as fn8_t3_store_test part 2) --------------


@dataclasses.dataclass
class MockMoeConfig:
    num_experts: int
    num_local_experts: int
    num_logical_experts: int
    moe_parallel_config: object = None


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


class MockKernel:
    def __init__(self, experts):
        self.fused_experts = experts
        self.calls = []

    def apply(self, **kw):
        self.calls.append(kw)
        return "OUT"


E, N1, K, N2 = 512, 64, 32, 16
SLOTS = 6
DUMMY = SLOTS


def make_world(layer_idx):
    experts = SimpleNamespace(
        is_block_fp8=True,
        moe_config=MockMoeConfig(E, E, E),
        quant_config=MockQuantConfig(
            torch.arange(E * 4, dtype=torch.float32).reshape(E, 2, 2),
            torch.arange(E * 2, dtype=torch.float32).reshape(E, 2, 1),
        ),
    )
    kernel = MockKernel(experts)
    method = SimpleNamespace(moe_kernel=kernel)
    layer = SimpleNamespace(
        layer_name=f"model.layers.{layer_idx}.mlp.experts",
        global_num_experts=E,
        expert_map=None,
        activation="SILU_SENTINEL",
        apply_router_weight_on_input=False,
        w13_weight=torch.nn.Parameter(torch.randn(E, N1, K)),
        w2_weight=torch.nn.Parameter(torch.randn(E, N2, K // 2)),
    )
    snap_w13 = layer.w13_weight.data.clone()
    st = m._fn8_store_for(layer, method)
    assert st is not None and st.layer_idx == layer_idx
    return st, kernel, snap_w13, method, layer


def dispatch(world, ids_2d):
    st, _k, _snap, method, layer = world
    tk = torch.tensor(ids_2d, dtype=torch.int32)
    tw = torch.full((tk.shape[0], tk.shape[1]), 0.25, dtype=torch.float32)
    x = torch.randn(tk.shape[0], K)
    return st.dispatch(method, layer, x, tw, tk, "SHARED")


def slot_ids(st, ids):
    return [st.slot_of[i] for i in ids]


# ===========================================================================
if not m._FN8_PF:
    # ---- OFF-mode: default leaves every _pf_ dormant ----------------------
    w = make_world(7)
    st, kernel = w[0], w[1]
    md = object()
    _MD.attn_metadata = md
    dispatch(w, [[10, 20]])
    assert st.replay_dev is None
    assert st.rec["pf_fills"] == 0 and st.rec["pf_promotes"] == 0
    assert kernel.calls[-1]["topk_ids"].tolist() == [slot_ids(st, [10, 20])]
    print("PF_OFF_OK dispatch clean, no staging, pf counters zero")
    print("FN8_T3_PF_OFF_ALL_OK")
    sys.exit(0)

assert m._FN8_PF_LAYERS == 4 and m._FN8_PF_STREAM
assert m._FN8_PF_MIN_ROWS == 1
assert m._FN8_PF_BATCH is True

w7 = make_world(7)
st7, k7, snap7 = w7[0], w7[1], w7[2]
w11 = make_world(11)
st11, k11, snap11 = w11[0], w11[1], w11[2]
assert m._FN8_EPOCH == 0  # dummy build dispatches carry attn_metadata=None

# -- epoch 1: seed L11 with 2 residents (4 slots free) ----------------------
md1 = object()
_MD.attn_metadata = md1
dispatch(w11, [[1, 2]])
assert m._FN8_EPOCH == 1
assert sorted(st11.slot_of) == [1, 2]
# L11's own tail targets layer 15 (not built) -> no-op
assert st11.rec["pf_fills"] == 0
# same md object again -> no epoch bump
dispatch(w11, [[1, 2]])
assert m._FN8_EPOCH == 1
print("EPOCH_OK bumped on identity change, stable on repeat")

# -- epoch 2: L7 dispatch stages replay + tail fills L11 async --------------
md2 = object()
_MD.attn_metadata = md2
dispatch(w7, [[9, 10], [11, 12]])
assert m._FN8_EPOCH == 2
# L7's own misses loaded into ITS pool (4 of 6 slots)
assert st7.rec["misses"] == 4 and sorted(st7.slot_of) == [9, 10, 11, 12]
# staging wrote the fresh replay, epoch-tagged
assert st7.replay_epoch == 2 and st7.replay_n == 2
assert st7.replay_host[:2].tolist() == [[9, 10], [11, 12]]
# layer-ahead: L7+4 == L11 got all four fills in its 4 FREE slots
assert st11.rec["pf_fills"] == 4 and len(st11.filling) == 4
assert len(st11.filling_slots) == 4
assert st11.filling_slots.isdisjoint({st11.slot_of[1], st11.slot_of[2]})
# mirror law: un-promoted fills stay dummy in the host map
for g in (9, 10, 11, 12):
    assert st11.slot_map_host[g] == DUMMY, "fill must not claim residency"
    assert g not in st11.slot_of
# pool content already streamed (CPU fakes execute eagerly)
for g, (s, _ev) in st11.filling.items():
    assert torch.equal(st11.pool_w13[s], snap11[g]), f"fill slot {s}!={g}"
print("TAIL_FILL_OK 4 async fills, mirror dormant, slots disjoint")

# -- epoch 3: L11 needs 9,10 (wait path) + 11,12 complete (opportunistic) ---
for g, (_s, ev) in st11.filling.items():
    ev._q = g not in (9, 10)   # 9,10 incomplete -> must synchronize
md3 = object()
_MD.attn_metadata = md3
dispatch(w11, [[9, 10]])
assert m._FN8_EPOCH == 3
assert st11.rec["pf_promotes"] == 4 and st11.rec["pf_waits"] == 2
assert st11.rec["pf_fills"] == 4 and not st11.filling
assert not st11.filling_slots
for g in (9, 10, 11, 12):
    assert g in st11.slot_of
    assert st11.slot_map_host[g] == st11.slot_of[g], "promotion updates map"
got = k11.calls[-1]["topk_ids"].tolist()
assert got == [slot_ids(st11, [9, 10])], got
assert DUMMY not in got[0], "needed ids must NOT route to dummy"
print("PROMOTE_OK 2 waits + 2 opportunistic, gather uses real slots")

# -- epoch 4: authoritative miss evicts a resident, never a filling slot ----
md4 = object()
_MD.attn_metadata = md4
dispatch(w11, [[20, 9]])
assert m._FN8_EPOCH == 4
assert 20 in st11.slot_of and 9 in st11.slot_of
victim = next(g for g in (1, 2) if g not in st11.slot_of)
assert st11.slot_map_host[victim] == DUMMY
assert st11.rec["evictions"] == 1
got = k11.calls[-1]["topk_ids"].tolist()
assert got == [slot_ids(st11, [20, 9])], got
print("MISS_EVICT_OK victim=%d slot reused for 20" % victim)

# -- saturated pool -> tail DISPLACES the LRU residents (boot-8 law) --------
# L11 is full (6 residents after md4); the md5 tail must evict the LRU
# pair and async-fill 13,14 into their slots.
res_before = set(st11.slot_of)
md5 = object()
_MD.attn_metadata = md5
dispatch(w7, [[13, 14]])          # L7 replay fresh again
assert st7.replay_epoch == 5
assert st11.rec["pf_fills"] == 6 and set(st11.filling) == {13, 14}
assert st11.rec["evictions"] == 3  # md4 miss + 2 prefetch displacements
victims = res_before - set(st11.slot_of)
assert len(victims) == 2 and victims.isdisjoint({13, 14})
for v in victims:
    assert st11.slot_map_host[v] == DUMMY, "displaced id falls to dummy"
assert st11.filling_slots.isdisjoint(
    {s for g, s in ((g, st11.slot_of[g]) for g in st11.slot_of)})
print("SATURATED_EVICT_FILL_OK victims=%s" % sorted(victims))

# -- budget cap: 0 budget blocks fills even with capacity -------------------
m._FN8_PF_BUDGET = 0
md6 = object()
_MD.attn_metadata = md6
dispatch(w7, [[15, 16]])
assert st11.rec["pf_fills"] == 6, "budget 0 must block all fills"
m._FN8_PF_BUDGET = 8
print("BUDGET_OK")

# -- loud disable: a _pf_fill exception kills prefetch for THAT store only --
def _boom(g):
    raise RuntimeError("synthetic fill failure")

st11._pf_fill = _boom
md7 = object()
_MD.attn_metadata = md7
dispatch(w7, [[17, 18]])          # tail raises inside -> caught
assert st7._pf_broken is True, "tail exception must disable prefetch loudly"
assert st11._pf_broken is False
md8 = object()
_MD.attn_metadata = md8
out = dispatch(w7, [[19]])        # compute continues, no staging/tail
assert out == "OUT" and st7.replay_epoch == 7  # stage() early-outs (broken)
print("LOUD_DISABLE_OK compute unaffected")

# -- pf_drops: pool entirely in-flight -> oldest fill dropped with a sync ----
st15, _k15, _s15, _m15, _l15 = make_world(15)
for i in range(SLOTS):
    st15.filling[100 + i] = (i, FakeEvent())
    st15.filling_slots.add(i)
assert not st15.slot_of
st15._load_slot(200)
assert st15.rec.get("pf_drops", 0) == 1
assert 100 not in st15.filling and 0 not in st15.filling_slots
assert st15.slot_map_host[100] == DUMMY
assert 200 in st15.slot_of and st15.slot_of[200] == 0
assert len(st15.filling) == SLOTS - 1
print("PF_DROPS_OK oldest fill dropped+synchronized, slot reused")

# -- v4 batched demand loads: ONE device sync for the whole miss set -------
w19 = make_world(19)
st19, k19, snap19 = w19[0], w19[1], w19[2]
mdX = object()
_MD.attn_metadata = mdX                # epoch 9
SYNCS[0] = 0
dispatch(w19, [[31, 32, 33, 34, 35, 36, 37, 38]])  # 8 missing, pool 6
assert SYNCS[0] == 1, "one device sync for the whole batch"
assert st19.rec["misses"] == 8 and st19.rec["h2d_copies"] == 6
assert len(st19.slot_of) == 6, "oversize set: excess routed to dummy"
routed = k19.calls[-1]["topk_ids"].tolist()[0]
assert routed.count(DUMMY) == 2, "exactly the overflow ids hit dummy"
assert DUMMY not in [st19.slot_map_host[g] for g in st19.slot_of]
for g in st19.slot_of:
    assert torch.equal(
        st19.pool_w13[st19.slot_of[g]], snap19[g]), f"batch slot {g} wrong"
print("BATCH_LOAD_OK 8-miss batch: 1 sync, 6 resident, 2 dummy")

mdY = object()
_MD.attn_metadata = mdY                # epoch 10
SYNCS[0] = 0
dispatch(w19, [[39, 40]])              # pool full: in-batch LRU eviction
assert SYNCS[0] == 1
assert st19.rec["evictions"] == 2 and len(st19.slot_of) == 6
assert 39 in st19.slot_of and 40 in st19.slot_of
got = k19.calls[-1]["topk_ids"].tolist()
assert got == [slot_ids(st19, [39, 40])] and DUMMY not in got[0]
print("BATCH_EVICT_OK 2 more via one batch, gather routes real slots")

# -- telemetry carries the pf counters ---------------------------------------
m._FN8_TELE.flush()
with open(m._FN8_TIER_DUMP) as f:
    payload = json.load(f)
r11 = payload["stores"]["11"]
for key in ("pf_fills", "pf_promotes", "pf_waits", "pf_drops"):
    assert key in r11, key
assert r11["pf_fills"] == 6 and r11["pf_promotes"] == 4
print("TELE_PF_OK", {k: r11[k] for k in
                     ("pf_fills", "pf_promotes", "pf_waits", "pf_drops")})

# -- decode gate: a dispatch with n < MIN_ROWS stages nothing (boot-9 law) --
m._FN8_PF_MIN_ROWS = 8
md_ds = object()
_MD.attn_metadata = md_ds
dispatch(w11, [[30, 31]])            # 2 rows < 8 -> skip staging + tail
assert m._FN8_EPOCH == 11 and st11.replay_epoch == 4, "no staging below MIN_ROWS"
assert st11.rec["pf_fills"] == 6, "dormant tail must not fill"
m._FN8_PF_MIN_ROWS = 1
print("DECODE_SKIP_OK small dispatch: no staging, no fills, tail dormant")

print("FN8_T3_PF_ALL_OK")
sys.exit(0)
