#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 patch_02a — TieredExpertsStore (P2 core).

Port of the Strata resident-set ideology to this fork's modular MoE path,
specialized for FlashNext-FP8 (XPUExpertsFp8Block, 128x128 block FP8):

- Host mirror: the UVA offloader's pageable expert tensors (patch_02b
  suppresses the per-step wholesale-copy closure for FusedMoE modules;
  the whitelisted params still page at load, budget accounting intact).
- Device slot pool: S slots/layer + 1 persistent ZERO-WEIGHT dummy slot.
  Seeded from the P1 routing profile (per-layer hot lists, rank-local).
- Runtime: apply() is interposed at FusedMoEModularMethod.apply (the
  single choke point — under --enable-expert-parallel every monolithic
  ESIMD fast path self-disables, so the modular path carries prefill AND
  decode).  Per dispatch: sync D2H of topk_ids (tiny), host-side
  ensure-resident (blocking H2D of misses into LRU slots), device remap
  global->slot, then the UNMODIFIED kernel stack runs on pool tensors:
  w1.shape[0] drives local_num_experts downstream, and with the experts
  object's moe_config copy set to (num_local=S+1, ep_rank=0, ep_size=1)
  _xpu_block_moe_forward passes expert_map=None and treats slot ids as
  direct weight-row indices.  Zero kernel / xpu_moe.py edits.
- Non-owned and not-resident ids map to the dummy zero slot: fp8 zeros
  dequant to 0, so the gathered contribution is topk_weight * 0 = 0 —
  mathematically exact drop, identical partial-sum + TP-allreduce
  contract as expert_map=-1 dropping under EP.
- Dummy run (determine_available_memory, attn_metadata is None — the
  fork's PLE/GDN idiom): every id routed to the dummy slot.  No misses,
  no copies, VRAM-safe, and the pool is resident during measurement so
  KV sizing accounts for it.  The store builds lazily at first apply,
  which is exactly that call — pre-warm rides the dummy run per layer.
- Telemetry (V1X_TIER_DUMP, patch_01 atomic-JSON law): per-layer
  dispatches, dummy share, D2H/ensure/H2D ms, copies+bytes, hits/misses/
  evictions, resident count; flushed every V1X_TIER_FLUSH + atexit;
  telemetry failures can never take down the engine (compute-path
  errors propagate — a broken store must be loud, not silently wrong).

Env: FN8_TIERED=1 master, FN8_TIER_SEED=<routing_hist.json>,
FN8_TIER_SLOTS=<per-layer slots, default 96>, V1X_TIER_DUMP=<file>,
V1X_TIER_FLUSH=<n>.

Idempotent (FN8_02A marker), backs up .q4dbak, --restore undoes.
Companion: patch_02b (uva.py no-wrap flag + offload whitelist).
"""
import sys

TARGET = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
          "layers/fused_moe/fused_moe_modular_method.py")
MARKER = "FN8_02A"

ANCHOR_LOGGER = "logger = init_logger(__name__)\n"

STORE_BLOCK = '''logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# FN8_02A (flashnext-fp8-tier-v1 P2): TieredExpertsStore.
# See patch_02a docstring in the repo (patches/prod/flashnext-fp8-tier-v1)
# for the full design law.  Env-gated; inert unless FN8_TIERED=1.
# ---------------------------------------------------------------------------
import atexit
import dataclasses
import json as _fn8_json
import os as _fn8_os
import time as _fn8_time

_FN8_TIERED = _fn8_os.environ.get("FN8_TIERED", "") == "1"
_FN8_TIER_SEED = _fn8_os.environ.get("FN8_TIER_SEED", "")
_FN8_TIER_SLOTS = int(_fn8_os.environ.get("FN8_TIER_SLOTS", "96"))
_FN8_TIER_DUMP = _fn8_os.environ.get("V1X_TIER_DUMP", "")
_FN8_TIER_FLUSH = int(_fn8_os.environ.get("V1X_TIER_FLUSH", "200"))


class _Fn8TierTelemetry:
    """Per-process store counters (patch_01 atomic-JSON flush law)."""

    def __init__(self) -> None:
        self.stores: dict[int, dict] = {}

    def register(self, layer_idx: int, rec: dict) -> None:
        self.stores[layer_idx] = rec

    def flush(self) -> None:
        if not _FN8_TIER_DUMP or not self.stores:
            return
        payload = {
            "v": 1,
            "ts": _fn8_time.time(),
            "slots": _FN8_TIER_SLOTS,
            "seed": _FN8_TIER_SEED,
            "stores": {str(k): v for k, v in self.stores.items()},
        }
        tmp = _FN8_TIER_DUMP + ".tmp"
        with open(tmp, "w") as f:
            _fn8_json.dump(payload, f)
        _fn8_os.replace(tmp, _FN8_TIER_DUMP)


_FN8_TELE = _Fn8TierTelemetry() if _FN8_TIER_DUMP else None
if _FN8_TELE is not None:
    atexit.register(_FN8_TELE.flush)


def _fn8_seed_hot_lists(path: str) -> dict[int, list[int]]:
    """routing_hist.json -> {layer_idx: [expert ids by descending heat]}."""
    try:
        with open(path) as f:
            hist = _fn8_json.load(f)
        out = {}
        for li, row in enumerate(hist["counts"]):
            order = sorted(range(len(row)), key=lambda e: -row[e])
            out[li] = [e for e in order if row[e] > 0]
        return out
    except Exception:
        return {}


class _Fn8TieredStore:
    """Device slot pool + host mirror for ONE FusedMoE layer's experts."""

    def __init__(self, layer, method) -> None:
        self.layer = layer
        self.method = method
        kernel = method.moe_kernel
        self.kernel = kernel
        self.experts = kernel.fused_experts
        self.global_num = layer.global_num_experts

        # Host mirror = the offloader's pageable tensors (post-load,
        # post-process_weights_after_loading layout, exactly what the
        # unpatched path would have consumed via the per-step closure).
        self.host_w13 = layer.w13_weight.data
        self.host_w2 = layer.w2_weight.data
        qc = self.experts.quant_config
        self.host_s13 = qc.w1_scale
        self.host_s2 = qc.w2_scale

        local_num, hidden_dim = self.host_w13.shape[1], self.host_w13.shape[2]
        n2, k2 = self.host_w2.shape[1], self.host_w2.shape[2]
        self.pool_slots = _FN8_TIER_SLOTS
        self.dummy = self.pool_slots  # last slot = persistent zero slot
        dev = self.host_w13.device if self.host_w13.is_xpu else \\
            __import__("torch").xpu.current_device()

        # Pool geometry mirrors the host mirror row-for-row.
        import torch as _t
        self.pool_w13 = _t.zeros(
            self.pool_slots + 1, local_num, hidden_dim,
            dtype=self.host_w13.dtype, device=dev)
        self.pool_w2 = _t.zeros(
            self.pool_slots + 1, n2, k2,
            dtype=self.host_w2.dtype, device=dev)
        self.pool_s13 = _t.zeros(
            self.pool_slots + 1, *self.host_s13.shape[1:],
            dtype=self.host_s13.dtype, device=dev)
        self.pool_s2 = _t.zeros(
            self.pool_slots + 1, *self.host_s2.shape[1:],
            dtype=self.host_s2.dtype, device=dev)

        # Ownership from the authoritative expert_map (-1 = other ranks).
        em = layer.expert_map
        if em is not None:
            em = em.detach().to("cpu")
            self.owned = {int(g): int(l) for g, l in enumerate(em.tolist())
                          if l >= 0}
        else:
            self.owned = {g: g for g in range(self.global_num)}

        # Slot bookkeeping: slot -> global, global -> (slot, last_use).
        self.slot_of: dict[int, int] = {}
        self.global_of: dict[int, int] = {}
        self.use_tick = 0
        self.last_use: dict[int, int] = {}
        self.slot_map_host = _t.full(
            (self.global_num,), self.dummy, dtype=_t.int32)
        self.slot_map_dev = self.slot_map_host.to(dev)

        # Kernel-stack view: pool geometry + no expert_map.  A COPY of
        # the experts' moe_config (never mutate the shared layer/runner
        # config); scales re-pointed at the pool on the quant_config
        # (block branch reads them live off the experts object).
        self.experts.moe_config = dataclasses.replace(
            self.experts.moe_config,
            num_local_experts=self.pool_slots + 1, ep_rank=0, ep_size=1)
        qc.w1_scale = self.pool_s13
        qc.w2_scale = self.pool_s2

        # Telemetry record.
        self.layer_idx = -1
        try:
            from vllm.model_executor.models.utils import (
                extract_layer_index)
            self.layer_idx = extract_layer_index(layer.layer_name)
        except Exception:
            pass
        self.rec = {
            "layer": self.layer_idx,
            "dispatches": 0, "dummy_dispatches": 0,
            "hits": 0, "misses": 0, "evictions": 0,
            "h2d_copies": 0, "h2d_bytes": 0,
            "d2h_ms": 0.0, "ensure_ms": 0.0, "h2d_ms": 0.0,
            "resident": 0, "pool_slots": self.pool_slots,
            "local_num": int(local_num), "hidden": int(hidden_dim),
        }
        if _FN8_TELE is not None:
            _FN8_TELE.register(self.layer_idx, self.rec)

        # Seed pre-warm (rides the dummy run: this ctor runs at the
        # layer's first apply, which is determine_available_memory).
        seed = _fn8_seed_hot_lists(_FN8_TIER_SEED) if _FN8_TIER_SEED else {}
        hot = [g for g in seed.get(self.layer_idx, [])
               if g in self.owned][: self.pool_slots]
        t0 = _fn8_time.time()
        for g in hot:
            self._load_slot(g)
        if hot:
            self._sync_slot_map()
            logger.info(
                "FN8_02A store layer %s: %s/%s slots pre-warmed from seed "
                "in %.1fs (pool %.2f GiB)",
                self.layer_idx, len(hot), self.pool_slots,
                _fn8_time.time() - t0,
                (self.pool_w13.numel() + self.pool_w2.numel()) / 2**30)
        else:
            logger.info(
                "FN8_02A store layer %s: cold start (no seed), "
                "pool %.2f GiB, %s owned experts",
                self.layer_idx,
                (self.pool_w13.numel() + self.pool_w2.numel()) / 2**30,
                len(self.owned))

    # -- slot machinery ----------------------------------------------------

    def _load_slot(self, global_id: int) -> None:
        """Blocking H2D of one expert into a slot (evicting LRU if full)."""
        import torch as _t
        self.use_tick += 1
        if global_id in self.slot_of:
            self.last_use[global_id] = self.use_tick
            return
        if len(self.slot_of) >= self.pool_slots:
            victim = min(
                (g for g in self.slot_of),
                key=lambda g: self.last_use.get(g, 0), default=None)
            if victim is not None:
                slot = self.slot_of.pop(victim)
                del self.global_of[slot]
                del self.last_use[victim]
                # keep slot_map_host a trustworthy mirror (P3 prefetch
                # reads it outside ensure_resident)
                self.slot_map_host[victim] = self.dummy
                self.rec["evictions"] += 1
        # first free slot, else reuse evicted
        slot = next(
            (s for s in range(self.pool_slots) if s not in self.global_of),
            None)
        if slot is None:
            return  # unreachable when eviction succeeded
        local = self.owned[global_id]
        t0 = _fn8_time.time()
        self.pool_w13[slot].copy_(self.host_w13[local])
        self.pool_w2[slot].copy_(self.host_w2[local])
        self.pool_s13[slot].copy_(self.host_s13[local])
        self.pool_s2[slot].copy_(self.host_s2[local])
        _t.xpu.synchronize()
        dt = _fn8_time.time() - t0
        self.slot_of[global_id] = slot
        self.global_of[slot] = global_id
        self.last_use[global_id] = self.use_tick
        self.rec["h2d_copies"] += 1
        self.rec["h2d_ms"] += dt
        self.rec["h2d_bytes"] += (
            self.host_w13[local].numel() * self.host_w13.element_size()
            + self.host_w2[local].numel() * self.host_w2.element_size())
        self.slot_map_host[global_id] = slot

    def _sync_slot_map(self) -> None:
        self.slot_map_dev.copy_(self.slot_map_host)
        self.rec["resident"] = len(self.slot_of)

    def ensure_resident(self, ids_host) -> None:
        """All needed OWNED ids resident before compute (correctness)."""
        t0 = _fn8_time.time()
        self.use_tick += 1
        owned = self.owned
        needed = set()
        for row in ids_host:
            for g in row:
                if g in owned:
                    needed.add(g)
        changed = False
        for g in needed:
            if g in self.slot_of:
                self.last_use[g] = self.use_tick
                self.rec["hits"] += 1
            else:
                self.rec["misses"] += 1
                self._load_slot(g)
                changed = True
        if changed:
            self._sync_slot_map()
        self.rec["ensure_ms"] += _fn8_time.time() - t0

    # -- dispatch ----------------------------------------------------------

    def dispatch(self, method, layer, x, topk_weights, topk_ids,
                 shared_experts_input):
        import torch as _t
        self.rec["dispatches"] += 1
        if _FN8_TELE is not None and (
            self.rec["dispatches"] % _FN8_TIER_FLUSH == 0):
            try:
                _FN8_TELE.flush()
            except Exception:
                pass

        try:
            from vllm.forward_context import get_forward_context
            attn_metadata = get_forward_context().attn_metadata
        except Exception:
            attn_metadata = "unknown"
        if attn_metadata is None:
            # Boot dummy run (fork idiom): all-zero weights, no misses,
            # pool resident for the memory measurement.
            self.rec["dummy_dispatches"] += 1
            slot_ids = _t.full_like(topk_ids, self.dummy)
        else:
            t0 = _fn8_time.time()
            ids_host = topk_ids.detach().to("cpu").tolist()
            self.rec["d2h_ms"] += _fn8_time.time() - t0
            self.ensure_resident(ids_host)
            slot_ids = self.slot_map_dev[topk_ids.detach().long()].to(
                topk_ids.dtype)

        return self.kernel.apply(
            hidden_states=x,
            w1=self.pool_w13,
            w2=self.pool_w2,
            topk_weights=topk_weights,
            topk_ids=slot_ids,
            activation=layer.activation,
            global_num_experts=self.global_num,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=None,
            shared_experts_input=shared_experts_input,
        )


_FN8_BUILT: set[int] = set()
_FN8_DISABLED: set[int] = set()


def _fn8_store_for(layer, method):
    """Lazy per-layer build at first apply (== the boot dummy run)."""
    if not _FN8_TIERED:
        return None
    lid = id(layer)
    if lid in _FN8_DISABLED:
        return None
    if lid not in _FN8_BUILT:
        experts = getattr(method.moe_kernel, "fused_experts", None)
        if not getattr(experts, "is_block_fp8", False):
            _FN8_DISABLED.add(lid)
            return None
        try:
            layer._fn8_store = _Fn8TieredStore(layer, method)
        except Exception as e:
            logger.exception("FN8_02A store build failed (layer %s): %s",
                             getattr(layer, "layer_name", "?"), e)
            _FN8_DISABLED.add(lid)
            return None
        _FN8_BUILT.add(lid)
    return getattr(layer, "_fn8_store", None)

'''

ANCHOR_APPLY_HEAD = '''    def apply(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
'''

NEW_APPLY_HEAD = '''    def apply(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # FN8_02A: tiered-experts dispatch (inert unless FN8_TIERED=1).
        _fn8_st = _fn8_store_for(layer, self)
        if _fn8_st is not None:
            return _fn8_st.dispatch(
                self, layer, x, topk_weights, topk_ids,
                shared_experts_input)
        assert self.moe_kernel is not None
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
        print("FN8_02A RESTORED", TARGET, md5(TARGET))
        return

    with open(TARGET) as f:
        src = f.read()

    if MARKER in src:
        print("FN8_02A already applied, nothing to do", md5(TARGET))
        return

    for name, anchor, count_expected in (
        ("logger", ANCHOR_LOGGER, 1),
        ("apply-head", ANCHOR_APPLY_HEAD, 1),
    ):
        n = src.count(anchor)
        assert n == count_expected, f"anchor {name} count {n} != {count_expected}"

    import shutil
    shutil.copyfile(TARGET, TARGET + ".q4dbak")
    before = md5(TARGET)

    src = src.replace(ANCHOR_LOGGER, STORE_BLOCK)
    src = src.replace(ANCHOR_APPLY_HEAD, NEW_APPLY_HEAD)
    assert src.count(MARKER) >= 5, "marker missing after edit?!"

    with open(TARGET, "w") as f:
        f.write(src)
    print(f"FN8_02A backup written: {TARGET}.q4dbak (md5 {before})")
    print(f"FN8_02A PATCHED: {TARGET} md5 {before} -> {md5(TARGET)}")


if __name__ == "__main__":
    main()
