#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 patch_02a — TieredExpertsStore (P2 core).

Port of the Strata resident-set ideology to this fork's modular MoE path,
specialized for FlashNext-FP8 (XPUExpertsFp8Block, 128x128 block FP8):

- Host mirror: the UVA offloader's pageable expert tensors (patch_02b
  suppresses the per-step wholesale-copy closure for modules whose subtree
  carries whitelisted expert tensors; the params still page at load,
  budget accounting intact).
- Device slot pool: S slots/layer + 1 persistent ZERO-WEIGHT dummy slot.
  Seeded from the P1 routing profile (per-layer hot lists, rank-local).
- Runtime: apply() is interposed at BOTH apply choke points —
  FusedMoEModularMethod.apply (the layer.py maybe_init_modular_kernel swap
  path) AND Fp8MoEMethod.apply in quantization/fp8.py.  OWNS-MK LAW
  (boot-3 DEVICE_LOST, 2026-10-11): flash-next-fp8 takes the SECOND path —
  Fp8MoEMethod builds its own moe_kernel in process_weights_after_loading
  (make_fp8_moe_kernel), layer.py sees supports_internal_mk=True and SKIPS
  the modular-method swap, so the runner calls Fp8MoEMethod.apply directly
  and the Round-9-only interposition never ran: zero store lines,
  kernel-direct MoE against CPU-paged weights, UR error 20 (DEVICE_LOST)
  at the first sync after warmup.  fp8.py bridges to the shared
  _fn8_store_for via a lazily-bound resolver (no import-order coupling).
  Per dispatch: sync D2H of topk_ids (tiny), host-side ensure-resident
  (blocking H2D of misses into LRU slots), device remap global->slot,
  then the UNMODIFIED kernel stack runs on pool tensors.
- Non-owned and not-resident ids map to the dummy zero slot: fp8 zeros
  dequant to 0, so the gathered contribution is topk_weight * 0 = 0 —
  mathematically exact drop, identical partial-sum + TP-allreduce
  contract as expert_map=-1 dropping under EP.
- Dummy run (determine_available_memory, attn_metadata is None — the
  fork's PLE/GDN idiom): every id routed to the dummy slot.  No misses,
  no copies, VRAM-safe, and the pool is resident during measurement so
  KV sizing accounts for it.  The store builds lazily at first apply,
  which is exactly that call — pre-warm rides the dummy run per layer.
- Guard law: the store disables only LOUDLY.  The is_block_fp8 gate and
  build failures log (warning / exception); a silent disable is what made
  boot-3 fail card-killing instead of legible.
- Telemetry (V1X_TIER_DUMP, patch_01 atomic-JSON law): per-layer
  dispatches, dummy share, D2H/ensure/H2D ms, copies+bytes, hits/misses/
  evictions, resident count; flushed every V1X_TIER_FLUSH + atexit;
  telemetry failures can never take down the engine (compute-path
  errors propagate — a broken store must be loud, not silently wrong).

Env: FN8_TIERED=1 master, FN8_TIER_SEED=<routing_hist.json>,
FN8_TIER_SLOTS=<per-layer slots, default 96>, V1X_TIER_DUMP=<file>,
V1X_TIER_FLUSH=<n>.

Idempotent (FN8_02A marker per target, independent), backs up .q4dbak,
--restore undoes both targets.  Companion: patch_02b (uva.py no-wrap +
offload whitelist).
"""
import sys

TARGET_MOD = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
              "layers/fused_moe/fused_moe_modular_method.py")
TARGET_FP8 = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
              "layers/quantization/fp8.py")
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
        # Block-scale host mirror + repoint.  Primary path: the experts
        # object's quant_config (the modular FusedMoEQuantConfig the block
        # branch reads live).  Fallback: the layer scale_inv params (both
        # are whitelisted/pageable in patch_02b; whichever the kernel
        # stack consumes must end up pointing at the pool).
        qc = getattr(self.experts, "quant_config", None)
        self.host_s13 = getattr(qc, "w1_scale", None)
        self.host_s2 = getattr(qc, "w2_scale", None)
        self.scale_on_layer = False
        if self.host_s13 is None or self.host_s2 is None:
            self.host_s13 = layer.w13_weight_scale_inv.data
            self.host_s2 = layer.w2_weight_scale_inv.data
            self.scale_on_layer = True

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
        # config); scales re-pointed at the pool on whichever object the
        # block branch reads them from.  REAL FusedMoEConfig law (boot-5):
        # ep_rank/ep_size are PROPERTIES over moe_parallel_config (not
        # fields — dataclasses.replace raised "unexpected keyword 'ep_rank'"
        # on all 48 layers); the pool is one flat local expert set, so the
        # consistent trio is num_experts == num_local_experts ==
        # num_logical_experts = S+1.  EP neutralization only when needed
        # (TP1 already runs ep 1/0; the TP2 window exercises the replace).
        _fn8_pc = self.experts.moe_config.moe_parallel_config
        if getattr(_fn8_pc, "ep_size", 1) != 1:
            try:
                _fn8_pc = dataclasses.replace(_fn8_pc, ep_size=1, ep_rank=0)
            except TypeError:
                pass  # non-dataclass parallel config; TP2 gate must catch
        self.experts.moe_config = dataclasses.replace(
            self.experts.moe_config,
            num_experts=self.pool_slots + 1,
            num_local_experts=self.pool_slots + 1,
            num_logical_experts=self.pool_slots + 1,
            moe_parallel_config=_fn8_pc)
        if self.scale_on_layer:
            layer.w13_weight_scale_inv = self.pool_s13
            layer.w2_weight_scale_inv = self.pool_s2
        else:
            # REAL FusedMoEQuantConfig law (boot-6): w1_scale/w2_scale are
            # read-only properties over the _w1/_w2 desc dataclasses —
            # repoint the desc FIELDS (plain, per-layer, no setter).
            qc._w1.scale = self.pool_s13
            qc._w2.scale = self.pool_s2

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
            "scale_on_layer": self.scale_on_layer,
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
_FN8_WARNED_GATE = False


def _fn8_store_for(layer, method):
    """Lazy per-layer build at first apply (== the boot dummy run)."""
    global _FN8_WARNED_GATE
    if not _FN8_TIERED:
        return None
    lid = id(layer)
    if lid in _FN8_DISABLED:
        return None
    if lid not in _FN8_BUILT:
        experts = getattr(method.moe_kernel, "fused_experts", None)
        if not getattr(experts, "is_block_fp8", False):
            _FN8_DISABLED.add(lid)
            if not _FN8_WARNED_GATE:
                # Guard law: disables are LOUD.  Silent disable = the
                # boot-3 DEVICE_LOST shape (kernel-direct MoE against
                # pageable host weights, UR error 20 at first sync).
                _FN8_WARNED_GATE = True
                logger.warning(
                    "FN8_02A store DISABLED (first at layer %s): "
                    "moe_kernel=%r fused_experts=%r is_block_fp8=%s — "
                    "kernel-direct path keeps CPU-paged weights",
                    getattr(layer, "layer_name", "?"),
                    getattr(method, "moe_kernel", None), experts,
                    getattr(experts, "is_block_fp8", None))
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

# -- fp8.py: the path flash-next-fp8 actually takes (owns-MK law) ----------

ANCHOR_FP8_CLASS = "class Fp8MoEMethod(FusedMoEMethodBase):\n"

NEW_FP8_CLASS = '''_FN8_RESOLVER = None


def _fn8_store_for(layer, method):
    """FN8_02A: lazy bridge to the shared TieredExpertsStore builder.

    quantization/fp8.py must not import fused_moe_modular_method at module
    level (import-order coupling); the resolver binds on first call and
    caches.  Both apply() choke points share one store registry keyed by
    id(layer), so whichever path a layer takes, its store builds once.
    """
    global _FN8_RESOLVER
    if _FN8_RESOLVER is None:
        from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (  # noqa: E501
            _fn8_store_for as _resolver,
        )
        _FN8_RESOLVER = _resolver
    return _FN8_RESOLVER(layer, method)


class Fp8MoEMethod(FusedMoEMethodBase):
'''

ANCHOR_FP8_APPLY = '''    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
'''

NEW_FP8_APPLY = '''    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # FN8_02A: tiered-experts dispatch on the fp8-owns-MK path — the
        # one flash-next-fp8 actually takes.  Fp8MoEMethod builds its own
        # moe_kernel in process_weights_after_loading, layer.py's
        # maybe_init_modular_kernel sees supports_internal_mk=True and
        # SKIPS the FusedMoEModularMethod swap, so without this head the
        # store never builds and moe_kernel.apply reads CPU-paged weights
        # kernel-direct (boot-3 DEVICE_LOST, 2026-10-11).  Inert unless
        # FN8_TIERED=1.
        _fn8_st = _fn8_store_for(layer, self)
        if _fn8_st is not None:
            return _fn8_st.dispatch(
                self, layer, x, topk_weights, topk_ids,
                shared_experts_input)
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
'''


def md5(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def patch_file(label, target, edits, marker_min):
    with open(target) as f:
        src = f.read()

    if MARKER in src:
        print(f"{label} already applied, nothing to do", md5(target))
        return

    for name, anchor, _repl in edits:
        n = src.count(anchor)
        assert n == 1, f"{label} anchor {name} count {n} != 1"

    import shutil
    shutil.copyfile(target, target + ".q4dbak")
    before = md5(target)

    for _name, anchor, repl in edits:
        src = src.replace(anchor, repl)
    assert src.count(MARKER) >= marker_min, f"{label} marker missing after edit?!"

    with open(target, "w") as f:
        f.write(src)
    print(f"{label} backup written: {target}.q4dbak (md5 {before})")
    print(f"{label} PATCHED: {target} md5 {before} -> {md5(target)}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        import shutil
        for label, target in (("FN8_02A-MOD", TARGET_MOD),
                              ("FN8_02A-FP8", TARGET_FP8)):
            shutil.copyfile(target + ".q4dbak", target)
            print("FN8_02A RESTORED", target, md5(target))
        return

    patch_file(
        "FN8_02A-MOD", TARGET_MOD,
        (("logger", ANCHOR_LOGGER, STORE_BLOCK),
         ("apply-head", ANCHOR_APPLY_HEAD, NEW_APPLY_HEAD)),
        marker_min=5)
    patch_file(
        "FN8_02A-FP8", TARGET_FP8,
        (("fp8-class", ANCHOR_FP8_CLASS, NEW_FP8_CLASS),
         ("fp8-apply", ANCHOR_FP8_APPLY, NEW_FP8_APPLY)),
        marker_min=2)


if __name__ == "__main__":
    main()
