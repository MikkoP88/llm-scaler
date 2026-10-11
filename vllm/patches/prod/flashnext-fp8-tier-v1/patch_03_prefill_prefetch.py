#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 patch_03 — P3 prefill streaming (layer-ahead prefetch).

Chases the PHASES P3 gate (prompt tok/s >=600 @4K, no TTFT regression vs P2)
on top of the patch_02a tiered store.  All edits live inside the FN8_02A
injected block of fused_moe_modular_method.py (no other file — deliberate
deviation from the PHASES sketch that anchored gpu_model_runner: chunk
epochs are derivable from forward_context attn_metadata identity, and
keeping the lane-hot file out of the patch set removes a whole md5 gate):

1. Replay staging (write side): every dispatch stashes its topk_ids into a
   per-store device int16 buffer + host mirror, epoch-tagged (epoch = a
   module counter bumped whenever the forward_context attn_metadata object
   identity changes = one scheduler step / prefill chunk).
2. Layer-ahead prediction (read side): at store L's dispatch tail, warm
   store L+V1X_TIER_PF_LAYERS from L's OWN fresh replay — adjacent-layer
   routing correlation stands in for the target's not-yet-computed routing
   (the target's own stale replay predicts nothing: its ensure already
   made those ids resident).  Budget-capped, ownership-filtered.
3. Side-stream async fills: predicted experts stream into FREE pool slots
   on a dedicated torch.xpu.Stream ordered after all default-stream work
   enqueued so far, recorded with an event; ensure_resident PROMOTES
   filling ids (event.synchronize) before the kernel runs and
   opportunistically promotes completed fills so they age out via LRU.

SLOT-OWNERSHIP INVARIANTS (the two-writer and stale-map races):
- Prefetch NEVER evicts: _pf_fill uses only genuinely free slots (excl.
  filling); a full pool makes the tail a no-op (prediction is advisory).
  _load_slot stays the SINGLE eviction site.
- _load_slot counts filling slots in its capacity check, never scans a
  filling slot as free, and when the pool is entirely in-flight fills it
  drops the OLDEST fill (event-synchronized first — two async writers
  must never share a slot) before reloading.
- Invariant that makes the device gather safe: every id in slot_of has a
  correct slot_map_dev entry (entries are only written at _load_slot /
  promotion time, both followed by a map sync), and every id NOT in
  slot_of is routed through the authoritative miss path when current.

CORRECTNESS LAW: prediction is advisory only — ensure_resident still
checks the authoritative current-step needed-set.  Any _pf_* exception
logs one loud warning and disables prefetch for that store (compute
continues — unlike a build failure, a prefetch bug must not take the
layer down).

Env: V1X_TIER_PREFETCH=1 (default off), V1X_TIER_PF_LAYERS=4,
V1X_TIER_PF_BUDGET=8, V1X_TIER_PF_STREAM=1 (0 = fill on the default
stream via the blocking _load_slot = prediction-only mode).

Idempotent (FN8_03 marker), .q4dbak3, --restore undoes.  MUST run after
patch_02a (asserts the FN8_02A marker).
"""
import sys

TARGET = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
          "layers/fused_moe/fused_moe_modular_method.py")
MARKER = "FN8_03"

# -- anchors (must each appear exactly once in the 02a-patched source) ------

A_ENV = '_FN8_TIER_FLUSH = int(_fn8_os.environ.get("V1X_TIER_FLUSH", "200"))\n'
N_ENV = A_ENV + '''# FN8_03 (P3 prefill streaming): layer-ahead prefetch knobs + chunk
# epoch counter.  PREFETCH defaults OFF — the P2 window runs unprefetched;
# the P3 serve script opts in.
_FN8_PF = _fn8_os.environ.get("V1X_TIER_PREFETCH", "0") == "1"
_FN8_PF_LAYERS = int(_fn8_os.environ.get("V1X_TIER_PF_LAYERS", "4"))
_FN8_PF_BUDGET = int(_fn8_os.environ.get("V1X_TIER_PF_BUDGET", "8"))
_FN8_PF_STREAM = _fn8_os.environ.get("V1X_TIER_PF_STREAM", "1") == "1"
_FN8_EPOCH = 0
_FN8_LAST_MD = None
_FN8_PF_SIDE = None  # lazy per-process side stream
_FN8_PF_REG = []  # patch_03 store registry (02a keeps only id(layer) ints)
_FN8_PF_BY_LAYER = {}  # layer_idx -> store (rebuilt when registry grows)
'''

A_INIT = "        self.slot_map_dev = self.slot_map_host.to(dev)\n"
N_INIT = A_INIT + '''        # FN8_03: replay staging + async-fill bookkeeping (P3).
        self.replay_epoch = -1
        self.replay_n = 0
        self.replay_dev = None    # (MAXTOK, K) int16, lazy
        self.replay_host = None
        self.filling = {}         # global_id -> (slot, event)
        self.filling_slots = set()
        self._pf_broken = False
        _FN8_PF_REG.append(self)  # 02a registry stores only id(layer) ints
'''

A_METHODS = "    # -- slot machinery ----------------------------------------------------\n"
N_METHODS = '''    # -- FN8_03: prefill streaming (layer-ahead prefetch) -------------------

    def _pf_promote(self, global_id, slot) -> None:
        """Move a completed async fill into the resident set (mirror law)."""
        self.slot_of[global_id] = slot
        self.global_of[slot] = global_id
        self.last_use[global_id] = self.use_tick
        self.slot_map_host[global_id] = slot

    def _pf_stage(self, topk_ids) -> None:
        """Stash this dispatch's routing into the replay buffers."""
        import torch as _t
        if not _FN8_PF or self._pf_broken:
            return
        try:
            n, k = topk_ids.shape
            if self.replay_dev is None or self.replay_dev.shape[1] != k:
                # one buffer sized for max_num_batched_tokens headroom; a
                # bigger n than that simply skips staging (stale stays None)
                self.replay_dev = _t.empty(
                    (4096, k), dtype=_t.int16, device=self.pool_w13.device)
                try:
                    self.replay_host = _t.empty_like(
                        self.replay_dev, device="cpu").pin_memory()
                except Exception:
                    self.replay_host = _t.empty_like(
                        self.replay_dev, device="cpu")
            if n > self.replay_dev.shape[0]:
                self.replay_epoch = -1
                return
            self.replay_dev[:n].copy_(topk_ids.to(_t.int16))
            # tiny copy (<=4096*K*2B); consumed one full model step later —
            # no event needed, the epoch gap dwarfs the transfer.
            self.replay_host[:n].copy_(self.replay_dev[:n])
            self.replay_epoch = _FN8_EPOCH
            self.replay_n = n
        except Exception as _e:
            self._pf_broken = True
            logger.warning(
                "FN8_03 prefetch DISABLED for layer %s (stage): %s",
                self.layer_idx, _e)

    def _pf_fill(self, global_id: int) -> None:
        """Async side-stream fill of one expert into a FREE pool slot.

        Advisory: NEVER evicts — with no free slot the prediction is
        skipped (eviction stays the sole province of _load_slot)."""
        import torch as _t
        global _FN8_PF_SIDE
        if not _FN8_PF_STREAM:
            # prediction-only mode: blocking load on the default stream
            self._load_slot(global_id)
            self.rec["pf_fills"] += 1
            return
        slot = next(
            (s for s in range(self.pool_slots)
             if s not in self.global_of and s not in self.filling_slots),
            None)
        if slot is None:
            return  # pool full of residents+fills: prediction skipped
        if _FN8_PF_SIDE is None:
            _FN8_PF_SIDE = _t.xpu.Stream()
        # order after ALL default-stream work enqueued so far: earlier
        # steps' kernels may still be reading pool slots.
        ev_def = _t.xpu.Event()
        ev_def.record()
        _FN8_PF_SIDE.wait_event(ev_def)
        local = self.owned[global_id]
        with _t.xpu.stream(_FN8_PF_SIDE):
            self.pool_w13[slot].copy_(self.host_w13[local], non_blocking=True)
            self.pool_w2[slot].copy_(self.host_w2[local], non_blocking=True)
            self.pool_s13[slot].copy_(self.host_s13[local], non_blocking=True)
            self.pool_s2[slot].copy_(self.host_s2[local], non_blocking=True)
            ev = _t.xpu.Event()
            ev.record()
        self.filling[global_id] = (slot, ev)
        self.filling_slots.add(slot)
        self.rec["pf_fills"] += 1
        self.rec["h2d_bytes"] += (
            self.host_w13[local].numel() * self.host_w13.element_size()
            + self.host_w2[local].numel() * self.host_w2.element_size())

    def _pf_dispatch_tail(self) -> None:
        """Layer-ahead: warm store L+PF_LAYERS from MY OWN fresh replay.

        Cross-layer routing correlation stands in for the target's
        not-yet-computed routing (the target's own last replay is useless
        as a prediction: its ensure already made those ids resident)."""
        global _FN8_PF_BY_LAYER
        if self._pf_broken:
            return
        try:
            if self.replay_epoch != _FN8_EPOCH or self.replay_n == 0:
                return
            if len(_FN8_PF_BY_LAYER) != len(_FN8_PF_REG):
                _FN8_PF_BY_LAYER = {st.layer_idx: st for st in _FN8_PF_REG}
            tgt = _FN8_PF_BY_LAYER.get(self.layer_idx + _FN8_PF_LAYERS)
            if tgt is None or tgt is self or tgt._pf_broken:
                return
            budget = _FN8_PF_BUDGET
            seen = set()
            for row in self.replay_host[: self.replay_n].tolist():
                for g in row:
                    if g in seen:
                        continue
                    seen.add(g)
                    if (g not in tgt.owned or g in tgt.slot_of
                            or g in tgt.filling):
                        continue
                    if budget <= 0:
                        return
                    budget -= 1
                    tgt._pf_fill(g)
        except Exception as _e:
            self._pf_broken = True
            logger.warning(
                "FN8_03 prefetch DISABLED for layer %s (tail): %s",
                self.layer_idx, _e)

''' + A_METHODS

A_LOAD = '''        if len(self.slot_of) >= self.pool_slots:
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
'''
N_LOAD = '''        # FN8_03: filling slots count toward capacity, are never scanned
        # as free, and (pool entirely in-flight) are dropped oldest-first
        # with an event sync — two async writers must never share a slot.
        if len(self.slot_of) + len(self.filling) >= self.pool_slots:
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
            elif self.filling:
                fg = next(iter(self.filling))
                slot_fg, ev_fg = self.filling.pop(fg)
                try:
                    ev_fg.synchronize()
                except Exception:
                    pass
                self.filling_slots.discard(slot_fg)
                self.slot_map_host[fg] = self.dummy
                self.rec["pf_drops"] = self.rec.get("pf_drops", 0) + 1
        # first free slot, else reuse evicted (never a filling slot)
        slot = next(
            (s for s in range(self.pool_slots)
             if s not in self.global_of and s not in self.filling_slots),
            None)
        if slot is None:
            return  # unreachable when eviction succeeded
'''

A_REC = '            "resident": 0, "pool_slots": self.pool_slots,\n'
N_REC = ('            "resident": 0, "pool_slots": self.pool_slots,\n'
         '            "pf_fills": 0, "pf_promotes": 0, "pf_waits": 0,\n'
         '            "pf_drops": 0,\n')

A_MD = "        if attn_metadata is None:\n"
N_MD = '''        global _FN8_EPOCH, _FN8_LAST_MD
        if (attn_metadata is not None and attn_metadata != "unknown"
                and attn_metadata is not _FN8_LAST_MD):
            _FN8_EPOCH += 1
            _FN8_LAST_MD = attn_metadata
        if attn_metadata is None:
'''

A_IDS = '            ids_host = topk_ids.detach().to("cpu").tolist()\n'
N_IDS = ('            self._pf_stage(topk_ids)\n'
         + A_IDS)

A_RET = "        return self.kernel.apply(\n"
N_RET = ('        if _FN8_PF:\n'
         '            self._pf_dispatch_tail()\n'
         + A_RET)

A_ENSURE = '''        changed = False
        for g in needed:
            if g in self.slot_of:
'''
N_ENSURE = '''        changed = False
        # FN8_03: promote async fills first — a needed filling id MUST be
        # event-synchronized before its slot is readable; completed
        # unneeded fills are promoted opportunistically so they age out
        # via LRU instead of pinning pool capacity forever.
        try:
            for g, (slot_g, ev_g) in list(self.filling.items()):
                done = False
                try:
                    done = bool(ev_g.query())
                except Exception:
                    done = False
                if done or g in needed:
                    if not done:
                        ev_g.synchronize()
                        self.rec["pf_waits"] += 1
                    self.filling.pop(g)
                    self.filling_slots.discard(slot_g)
                    self._pf_promote(g, slot_g)
                    self.rec["pf_promotes"] += 1
                    changed = True
        except Exception as _e:
            self._pf_broken = True
            logger.warning(
                "FN8_03 prefetch DISABLED for layer %s (ensure): %s",
                self.layer_idx, _e)
        for g in needed:
            if g in self.slot_of:
'''


def md5(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        bak = TARGET + ".q4dbak3"
        import shutil
        shutil.copyfile(bak, TARGET)
        print("FN8_03 RESTORED", TARGET, md5(TARGET))
        return

    with open(TARGET) as f:
        src = f.read()

    if MARKER in src:
        print("FN8_03 already applied, nothing to do", md5(TARGET))
        return

    assert "FN8_02A" in src, (
        "patch_03 requires patch_02a applied first (FN8_02A marker absent)")

    for name, anchor in (
        ("env", A_ENV),
        ("init", A_INIT),
        ("methods", A_METHODS),
        ("load", A_LOAD),
        ("rec", A_REC),
        ("attn-md", A_MD),
        ("ids", A_IDS),
        ("return", A_RET),
        ("ensure", A_ENSURE),
    ):
        n = src.count(anchor)
        assert n == 1, f"anchor {name} count {n} != 1"

    import shutil
    shutil.copyfile(TARGET, TARGET + ".q4dbak3")
    before = md5(TARGET)

    src = src.replace(A_ENV, N_ENV)
    src = src.replace(A_INIT, N_INIT)
    src = src.replace(A_METHODS, N_METHODS)
    src = src.replace(A_LOAD, N_LOAD)
    src = src.replace(A_REC, N_REC)
    src = src.replace(A_MD, N_MD)
    src = src.replace(A_IDS, N_IDS)
    src = src.replace(A_RET, N_RET)
    src = src.replace(A_ENSURE, N_ENSURE)
    assert src.count(MARKER) >= 8, "marker missing after edit?!"

    with open(TARGET, "w") as f:
        f.write(src)
    print(f"FN8_03 backup written: {TARGET}.q4dbak3 (md5 {before})")
    print(f"FN8_03 PATCHED: {TARGET} md5 {before} -> {md5(TARGET)}")


if __name__ == "__main__":
    main()
