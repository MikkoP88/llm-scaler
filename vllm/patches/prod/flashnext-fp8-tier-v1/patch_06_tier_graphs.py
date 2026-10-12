#!/usr/bin/env python3
"""FN8_06 (DEC40 D8): capture-safe tier under XPU graphs + post-replay
residency hook + env-gated H2D sync-kill.

Evidence chain (D7, 2026-10-12):
  * step_ms ~250 invariant under patch_05 (PLE sync-free) + patch_05b
    (Parameter hook) + --async-scheduling -> the removed host tax was
    never the critical path;
  * tier_d7.json: h2d_ms totals ~2 ms/layer over 750 dispatches while
    20+ GB/layer were copied -> the per-miss `_t.xpu.synchronize()`
    drains return INSTANTLY -> the device is idle at every miss point ->
    the step is serialized by HOST work (py-spy: 96.9% of samples inside
    forward; _ops.py 41% / moe 42.8% / linear 23.7% presence);
  * the only lever that removes per-op python dispatch entirely is
    FULL_DECODE_ONLY graph replay (D3 proved the graph machinery boots
    and captures on this stack — it crashed INSIDE the tier's
    tolist/ensure_resident python, i.e. exactly the piece this patch
    makes capture-legal).

Design:
  fused_moe_modular_method.py
    E1 module globals + store registry + fn8_tier_post_replay() hook
       (one device sync per STEP; ensure_resident + _sync_slot_map on
       the default stream, same-stream ordered before the next replay;
       residency stale-by-one on newly-routed experts, slot_map values
       always valid slots or dummy)
    E2 __init__: gr_dev/gr_host/_graphs_armed + registry append
    E3 rec dict: graph_dispatches/hook_calls/hook_errors/hook_ms/
       nosync/syncs keys (telemetry directive)
    E4 dispatch(): capture branch first — async D2H of topk_ids into the
       pinned int16 ring (recorded IN the graph; every replay refreshes
       it with CURRENT ids), then the same capture-legal gather; eager
       path untouched; _pf_dispatch_tail gated off during capture
    E5 _gr_alloc helper (pinned via .pin_memory() — proven at line 371)
    E6a/E6b _load_many + _load_slot: `_t.xpu.synchronize()` skipped when
       FN8_TIER_NO_SYNC=1 (same-stream ordering already guarantees the
       copies land before same-stream readers; counter per mode)
  gpu_ubatch_wrapper.py
    E7 hook call after FULL replay (async scheduler site; rows=num_tokens)
  cudagraph_utils.py
    E8 hook call after run_fullgraph replay (rows=desc.num_tokens)

Env: FN8_TIER_GRAPHS=1 arms the capture branch + hook; FN8_TIER_NO_SYNC=1
skips the two drains.  Both default OFF (D4/D7 behavior byte-identical).

Idempotent; per-file backups .fn8_06.bak; py_compile gate; EOL aware;
anchor==1 gates; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)

F1 = "model_executor/layers/fused_moe/fused_moe_modular_method.py"
F2 = "v1/worker/gpu_ubatch_wrapper.py"
F3 = "v1/worker/gpu/cudagraph_utils.py"

# ============================ FILE 1 edits ==============================

# --- E1: module globals + registry + post-replay hook ----------------------
OLD1_G_LF = '''_FN8_PF_BY_LAYER = {}  # layer_idx -> store (rebuilt when registry grows)
'''
NEW1_G_LF = '''_FN8_PF_BY_LAYER = {}  # layer_idx -> store (rebuilt when registry grows)
_FN8_TIER_GRAPHS = _fn8_os.environ.get("FN8_TIER_GRAPHS", "") == "1"
_FN8_TIER_NO_SYNC = _fn8_os.environ.get("FN8_TIER_NO_SYNC", "") == "1"
_FN8_TIER_CAPTURING = getattr(torch.xpu, "is_current_stream_capturing", None)
_FN8_TIER_STORES = []  # FN8_06: every store, for the post-replay hook


def fn8_tier_post_replay(rows):
    """FN8_06 (D8 graphs): post-replay tier residency update.

    Called by the FULL-graph replay sites (gpu_ubatch_wrapper /
    cudagraph_utils) after every replay.  The captured graphs D2H each
    step's topk_ids into every store's pinned int16 ring (in-graph async
    copy); ONE device sync per step (not per layer) makes the ring
    host-visible, then ensure_resident + _sync_slot_map run on the
    host/default stream — same-stream ordered before the next replay's
    gather and kernel reads.  Residency is stale-by-one on newly-routed
    experts (dummy slot for one step); slot_map values are always valid
    slots or the dummy, so a racing replay can never read a torn map.
    """
    if not _FN8_TIER_GRAPHS:
        return
    if _FN8_TIER_CAPTURING is not None and _FN8_TIER_CAPTURING():
        return
    armed = [s for s in _FN8_TIER_STORES if s._graphs_armed]
    if not armed:
        return
    import torch as _t
    _t.xpu.synchronize()
    for s in armed:
        _t0 = _fn8_time.time()
        try:
            _n = min(int(rows), s.gr_host.shape[0])
            s.ensure_resident(s.gr_host[:_n].tolist())
            s._sync_slot_map()
            s.rec["hook_calls"] += 1
        except Exception:
            s.rec["hook_errors"] += 1
        s.rec["hook_ms"] += _fn8_time.time() - _t0
'''

# --- E2: __init__ ring state + registry -------------------------------------
OLD1_I_LF = '''        self.filling_slots = set()
'''
NEW1_I_LF = '''        self.filling_slots = set()
        # FN8_06 (D8 graphs): pinned topk D2H ring + post-replay state
        self.gr_dev = None
        self.gr_host = None
        self._graphs_armed = False
        _FN8_TIER_STORES.append(self)
'''

# --- E3: rec dict telemetry keys ---------------------------------------------
OLD1_R_LF = '''            "pf_drops": 0,
'''
NEW1_R_LF = '''            "pf_drops": 0,
            "graph_dispatches": 0, "hook_calls": 0, "hook_errors": 0,
            "hook_ms": 0.0, "nosync": 0, "syncs": 0,
'''

# --- E4: dispatch capture branch ----------------------------------------------
OLD1_D_LF = '''        if attn_metadata is None:
            # Boot dummy run (fork idiom): all-zero weights, no misses,
            # pool resident for the memory measurement.
            self.rec["dummy_dispatches"] += 1
            slot_ids = _t.full_like(topk_ids, self.dummy)
        else:
'''
NEW1_D_LF = '''        _capt = (_FN8_TIER_GRAPHS and _FN8_TIER_CAPTURING is not None
                 and _FN8_TIER_CAPTURING())
        if _capt:
            # FN8_06: graph capture — the eager tier path below is
            # capture-ILLEGAL (tolist D2H + host ensure + device sync
            # inside the piece = the D3 capture crash).  Record an async
            # D2H of this step's topk into the store's pinned ring: the
            # copy becomes a graph node, so EVERY replay refreshes the
            # ring with the CURRENT ids; the post-replay hook consumes
            # it one step later.  The gather below is capture-legal and
            # re-executes per replay against the live slot_map_dev.
            _n, _k = topk_ids.shape
            if (self.gr_dev is None or self.gr_dev.shape[0] < _n
                    or self.gr_dev.shape[1] != _k):
                self._gr_alloc(_n, _k)
            self.gr_dev[:_n].copy_(topk_ids.to(_t.int16))
            self.gr_host[:_n].copy_(self.gr_dev[:_n], non_blocking=True)
            self._graphs_armed = True
            self.rec["graph_dispatches"] += 1
            slot_ids = self.slot_map_dev[topk_ids.detach().long()].to(
                topk_ids.dtype)
        elif attn_metadata is None:
            # Boot dummy run (fork idiom): all-zero weights, no misses,
            # pool resident for the memory measurement.
            self.rec["dummy_dispatches"] += 1
            slot_ids = _t.full_like(topk_ids, self.dummy)
        else:
'''

# --- E4b: PF tail gated off during capture ------------------------------------
OLD1_T_LF = '''        if _FN8_PF:
            self._pf_dispatch_tail()
'''
NEW1_T_LF = '''        if _FN8_PF and not _capt:
            self._pf_dispatch_tail()
'''

# --- E5: _gr_alloc helper ------------------------------------------------------
OLD1_A_LF = '''    def _sync_slot_map(self) -> None:
        self.slot_map_dev.copy_(self.slot_map_host)
        self.rec["resident"] = len(self.slot_of)
'''
NEW1_A_LF = '''    def _sync_slot_map(self) -> None:
        self.slot_map_dev.copy_(self.slot_map_host)
        self.rec["resident"] = len(self.slot_of)

    def _gr_alloc(self, rows, k):
        """FN8_06: pinned D2H ring for graph replay (int16 expert ids;
        global ids < 32768 — same law as the _pf_stage staging buffer)."""
        import torch as _t
        rows = max(128, int(rows))
        self.gr_dev = _t.empty((rows, k), dtype=_t.int16,
                               device=self.pool_w13.device)
        self.gr_host = _t.empty((rows, k), dtype=_t.int16,
                                device="cpu").pin_memory()
'''

# --- E6a: _load_many drain env-gated --------------------------------------------
OLD1_S1_LF = '''            self.pool_s2[slot].copy_(self.host_s2[local])
        _t.xpu.synchronize()
        dt = _fn8_time.time() - t0
        for g, slot, local in plan:
'''
NEW1_S1_LF = '''            self.pool_s2[slot].copy_(self.host_s2[local])
        # FN8_06: same-stream ordering already lands these copies before
        # any same-stream reader; the drain is only kept for the default
        # (eager) mode.  Under D8 the hook issues these between replays
        # and a device-wide drain would stall the async-scheduler host.
        if _FN8_TIER_NO_SYNC:
            self.rec["nosync"] += 1
        else:
            _t.xpu.synchronize()
            self.rec["syncs"] += 1
        dt = _fn8_time.time() - t0
        for g, slot, local in plan:
'''

# --- E6b: _load_slot drain env-gated ---------------------------------------------
OLD1_S2_LF = '''        self.pool_s2[slot].copy_(self.host_s2[local])
        _t.xpu.synchronize()
        dt = _fn8_time.time() - t0
        self.slot_of[global_id] = slot
'''
NEW1_S2_LF = '''        self.pool_s2[slot].copy_(self.host_s2[local])
        if _FN8_TIER_NO_SYNC:
            self.rec["nosync"] += 1
        else:
            _t.xpu.synchronize()
            self.rec["syncs"] += 1
        dt = _fn8_time.time() - t0
        self.slot_of[global_id] = slot
'''

# ============================ FILE 2 edit ===============================

OLD2_LF = '''            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs
'''
NEW2_LF = '''            cudagraph_metadata.cudagraph.replay()
            # FN8_06: tier residency hook after FULL replay (no-op
            # unless FN8_TIER_GRAPHS armed; one host sync per step,
            # replaces the per-layer tolist/ensure/drain of the eager
            # tier path).
            try:
                from vllm.model_executor.layers.fused_moe import (
                    fused_moe_modular_method as _fmmm)
                _fmmm.fn8_tier_post_replay(num_tokens)
            except Exception:
                pass
            return cudagraph_metadata.outputs
'''

# ============================ FILE 3 edit ===============================

OLD3_LF = '''        get_offloader().sync_prev_onload()
        self.graphs[desc].replay()
'''
NEW3_LF = '''        get_offloader().sync_prev_onload()
        self.graphs[desc].replay()
        # FN8_06: tier residency hook after FULL replay (see
        # fused_moe_modular_method.fn8_tier_post_replay).
        try:
            from vllm.model_executor.layers.fused_moe import (
                fused_moe_modular_method as _fmmm)
            _fmmm.fn8_tier_post_replay(desc.num_tokens)
        except Exception:
            pass
'''

EDITS = {
    F1: [
        ("globals_hook", OLD1_G_LF, NEW1_G_LF),
        ("init_ring", OLD1_I_LF, NEW1_I_LF),
        ("rec_keys", OLD1_R_LF, NEW1_R_LF),
        ("dispatch_capture", OLD1_D_LF, NEW1_D_LF),
        ("pf_tail_gate", OLD1_T_LF, NEW1_T_LF),
        ("gr_alloc", OLD1_A_LF, NEW1_A_LF),
        ("drain_load_many", OLD1_S1_LF, NEW1_S1_LF),
        ("drain_load_slot", OLD1_S2_LF, NEW1_S2_LF),
    ],
    F2: [("replay_hook_async", OLD2_LF, NEW2_LF)],
    F3: [("replay_hook_fullgraph", OLD3_LF, NEW3_LF)],
}


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def restore(path, rel):
    bak = path + ".fn8_06.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {rel} md5={md5(path)}")
    else:
        print(f"{TAG} no backup for {rel}")


if "--restore" in sys.argv:
    for rel in (F1, F2, F3):
        restore(os.path.join(PKG, rel), rel)
    sys.exit(0)

ok_all = True
for rel, edits in EDITS.items():
    path = os.path.join(PKG, rel)
    with open(path, "r", encoding="utf-8", newline="") as f:
        src = f.read()

    if "_FN8_TIER_GRAPHS" in src or "fn8_tier_post_replay" in src:
        print(f"{TAG} [{rel}] edit already present, skip (md5={md5(path)})")
        continue

    eol = "\r\n" if "\r\n" in src else "\n"

    def ed(lf_text):
        return lf_text.replace("\n", eol)

    fail = False
    for name, old_lf, _ in edits:
        n = src.count(ed(old_lf))
        if n != 1:
            print(f"{TAG} [{rel}] FAIL: anchor '{name}' occurs {n}x (want 1)")
            fail = True
    if fail:
        ok_all = False
        continue

    bak = path + ".fn8_06.bak"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
        print(f"{TAG} [{rel}] backup {bak} md5={md5(bak)}")

    out = src
    for _, old_lf, new_lf in edits:
        out = out.replace(ed(old_lf), ed(new_lf))

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8", newline=""
    ) as tf:
        tf.write(out)
        tmp = tf.name
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"{TAG} [{rel}] FAIL: compile error -> {e}")
        ok_all = False
        os.unlink(tmp)
        continue
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(
        f"{TAG} [{rel}] edited "
        f"(eol={'CRLF' if eol == chr(13) + chr(10) else 'LF'}), "
        f"md5={md5(path)}"
    )

if not ok_all:
    print(f"{TAG} FAILURES — restore with --restore")
    sys.exit(2)
print(f"{TAG} ALL_EDITS_OK")
