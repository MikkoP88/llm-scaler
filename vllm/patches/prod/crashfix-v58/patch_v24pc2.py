#!/opt/venv/bin/python3
# llm-scaler v24pc2 — §24 S ARENA COPY-ON-HIT (COH v2) boot patcher.
#
# Mechanism (§24 R evidence): warm prefix-cache hits decode -15..17% because
# the inherited block TABLE is id-scattered (decode tok/s tracks physical id
# density/span monotonically: cold span 163 -> 359 tok/s, warm wide -> 301,
# 128k ~98% span -> 277, half65k maxgap 616 -> 209). Manager blocks are
# 1024 tokens; pool ~582 blocks.
#
# v1 (§24 Q) was REJECTED: get_new_blocks() at hit time popped hash-carrying
# LRU blocks -> silent eviction -> 128k prefix chain broke (warm TTFT 140s),
# and the copies never executed: the pending queue lives in EngineCore while
# drain_copies ran in worker processes (module state is per-process).
#
# v2 fixes both:
#   ARENA   — one-time contiguous reservation of VLLM_COH2_ARENA (160)
#             manager blocks, taken at the FIRST schedule() tick when the
#             pool is pristine (no hash-carrying blocks in the free queue ->
#             zero eviction). Arena blocks are held forever (ref_cnt 1,
#             never re-enter the BlockPool free queue). Hits >= threshold
#             are served from the arena via a bump allocator and recycled at
#             request free (filtered out of req_to_blocks BEFORE base free).
#   IPC     — copy pairs (dst_ids, src_ids) ride the SchedulerOutput
#             (pickled to both TP workers). Worker drain runs AFTER the
#             runner's new-block zeroing and BEFORE the forward kernels,
#             mirroring KVBlockZeroer's own id->page mapping
#             (ratio = spec.block_size // kernel_bs; block_dim from the
#             attention backend; kv tensors from static_forward_context).
#
# Numerics: the copy is bitwise D2D per layer (index_copy_ of index_select
# along the block dim). Cache-hash space is untouched: the copied arena
# prefix sits in [0, num_cached_block) so cache_blocks() never re-hashes it,
# and the original cached blocks stay registered + evictable (touch() is
# deliberately skipped).
#
# Patches:
#   P1 vllm/v1/core/single_type_kv_cache_manager.py:
#      FullAttentionManager.allocate_new_computed_blocks -> _v24pc2.coh_allocate
#      FullAttentionManager.free                        -> _v24pc2.coh_free
#   P2 vllm/v1/worker/gpu_model_runner.py: drain at the zero-blocks site in
#      _update_states (zero -> copy order guaranteed).
#   P3 vllm/v1/core/sched/output.py: SchedulerOutput._v24pc2_copies field.
#   P4 vllm/v1/core/sched/scheduler.py: attach_stash(scheduler_output)
#      right before `return scheduler_output` (also builds the arena early).
# Installs vllm/_v24pc2.py.
#
# NOTE: requires non-async scheduling (standing lane config) so that request
# frees are strictly ordered after the worker's last use of the blocks.
#
# Env: VLLM_COH2=0 kill-switch; VLLM_COH2_MIN_BLOCKS (default 16 manager
# blocks = 16k tokens); VLLM_COH2_ARENA (default 160); VLLM_COH2_ARENA_MARGIN
# (default 64). Usage: patch_v24pc2.py [--check | --revert]
import pathlib
import shutil
import sys

SP = pathlib.Path("/opt/venv/lib/python3.12/site-packages")

MODULE = r'''# llm-scaler v24pc2 module (installed by patch_v24pc2.py) — arena COH v2
import os
from collections import deque

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec

_ENABLED = os.environ.get("VLLM_COH2", "1") != "0"
_MIN_BLOCKS = int(os.environ.get("VLLM_COH2_MIN_BLOCKS", "16"))
_ARENA_BLOCKS = int(os.environ.get("VLLM_COH2_ARENA", "160"))
_ARENA_MARGIN = int(os.environ.get("VLLM_COH2_ARENA_MARGIN", "64"))

# EngineCore-side state (scheduler / kv cache manager process)
_st = {
    "arena": None,         # list[KVCacheBlock] held forever
    "arena_ids": set(),    # block ids owned by the arena
    "arena_free": deque(), # recycle list (bump order)
    "req": {},             # request_id -> arena blocks handed out
    "pending": [],         # (dst_ids, src_ids, rid, n) -> SchedulerOutput
    "logs": 0,
}
# Worker-side state (model runner process)
_w = {"meta": None, "runner": None, "logs": 0, "shape_logged": False}


def _log(msg):
    if _st["logs"] < 400:
        _st["logs"] += 1
        print(f"[v24pc2] {msg}", flush=True)


def _wlog(msg):
    if _w["logs"] < 200:
        _w["logs"] += 1
        print(f"[v24pc2w] {msg}", flush=True)


def _runs_desc(ids):
    if not ids:
        return "len=0"
    runs = 1
    gaps = 0
    maxgap = 0
    for a, b in zip(ids, ids[1:]):
        d = b - a
        if d != 1:
            runs += 1
            gaps += 1
            if d > maxgap:
                maxgap = d
    return (f"len={len(ids)} first={ids[0]} last={ids[-1]} runs={runs} "
            f"gaps={gaps} maxgap={maxgap}")


# ---- EngineCore side -------------------------------------------------------

def _ensure_arena(pool):
    """Reserve the arena. Called at the FIRST schedule() tick via
    attach_stash(scheduler_output, scheduler): the pool is pristine then
    (no hash-carrying blocks in the free queue -> zero prefix evictions),
    unlike a lazy build at first-hit time which would pop LRU hashes."""
    if not _ENABLED or _st["arena"] is not None:
        return
    if pool.get_num_free_blocks() < _ARENA_BLOCKS + _ARENA_MARGIN:
        return  # retry on a later schedule tick
    blocks = pool.get_new_blocks(_ARENA_BLOCKS)
    _st["arena"] = list(blocks)
    _st["arena_ids"] = {b.block_id for b in blocks}
    _st["arena_free"] = deque(blocks)
    ids = [b.block_id for b in blocks]
    _log(f"ARENA n={_ARENA_BLOCKS} poolfree_after={pool.get_num_free_blocks()} "
         f"[{_runs_desc(ids)}]")


def _alloc_arena(n):
    af = _st["arena_free"]
    if len(af) < n:
        return None
    return [af.popleft() for _ in range(n)]


def coh_allocate(base, self, request_id, new_computed_blocks,
                 num_local_computed_tokens, num_external_computed_tokens):
    """llm-scaler v24pc2: arena copy-on-hit for FullAttentionManager.

    Swaps a >= threshold prefix-cache hit for arena blocks plus a one-shot
    D2D copy (stashed to SchedulerOutput, executed by the worker drain).
    Falls back to the base shared-block attach on any doubt.
    """
    if not _ENABLED or request_id in self.num_cached_block:
        # kill-switch, or base's running-request fast path
        return base(self, request_id, new_computed_blocks,
                    num_local_computed_tokens, num_external_computed_tokens)
    try:
        n = len(new_computed_blocks)
        if (n >= _MIN_BLOCKS and num_external_computed_tokens == 0
                and self.enable_caching and _st["arena"] is not None):
            dst = _alloc_arena(n)
            if dst is not None:
                req_blocks = self.req_to_blocks[request_id]
                if len(req_blocks) == 0:
                    dst_ids = [b.block_id for b in dst]
                    src_ids = [b.block_id for b in new_computed_blocks]
                    # Replicate base attach EXCEPT touch(): arena blocks keep
                    # ref_cnt=1 (never re-enter the pool); the cached
                    # originals stay hash-registered + evictable, and the
                    # anonymous copies are never re-hashed (cache_blocks
                    # hashes only [num_cached_block, num_full)).
                    req_blocks.extend(dst)
                    self.num_cached_block[request_id] = len(req_blocks)
                    self.new_block_ids.extend(dst_ids)
                    _st["req"][request_id] = list(dst)
                    _st["pending"].append((dst_ids, src_ids, request_id, n))
                    _log(f"HIT2 req={request_id[:14]} n={n} "
                         f"tok={num_local_computed_tokens} "
                         f"src[{_runs_desc(src_ids)}] "
                         f"dst[{_runs_desc(dst_ids)}] "
                         f"af={len(_st['arena_free'])}")
                    return None
    except Exception as e:  # containment: fall back to the shared attach
        _log(f"COH2 fallback ({type(e).__name__}: {e}); shared attach")
    return base(self, request_id, new_computed_blocks,
                num_local_computed_tokens, num_external_computed_tokens)


def coh_free(base, self, request_id):
    """llm-scaler v24pc2: reclaim arena blocks, then base free."""
    try:
        mine = _st["req"].pop(request_id, None)
        if mine:
            blocks = self.req_to_blocks.get(request_id)
            if blocks:
                keep = [b for b in blocks
                        if b.block_id not in _st["arena_ids"]]
                if len(keep) != len(blocks):
                    blocks[:] = keep
            for b in mine:
                _st["arena_free"].append(b)
            ids = [b.block_id for b in blocks] if blocks else []
            _log(f"FREE2 req={request_id[:14]} table[{_runs_desc(ids)}] "
                 f"af={len(_st['arena_free'])}")
    except Exception as e:
        _log(f"coh_free err ({type(e).__name__}: {e})")
    return base(self, request_id)


def attach_stash(scheduler_output, scheduler=None):
    """llm-scaler v24pc2: build the arena early; stash copies for workers.

    The scheduler hook passes `scheduler` so the arena reserves from
    scheduler.kv_cache_manager.block_pool at the FIRST schedule() tick —
    before any prefix caching exists, so the 160-block pop evicts nothing.
    """
    if not _ENABLED:
        return
    try:
        if scheduler is not None:
            _ensure_arena(scheduler.kv_cache_manager.block_pool)
        if _st["pending"]:
            scheduler_output._v24pc2_copies = _st["pending"]
            _st["pending"] = []
    except Exception as e:
        _log(f"stash fail ({type(e).__name__}: {e})")


# ---- Worker side -----------------------------------------------------------

def _worker_meta(runner):
    if _w["meta"] is not None and _w["runner"] == id(runner):
        return _w["meta"]
    meta = []
    seen = set()
    try:
        kbs = runner._kernel_block_sizes
        for group in runner._kv_cache_spec_attn_group_iterator():
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kbs):
                continue
            kernel_bs = kbs[group.kv_cache_group_id]
            ratio = spec.block_size // kernel_bs
            block_dim = group.backend.get_kv_cache_block_dim(
                kernel_bs, spec.num_kv_heads, spec.head_size,
                cache_dtype_str=runner.cache_config.cache_dtype)
            for layer_name in group.layer_names:
                if layer_name in runner.runner_only_attn_layers:
                    continue
                kv = (runner.compilation_config
                      .static_forward_context[layer_name].kv_cache)
                if not isinstance(kv, torch.Tensor):
                    continue
                if kv.data_ptr() in seen:
                    continue
                seen.add(kv.data_ptr())
                meta.append((kv, block_dim, ratio))
    except Exception as e:
        _wlog(f"meta fail ({type(e).__name__}: {e})")
        meta = []
    _w["meta"] = meta or None
    _w["runner"] = id(runner)
    if _w["meta"] and not _w["shape_logged"]:
        _w["shape_logged"] = True
        t0, bd, r = _w["meta"][0]
        _wlog(f"meta tensors={len(_w['meta'])} shape0={tuple(t0.shape)} "
              f"bd={bd} ratio={r} dim{bd}={t0.shape[bd]}")
    return _w["meta"]


def drain_copies(runner, scheduler_output):
    """llm-scaler v24pc2: D2D-copy arena dst blocks from cached src blocks.

    Runs after the runner's new-block zeroing and before the step's forward
    kernels. Mapping mirrors KVBlockZeroer.init_meta: manager block id * ratio
    = kernel page offset along the tensor's block dim.
    """
    if not _ENABLED:
        return
    ev = getattr(scheduler_output, "_v24pc2_copies", None)
    if not ev:
        return
    meta = _worker_meta(runner)
    if not meta:
        _wlog("no full-attn kv tensors; drops pending")
        return
    groups = {}
    for t, bd, r in meta:
        groups.setdefault((bd, r), []).append(t)
    for dst_ids, src_ids, rid, n in ev:
        plan = []
        caps_ok = True
        pages = 0
        for (bd, r), ts in groups.items():
            dp = [d * r + k for d in dst_ids for k in range(r)]
            sp = [s * r + k for s in src_ids for k in range(r)]
            pages = len(dp)
            cap = ts[0].shape[bd]
            if max(dp) >= cap or max(sp) >= cap:
                _wlog(f"CAP FAIL bd={bd} cap={cap} maxd={max(dp)} "
                      f"maxs={max(sp)} req={rid[:14]} SKIP EVENT")
                caps_ok = False
                break
            plan.append((bd, ts, dp, sp))
        if not caps_ok:
            continue
        for bd, ts, dp, sp in plan:
            di = torch.tensor(dp, device=ts[0].device, dtype=torch.long)
            si = torch.tensor(sp, device=ts[0].device, dtype=torch.long)
            for t in ts:
                t.index_copy_(bd, di, t.index_select(bd, si))
        _wlog(f"drained req={rid[:14]} n={n} pages={pages} "
              f"groups={len(groups)} tensors={len(meta)}")
'''

# ---- P1: FullAttentionManager arena copy-on-hit overrides ------------------
P1_FILE = SP / "vllm/v1/core/single_type_kv_cache_manager.py"
P1_ANCHOR = "class FullAttentionManager(SingleTypeKVCacheManager):\n"
P1_REPLACEMENT = (
    "class FullAttentionManager(SingleTypeKVCacheManager):\n"
    "    # llm-scaler v24pc2: arena copy-on-hit (see vllm/_v24pc2.py)\n"
    "    def allocate_new_computed_blocks(\n"
    "        self,\n"
    "        request_id: str,\n"
    "        new_computed_blocks: Sequence[KVCacheBlock],\n"
    "        num_local_computed_tokens: int,\n"
    "        num_external_computed_tokens: int,\n"
    "    ) -> None:\n"
    "        from vllm import _v24pc2 as _v24pc2_mod  # llm-scaler v24pc2\n"
    "        return _v24pc2_mod.coh_allocate(\n"
    "            SingleTypeKVCacheManager.allocate_new_computed_blocks,\n"
    "            self,\n"
    "            request_id,\n"
    "            new_computed_blocks,\n"
    "            num_local_computed_tokens,\n"
    "            num_external_computed_tokens,\n"
    "        )\n"
    "\n"
    "    def free(self, request_id: str) -> None:\n"
    "        # llm-scaler v24pc2\n"
    "        from vllm import _v24pc2 as _v24pc2_mod\n"
    "        return _v24pc2_mod.coh_free(\n"
    "            SingleTypeKVCacheManager.free, self, request_id)\n"
    "\n"
)

# ---- P2: worker drain (after zeroing, before forward) ----------------------
P2_FILE = SP / "vllm/v1/worker/gpu_model_runner.py"
P2_ANCHOR = (
    "        if scheduler_output.new_block_ids_to_zero:\n"
    "            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)\n"
)
P2_REPLACEMENT = P2_ANCHOR + (
    "        # llm-scaler v24pc2: D2D arena copies (zero -> copy -> forward)\n"
    "        from vllm import _v24pc2 as _v24pc2_mod\n"
    "        _v24pc2_mod.drain_copies(self, scheduler_output)\n"
)

# ---- P3: SchedulerOutput stash field ---------------------------------------
P3_FILE = SP / "vllm/v1/core/sched/output.py"
P3_ANCHOR = "    new_block_ids_to_zero: list[int] | None = None\n"
P3_REPLACEMENT = (
    "    new_block_ids_to_zero: list[int] | None = None\n"
    "\n"
    "    # llm-scaler v24pc2: pending arena copy-on-hit (dst, src) id pairs.\n"
    "    _v24pc2_copies: list | None = None\n"
)

# ---- P4: scheduler-side stash attach ---------------------------------------
P4_FILE = SP / "vllm/v1/core/sched/scheduler.py"
P4_ANCHOR = "        return scheduler_output\n"
P4_REPLACEMENT = (
    "        # llm-scaler v24pc2\n"
    "        from vllm import _v24pc2 as _v24pc2_mod\n"
    "        _v24pc2_mod.attach_stash(scheduler_output, self)\n"
    "        return scheduler_output\n"
)


def apply_patch(path, anchor, replacement, tag):
    txt = path.read_text()
    if "llm-scaler v24pc2" in txt:
        print(f"{tag}: already patched, skip")
        return
    cnt = txt.count(anchor)
    if cnt != 1:
        print(f"{tag}: ANCHOR_COUNT={cnt} (need 1) ABORT")
        sys.exit(3)
    bak = path.with_suffix(path.suffix + ".v24pc2bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(txt.replace(anchor, replacement, 1))
    print(f"{tag}: patched (backup {bak.name})")


def main():
    if "--revert" in sys.argv:
        for f in (P1_FILE, P2_FILE, P3_FILE, P4_FILE):
            bak = f.with_suffix(f.suffix + ".v24pc2bak")
            if bak.exists():
                shutil.copy2(bak, f)
                print(f"reverted {f.name}")
            else:
                print(f"no backup for {f.name}")
        (SP / "vllm/_v24pc2.py").unlink(missing_ok=True)
        print("removed _v24pc2.py")
        print("V24PC2_REVERTED")
        return
    if "--check" in sys.argv:
        compile(MODULE, "vllm/_v24pc2.py", "exec")
        for path, anchor, tag in ((P1_FILE, P1_ANCHOR, "P1"),
                                  (P2_FILE, P2_ANCHOR, "P2"),
                                  (P3_FILE, P3_ANCHOR, "P3"),
                                  (P4_FILE, P4_ANCHOR, "P4")):
            txt = path.read_text()
            print(f"{tag}: anchor_count={txt.count(anchor)} "
                  f"already={'llm-scaler v24pc2' in txt}")
        print("CHECK_OK (module compiles)")
        return
    (SP / "vllm/_v24pc2.py").write_text(MODULE)
    print("installed vllm/_v24pc2.py")
    apply_patch(P1_FILE, P1_ANCHOR, P1_REPLACEMENT, "P1")
    apply_patch(P2_FILE, P2_ANCHOR, P2_REPLACEMENT, "P2")
    apply_patch(P3_FILE, P3_ANCHOR, P3_REPLACEMENT, "P3")
    apply_patch(P4_FILE, P4_ANCHOR, P4_REPLACEMENT, "P4")
    print("V24PC2_INSTALLED")


if __name__ == "__main__":
    main()
