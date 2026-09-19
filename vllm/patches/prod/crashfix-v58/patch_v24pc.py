#!/opt/venv/bin/python3
# llm-scaler v24pc — §24 Q COPY-ON-HIT (COH) boot patcher (prototype).
#
# Problem (§24 M/P evidence): with --enable-prefix-caching, a warm 65k-ctx
# cache hit makes the request decode route AROUND ~1.1k pinned cached blocks
# -> scattered block tables -> gather-bound KV reads -> decode -15..17%
# (301-306 vs 360-362 tok/s vs cold). PC OFF kills the dip but costs 12.4x
# cache-eligible TTFT (65k warm 4.6s -> 57s).
#
# Fix: on a full-attention prefix hit >= VLLM_COH_MIN_BLOCKS blocks, allocate
# FRESH blocks (front of free queue -> compact run) and D2D-copy the cached
# KV into them ONCE at the next execute_model entry (bitwise; stream-ordered
# before the forward kernels). The request then decodes from a clean layout
# (dip gone by construction) while keeping ~all of the PC TTFT win (copy of
# ~2k blocks x N layers x 64 tok x KV bytes = single-digit GB split TP2,
# tens of ms, vs 47-110s prefill).
#
# Bookkeeping (verified against pinned tree 0.21.1.dev0+gad7125a43.d20260826):
#   - fresh blocks carry ref_cnt=1 from BlockPool.get_new_blocks; the base
#     attach's touch() would double-ref and LEAK them at free() -> the COH
#     override replicates the attach (req_blocks.extend + num_cached_block)
#     WITHOUT touch(). free() then returns them cleanly to the pool.
#   - cache_blocks() hashes only blocks [num_cached_block, num_full) -> the
#     anonymous copies are never re-registered; original cached blocks stay
#     hash-registered + evictable (deliberately NOT touch()-ed).
#   - new_block_ids extended (mirrors allocate_new_blocks for FullAttention).
#   - mamba/linear-attention state tensors are never copied: drain selects
#     tensors by shape signature (ndim==4 and shape[0]==2) = K/V pairs only.
#
# Patches:
#   P1 vllm/v1/core/single_type_kv_cache_manager.py:
#      FullAttentionManager.allocate_new_computed_blocks override -> _v24pc.
#   P2 vllm/v1/worker/gpu_model_runner.py: drain pending copies at
#      execute_model entry (right after the f15b execute_model mark).
# Installs vllm/_v24pc.py (pending-copy queue + both hooks).
#
# Env kill-switch/knobs: VLLM_COH=0 off; VLLM_COH_MIN_BLOCKS (default 16
# blocks = 1024 tok @ block-size 64); VLLM_COH_FREE_MARGIN (default 64).
# Usage: patch_v24pc.py [--check | --revert]
#   --check : read-only dry-run (compile module text, verify anchors)
#   --revert: restore *.v24pcbak backups, remove _v24pc.py
import pathlib
import shutil
import sys

SP = pathlib.Path("/opt/venv/lib/python3.12/site-packages")

MODULE = r'''# llm-scaler v24pc module (installed by patch_v24pc.py) — copy-on-hit
import os

import torch

_ENABLED = os.environ.get("VLLM_COH", "1") != "0"
_MIN_BLOCKS = int(os.environ.get("VLLM_COH_MIN_BLOCKS", "16"))
_FREE_MARGIN = int(os.environ.get("VLLM_COH_FREE_MARGIN", "64"))

_st = {"pending": [], "fa": None, "kv_id": None, "fa_shape": None, "logs": 0}


def _log(msg):
    if _st["logs"] < 40:
        _st["logs"] += 1
        print(f"[v24pc] {msg}", flush=True)


def coh_allocate(base, self, request_id, new_computed_blocks,
                 num_local_computed_tokens, num_external_computed_tokens):
    """llm-scaler v24pc: copy-on-hit for FullAttentionManager.

    Swaps a >= threshold prefix-cache hit for freshly allocated blocks plus
    a one-shot D2D copy (queued here, executed by drain_copies in the model
    runner). Falls back to the base shared-block attach on any doubt.
    """
    if not _ENABLED or request_id in self.num_cached_block:
        # kill-switch, or base's running-request fast path (asserts empty)
        return base(self, request_id, new_computed_blocks,
                    num_local_computed_tokens, num_external_computed_tokens)
    try:
        n = len(new_computed_blocks)
        if (n >= _MIN_BLOCKS
                and num_external_computed_tokens == 0
                and self.enable_caching
                and self.block_pool.get_num_free_blocks() >= n + _FREE_MARGIN):
            req_blocks = self.req_to_blocks[request_id]
            if len(req_blocks) == 0:
                fresh = self.block_pool.get_new_blocks(n)
                src_ids = [b.block_id for b in new_computed_blocks]
                dst_ids = [b.block_id for b in fresh]
                if set(dst_ids) & set(src_ids):
                    # Pathological: pool front popped the cached blocks
                    # themselves. Free the duplicates and attach originals.
                    self.block_pool.free_blocks(fresh)
                else:
                    # Replicate base attach EXCEPT touch(): fresh blocks
                    # already carry ref_cnt=1 from get_new_blocks; touch()
                    # would double-ref and leak them at free() time. The
                    # cached originals stay hash-registered + evictable.
                    req_blocks.extend(fresh)
                    self.num_cached_block[request_id] = len(req_blocks)
                    self.new_block_ids.extend(dst_ids)
                    _st["pending"].append((dst_ids, src_ids, request_id, n))
                    _log(f"copy-on-hit req={request_id} blocks={n} "
                         f"src0={src_ids[0]} dst0={dst_ids[0]} "
                         f"free={self.block_pool.get_num_free_blocks()}")
                    return None
    except Exception as e:  # prototype containment: fall back to shared
        _log(f"COH fallback ({type(e).__name__}: {e}); shared attach")
    return base(self, request_id, new_computed_blocks,
                num_local_computed_tokens, num_external_computed_tokens)


def drain_copies(runner):
    """llm-scaler v24pc: D2D-copy pending COH blocks.

    Called at execute_model entry — stream-ordered before the forward
    kernels, so the copied KV is in place before any attention reads it.
    """
    if not _st["pending"] or not _ENABLED:
        return
    kv = getattr(runner, "kv_caches", None)
    if not kv:
        return
    if _st["fa"] is None or _st["kv_id"] != id(kv):
        fa = [t for t in kv
              if isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[0] == 2]
        _st["fa"] = fa
        _st["kv_id"] = id(kv)
        _st["fa_shape"] = tuple(fa[0].shape) if fa else None
        _log(f"kv_caches={len(kv)} full_attn_tensors={len(fa)} "
             f"shape={_st['fa_shape']}")
    fa = _st["fa"]
    if not fa:
        _st["pending"].clear()
        return
    dev = fa[0].device
    for dst_ids, src_ids, rid, n in _st["pending"]:
        dst = torch.tensor(dst_ids, device=dev, dtype=torch.long)
        src = torch.tensor(src_ids, device=dev, dtype=torch.long)
        for t in fa:
            t[:, dst] = t[:, src]  # bitwise D2D (K and V), stream-ordered
        _log(f"drained {n} blocks req={rid} tensors={len(fa)}")
    _st["pending"].clear()
'''

# ---- P1: FullAttentionManager copy-on-hit override ------------------------
P1_FILE = SP / "vllm/v1/core/single_type_kv_cache_manager.py"
P1_ANCHOR = "class FullAttentionManager(SingleTypeKVCacheManager):\n"
P1_REPLACEMENT = (
    "class FullAttentionManager(SingleTypeKVCacheManager):\n"
    "    # llm-scaler v24pc: copy-on-hit override (see vllm/_v24pc.py)\n"
    "    def allocate_new_computed_blocks(\n"
    "        self,\n"
    "        request_id: str,\n"
    "        new_computed_blocks: Sequence[KVCacheBlock],\n"
    "        num_local_computed_tokens: int,\n"
    "        num_external_computed_tokens: int,\n"
    "    ) -> None:\n"
    "        from vllm import _v24pc as _v24pc_mod  # llm-scaler v24pc\n"
    "        return _v24pc_mod.coh_allocate(\n"
    "            SingleTypeKVCacheManager.allocate_new_computed_blocks,\n"
    "            self,\n"
    "            request_id,\n"
    "            new_computed_blocks,\n"
    "            num_local_computed_tokens,\n"
    "            num_external_computed_tokens,\n"
    "        )\n"
    "\n"
)

# ---- P2: drain pending copies at execute_model entry ----------------------
P2_FILE = SP / "vllm/v1/worker/gpu_model_runner.py"
P2_ANCHOR = (
    '        from vllm import _f15b as _f15b_mod  # llm-scaler f15b\n'
    '        _f15b_mod.mark("execute_model")\n'
)
P2_REPLACEMENT = P2_ANCHOR + (
    '        from vllm import _v24pc as _v24pc_mod  # llm-scaler v24pc\n'
    '        _v24pc_mod.drain_copies(self)\n'
)


def apply_patch(path, anchor, replacement, tag):
    txt = path.read_text()
    if "llm-scaler v24pc" in txt:
        print(f"{tag}: already patched, skip")
        return
    cnt = txt.count(anchor)
    if cnt != 1:
        print(f"{tag}: ANCHOR_COUNT={cnt} (need 1) ABORT")
        sys.exit(3)
    bak = path.with_suffix(path.suffix + ".v24pcbak")
    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(txt.replace(anchor, replacement, 1))
    print(f"{tag}: patched (backup {bak.name})")


def main():
    if "--revert" in sys.argv:
        for f in (P1_FILE, P2_FILE):
            bak = f.with_suffix(f.suffix + ".v24pcbak")
            if bak.exists():
                shutil.copy2(bak, f)
                print(f"reverted {f.name}")
            else:
                print(f"no backup for {f.name}")
        mod = SP / "vllm/_v24pc.py"
        mod.unlink(missing_ok=True)
        print("removed _v24pc.py")
        print("V24PC_REVERTED")
        return
    if "--check" in sys.argv:
        compile(MODULE, "vllm/_v24pc.py", "exec")
        for path, anchor, tag in ((P1_FILE, P1_ANCHOR, "P1"),
                                  (P2_FILE, P2_ANCHOR, "P2")):
            txt = path.read_text()
            print(f"{tag}: anchor_count={txt.count(anchor)} "
                  f"already={'llm-scaler v24pc' in txt}")
        print("CHECK_OK (module compiles)")
        return
    (SP / "vllm/_v24pc.py").write_text(MODULE)
    print("installed vllm/_v24pc.py")
    apply_patch(P1_FILE, P1_ANCHOR, P1_REPLACEMENT, "P1")
    apply_patch(P2_FILE, P2_ANCHOR, P2_REPLACEMENT, "P2")
    print("V24PC_INSTALLED")


if __name__ == "__main__":
    main()
