#!/opt/venv/bin/python3
# llm-scaler v24r — §24 R BLOCK-TABLE DIAGNOSTIC boot patcher (read-only).
#
# Question (§24 Q follow-up): why does a warm 65k prefix-cache hit decode
# -15..17% when the cached blocks were themselves allocated as ONE
# contiguous run during cold prefill? Leading theory (i): the warm block
# TABLE is bimodal — [~1,154 shared blocks @ mid ids] + [~17 fresh tail
# blocks @ recycled low ids] — and the paged-attention kernel is
# contiguity-sensitive (penalty scales with prefix depth: 16k below the
# knee, 65k dipped, 128k volume-saturated). Theory (ii): kernel path
# switch at large KV regardless of table shape.
#
# This patch makes the engine answer directly: on every full-attention
# cache hit and at request free, dump the block-id table structure
# (first/last id, run count, gaps, max gap) plus the hit prefix and the
# fresh tail separately. NO behavior change — wraps call through to base.
#   P1 vllm/v1/core/single_type_kv_cache_manager.py:
#      FullAttentionManager.allocate_new_computed_blocks -> _v24r.dbg_allocate
#      FullAttentionManager.free                          -> _v24r.dbg_free
# Installs vllm/_v24r.py.
# Env: VLLM_V24R=0 kill-switch; log cap 300 lines.
# Usage: patch_v24r.py [--check | --revert]
import pathlib
import shutil
import sys

SP = pathlib.Path("/opt/venv/lib/python3.12/site-packages")

MODULE = r'''# llm-scaler v24r module (installed by patch_v24r.py) — table dump
import os

_ENABLED = os.environ.get("VLLM_V24R", "1") != "0"
_st = {"hits": {}, "logs": 0}


def _log(msg):
    if _st["logs"] < 300:
        _st["logs"] += 1
        print(f"[v24r] {msg}", flush=True)


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


def dbg_allocate(base, self, request_id, new_computed_blocks,
                 num_local_computed_tokens, num_external_computed_tokens):
    """llm-scaler v24r: dump cache-hit structure, then base attach."""
    n = len(new_computed_blocks)
    if n > 0 and _ENABLED:
        ids = [b.block_id for b in new_computed_blocks]
        _st["hits"][request_id] = n
        _log(f"HIT req={request_id} n={n} tok={num_local_computed_tokens} "
             f"{_runs_desc(ids)}")
    return base(self, request_id, new_computed_blocks,
                num_local_computed_tokens, num_external_computed_tokens)


def dbg_free(base, self, request_id):
    """llm-scaler v24r: dump the request's final block table, then free."""
    if _ENABLED:
        blocks = self.req_to_blocks.get(request_id)
        if blocks:
            ids = [b.block_id for b in blocks if not b.is_null]
            hit = _st["hits"].pop(request_id, 0)
            tail = ids[hit:] if 0 < hit <= len(ids) else []
            _log(f"FREE req={request_id} hit={hit} table[{_runs_desc(ids)}] "
                 f"tail[{_runs_desc(tail)}]")
        else:
            _st["hits"].pop(request_id, None)
    return base(self, request_id)
'''

# ---- P1: FullAttentionManager diagnostic overrides ------------------------
P1_FILE = SP / "vllm/v1/core/single_type_kv_cache_manager.py"
P1_ANCHOR = "class FullAttentionManager(SingleTypeKVCacheManager):\n"
P1_REPLACEMENT = (
    "class FullAttentionManager(SingleTypeKVCacheManager):\n"
    "    # llm-scaler v24r: block-table diagnostic (see vllm/_v24r.py)\n"
    "    def allocate_new_computed_blocks(\n"
    "        self,\n"
    "        request_id: str,\n"
    "        new_computed_blocks: Sequence[KVCacheBlock],\n"
    "        num_local_computed_tokens: int,\n"
    "        num_external_computed_tokens: int,\n"
    "    ) -> None:\n"
    "        from vllm import _v24r as _v24r_mod  # llm-scaler v24r\n"
    "        return _v24r_mod.dbg_allocate(\n"
    "            SingleTypeKVCacheManager.allocate_new_computed_blocks,\n"
    "            self,\n"
    "            request_id,\n"
    "            new_computed_blocks,\n"
    "            num_local_computed_tokens,\n"
    "            num_external_computed_tokens,\n"
    "        )\n"
    "\n"
    "    def free(self, request_id: str) -> None:\n"
    "        # llm-scaler v24r\n"
    "        from vllm import _v24r as _v24r_mod\n"
    "        return _v24r_mod.dbg_free(\n"
    "            SingleTypeKVCacheManager.free, self, request_id)\n"
    "\n"
)


def apply_patch(path, anchor, replacement, tag):
    txt = path.read_text()
    if "llm-scaler v24r" in txt:
        print(f"{tag}: already patched, skip")
        return
    cnt = txt.count(anchor)
    if cnt != 1:
        print(f"{tag}: ANCHOR_COUNT={cnt} (need 1) ABORT")
        sys.exit(3)
    bak = path.with_suffix(path.suffix + ".v24rbak")
    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(txt.replace(anchor, replacement, 1))
    print(f"{tag}: patched (backup {bak.name})")


def main():
    if "--revert" in sys.argv:
        bak = P1_FILE.with_suffix(P1_FILE.suffix + ".v24rbak")
        if bak.exists():
            shutil.copy2(bak, P1_FILE)
            print("reverted single_type_kv_cache_manager.py")
        else:
            print("no backup")
        (SP / "vllm/_v24r.py").unlink(missing_ok=True)
        print("removed _v24r.py")
        print("V24R_REVERTED")
        return
    if "--check" in sys.argv:
        compile(MODULE, "vllm/_v24r.py", "exec")
        txt = P1_FILE.read_text()
        print(f"P1: anchor_count={txt.count(P1_ANCHOR)} "
              f"already={'llm-scaler v24r' in txt}")
        print("CHECK_OK (module compiles)")
        return
    (SP / "vllm/_v24r.py").write_text(MODULE)
    print("installed vllm/_v24r.py")
    apply_patch(P1_FILE, P1_ANCHOR, P1_REPLACEMENT, "P1")
    print("V24R_INSTALLED")


if __name__ == "__main__":
    main()
