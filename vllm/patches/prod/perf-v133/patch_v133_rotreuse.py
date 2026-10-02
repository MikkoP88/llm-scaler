#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# llm-scaler perf-v133 P68 — SELF-SERVICE ROTATION PAIR patcher.
#
# Patches vllm/v1/core/single_type_kv_cache_manager.py (v1.2.27 lane
# image, vLLM v0.21.1.dev0+gad7125a43) with the alloc-side mamba
# rotation-reuse lever sealed in P67:
#
#   FREE  (M3, MambaManager.remove_skipped_blocks): the two-steps-ago
#   state block takes the COMPLETELY STOCK free path (free_block_queue
#   tail, ref_cnt 0, mapping alive — an ordinary free member in every
#   respect) and its identity is parked in a per-request dict.
#
#   ALLOC (M4, MambaManager.allocate_new_blocks): when armed and this
#   call needs exactly one steady block (blocks_allocated and
#   num_new_blocks == 1 — the spec-block dance guarantees <= 1 per
#   call at any chunk size), the parked block is consumed by
#   IDENTITY-REMOVING it from the free queue (the same O(1) unlink
#   primitive touch() uses) instead of popping the front, then given
#   the exact stock pop-time treatment: _maybe_evict_cached_block
#   destroys the block's OWN spent mapping, assert ref_cnt == 0,
#   ref_cnt += 1, metrics on_block_allocated.
#
#   Only delta vs stock: WHICH mapping dies — the rotation's own spent
#   state block instead of whatever innocent cached chain sits at the
#   queue front (CHURN-DESTRUCTION LAW: mapping death tracks popped-
#   block count). LRU order for every other block untouched
#   (RECENCY-PROTECTION LAW); free-block accounting stock at every
#   instant (the parked block IS a queue member while parked, so no
#   v132-ring-style accounting surgery is needed).
#
#   Guards: ref_cnt == 0, live queue membership (prev/next_free_block
#   not None — remove()/popleft*() clear the links of non-members),
#   not null. Any guard failure routes to the stock call (counter
#   stock_guard); first allocs (1 + num_speculative_blocks) are always
#   stock (counter stock_first). If the parked block is matched
#   (touched) between park and consume, ref_cnt != 0 fails the guard,
#   the match wins, and the alloc takes the stock path.
#
#   Match-economy safety (P67): resend matches need mamba state
#   mappings alive at the LAST aligned boundary — those blocks free at
#   request end via the wholesale path (base free, untouched here).
#   Rotation-freed intermediate states are spent currency; consuming
#   them one step early cannot degrade the match economy.
#
# Knob: VLLM_V133_ROTREUSE (read once at module import; EngineCore
# inherits serve_user.sh env — same class as V1227_PERREQ).
#   0 / unset = OFF: every patched branch falls through to the stock
#   lines verbatim (quiet-inert, one extra `if` per site).
#   1 = armed. ARMED + V133_ROT telemetry lines via vllm logger.
#
# Telemetry: "V133_ROT_ARMED" at MambaManager init; "V133_ROT
# reused=... stock_first=... stock_guard=... parked=... parks=...
# freeq=..." on the first reuse and every 256 after (first line =
# arming proof within one alloc step of the first park+consume).
#
# Usage:
#   patch_v133_rotreuse.py            # patch the in-container path
#   patch_v133_rotreuse.py PATH       # patch an explicit file (dry-cert)
#   patch_v133_rotreuse.py --restore [PATH]   # restore .v133rbak
#
# Idempotent (re-run prints ALREADY and exits 0). Backup .v133rbak is
# written only if absent (it always holds the PRE-patch pristine).
# Patched text is py_compile-checked on a temp file before an atomic
# os.replace — the target is never left half-written.

import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/"
    "single_type_kv_cache_manager.py"
)
MARKER = "# llm-scaler v133 ROTREUSE"
EXPECTED_MARKERS = 5
BACKUP_SUFFIX = ".v133rbak"

# --- M1: module knob + logger (after imports, before the ABC) ------
A1_ANCHOR = '''from vllm.v1.request import Request


class SingleTypeKVCacheManager(ABC):
'''
A1_REPL = '''from vllm.v1.request import Request

# llm-scaler v133 ROTREUSE: alloc-side mamba rotation reuse knob.
# Default 0 = OFF -> every patched branch below falls through to the
# stock free/alloc lines verbatim. VLLM_V133_ROTREUSE=1 arms the
# self-service rotation pair in MambaManager (see the M3/M4 comments).
import os as _v133_os

_v133_rotreuse = 0
try:
    _v133_rotreuse = int(_v133_os.environ.get("VLLM_V133_ROTREUSE", "0"))
except Exception:
    _v133_rotreuse = 0
if _v133_rotreuse not in (0, 1):
    _v133_rotreuse = 0

from vllm.logger import init_logger as _v133_init_logger

_v133_logger = _v133_init_logger(__name__)


class SingleTypeKVCacheManager(ABC):
'''

# --- M2: MambaManager.__init__ align branch — parked dict + counters
A2_ANCHOR = '''        if self.mamba_cache_mode == "align":
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()
'''
A2_REPL = '''        if self.mamba_cache_mode == "align":
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()
            # llm-scaler v133 ROTREUSE: per-request parked rotation
            # block (freed by remove_skipped_blocks at this call or an
            # earlier one; consumed by the matching steady alloc) and
            # readout counters. A parked block is ALWAYS a normal
            # free-queue member (the stock free path ran at park
            # time); this dict holds only its identity.
            self._v133_parked: dict[str, KVCacheBlock] = {}
            self._v133_reused = 0
            self._v133_stock_first = 0
            self._v133_stock_guard = 0
            self._v133_park_total = 0
            if _v133_rotreuse > 0:
                _v133_logger.info(
                    "V133_ROT_ARMED mode=%s spec=%d",
                    self.mamba_cache_mode,
                    self.num_speculative_blocks,
                )
'''

# --- M3: remove_skipped_blocks — park + stock free VERBATIM ---------
A3_ANCHOR = '''                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block
'''
A3_REPL = '''                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    # llm-scaler v133 ROTREUSE: park the identity,
                    # then run the stock free VERBATIM. The block
                    # becomes an ordinary free-queue member (tail,
                    # ref_cnt 0, mapping alive). If a later park
                    # overwrites an unconsumed entry, the old block is
                    # already a normal queue member - no leak, no held
                    # ref_cnt, nothing to unwind.
                    if _v133_rotreuse > 0:
                        self._v133_parked[request_id] = blocks[last_state_block_idx]
                        self._v133_park_total += 1
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block
'''

# --- M4: allocate_new_blocks — consume the parked block -------------
A4_ANCHOR = '''                num_new_blocks = num_required_blocks - len(req_blocks)
                if blocks_allocated:
                    assert num_new_blocks <= 1
                else:
                    assert num_new_blocks <= self.num_speculative_blocks + 1
                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
                req_blocks.extend(new_blocks)
'''
A4_REPL = '''                num_new_blocks = num_required_blocks - len(req_blocks)
                if blocks_allocated:
                    assert num_new_blocks <= 1
                else:
                    assert num_new_blocks <= self.num_speculative_blocks + 1
                # llm-scaler v133 ROTREUSE: the steady rotation alloc.
                # When armed and this call needs exactly one block,
                # consume this request's parked block instead of
                # popping the queue front: identity-remove it (the O(1)
                # unlink touch() uses; remove() is loud on
                # non-members), then apply the exact stock pop-time
                # treatment - _maybe_evict_cached_block destroys the
                # block's OWN spent mapping, ref_cnt 0->1,
                # on_block_allocated. Queue order and accounting for
                # every other block are untouched; the only delta is
                # which mapping dies. Guards (ref_cnt == 0, live queue
                # membership, not null) route any exotic case to the
                # stock call; first allocs (1+spec) are always stock.
                # If the parked block was matched (touched) between
                # park and consume, the ref_cnt guard fails, the match
                # wins, and this takes the stock path.
                _v133_blk = None
                if (
                    _v133_rotreuse > 0
                    and blocks_allocated
                    and num_new_blocks == 1
                ):
                    _v133_blk = self._v133_parked.pop(request_id, None)
                    if not (
                        _v133_blk is not None
                        and _v133_blk.ref_cnt == 0
                        and not _v133_blk.is_null
                        and _v133_blk.prev_free_block is not None
                        and _v133_blk.next_free_block is not None
                    ):
                        _v133_blk = None
                        self._v133_stock_guard += 1
                if _v133_blk is not None:
                    self.block_pool.free_block_queue.remove(_v133_blk)
                    if self.block_pool.enable_caching:
                        self.block_pool._maybe_evict_cached_block(_v133_blk)
                    assert _v133_blk.ref_cnt == 0
                    _v133_blk.ref_cnt += 1
                    if self.block_pool.metrics_collector:
                        self.block_pool.metrics_collector.on_block_allocated(
                            _v133_blk
                        )
                    new_blocks = [_v133_blk]
                    self._v133_reused += 1
                    if self._v133_reused % 256 == 1:
                        _v133_logger.info(
                            "V133_ROT reused=%d stock_first=%d "
                            "stock_guard=%d parked=%d parks=%d freeq=%d",
                            self._v133_reused,
                            self._v133_stock_first,
                            self._v133_stock_guard,
                            len(self._v133_parked),
                            self._v133_park_total,
                            self.block_pool.get_num_free_blocks(),
                        )
                else:
                    if _v133_rotreuse > 0 and not blocks_allocated:
                        self._v133_stock_first += 1
                    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
                req_blocks.extend(new_blocks)
'''

# --- M5: free — drop any unconsumed parked entry --------------------
A5_ANCHOR = '''    def free(self, request_id: str) -> None:
        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
        super().free(request_id)
'''
A5_REPL = '''    def free(self, request_id: str) -> None:
        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
            # llm-scaler v133 ROTREUSE: drop any unconsumed parked
            # entry at request end / preemption. The block itself is
            # already a normal free-queue member (the stock free path
            # ran at park time) - zero pool-side cleanup needed.
            self._v133_parked.pop(request_id, None)
        super().free(request_id)
'''

SITES = [
    ("M1 module knob + logger", A1_ANCHOR, A1_REPL),
    ("M2 init parked dict + counters", A2_ANCHOR, A2_REPL),
    ("M3 park (remove_skipped_blocks)", A3_ANCHOR, A3_REPL),
    ("M4 consume (allocate_new_blocks)", A4_ANCHOR, A4_REPL),
    ("M5 free (drop parked entry)", A5_ANCHOR, A5_REPL),
]

# The anchors must match the image file byte-exactly. Normalize away
# any accidental CRLF so a Windows-side edit of THIS script cannot
# desynchronize the templates from the LF target file.
for _i, (_name, _anchor, _repl) in enumerate(SITES):
    SITES[_i] = (
        _name,
        _anchor.replace("\r\n", "\n"),
        _repl.replace("\r\n", "\n"),
    )


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def apply_patch(path: str) -> int:
    raw = read_bytes(path)
    src = raw.decode("utf-8")
    if MARKER in src:
        n = src.count(MARKER)
        if n != EXPECTED_MARKERS:
            print(
                f"V133 FAIL: {path} already carries {n} markers, "
                f"expected {EXPECTED_MARKERS} - refusing to touch"
            )
            return 2
        print(f"V133 ALREADY: {path} markers={n}")
        return 0

    backup = path + BACKUP_SUFFIX
    if not os.path.exists(backup):
        with open(backup, "wb") as f:
            f.write(raw)
        print(f"V133 backup written: {backup} (md5 {md5(raw)})")

    patched = src
    for name, anchor, repl in SITES:
        cnt = patched.count(anchor)
        if cnt != 1:
            print(
                f"V133 FAIL: anchor {name} matches {cnt} times "
                f"(need exactly 1) in {path} - aborting, file untouched"
            )
            return 3
        patched = patched.replace(anchor, repl, 1)

    n = patched.count(MARKER)
    if n != EXPECTED_MARKERS:
        print(
            f"V133 FAIL: post-patch marker count {n} != "
            f"{EXPECTED_MARKERS} - aborting, file untouched"
        )
        return 4

    # Compile-check the patched text on a temp file, then swap in
    # atomically so the target is never half-written.
    fd, tmp = tempfile.mkstemp(
        suffix=".py", prefix="v133_patch_", dir=os.path.dirname(path) or "."
    )
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(patched.encode("utf-8"))
        py_compile.compile(tmp, doraise=True)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - report and stay pristine
        if os.path.exists(tmp):
            os.unlink(tmp)
        print(f"V133 FAIL: compile/replace error: {exc!r} - file untouched")
        return 5

    print(
        f"V133 PATCHED: {path} markers={n} "
        f"md5 {md5(raw)} -> {md5(patched.encode('utf-8'))} "
        f"size {len(raw)} -> {len(patched.encode('utf-8'))}"
    )
    return 0


def restore(path: str) -> int:
    backup = path + BACKUP_SUFFIX
    if not os.path.exists(backup):
        print(f"V133 RESTORE FAIL: no backup at {backup}")
        return 6
    braw = read_bytes(backup)
    shutil.copyfile(backup, path)
    got = read_bytes(path)
    if md5(got) != md5(braw):
        print(f"V133 RESTORE FAIL: md5 mismatch after copy for {path}")
        return 7
    print(f"V133 RESTORED: {path} from backup (md5 {md5(got)})")
    return 0


def main() -> int:
    args = [a for a in sys.argv[1:]]
    do_restore = "--restore" in args
    target = PATH
    for a in args:
        if not a.startswith("--"):
            target = a
    if do_restore:
        return restore(target)
    return apply_patch(target)


if __name__ == "__main__":
    sys.exit(main())
