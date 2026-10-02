#!/usr/bin/env python3
"""patch_v132_ring.py — perf-v132 P64 scratch-ring segregation patcher.

Implements the v131-handed-forward churn-event-elimination lever as a
four-site surgery in vllm/v1/core/block_pool.py of the running lane
container (llm-scaler-exp:v1.2.27 posture).

Rationale (v131 CHURN-DESTRUCTION + RECENCY-PROTECTION LAWS): cached
mappings die at free-queue pop time in proportion to allocation EVENT
count — the per-step mamba rolling-state alloc/free pair rides the
free-queue conveyor, and every pop risks destroying the hash mapping
sitting at the queue front. v131 leg A proved pop-time REORDERING is
dead (soft-pop: +13% computed, HEAD_EVICTED). The ring therefore does
not touch queue ORDER at all — it SEGREGATES hash-free scratch:

  free_blocks:   blocks reaching ref_cnt==0 with block_hash is None
                 park in a bounded ring OUTSIDE the free queue
                 (cap VLLM_V132_RING); hash-bearing frees take the
                 queue exactly as stock (mappings survive, touch()
                 stays sound — ring blocks are never in the hash
                 table, so never matched, so never touched).
  get_new_blocks: drains the ring FIRST; only the remainder pops
                 the queue front. With ring-inclusive free-block
                 accounting, num_blocks - ring_served <= queue_len
                 always holds, so the stock popleft_n can never
                 over-pop (algebraic no-assert proof).
  get_num_free_blocks: queue length + ring length when armed.

Knob VLLM_V132_RING (default 0 = OFF): the stock lines run verbatim
on the knob-off path (one extra `if` test per call, no state).
Armed 1..4096 = ring capacity in blocks.

Telemetry: V132_RING_ARMED at import when armed; V132_RING
parked/served/capfull/ringlen/qlen every 4096 ring-served blocks
(the supply/demand readout P63 would otherwise have needed a
separate trace leg for).

Sites (anchors verified unique in the pristine container file,
matched through EOL, LF):
  S1  module knob block, after the module logger line
  S2  free_blocks body — the append_n comprehension (stock line
      kept verbatim as the knob-off branch)
  S3  get_new_blocks — the popleft_n assignment (stock line kept
      verbatim as the else branch)
  S4  get_num_free_blocks body — the return through EOL

Backs up to block_pool.py.v132rbak (first apply only). Idempotent
(V132_RING_ALREADY), compile-checked temp-write -> os.replace,
--restore copies the backup back. Optional positional target path
dry-tests against a copy (derived .v132rbak/.v132rtmp follow it).
Caps: V132_RING_OK markers=<n> bak=<path> ring_env=<value or unset>.
"""

import os
import py_compile
import shutil
import sys

PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/"
    "block_pool.py"
)
BAK = PATH + ".v132rbak"
TMP = PATH + ".v132rtmp"
MARKER = "# llm-scaler v132 RING"
EXPECTED_MARKERS = 4

# ---------------------------------------------------------------- S1
S1_ANCHOR = (
    "from vllm.v1.request import Request\n"
    "\n"
    "logger = init_logger(__name__)\n"
)

S1_INSERT = (
    "from vllm.v1.request import Request\n"
    "\n"
    "logger = init_logger(__name__)\n"
    "\n"
    "# llm-scaler v132 RING: scratch-ring segregation knob. Default\n"
    "# 0 = OFF -> the stock free/alloc/accounting lines in\n"
    "# free_blocks, get_new_blocks and get_num_free_blocks run\n"
    "# verbatim (see the RING blocks at each site). Armed via\n"
    "# VLLM_V132_RING=<cap 1..4096> in serve_user.sh: hash-free\n"
    "# freed blocks (the mamba rolling-state churn) park in a\n"
    "# bounded ring outside the free queue, and get_new_blocks\n"
    "# drains the ring before popping the queue — the queue keeps\n"
    "# pure LRU semantics for everything hash-bearing, and the\n"
    "# churn conveyor stops carrying cached chains into the popper.\n"
    "# Ring blocks are hash-free by construction: never in the\n"
    "# cached-hash table, never matched, never touch()ed.\n"
    "_os132 = __import__(\"os\")\n"
    "try:\n"
    "    _V132_RING_CAP = int(_os132.environ.get(\"VLLM_V132_RING\", \"0\"))\n"
    "except Exception:\n"
    "    _V132_RING_CAP = 0\n"
    "if not (1 <= _V132_RING_CAP <= 4096):\n"
    "    _V132_RING_CAP = 0\n"
    "_v132_deque = __import__(\"collections\").deque\n"
    "if _V132_RING_CAP > 0:\n"
    "    logger.info(\"V132_RING_ARMED cap=%d\", _V132_RING_CAP)\n"
)

# ---------------------------------------------------------------- S2
S2_ANCHOR = (
    "        # Materialize the iterable to allow multiple passes.\n"
    "        blocks_list = list(ordered_blocks)\n"
    "        for block in blocks_list:\n"
    "            block.ref_cnt -= 1\n"
    "        self.free_block_queue.append_n(\n"
    "            [block for block in blocks_list if block.ref_cnt == 0"
    " and not block.is_null]\n"
    "        )\n"
)

S2_INSERT = (
    "        # Materialize the iterable to allow multiple passes.\n"
    "        blocks_list = list(ordered_blocks)\n"
    "        for block in blocks_list:\n"
    "            block.ref_cnt -= 1\n"
    "        # llm-scaler v132 RING: park hash-free freed blocks in\n"
    "        # the scratch ring (outside the free queue) when armed;\n"
    "        # hash-bearing frees always take the queue exactly as\n"
    "        # stock so their mappings survive and touch() stays\n"
    "        # sound. Cap overflow (capfull) falls back to the queue.\n"
    "        if _V132_RING_CAP > 0 and self.enable_caching:\n"
    "            _v132_ring = getattr(self, \"_v132_ring\", None)\n"
    "            if _v132_ring is None:\n"
    "                _v132_ring = self._v132_ring = _v132_deque()\n"
    "            _v132_to_queue = []\n"
    "            for block in blocks_list:\n"
    "                if block.ref_cnt == 0 and not block.is_null:\n"
    "                    if (\n"
    "                        block.block_hash is None\n"
    "                        and len(_v132_ring) < _V132_RING_CAP\n"
    "                    ):\n"
    "                        _v132_ring.append(block)\n"
    "                        self._v132_parked = (\n"
    "                            getattr(self, \"_v132_parked\", 0) + 1\n"
    "                        )\n"
    "                    else:\n"
    "                        if block.block_hash is None:\n"
    "                            self._v132_capfull = (\n"
    "                                getattr(self, \"_v132_capfull\", 0) + 1\n"
    "                            )\n"
    "                        _v132_to_queue.append(block)\n"
    "            self.free_block_queue.append_n(_v132_to_queue)\n"
    "        else:\n"
    "            self.free_block_queue.append_n(\n"
    "                [block for block in blocks_list if block.ref_cnt == 0"
    " and not block.is_null]\n"
    "            )\n"
)

# ---------------------------------------------------------------- S3
S3_ANCHOR = (
    "        ret: list[KVCacheBlock] = "
    "self.free_block_queue.popleft_n(num_blocks)\n"
)

S3_INSERT = (
    "        # llm-scaler v132 RING: drain the scratch ring first;\n"
    "        # only the remainder pops the free-queue front. Ring\n"
    "        # blocks are hash-free (see free_blocks), so the\n"
    "        # stock tail loop below is a no-op eviction check for\n"
    "        # them, exactly like stock hash-free front pops. With\n"
    "        # the ring counted in get_num_free_blocks, the guard\n"
    "        # above passed for queue+ring, and ring_served >=\n"
    "        # num_blocks - queue_len holds, so the remainder pop\n"
    "        # can never over-pop the queue.\n"
    "        if (\n"
    "            _V132_RING_CAP > 0\n"
    "            and self.enable_caching\n"
    "            and num_blocks > 0\n"
    "        ):\n"
    "            _v132_ring = getattr(self, \"_v132_ring\", None)\n"
    "            if _v132_ring is None:\n"
    "                _v132_ring = self._v132_ring = _v132_deque()\n"
    "            _v132_take = num_blocks\n"
    "            if _v132_take > len(_v132_ring):\n"
    "                _v132_take = len(_v132_ring)\n"
    "            _v132_ring_blocks = []\n"
    "            for _ in range(_v132_take):\n"
    "                _v132_ring_blocks.append(_v132_ring.popleft())\n"
    "            num_blocks -= _v132_take\n"
    "            if _v132_take > 0:\n"
    "                self._v132_served = (\n"
    "                    getattr(self, \"_v132_served\", 0) + _v132_take\n"
    "                )\n"
    "                if self._v132_served % 4096 == 1:\n"
    "                    logger.info(\n"
    "                        \"V132_RING parked=%d served=%d capfull=%d\"\n"
    "                        \" ringlen=%d qlen=%d\",\n"
    "                        getattr(self, \"_v132_parked\", 0),\n"
    "                        self._v132_served,\n"
    "                        getattr(self, \"_v132_capfull\", 0),\n"
    "                        len(_v132_ring),\n"
    "                        self.free_block_queue.num_free_blocks,\n"
    "                    )\n"
    "            if num_blocks > 0:\n"
    "                ret = _v132_ring_blocks + "
    "self.free_block_queue.popleft_n(num_blocks)\n"
    "            else:\n"
    "                ret = _v132_ring_blocks\n"
    "        else:\n"
    "            ret: list[KVCacheBlock] = "
    "self.free_block_queue.popleft_n(num_blocks)\n"
)

# ---------------------------------------------------------------- S4
S4_ANCHOR = (
    "    def get_num_free_blocks(self) -> int:\n"
    "        \"\"\"Get the number of free blocks in the pool.\n"
    "\n"
    "        Returns:\n"
    "            The number of free blocks.\n"
    "        \"\"\"\n"
    "        return self.free_block_queue.num_free_blocks\n"
)

S4_INSERT = (
    "    def get_num_free_blocks(self) -> int:\n"
    "        \"\"\"Get the number of free blocks in the pool.\n"
    "\n"
    "        Returns:\n"
    "            The number of free blocks.\n"
    "        \"\"\"\n"
    "        # llm-scaler v132 RING: ring-parked blocks are free —\n"
    "        # count them so every capacity check (the ValueError\n"
    "        # guard in get_new_blocks, the used-block accounting,\n"
    "        # the scheduler's allocation headroom) keeps its exact\n"
    "        # stock semantics.\n"
    "        if _V132_RING_CAP > 0:\n"
    "            _v132_ring = getattr(self, \"_v132_ring\", None)\n"
    "            if _v132_ring:\n"
    "                return (\n"
    "                    self.free_block_queue.num_free_blocks\n"
    "                    + len(_v132_ring)\n"
    "                )\n"
    "        return self.free_block_queue.num_free_blocks\n"
)


def compile_check(path: str) -> None:
    py_compile.compile(path, doraise=True)


def restore() -> int:
    if not os.path.exists(BAK):
        print("V132_RING_RESTORE_FAIL no_backup=%s" % BAK)
        return 1
    shutil.copy2(BAK, TMP)
    compile_check(TMP)
    st = os.stat(BAK)
    os.chmod(TMP, st.st_mode & 0o7777)
    os.replace(TMP, PATH)
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    print("V132_RING_RESTORED markers_left=%d" % src.count(MARKER))
    return 0


def apply() -> int:
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        print("V132_RING_ALREADY marker_count=%d" % src.count(MARKER))
        return 0
    for name, anchor in (
        ("S1", S1_ANCHOR),
        ("S2", S2_ANCHOR),
        ("S3", S3_ANCHOR),
        ("S4", S4_ANCHOR),
    ):
        n = src.count(anchor)
        if n != 1:
            print("V132_RING_FAIL anchor_%s count=%d (need 1)" % (name, n))
            return 1
    if not os.path.exists(BAK):
        shutil.copy2(PATH, BAK)
    out = src.replace(S4_ANCHOR, S4_INSERT)
    out = out.replace(S3_ANCHOR, S3_INSERT)
    out = out.replace(S2_ANCHOR, S2_INSERT)
    out = out.replace(S1_ANCHOR, S1_INSERT)
    n_markers = out.count(MARKER)
    if n_markers != EXPECTED_MARKERS:
        print(
            "V132_RING_FAIL markers=%d (need %d)"
            % (n_markers, EXPECTED_MARKERS)
        )
        return 1
    with open(TMP, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    compile_check(TMP)
    st = os.stat(PATH)
    os.chmod(TMP, st.st_mode & 0o7777)
    os.replace(TMP, PATH)
    print(
        "V132_RING_OK markers=%d bak=%s ring_env=%r"
        % (
            n_markers,
            BAK,
            os.environ.get("VLLM_V132_RING", "unset"),
        )
    )
    return 0


if __name__ == "__main__":
    _args = sys.argv[1:]
    # Optional positional target path (dry-test against a copy; the
    # derived .v132rbak/.v132rtmp follow it). Default = lane target.
    for _a in _args:
        if not _a.startswith("-"):
            PATH = _a
            BAK = _a + ".v132rbak"
            TMP = _a + ".v132rtmp"
    if "--restore" in _args:
        sys.exit(restore())
    sys.exit(apply())
