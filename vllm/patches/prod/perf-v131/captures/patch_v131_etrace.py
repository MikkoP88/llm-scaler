#!/usr/bin/env python3
"""patch_v131_etrace.py — perf-v131 P59 eviction-count patcher.

Instruments vllm/v1/core/block_pool.py of the running lane
container (llm-scaler-exp:v1.2.27 posture). Same knob as the MTRACE
patcher (VLLM_V131_TRACE, default 0 = OFF -> one false `if` on the
allocation fast path and one on the eviction path; no state, no
behavior delta).

Sites (anchors verified unique in-container, through EOL, LF):
  S1  module knob block, after the module logger line
  S2  eviction counter, after block.reset_hash() inside
      _maybe_evict_cached_block (anchored on the distinctive
      pop(...) is None guard — bare reset_hash() is NOT unique)
  S3  rate-limited snapshot, before the `return ret` of
      get_new_blocks' non-caching branch tail (anchored on the
      else-branch loop body — on_block_allocated appears twice)

S2/S3 semantics — the V131_EVICT line fires whenever the running
eviction total crosses a fresh bucket of 32 destroyed hash
mappings:
  ev_total  cumulative evictions since boot (hash-bearing blocks
            whose mapping was destroyed at free-queue pop time or
            via the explicit evict_blocks path)
  freeq     free-queue length AFTER this allocation
  new       blocks this allocation consumed

Silence (no V131_EVICT lines) + V131_MATCH hit=0/kmiss=0 would be
the smoking gun for never-inserted hashes; eviction bursts on the
storm timeline paired with kmiss=0 admissions are hypothesis (a)
evicted-before-resend.

Idempotent: exits V131_ETRACE_ALREADY if the marker is present.
Backs up to block_pool.py.v131bak (first apply only). Compile-
checks a temp copy before os.replace; --restore copies the backup
back (same discipline). Optional positional target path dry-tests
against a copy (derived .v131bak/.v131tmp follow it).

Caps line on success: V131_ETRACE_OK markers=<n> bak=<path>
"""

import os
import py_compile
import shutil
import sys

PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/"
    "block_pool.py"
)
BAK = PATH + ".v131bak"
TMP = PATH + ".v131tmp"
MARKER = "# llm-scaler v131 ETRACE"
EXPECTED_MARKERS = 3

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
    "# llm-scaler v131 ETRACE: eviction-count knob (see the counter in\n"
    "# _maybe_evict_cached_block and the snapshot in get_new_blocks).\n"
    "# Default 0 = OFF -> allocation/eviction behavior byte-identical.\n"
    "# Armed via VLLM_V131_TRACE=1 in serve_user.sh (same knob as the\n"
    "# kv_cache_manager MTRACE patcher; EngineCore inherits the env).\n"
    "_os131 = __import__(\"os\")\n"
    "try:\n"
    "    _V131_TRACE = int(_os131.environ.get(\"VLLM_V131_TRACE\", \"0\"))\n"
    "except Exception:\n"
    "    _V131_TRACE = 0\n"
    "if _V131_TRACE:\n"
    "    logger.info(\"V131_ETRACE_ARMED\")\n"
)

# ---------------------------------------------------------------- S2
S2_ANCHOR = (
    "        if self.cached_block_hash_to_block.pop("
    "block_hash, block.block_id) is None:\n"
    "            # block not found in cached_block_hash_to_block,\n"
    "            # eviction is not needed\n"
    "            return False\n"
    "\n"
    "        block.reset_hash()\n"
)

S2_INSERT = (
    "        if self.cached_block_hash_to_block.pop("
    "block_hash, block.block_id) is None:\n"
    "            # block not found in cached_block_hash_to_block,\n"
    "            # eviction is not needed\n"
    "            return False\n"
    "\n"
    "        block.reset_hash()\n"
    "        # llm-scaler v131 ETRACE: count the destroyed mapping.\n"
    "        if _V131_TRACE:\n"
    "            self._v131_ev_total = (\n"
    "                getattr(self, \"_v131_ev_total\", 0) + 1\n"
    "            )\n"
)

# ---------------------------------------------------------------- S3
S3_ANCHOR = (
    "            for block in ret:\n"
    "                assert block.ref_cnt == 0\n"
    "                block.ref_cnt += 1\n"
    "                if self.metrics_collector:\n"
    "                    self.metrics_collector.on_block_allocated(block)\n"
    "        return ret\n"
)

S3_INSERT = (
    "            for block in ret:\n"
    "                assert block.ref_cnt == 0\n"
    "                block.ref_cnt += 1\n"
    "                if self.metrics_collector:\n"
    "                    self.metrics_collector.on_block_allocated(block)\n"
    "        # llm-scaler v131 ETRACE: rate-limited eviction snapshot\n"
    "        # (one line per fresh bucket of 32 evictions; silent when\n"
    "        # nothing is evicted). Host-side queue state only —\n"
    "        # READOUT-LEGAL, no device syncs.\n"
    "        if _V131_TRACE:\n"
    "            _v131_ev = getattr(self, \"_v131_ev_total\", 0)\n"
    "            if _v131_ev // 32 > getattr(self, \"_v131_ev_bucket\", 0):\n"
    "                self._v131_ev_bucket = _v131_ev // 32\n"
    "                logger.info(\n"
    "                    \"V131_EVICT ev_total=%d freeq=%d new=%d\",\n"
    "                    _v131_ev,\n"
    "                    self.get_num_free_blocks(),\n"
    "                    num_blocks,\n"
    "                )\n"
    "        return ret\n"
)


def compile_check(path: str) -> None:
    py_compile.compile(path, doraise=True)


def restore() -> int:
    if not os.path.exists(BAK):
        print("V131_ETRACE_RESTORE_FAIL no_backup=%s" % BAK)
        return 1
    shutil.copy2(BAK, TMP)
    compile_check(TMP)
    st = os.stat(BAK)
    os.chmod(TMP, st.st_mode & 0o7777)
    os.replace(TMP, PATH)
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    print("V131_ETRACE_RESTORED markers_left=%d" % src.count(MARKER))
    return 0


def apply() -> int:
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        print("V131_ETRACE_ALREADY marker_count=%d" % src.count(MARKER))
        return 0
    for name, anchor in (
        ("S1", S1_ANCHOR),
        ("S2", S2_ANCHOR),
        ("S3", S3_ANCHOR),
    ):
        n = src.count(anchor)
        if n != 1:
            print("V131_ETRACE_FAIL anchor_%s count=%d (need 1)" % (name, n))
            return 1
    if not os.path.exists(BAK):
        shutil.copy2(PATH, BAK)
    out = src.replace(S3_ANCHOR, S3_INSERT)
    out = out.replace(S2_ANCHOR, S2_INSERT)
    out = out.replace(S1_ANCHOR, S1_INSERT)
    n_markers = out.count(MARKER)
    if n_markers != EXPECTED_MARKERS:
        print(
            "V131_ETRACE_FAIL markers=%d (need %d)"
            % (n_markers, EXPECTED_MARKERS)
        )
        return 1
    with open(TMP, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    compile_check(TMP)
    st = os.stat(PATH)
    os.chmod(TMP, st.st_mode & 0o7777)
    os.replace(TMP, PATH)
    print("V131_ETRACE_OK markers=%d bak=%s" % (n_markers, BAK))
    return 0


if __name__ == "__main__":
    _args = sys.argv[1:]
    # Optional positional target path (dry-test against a copy; the
    # derived .v131bak/.v131tmp follow it). Default = the lane target.
    for _a in _args:
        if not _a.startswith("-"):
            PATH = _a
            BAK = _a + ".v131bak"
            TMP = _a + ".v131tmp"
    if "--restore" in _args:
        sys.exit(restore())
    sys.exit(apply())
