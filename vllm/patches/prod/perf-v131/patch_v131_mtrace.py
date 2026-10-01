#!/usr/bin/env python3
"""patch_v131_mtrace.py — perf-v131 P59 admission-match forensics patcher.

Instruments vllm/v1/core/kv_cache_manager.py of the running lane
container (llm-scaler-exp:v1.2.27 posture). Default
VLLM_V131_TRACE=0 = OFF -> the only delta is one `if _V131_TRACE`
false test per get_computed_blocks call (admission path only; the
skip_reading_prefix_cache early-return precedes the hook, so the
decode path never pays anything). Arm at serve relaunch by
exporting the env knob in serve_user.sh (scheduler-side code runs
in EngineCore, which inherits it — marker-file law N/A).

Sites (anchors verified unique in-container, through EOL, LF):
  S1  module knob block, after the module logger line
  S2  match hook, between the find_longest_cache_hit call and the
      `if self.log_stats:` tail of get_computed_blocks (L~203)

S2 semantics — the V131_MATCH line discriminates the round's open
question (v130 close: WS ~344 pages fits the ~465-498-page pool,
yet PRIV turn>=2 lands cached=0 deterministically):
  rid   request_id[:12]            (join key vs driver turns)
  ptok  request.num_tokens         (admission prompt tokens)
  hit   num_new_computed_tokens    (what the hybrid fixed-point returned)
  kmiss first hash-chain index ABSENT from the cached-hash table,
        probed exactly like the real match: plain hashes isliced
        from request.block_hashes, salted per group inside
        block_pool.get_cached_block against ALL group ids (the 4
        hybrid groups mirror allocations — HYBRID ANATOMY LAW).
        Bounded to the first 64 hashes. -2 = probe itself failed.
        hit=0 with kmiss>0   -> present-but-unmatched (mamba-align /
                                hash-chain lever, hypothesis (b))
        hit=0 with kmiss=0   -> cache absent (evicted earlier or
                                never inserted; pair with the ETRACE
                                counters, hypotheses (a)/(c))
  freeq free-queue length (includes reclaimable cached blocks —
        the scheduler's true allocation headroom)
  pre   request.num_preemptions (preempt re-admissions stand out)

Idempotent: exits V131_MTRACE_ALREADY if the marker is present.
Backs up to kv_cache_manager.py.v131bak (first apply only).
Compile-checks a temp copy before os.replace; --restore copies the
backup back (same discipline). Optional positional target path
dry-tests against a copy (derived .v131bak/.v131tmp follow it).

Caps line on success: V131_MTRACE_OK markers=<n> bak=<path>
"""

import os
import py_compile
import shutil
import sys

PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/"
    "kv_cache_manager.py"
)
BAK = PATH + ".v131bak"
TMP = PATH + ".v131tmp"
MARKER = "# llm-scaler v131 MTRACE"
EXPECTED_MARKERS = 2

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
    "# llm-scaler v131 MTRACE: admission-match forensics knob (see the\n"
    "# V131_MATCH hook in get_computed_blocks). Default 0 = OFF ->\n"
    "# scheduling behavior byte-identical; one false `if` per admission.\n"
    "# Armed via VLLM_V131_TRACE=1 in serve_user.sh; scheduler-side\n"
    "# code runs in EngineCore, which inherits the env.\n"
    "_os131 = __import__(\"os\")\n"
    "try:\n"
    "    _V131_TRACE = int(_os131.environ.get(\"VLLM_V131_TRACE\", \"0\"))\n"
    "except Exception:\n"
    "    _V131_TRACE = 0\n"
    "if _V131_TRACE:\n"
    "    logger.info(\"V131_MTRACE_ARMED\")\n"
)

# ---------------------------------------------------------------- S2
S2_ANCHOR = (
    "        computed_blocks, num_new_computed_tokens = (\n"
    "            self.coordinator.find_longest_cache_hit(\n"
    "                request.block_hashes, max_cache_hit_length\n"
    "            )\n"
    "        )\n"
    "\n"
    "        if self.log_stats:\n"
)

S2_INSERT = (
    "        computed_blocks, num_new_computed_tokens = (\n"
    "            self.coordinator.find_longest_cache_hit(\n"
    "                request.block_hashes, max_cache_hit_length\n"
    "            )\n"
    "        )\n"
    "\n"
    "        # llm-scaler v131 MTRACE: match forensics (see module-top\n"
    "        # knob). kmiss = first hash-chain index absent from the\n"
    "        # cached-hash table, probed exactly like the real match\n"
    "        # (plain hashes, per-group salting inside get_cached_block,\n"
    "        # all mirroring group ids). Bounded 64 probes. Host-side\n"
    "        # dict/queue state only — READOUT-LEGAL, no device syncs.\n"
    "        if _V131_TRACE:\n"
    "            try:\n"
    "                _v131_gids = list(\n"
    "                    range(\n"
    "                        len(\n"
    "                            self.coordinator.kv_cache_config.\n"
    "                            kv_cache_groups\n"
    "                        )\n"
    "                    )\n"
    "                )\n"
    "                _v131_i = 0\n"
    "                _v131_nprobe = min(len(request.block_hashes), 64)\n"
    "                for _v131_h in itertools.islice(\n"
    "                    request.block_hashes, _v131_nprobe\n"
    "                ):\n"
    "                    if (\n"
    "                        self.block_pool.get_cached_block(\n"
    "                            _v131_h, _v131_gids\n"
    "                        )\n"
    "                        is None\n"
    "                    ):\n"
    "                        break\n"
    "                    _v131_i += 1\n"
    "                else:\n"
    "                    _v131_i = _v131_nprobe\n"
    "                _v131_kmiss = _v131_i\n"
    "            except Exception:\n"
    "                _v131_kmiss = -2\n"
    "            logger.info(\n"
    "                \"V131_MATCH rid=%s ptok=%d hit=%d kmiss=%d\"\n"
    "                \" freeq=%d pre=%d\",\n"
    "                request.request_id[:12],\n"
    "                request.num_tokens,\n"
    "                num_new_computed_tokens,\n"
    "                _v131_kmiss,\n"
    "                self.block_pool.get_num_free_blocks(),\n"
    "                request.num_preemptions,\n"
    "            )\n"
    "\n"
    "        if self.log_stats:\n"
)


def compile_check(path: str) -> None:
    py_compile.compile(path, doraise=True)


def restore() -> int:
    if not os.path.exists(BAK):
        print("V131_MTRACE_RESTORE_FAIL no_backup=%s" % BAK)
        return 1
    shutil.copy2(BAK, TMP)
    compile_check(TMP)
    st = os.stat(BAK)
    os.chmod(TMP, st.st_mode & 0o7777)
    os.replace(TMP, PATH)
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    print("V131_MTRACE_RESTORED markers_left=%d" % src.count(MARKER))
    return 0


def apply() -> int:
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        print("V131_MTRACE_ALREADY marker_count=%d" % src.count(MARKER))
        return 0
    for name, anchor in (("S1", S1_ANCHOR), ("S2", S2_ANCHOR)):
        n = src.count(anchor)
        if n != 1:
            print("V131_MTRACE_FAIL anchor_%s count=%d (need 1)" % (name, n))
            return 1
    if not os.path.exists(BAK):
        shutil.copy2(PATH, BAK)
    out = src.replace(S2_ANCHOR, S2_INSERT)
    out = out.replace(S1_ANCHOR, S1_INSERT)
    n_markers = out.count(MARKER)
    if n_markers != EXPECTED_MARKERS:
        print(
            "V131_MTRACE_FAIL markers=%d (need %d)"
            % (n_markers, EXPECTED_MARKERS)
        )
        return 1
    with open(TMP, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    compile_check(TMP)
    st = os.stat(PATH)
    os.chmod(TMP, st.st_mode & 0o7777)
    os.replace(TMP, PATH)
    print("V131_MTRACE_OK markers=%d bak=%s" % (n_markers, BAK))
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
