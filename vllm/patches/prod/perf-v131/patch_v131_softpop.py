#!/usr/bin/env python3
"""patch_v131_softpop.py — perf-v131 P60 pop-time protection patcher.

Implements the P59-mandated lever (CHURN-DESTRUCTION LAW: cached
mappings die at free-queue pop time in proportion to allocation
EVENT count, not capacity shortfall — the per-request mamba-align
rotation + MTP draft setup (new=5-6 pops, continuous) is the
guillotine) as a one-site surgery in vllm/v1/core/block_pool.py of
the running lane container (llm-scaler-exp:v1.2.27 posture).

Policy: in get_new_blocks, when VLLM_V131_SOFTPOP_SLACK > 0 AND
enable_caching AND freeq - num_blocks >= SLACK, pop one block at a
time; hash-bearing (cached) head blocks are append()ed to the queue
TAIL — mapping survives, eviction skipped; non-hash state blocks
(mamba rotation, draft setup — the expendable churn) are consumed.
Guard: 4*num+1024 skip-rotations, then stock popleft_n for the
remainder (queue-entirely-cached corner, counted + logged as
V131_SOFTPOP hard=). Genuine pressure (freeq - num < SLACK) or
SLACK=0 (default): stock front-pop path, byte-identical.

Self-organization: expendable state blocks never rotate and stay
consumable; cached blocks migrate tailward under slack; genuine
exhaustion still evicts LRU-first (the front converges to
non-hash blocks). Liveness: pops-minus-appends never exceed
num_blocks, so queue length never drops below the checked margin —
the ValueError guard above get_new_blocks keeps its exact
semantics. Host-side queue state only — READOUT-LEGAL.

Sites (through EOL, LF):
  S1  module knob block — anchored AFTER the v131 ETRACE knob tail
      (post-etrace file), with a pristine-file fallback anchor
      (both orders supported: this patcher works on a fresh
      container re-created by the watchdog as well)
  S2  get_new_blocks pop site — the annotated ret line (unique)

NOTE on backups: uses block_pool.py.v131sbak (distinct from the
ETRACE patcher's .v131bak) — restore order is softpop first,
etrace second, back to pristine.

Idempotent (V131_SOFTPOP_ALREADY), compile-checked temp-write ->
os.replace, --restore. Optional positional target path dry-tests
against a copy. Caps: V131_SOFTPOP_OK markers=<n> bak=<path>
slack_env=<value or unset>.
"""

import os
import py_compile
import shutil
import sys

PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/"
    "block_pool.py"
)
BAK = PATH + ".v131sbak"
TMP = PATH + ".v131stmp"
MARKER = "# llm-scaler v131 SOFTPOP"
ETRACE_MARKER = "# llm-scaler v131 ETRACE"
EXPECTED_MARKERS = 2

# ---------------------------------------------------------------- S1
# Primary anchor: file already carrying the ETRACE knob (live order).
S1_ANCHOR_POST = (
    "    _V131_TRACE = 0\n"
    "if _V131_TRACE:\n"
    "    logger.info(\"V131_ETRACE_ARMED\")\n"
)
# Fallback anchor: pristine file (watchdog re-create order).
S1_ANCHOR_PRISTINE = (
    "from vllm.v1.request import Request\n"
    "\n"
    "logger = init_logger(__name__)\n"
)

S1_INSERT_BODY = (
    "\n"
    "# llm-scaler v131 SOFTPOP: pop-time protection knob. Default\n"
    "# 0 = OFF -> the soft path is dead code behind a false test;\n"
    "# get_new_blocks is byte-identical to upstream. Armed via\n"
    "# VLLM_V131_SOFTPOP_SLACK=<n> (1..4096) in serve_user.sh: when\n"
    "# the free queue holds >= n blocks beyond this allocation's\n"
    "# demand, hash-bearing head blocks rotate to the tail instead\n"
    "# of being evicted (see the SOFTPOP block in get_new_blocks).\n"
    "_os131s = __import__(\"os\")\n"
    "try:\n"
    "    _V131_SOFTPOP_SLACK = int(\n"
    "        _os131s.environ.get(\"VLLM_V131_SOFTPOP_SLACK\", \"0\")\n"
    "    )\n"
    "except Exception:\n"
    "    _V131_SOFTPOP_SLACK = 0\n"
    "if not (1 <= _V131_SOFTPOP_SLACK <= 4096):\n"
    "    _V131_SOFTPOP_SLACK = 0\n"
    "if _V131_SOFTPOP_SLACK > 0:\n"
    "    logger.info(\n"
    "        \"V131_SOFTPOP_ARMED slack=%d\", _V131_SOFTPOP_SLACK\n"
    "    )\n"
)

# ---------------------------------------------------------------- S2
S2_ANCHOR = (
    "        ret: list[KVCacheBlock] = "
    "self.free_block_queue.popleft_n(num_blocks)\n"
)

S2_INSERT = (
    "        # llm-scaler v131 SOFTPOP: pop-time protection. Under\n"
    "        # freeq slack, rotate hash-bearing head blocks to the\n"
    "        # queue TAIL (mapping survives) and consume only\n"
    "        # non-hash state blocks; stock front-pop otherwise\n"
    "        # (pressure or SLACK=0) — byte-identical to upstream.\n"
    "        _v131_ret = None\n"
    "        if (\n"
    "            _V131_SOFTPOP_SLACK > 0\n"
    "            and self.enable_caching\n"
    "            and self.get_num_free_blocks() - num_blocks\n"
    "            >= _V131_SOFTPOP_SLACK\n"
    "        ):\n"
    "            _v131_ret = []\n"
    "            _v131_guard = 4 * num_blocks + 1024\n"
    "            while len(_v131_ret) < num_blocks and _v131_guard > 0:\n"
    "                _v131_guard -= 1\n"
    "                _v131_b = self.free_block_queue.popleft_n(1)[0]\n"
    "                if _v131_b.block_hash is not None:\n"
    "                    self.free_block_queue.append(_v131_b)\n"
    "                    self._v131_sp_rot = (\n"
    "                        getattr(self, \"_v131_sp_rot\", 0) + 1\n"
    "                    )\n"
    "                    if self._v131_sp_rot % 4096 == 1:\n"
    "                        logger.info(\n"
    "                            \"V131_SOFTPOP rot=%d hard=%d\"\n"
    "                            \" freeq=%d\",\n"
    "                            self._v131_sp_rot,\n"
    "                            getattr(self, \"_v131_sp_hard\", 0),\n"
    "                            self.get_num_free_blocks(),\n"
    "                        )\n"
    "                    continue\n"
    "                _v131_ret.append(_v131_b)\n"
    "            if len(_v131_ret) < num_blocks:\n"
    "                _v131_ret.extend(\n"
    "                    self.free_block_queue.popleft_n(\n"
    "                        num_blocks - len(_v131_ret)\n"
    "                    )\n"
    "                )\n"
    "                self._v131_sp_hard = (\n"
    "                    getattr(self, \"_v131_sp_hard\", 0) + 1\n"
    "                )\n"
    "        if _v131_ret is not None:\n"
    "            ret = _v131_ret\n"
    "        else:\n"
    "            ret: list[KVCacheBlock] = "
    "self.free_block_queue.popleft_n(num_blocks)\n"
)


def compile_check(path: str) -> None:
    py_compile.compile(path, doraise=True)


def restore() -> int:
    if not os.path.exists(BAK):
        print("V131_SOFTPOP_RESTORE_FAIL no_backup=%s" % BAK)
        return 1
    shutil.copy2(BAK, TMP)
    compile_check(TMP)
    st = os.stat(BAK)
    os.chmod(TMP, st.st_mode & 0o7777)
    os.replace(TMP, PATH)
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    print("V131_SOFTPOP_RESTORED markers_left=%d" % src.count(MARKER))
    return 0


def apply() -> int:
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        print("V131_SOFTPOP_ALREADY marker_count=%d" % src.count(MARKER))
        return 0
    n_post = src.count(S1_ANCHOR_POST)
    n_pris = src.count(S1_ANCHOR_PRISTINE)
    if n_post == 1:
        s1_anchor, s1_insert = S1_ANCHOR_POST, (
            S1_ANCHOR_POST + S1_INSERT_BODY
        )
    elif n_pris == 1 and ETRACE_MARKER not in src:
        s1_anchor, s1_insert = S1_ANCHOR_PRISTINE, (
            S1_ANCHOR_PRISTINE + S1_INSERT_BODY
        )
    else:
        print(
            "V131_SOFTPOP_FAIL anchor_S1 post=%d pristine=%d"
            " etrace_marker=%d" % (
                n_post, n_pris, src.count(ETRACE_MARKER)
            )
        )
        return 1
    n_s2 = src.count(S2_ANCHOR)
    if n_s2 != 1:
        print("V131_SOFTPOP_FAIL anchor_S2 count=%d (need 1)" % n_s2)
        return 1
    if not os.path.exists(BAK):
        shutil.copy2(PATH, BAK)
    out = src.replace(S2_ANCHOR, S2_INSERT)
    out = out.replace(s1_anchor, s1_insert)
    n_markers = out.count(MARKER)
    if n_markers != EXPECTED_MARKERS:
        print(
            "V131_SOFTPOP_FAIL markers=%d (need %d)"
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
        "V131_SOFTPOP_OK markers=%d bak=%s slack_env=%r"
        % (
            n_markers,
            BAK,
            os.environ.get("VLLM_V131_SOFTPOP_SLACK", "unset"),
        )
    )
    return 0


if __name__ == "__main__":
    _args = sys.argv[1:]
    # Optional positional target path (dry-test against a copy; the
    # derived .v131sbak/.v131stmp follow it). Default = lane target.
    for _a in _args:
        if not _a.startswith("-"):
            PATH = _a
            BAK = _a + ".v131sbak"
            TMP = _a + ".v131stmp"
    if "--restore" in _args:
        sys.exit(restore())
    sys.exit(apply())
