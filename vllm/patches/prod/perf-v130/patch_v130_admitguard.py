#!/usr/bin/env python3
"""patch_v130_admitguard.py — perf-v130 P56 admission-guard patcher.

Inserts the v130 ADMGUARD (aggregate-KV-demand admission cap) into
vllm/v1/core/sched/scheduler.py of the running lane container
(llm-scaler-exp:v1.2.27 posture). Default VLLM_V130_ADMIT_HIGH=0 =
OFF -> constant-false, scheduling behavior byte-identical (the P56
knob-off inertness gate certifies this). Arm at serve relaunch by
exporting the env knob in serve_user.sh.

Sites (all anchors verified unique, through EOL, LF endings):
  S1  module knob block, after the _V64 tail (L128-130)
  S2  per-step state init, after the _v66_bypass_now line (L578)
  S3  waiting-loop hold check, between the num_encoder_tokens sum and
      the allocate_slots call (L1017-1025) — anchored on the encoder
      context so the L750 running-loop allocate_slots cannot match

v2 (Leg C) — page-units fix: the scheduler's self.block_size is the
LCM of the hybrid groups' page sizes (1024 tok here — the hybrid
interface pads attention pages byte-equal to mamba pages), so
cand_blk and total_blk were already page units. The bug was the run
term: v1 SUMMED every KV cache group's block-id list, double-
counting mirrored hybrid groups. v2 takes the per-request MAX over
groups (correct for both mirrored and O(1)-mamba layouts) and logs
a one-shot V130_ADMGUARD_ANAT anatomy line at the first armed check.

Idempotent: exits V130_ADMGUARD_ALREADY if the marker comment is
present. Backs up to scheduler.py.v130bak (first apply only).
Compile-checks a temp copy before atomically replacing the target;
--restore copies the backup back (same compile discipline).

Caps line on success: V130_ADMGUARD_OK markers=<n> bak=<path>
high_env=<value or unset>
"""

import os
import py_compile
import shutil
import sys

PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/"
    "scheduler.py"
)
BAK = PATH + ".v130bak"
TMP = PATH + ".v130tmp"
MARKER = "# llm-scaler v130 ADMGUARD"
EXPECTED_MARKERS = 4

# ---------------------------------------------------------------- S1
S1_ANCHOR = (
    "_V64_DECODE_BUDGET = max(64, min(_V64_DECODE_BUDGET, 1023))\n"
    "if _V64_DECODE_INTERLEAVE < 2:\n"
    "    _V64_DECODE_INTERLEAVE = 0\n"
)

S1_INSERT = (
    "_V64_DECODE_BUDGET = max(64, min(_V64_DECODE_BUDGET, 1023))\n"
    "if _V64_DECODE_INTERLEAVE < 2:\n"
    "    _V64_DECODE_INTERLEAVE = 0\n"
    "\n"
    "# llm-scaler v130 ADMGUARD: aggregate-KV-demand admission cap (see\n"
    "# the WAITING-loop needle in schedule() for the check). Default\n"
    "# 0 = OFF -> constant-false, scheduling byte-identical. Armed\n"
    "# (>0): a WAITING request is held (skipped WITHOUT eviction) until\n"
    "# the RUNNING block footprint + this step's admitted demand + the\n"
    "# candidate's FULL uncached demand fit under HIGH x total blocks.\n"
    "# Cache-only blocks are the evictable buffer and are deliberately\n"
    "# NOT counted (counting them would deadlock behind dead\n"
    "# conversations). Units: scheduler block_size = LCM of the hybrid\n"
    "# groups' page sizes (1024 tok here — attention pages padded\n"
    "# byte-equal to mamba pages), and num_gpu_blocks counts those\n"
    "# pages; the run term takes the per-request MAX over groups (v1\n"
    "# summed groups — a hybrid double count). Liveness: a lone\n"
    "# candidate always admits because max_model_len (262,144 tok =\n"
    "# 256 pages) < HIGH x 497 pages for any HIGH >= 0.52. v66 FAIRFIX\n"
    "# interplay: the starved waiting head\n"
    "# bypasses the cap ONE head per schedule() step (starvation always\n"
    "# wins; the wave still paces to drain rate). Known conservative\n"
    "# biases (both toward holding, i.e. stability): run_blk\n"
    "# double-counts shared blocks referenced by several RUNNING\n"
    "# requests, and the step accumulator reserves the candidate's\n"
    "# full demand even when chunked prefill allocates only part of it\n"
    "# this step.\n"
    "_os130 = __import__(\"os\")\n"
    "try:\n"
    "    _V130_ADMIT_HIGH = float(\n"
    "        _os130.environ.get(\"VLLM_V130_ADMIT_HIGH\", \"0\")\n"
    "    )\n"
    "except Exception:\n"
    "    _V130_ADMIT_HIGH = 0.0\n"
    "if not (0.05 <= _V130_ADMIT_HIGH <= 1.0):\n"
    "    _V130_ADMIT_HIGH = 0.0\n"
    "if _V130_ADMIT_HIGH > 0.0:\n"
    "    logger.info(\"V130_ADMGUARD_ARMED high=%.2f\", _V130_ADMIT_HIGH)\n"
)

# ---------------------------------------------------------------- S2
S2_ANCHOR = "        _v66_bypass_now = _v66_waiting_starved(self)\n"

S2_INSERT = (
    "        _v66_bypass_now = _v66_waiting_starved(self)\n"
    "        # llm-scaler v130 ADMGUARD per-step state (see module-top\n"
    "        # knob). run_blk uses -1 as a lazy sentinel: computed on the\n"
    "        # first armed check, only when the guard is ON.\n"
    "        _v130_bypass_admitted = False\n"
    "        _v130_run_blk = -1\n"
    "        _v130_admitted_blk = 0\n"
)

# ---------------------------------------------------------------- S3
S3_ANCHOR = (
    "                    num_encoder_tokens = sum(\n"
    "                        request.get_num_encoder_embeds(i)\n"
    "                        for i in encoder_inputs_to_schedule\n"
    "                    )\n"
    "\n"
    "                new_blocks = self.kv_cache_manager.allocate_slots(\n"
)

S3_INSERT = (
    "                    num_encoder_tokens = sum(\n"
    "                        request.get_num_encoder_embeds(i)\n"
    "                        for i in encoder_inputs_to_schedule\n"
    "                    )\n"
    "\n"
    "                # llm-scaler v130 ADMGUARD: aggregate-KV-demand\n"
    "                # admission check. Host-side block tables only\n"
    "                # (READOUT-LEGAL; no device syncs in any forward\n"
    "                # path). A held request queues in\n"
    "                # step_skipped_waiting WITHOUT evicting anyone; the\n"
    "                # skip uses continue (not break) so smaller waiting\n"
    "                # candidates can still fill this step (best-fit).\n"
    "                if _V130_ADMIT_HIGH > 0.0:\n"
    "                    _v130_cand_blk = -(\n"
    "                        -(\n"
    "                            request.num_tokens\n"
    "                            - request.num_computed_tokens\n"
    "                        )\n"
    "                        // self.block_size\n"
    "                    )\n"
    "                    if not (\n"
    "                        _v66_bypass_now and not _v130_bypass_admitted\n"
    "                    ):\n"
    "                        if _v130_run_blk < 0:\n"
    "                            _v130_run_blk = 0\n"
    "                            for _vr in self.running:\n"
    "                                _v130_run_blk += max(\n"
    "                                    (\n"
    "                                        len(_vg)\n"
    "                                        for _vg in (\n"
    "                                            self.\n"
    "                                            kv_cache_manager.\n"
    "                                            get_block_ids(\n"
    "                                                _vr.request_id\n"
    "                                            )\n"
    "                                        )\n"
    "                                    ),\n"
    "                                    default=0,\n"
    "                                )\n"
    "                            if not getattr(\n"
    "                                self, \"_v130_anat_done\", False\n"
    "                            ):\n"
    "                                self._v130_anat_done = True\n"
    "                                logger.info(\n"
    "                                    \"V130_ADMGUARD_ANAT bs=%d\"\n"
    "                                    \" total_blk=%d groups=%d\"\n"
    "                                    \" run_blk=%d running=%d\",\n"
    "                                    self.block_size,\n"
    "                                    self.kv_cache_manager.\n"
    "                                    block_pool.num_gpu_blocks\n"
    "                                    - 1,\n"
    "                                    self.kv_cache_manager.\n"
    "                                    num_kv_cache_groups,\n"
    "                                    _v130_run_blk,\n"
    "                                    len(self.running),\n"
    "                                )\n"
    "                        _v130_total_blk = (\n"
    "                            self.kv_cache_manager.block_pool.\n"
    "                            num_gpu_blocks\n"
    "                            - 1\n"
    "                        )\n"
    "                        if (\n"
    "                            _v130_run_blk\n"
    "                            + _v130_admitted_blk\n"
    "                            + _v130_cand_blk\n"
    "                            > _V130_ADMIT_HIGH * _v130_total_blk\n"
    "                        ):\n"
    "                            self._v130_holds = (\n"
    "                                getattr(self, \"_v130_holds\", 0) + 1\n"
    "                            )\n"
    "                            if (\n"
    "                                not getattr(\n"
    "                                    self, \"_v130_announced\", False\n"
    "                                )\n"
    "                                or self._v130_holds % 50 == 1\n"
    "                            ):\n"
    "                                self._v130_announced = True\n"
    "                                logger.info(\n"
    "                                    \"V130_ADMGUARD_ACTIVE\"\n"
    "                                    \" high=%.2f run_blk=%d\"\n"
    "                                    \" adm_blk=%d cand_blk=%d\"\n"
    "                                    \" total_blk=%d waiting=%d\"\n"
    "                                    \" holds=%d\",\n"
    "                                    _V130_ADMIT_HIGH,\n"
    "                                    _v130_run_blk,\n"
    "                                    _v130_admitted_blk,\n"
    "                                    _v130_cand_blk,\n"
    "                                    _v130_total_blk,\n"
    "                                    len(self.waiting)\n"
    "                                    + len(self.skipped_waiting),\n"
    "                                    self._v130_holds,\n"
    "                                )\n"
    "                            request_queue.pop_request()\n"
    "                            step_skipped_waiting.prepend_request(\n"
    "                                request\n"
    "                            )\n"
    "                            continue\n"
    "                    _v130_admitted_blk += _v130_cand_blk\n"
    "                if _v66_bypass_now and not _v130_bypass_admitted:\n"
    "                    _v130_bypass_admitted = True"
    "  # llm-scaler v130 ADMGUARD\n"
    "\n"
    "                new_blocks = self.kv_cache_manager.allocate_slots(\n"
)


def compile_check(path: str) -> None:
    py_compile.compile(path, doraise=True)


def restore() -> int:
    if not os.path.exists(BAK):
        print("V130_ADMGUARD_RESTORE_FAIL no_backup=%s" % BAK)
        return 1
    shutil.copy2(BAK, TMP)
    compile_check(TMP)
    st = os.stat(BAK)
    os.chmod(TMP, st.st_mode & 0o7777)
    os.replace(TMP, PATH)
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    print(
        "V130_ADMGUARD_RESTORED markers_left=%d" % src.count(MARKER)
    )
    return 0


def apply() -> int:
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        print(
            "V130_ADMGUARD_ALREADY marker_count=%d" % src.count(MARKER)
        )
        return 0
    for name, anchor in (
        ("S1", S1_ANCHOR),
        ("S2", S2_ANCHOR),
        ("S3", S3_ANCHOR),
    ):
        n = src.count(anchor)
        if n != 1:
            print("V130_ADMGUARD_FAIL anchor_%s count=%d (need 1)" % (name, n))
            return 1
    if not os.path.exists(BAK):
        shutil.copy2(PATH, BAK)
    out = src.replace(S3_ANCHOR, S3_INSERT)
    out = out.replace(S2_ANCHOR, S2_INSERT)
    out = out.replace(S1_ANCHOR, S1_INSERT)
    n_markers = out.count(MARKER)
    if n_markers != EXPECTED_MARKERS:
        print(
            "V130_ADMGUARD_FAIL markers=%d (need %d)"
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
        "V130_ADMGUARD_OK markers=%d bak=%s high_env=%r"
        % (
            n_markers,
            BAK,
            os.environ.get("VLLM_V130_ADMIT_HIGH", "unset"),
        )
    )
    return 0


if __name__ == "__main__":
    _args = sys.argv[1:]
    # Optional positional target path (dry-test against a copy; the
    # derived .v130bak/.v130tmp follow it). Default = the lane target.
    for _a in _args:
        if not _a.startswith("-"):
            PATH = _a
            BAK = _a + ".v130bak"
            TMP = _a + ".v130tmp"
    if "--restore" in _args:
        sys.exit(restore())
    sys.exit(apply())
