#!/usr/bin/env python3
"""dflash2_schedk_edit.py — v42c: honor DFLASH2_EMIT_K on the ASYNC path.

Under --async-scheduling the scheduler cannot consume the drafter's true
variable-width lists (the engine skips post_step -> update_draft_token_ids;
engine/core.py gate `not self.async_scheduling`). Instead AsyncScheduler
injects a shared placeholder list into every running request each step:

    self._spec_token_placeholders = [-1] * self.num_spec_tokens   # width 7
    request.spec_token_ids = self._spec_token_placeholders        # each step

That placeholder WIDTH is what schedule() turns into spec slots (verified
live: D3 probe total_num_spec_tokens=7 with k=4 emission; metrics
num_draft_tokens/drafts = exactly 7.0; positions 4-6 zero, never accepted).

Since DFLASH2_EMIT_K is a boot-time constant, the async path does not need
per-step width feedback — it needs the placeholder list to be k-wide. The
accounting is self-consistent: _update_after_schedule derives
cur_num_spec_tokens from the scheduled width, and the engine's
update_draft_token_ids_in_output trims/pads real drafts to the scheduled
(orig) width. Capping the placeholder list caps the whole chain.

This file is NOT staged by dflash2_edit.py, so this patcher stages the
stock copy from site-packages into /w/patched first, then patches it.
Run AFTER dflash2_emitk_edit.py in phase 1. Default (env unset):
byte-identical stock behavior.
"""
import py_compile
import shutil
from pathlib import Path

W = Path("/w")
SP = Path("/opt/venv/lib/python3.12/site-packages")
REL = "vllm/v1/core/sched/async_scheduler.py"


def edit(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    assert n == 1, f"{what}: expected exactly 1 occurrence, found {n}"
    print(f"  ok: {what}")
    return text.replace(old, new)


def main() -> int:
    dst = W / "patched" / REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Unconditional re-stage: /w/patched persists on the host bind mount
    # across boots, so a guarded copy would leave a previously-patched file
    # and the anchor assert below would fail on the second boot.
    shutil.copy2(SP / REL, dst)
    print("  ok: staged stock async_scheduler.py")

    t = dst.read_text()

    t = edit(
        t,
        "from vllm.logger import init_logger\n",
        "import os\n\nfrom vllm.logger import init_logger\n",
        "import os",
    )

    # NOTE: stock already defines `logger = init_logger(__name__)` at module
    # level — do NOT add another one.

    t = edit(
        t,
        "        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens\n",
        "        # llm-scaler v42c: honor DFLASH2_EMIT_K on the async path.\n"
        "        # The placeholder list width IS the scheduled spec width\n"
        "        # under async scheduling; everything downstream (spec slot\n"
        "        # count, num_output_placeholders accounting, engine-side\n"
        "        # -1 padding in update_draft_token_ids_in_output) derives\n"
        "        # from the scheduled width, so capping here is\n"
        "        # self-consistent end-to-end. Default: stock width.\n"
        "        _emit_k = self.num_spec_tokens\n"
        "        _env_k = os.environ.get(\"DFLASH2_EMIT_K\", \"\")\n"
        "        if _env_k:\n"
        "            _emit_k = int(_env_k)\n"
        "            if not 1 <= _emit_k <= self.num_spec_tokens:\n"
        "                raise ValueError(\n"
        "                    f\"DFLASH2_EMIT_K={_emit_k} out of range 1..\"\n"
        "                    f\"{self.num_spec_tokens} (num_spec_tokens)\"\n"
        "                )\n"
        "        self._spec_token_placeholders: list[int] = [-1] * _emit_k\n"
        "        if _emit_k < self.num_spec_tokens:\n"
        "            logger.info(\n"
        "                \"DFlash2 v42c async cap: scheduling %d of %d spec \"\n"
        "                \"slots per step (async scheduling preserved)\",\n"
        "                _emit_k,\n"
        "                self.num_spec_tokens,\n"
        "            )\n",
        "placeholder width cap",
    )

    dst.write_text(t)
    py_compile.compile(str(dst), doraise=True)
    assert t.count('os.environ.get("DFLASH2_EMIT_K"') == 1
    assert t.count("self._spec_token_placeholders: list[int] = [-1] * _emit_k") == 1
    assert t.count("logger = init_logger(__name__)") == 1
    print("  ok: py_compile + greps")
    print("DFLASH2_SCHEDK_EDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
