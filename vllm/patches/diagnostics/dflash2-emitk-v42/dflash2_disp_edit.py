#!/usr/bin/env python3
"""dflash2_disp_edit.py — v42d edit 3: dispatcher-side uniform width cap.

THIRD derivation site of the uniform decode width. The runner's
uniform_decode_query_len (edit 1) and the capture lattice (edit 2) are
already width-aware, but CudagraphDispatcher.__init__ derives its own:

    self.uniform_decode_query_len = (
        1
        if not self.vllm_config.speculative_config
        else 1 + self.vllm_config.speculative_config.num_speculative_tokens
    )

and _create_padded_batch_descriptor asserts

    num_tokens_padded % self.uniform_decode_query_len == 0

against the (rescaled, width-multiple) lattice. initialize_cudagraph_keys
receives the runner's patched width only as a PARAMETER (used for the size
filter); the assert reads the INSTANCE attribute. Convicted live on the
k=4 lane: boot assert at cudagraph_dispatcher.py:144 with dispatcher width
8 against the width-5 lattice. The proposer's dispatcher (PIECEWISE-only
keys) never reaches the uniform+FULL assert, so this single edit closes
the set of derivation sites.

This file is NOT staged by dflash2_edit.py — this patcher stages the stock
copy from site-packages first. Run AFTER dflash2_udql_edit.py. Default
(env unset): byte-identical stock behavior.
"""
import py_compile
import shutil
from pathlib import Path

W = Path("/w")
SP = Path("/opt/venv/lib/python3.12/site-packages")
REL = "vllm/v1/cudagraph_dispatcher.py"


def edit(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    assert n == 1, f"{what}: expected exactly 1 occurrence, found {n}"
    print(f"  ok: {what}")
    return text.replace(old, new)


def main() -> int:
    dst = W / "patched" / REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Unconditional re-stage (see dflash2_schedk_edit.py rationale).
    shutil.copy2(SP / REL, dst)
    print("  ok: staged stock cudagraph_dispatcher.py")

    t = dst.read_text()

    t = edit(
        t,
        "from collections.abc import Set as AbstractSet\n",
        "import os\nfrom collections.abc import Set as AbstractSet\n",
        "import os",
    )

    # NOTE: stock already defines `logger = init_logger(__name__)` at module
    # level — do NOT add another one.

    t = edit(
        t,
        "        self.uniform_decode_query_len = (\n"
        "            1\n"
        "            if not self.vllm_config.speculative_config\n"
        "            else 1 + self.vllm_config.speculative_config.num_speculative_tokens\n"
        "        )\n",
        "        self.uniform_decode_query_len = (\n"
        "            1\n"
        "            if not self.vllm_config.speculative_config\n"
        "            else 1 + self.vllm_config.speculative_config.num_speculative_tokens\n"
        "        )\n"
        "        # llm-scaler v42d: keep the dispatcher's uniform width in sync\n"
        "        # with the runner-side DFLASH2_EMIT_K cap. The dispatcher\n"
        "        # derives this independently from config (1+num_spec = 8)\n"
        "        # while _create_padded_batch_descriptor asserts\n"
        "        # num_tokens_padded %% self.uniform_decode_query_len == 0\n"
        "        # against the rescaled width-multiple lattice; leaving it at\n"
        "        # the config width fails boot at cudagraph_dispatcher.py:144\n"
        "        # when the cap is active (convicted live, k=4 lane). Default\n"
        "        # (env unset): stock width, byte-identical.\n"
        "        _env_k = os.environ.get(\"DFLASH2_EMIT_K\", \"\")\n"
        "        if _env_k and self.vllm_config.speculative_config is not None:\n"
        "            _k = int(_env_k)\n"
        "            _ns = self.vllm_config.speculative_config.num_speculative_tokens\n"
        "            if not 1 <= _k <= _ns:\n"
        "                raise ValueError(\n"
        "                    f\"DFLASH2_EMIT_K={_k} out of range 1..{_ns}\"\n"
        "                )\n"
        "            self.uniform_decode_query_len = 1 + _k\n"
        "            if _k < _ns:\n"
        "                logger.info(\n"
        "                    \"DFlash2 v42d dispatcher: uniform_decode_query_len=\"\n"
        "                    \"%d (config width %d)\",\n"
        "                    1 + _k,\n"
        "                    1 + _ns,\n"
        "                )\n",
        "dispatcher width cap",
    )

    dst.write_text(t)
    py_compile.compile(str(dst), doraise=True)
    assert t.count('os.environ.get("DFLASH2_EMIT_K"') == 1
    assert t.count("self.uniform_decode_query_len = 1 + _k") == 1
    assert t.count("\nimport os\n") == 1
    assert t.count("logger = init_logger(__name__)") == 1
    print("  ok: py_compile + greps")
    print("DFLASH2_DISP_EDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
