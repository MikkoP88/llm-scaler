#!/usr/bin/env python3
"""dflash2_udql_edit.py — v42d: width-aware graph capture / uniform-decode
classification for the DFLASH2_EMIT_K cap.

With the v42c async-scheduler cap (placeholder [-1]*k), the verify batch is
uniform at 1+k rows per request. But the runner classifies uniform decode
via `max_num_scheduled_tokens == self.uniform_decode_query_len` and sizes
graph capture from the same attribute, which is 1+num_spec_tokens (8) from
config. Consequences measured live on the k=4 capped lane with graphs on:

  - runtime batches (5 rows) classify as NON-uniform while graphs were
    captured at 8-row uniform shapes -> replay corrupts numerics
    (' Paris!!!!...', 100% garbage acceptance)
  - cudagraph_mode=NONE (eager) is correct but 2-5x slower (87.9 vs 411.6
    tok/s at 2k; 75.8 vs 123.6 conc8)

TWO coordinated edits:
  (1) gpu_model_runner: uniform_decode_query_len = 1+k (runner-side
      classification + capture shapes). Buffer sizing unaffected
      (max_num_tokens comes from max_num_batched_tokens; the drafter's
      8-row internal layout is driven by num_spec_tokens in the v40 port).
  (2) config/vllm.py _set_cudagraph_sizes: rescale the capture lattice to
      multiples of 1+k. Stock sizes {8,16,24,...} are implicitly multiples
      of the config width 8; the dispatcher asserts
      `padded_size % uniform_decode_query_len == 0` (boot-time conviction:
      AssertionError at cudagraph_dispatcher.py:144 with width 5 against
      the stock lattice), so the lattice must be rebuilt as bs*(1+k).

config/vllm.py is NOT staged by dflash2_edit.py — this patcher stages the
stock copy from site-packages first. Run AFTER dflash2_schedk_edit.py.
Default (env unset): byte-identical stock behavior.
"""
import py_compile
import shutil
from pathlib import Path

W = Path("/w")
SP = Path("/opt/venv/lib/python3.12/site-packages")
REL_RUNNER = "vllm/v1/worker/gpu_model_runner.py"
REL_CONFIG = "vllm/config/vllm.py"


def edit(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    assert n == 1, f"{what}: expected exactly 1 occurrence, found {n}"
    print(f"  ok: {what}")
    return text.replace(old, new)


def main() -> int:
    p = W / "patched" / REL_RUNNER
    t = p.read_text()

    assert "import os\n" in t, "gpu_model_runner.py: import os missing"

    t = edit(
        t,
        "        self.uniform_decode_query_len = 1 + self.num_spec_tokens\n",
        "        # llm-scaler v42d: width-aware graph capture / uniform-decode\n"
        "        # classification for the DFLASH2_EMIT_K cap. With the v42c\n"
        "        # scheduler cap the verify batch is uniform at 1+k rows; this\n"
        "        # attribute must match or runtime batches misclassify and\n"
        "        # graphs capture at config width 1+7 — replay then corrupts\n"
        "        # numerics (convicted on the k=4 lane; eager is correct but\n"
        "        # 2-5x slower). Buffer sizing is unaffected: max_num_tokens\n"
        "        # comes from max_num_batched_tokens and the drafter's 8-row\n"
        "        # internal layout is driven by num_spec_tokens (v40 port).\n"
        "        _udql = 1 + self.num_spec_tokens\n"
        "        _env_k = os.environ.get(\"DFLASH2_EMIT_K\", \"\")\n"
        "        if _env_k:\n"
        "            _k = int(_env_k)\n"
        "            if not 1 <= _k <= self.num_spec_tokens:\n"
        "                raise ValueError(\n"
        "                    f\"DFLASH2_EMIT_K={_k} out of range 1..\"\n"
        "                    f\"{self.num_spec_tokens} (num_spec_tokens)\"\n"
        "                )\n"
        "            _udql = 1 + _k\n"
        "        self.uniform_decode_query_len = _udql\n"
        "        if _udql < 1 + self.num_spec_tokens:\n"
        "            logger.info(\n"
        "                \"DFlash2 v42d: uniform_decode_query_len=%d (config \"\n"
        "                \"width %d) — graphs capture at effective width\",\n"
        "                _udql,\n"
        "                1 + self.num_spec_tokens,\n"
        "            )\n",
        "uniform_decode_query_len cap",
    )

    p.write_text(t)
    py_compile.compile(str(p), doraise=True)
    assert t.count("self.uniform_decode_query_len = _udql") == 1
    assert t.count("self.uniform_decode_query_len = 1 + self.num_spec_tokens") == 0
    print("  ok: py_compile + greps (runner)")

    # --- edit 2: config/vllm.py capture lattice rescale ---
    dst = W / "patched" / REL_CONFIG
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Unconditional re-stage (see dflash2_schedk_edit.py rationale).
    shutil.copy2(SP / REL_CONFIG, dst)
    print("  ok: staged stock config/vllm.py")

    c = dst.read_text()
    assert "import os\n" in c, "config/vllm.py: import os missing"

    c = edit(
        c,
        "                # de-duplicate and sort the sizes\n"
        "                cudagraph_capture_sizes = sorted(set(cudagraph_capture_sizes))\n",
        "                # de-duplicate and sort the sizes\n"
        "                cudagraph_capture_sizes = sorted(set(cudagraph_capture_sizes))\n"
        "                # llm-scaler v42d: rescale the capture lattice to the\n"
        "                # effective uniform width when the DFLASH2_EMIT_K cap\n"
        "                # is active. Stock sizes are multiples of 8 == config\n"
        "                # width 1+7; the dispatcher asserts\n"
        "                # padded_size % uniform_decode_query_len == 0\n"
        "                # (boot-time conviction at cudagraph_dispatcher.py:144\n"
        "                # with width 5 against the stock lattice), so rebuild\n"
        "                # the lattice as bs*(1+k).\n"
        "                _env_k = os.environ.get(\"DFLASH2_EMIT_K\", \"\")\n"
        "                if _env_k:\n"
        "                    _k = int(_env_k)\n"
        "                    if not 1 <= _k <= self.num_speculative_tokens:\n"
        "                        raise ValueError(\n"
        "                            f\"DFLASH2_EMIT_K={_k} out of range 1..\"\n"
        "                            f\"{self.num_speculative_tokens}\"\n"
        "                        )\n"
        "                    _w = 1 + _k\n"
        "                    _max_bs = max_cudagraph_capture_size // _w\n"
        "                    _bs = [i for i in [1, 2, 4] if i <= _max_bs]\n"
        "                    if _max_bs >= 8:\n"
        "                        _bs += list(range(8, _max_bs + 1, 8))\n"
        "                    cudagraph_capture_sizes = sorted(\n"
        "                        set(b * _w for b in _bs)\n"
        "                    )\n"
        "                    logger.info(\n"
        "                        \"DFlash2 v42d: capture lattice rescaled to \"\n"
        "                        \"uniform width %d: %d sizes, max %d\",\n"
        "                        _w,\n"
        "                        len(cudagraph_capture_sizes),\n"
        "                        cudagraph_capture_sizes[-1],\n"
        "                    )\n",
        "capture lattice rescale",
    )

    dst.write_text(c)
    py_compile.compile(str(dst), doraise=True)
    assert c.count("capture lattice rescaled to") == 1
    assert c.count("_bs += list(range(8, _max_bs + 1, 8))") == 1
    print("  ok: py_compile + greps (config)")
    print("DFLASH2_UDQL_EDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

