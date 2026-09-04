#!/usr/bin/env python3
"""dflash2_dbg_edit.py — DISPOSABLE v42 debug instrumentation.

Never commit. One-shot logs at the three draft-width decision points in
the (v40+v41+v42-patched) gpu_model_runner.py:
  D1 _get_draft_token_ids_cpu: type of self._draft_token_ids at take time
  D2 _copy_draft_token_ids_to_cpu: async-gate values + type
  D3 _prepare_input_ids: total_num_spec_tokens the scheduler actually
     scheduled (sum of per-request draft_len)
Run AFTER dflash2_emitk_edit.py in phase 1.
"""
import py_compile
from pathlib import Path

W = Path("/w")
REL = "vllm/v1/worker/gpu_model_runner.py"


def edit(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    assert n == 1, f"{what}: expected exactly 1 occurrence, found {n}"
    print(f"  ok: {what}")
    return text.replace(old, new)


def main() -> int:
    p = W / "patched" / REL
    t = p.read_text()

    t = edit(
        t,
        "    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:\n"
        "        if isinstance(self._draft_token_ids, list):\n",
        "    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:\n"
        "        if not getattr(self, \"_dbg_gdt\", False):\n"
        "            self._dbg_gdt = True\n"
        "            _d = self._draft_token_ids\n"
        "            logger.info(\n"
        "                \"V42DBG D1 get_draft_cpu type=%s shape=%s reqids=%s\",\n"
        "                type(_d).__name__,\n"
        "                tuple(_d.shape) if torch.is_tensor(_d) else \"n/a\",\n"
        "                self._draft_token_req_ids is not None,\n"
        "            )\n"
        "        if isinstance(self._draft_token_ids, list):\n",
        "D1 get_draft_cpu type probe",
    )

    t = edit(
        t,
        "        # Check if we need to copy draft tokens to CPU. In async scheduling,\n"
        "        # we only copy when needed for structured output, penalties or bad_words.\n"
        "        if self.use_async_scheduling and not (\n",
        "        # Check if we need to copy draft tokens to CPU. In async scheduling,\n"
        "        # we only copy when needed for structured output, penalties or bad_words.\n"
        "        if not getattr(self, \"_dbg_cp\", False):\n"
        "            self._dbg_cp = True\n"
        "            logger.info(\n"
        "                \"V42DBG D2 copy_gate async=%s structured=%s outtok=%s \"\n"
        "                \"type=%s\",\n"
        "                self.use_async_scheduling,\n"
        "                scheduler_output.has_structured_output_requests,\n"
        "                bool(self.input_batch.sampling_metadata.output_token_ids),\n"
        "                type(self._draft_token_ids).__name__,\n"
        "            )\n"
        "        if self.use_async_scheduling and not (\n",
        "D2 copy gate probe",
    )

    t = edit(
        t,
        "        num_common_tokens = len(sample_flattened_indices)\n"
        "        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens\n",
        "        if not getattr(self, \"_dbg_pii\", False):\n"
        "            self._dbg_pii = True\n"
        "            logger.info(\n"
        "                \"V42DBG D3 prepare_input_ids total_num_spec_tokens=%d \"\n"
        "                \"num_common=%d total_scheduled=%d\",\n"
        "                total_num_spec_tokens,\n"
        "                len(sample_flattened_indices),\n"
        "                total_num_scheduled_tokens,\n"
        "            )\n"
        "        num_common_tokens = len(sample_flattened_indices)\n"
        "        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens\n",
        "D3 scheduled spec width probe",
    )

    p.write_text(t)
    py_compile.compile(str(p), doraise=True)
    n = t.count("V42DBG")
    assert n == 3, f"V42DBG: expected 3 occurrences, found {n}"
    print("  ok: py_compile + greps")
    print("DFLASH2_DBG_EDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
