#!/usr/bin/env python3
"""dflash2_emitk_edit.py — DFlash2 emission-width fix (v42, Path A part 2).

Companion to the DFLASH2_EMIT_K proposer knob (new/dflash2_proposer.py):
without this hook the knob truncates the draft EMISSION but the verify
batch stays at full config width, so k<7 only loses acceptance.

Defect: gpu_model_runner._prepare_input_ids zero-pads variable-width draft
lists (returned by propose()) to [B, num_spec_tokens] and STORES the padded
tensor back into self._draft_token_ids. Every later draft consumer
(take_draft_token_ids -> scheduler update_draft_token_ids; async
update_async_spec_token_ids) then reads num_spec_tokens-wide lists with a
zero tail: the scheduler schedules a full-width verify batch whose last
num_spec_tokens-k draft rows are token id 0 (measured on the k=4 lane:
378 drafts x exactly 7.0 tokens/step; per-position acceptance 325/268/220/
190/0/0/0 — positions 4-6 drafted as zeros, never accepted). Outputs stay
greedy-correct (zeros are rejected), but the width reduction — the entire
point of emission truncation — is defeated.

Fix: pad into a LOCAL tensor. Only the first len(toks) columns are ever
consumed by this function (prev_draft_token_indices uses the scheduler's
true draft_len), so a local tensor is sufficient and the stored list keeps
its true widths for the draft consumers.

Input: /w/patched/vllm/v1/worker/gpu_model_runner.py as produced by
dflash2_edit.py (v40 hooks already applied) — run this patcher AFTER
dflash2_edit.py in the boot/bake phase.
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

    # R1: pad into a LOCAL tensor; never overwrite self._draft_token_ids.
    t = edit(
        t,
        "        # Adaptive truncation may return variable-length lists; pad to tensor.\n"
        "        if isinstance(self._draft_token_ids, list):\n"
        "            padded = torch.zeros(\n"
        "                len(self._draft_token_ids),\n"
        "                self.num_spec_tokens,\n"
        "                dtype=torch.int32,\n"
        "                pin_memory=self.pin_memory,\n"
        "            )\n"
        "            for i, toks in enumerate(self._draft_token_ids):\n"
        "                if toks:\n"
        "                    padded[i, : len(toks)] = torch.tensor(\n"
        "                        toks, dtype=torch.int32\n"
        "                    )\n"
        "            self._draft_token_ids = padded.to(self.device, non_blocking=True)\n"
        "\n"
        "        assert isinstance(self._draft_token_ids, torch.Tensor)\n"
        "        draft_tokens_index_tensor = torch.tensor(\n",
        "        # Adaptive truncation may return variable-length lists; pad to tensor.\n"
        "        # llm-scaler v42: pad into a LOCAL tensor. Stock stored the\n"
        "        # padded [B, num_spec_tokens] tensor back into\n"
        "        # self._draft_token_ids, which re-widened variable-width\n"
        "        # drafts (zero tail) for every later draft consumer\n"
        "        # (take_draft_token_ids / async spec token updates) — the\n"
        "        # scheduler then scheduled full-width verify batches whose\n"
        "        # trailing draft rows were token id 0, defeating\n"
        "        # emission-width knobs (DFlash2 DFLASH2_EMIT_K; measured on\n"
        "        # the k=4 lane: 378 drafts x exactly 7.0 tokens/step,\n"
        "        # per-position acceptance 325/268/220/190/0/0/0 — positions\n"
        "        # 4-6 drafted as zeros, never accepted). Only the first\n"
        "        # len(toks) columns are consumed here\n"
        "        # (prev_draft_token_indices uses the scheduler's true\n"
        "        # draft_len), so a local tensor suffices and the stored\n"
        "        # list keeps its true widths for the draft consumers.\n"
        "        _dtids_local_padded = None\n"
        "        if isinstance(self._draft_token_ids, list):\n"
        "            padded = torch.zeros(\n"
        "                len(self._draft_token_ids),\n"
        "                self.num_spec_tokens,\n"
        "                dtype=torch.int32,\n"
        "                pin_memory=self.pin_memory,\n"
        "            )\n"
        "            for i, toks in enumerate(self._draft_token_ids):\n"
        "                if toks:\n"
        "                    padded[i, : len(toks)] = torch.tensor(\n"
        "                        toks, dtype=torch.int32\n"
        "                    )\n"
        "            _dtids_local_padded = padded.to(self.device, non_blocking=True)\n"
        "\n"
        "        if _dtids_local_padded is not None:\n"
        "            _dtids_src = _dtids_local_padded\n"
        "        else:\n"
        "            assert isinstance(self._draft_token_ids, torch.Tensor)\n"
        "            _dtids_src = self._draft_token_ids\n"
        "        draft_tokens_index_tensor = torch.tensor(\n",
        "gpu_model_runner.py: local draft pad (no store-back)",
    )

    # R2: scatter from the local tensor.
    t = edit(
        t,
        "        # because input_ids dtype is torch.int32,\n"
        "        # so convert draft_token_ids to torch.int32 here.\n"
        "        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)\n",
        "        # because input_ids dtype is torch.int32,\n"
        "        # so convert draft_token_ids to torch.int32 here.\n"
        "        draft_token_ids = _dtids_src.to(dtype=torch.int32)\n",
        "gpu_model_runner.py: scatter source local tensor",
    )

    p.write_text(t)
    py_compile.compile(str(p), doraise=True)
    print("  ok: py_compile gpu_model_runner.py")

    t2 = p.read_text()
    n = t2.count("_dtids_local_padded")
    assert n == 4, f"_dtids_local_padded: expected 4 occurrences, found {n}"
    assert "self._draft_token_ids = padded.to" not in t2
    print("  ok: hook greps")
    print("DFLASH2_EMITK_EDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
