#!/usr/bin/env python3
"""dflash2_edit.py — prepare DFlash2-patched site-package files (era-3).

Runs inside a helper container from the lane image with /w bound to
/root/build/qwen38-dflash2. Reads the stock files from site-packages,
applies the era-3 edits with strict one-match assertions, writes patched
copies under /w/patched/<relpath>, then byte-compiles every patched and new
module and runs an import smoke test against a scratch overlay. Never
touches the live site-packages of a serving container.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

SP = Path("/opt/venv/lib/python3.12/site-packages")
W = Path("/w")

EDITED = [
    "vllm/v1/spec_decode/utils.py",
    "vllm/v1/spec_decode/dflash.py",
    "vllm/model_executor/models/qwen3_dflash.py",
    "vllm/model_executor/models/registry.py",
    "vllm/v1/worker/gpu_model_runner.py",
]


def edit(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    assert n == 1, f"{what}: expected exactly 1 occurrence, found {n}"
    print(f"  ok: {what}")
    return text.replace(old, new)


def main() -> int:
    # ---- vllm/v1/spec_decode/utils.py: SKIP_BONUS in the input kernel ----
    p = SP / "vllm/v1/spec_decode/utils.py"
    t = p.read_text()
    t = edit(
        t,
        "    BLOCK_SIZE: tl.constexpr,\n"
        "    HAS_NUM_REJECTED: tl.constexpr = False,\n"
        "):\n",
        "    BLOCK_SIZE: tl.constexpr,\n"
        "    HAS_NUM_REJECTED: tl.constexpr = False,\n"
        "    SKIP_BONUS: tl.constexpr = False,\n"
        "):\n",
        "utils.py: SKIP_BONUS constexpr",
    )
    t = edit(
        t,
        "    # --- Token indices to sample (all k draft slots at offsets 0..k-1) ---\n"
        "    is_sample = is_query\n"
        "    sample_out_idx = req_idx * num_speculative_tokens + query_off\n",
        "    # --- Token indices to sample (all k draft slots at offsets 0..k-1) ---\n"
        "    if SKIP_BONUS:\n"
        "        # DFlash2 (1+N row layout): the bonus anchor row is not\n"
        "        # sampled; the N mask rows fill slots 0..N-1.\n"
        "        is_sample = is_query & (query_off >= 1)\n"
        "        sample_out_idx = (\n"
        "            req_idx * num_speculative_tokens + query_off - 1\n"
        "        )\n"
        "    else:\n"
        "        is_sample = is_query\n"
        "        sample_out_idx = req_idx * num_speculative_tokens + query_off\n",
        "utils.py: skip-bonus sample indices",
    )
    (W / "patched/vllm/v1/spec_decode/utils.py").parent.mkdir(parents=True, exist_ok=True)
    (W / "patched/vllm/v1/spec_decode/utils.py").write_text(t)

    # ---- vllm/v1/spec_decode/dflash.py: parameterized num_query_per_req ----
    p = SP / "vllm/v1/spec_decode/dflash.py"
    t = p.read_text()
    t = edit(
        t,
        "        num_query_per_req = self.num_speculative_tokens"
        "  # OMP: SpecForge DSpark block layout (anchor + k-1 masks)\n",
        "        # llm-scaler era-3 (DFlash2): a proposer may declare"
        " num_query_per_req\n"
        "        # (DFlash2 = 1 + N: bonus anchor + N masks, matching the conv\n"
        "        # block_size); legacy SpecForge DSpark layout keeps the\n"
        "        # anchor + (k-1) masks = N rows.\n"
        "        num_query_per_req = (\n"
        "            getattr(self, \"num_query_per_req\", 0)"
        " or self.num_speculative_tokens\n"
        "        )\n",
        "dflash.py: num_query_per_req hook",
    )
    t = edit(
        t,
        "                HAS_NUM_REJECTED=has_num_rejected,\n",
        "                HAS_NUM_REJECTED=has_num_rejected,\n"
        "                SKIP_BONUS=(\n"
        "                    num_query_per_req"
        " == 1 + self.num_speculative_tokens\n"
        "                ),\n",
        "dflash.py: SKIP_BONUS kernel arg",
    )
    (W / "patched/vllm/v1/spec_decode/dflash.py").parent.mkdir(parents=True, exist_ok=True)
    (W / "patched/vllm/v1/spec_decode/dflash.py").write_text(t)

    # ---- qwen3_dflash.py: decoder_layer_cls / model_cls hooks ----
    p = SP / "vllm/model_executor/models/qwen3_dflash.py"
    t = p.read_text()
    t = edit(
        t,
        "class DFlashQwen3Model(nn.Module):\n    def __init__(\n",
        "class DFlashQwen3Model(nn.Module):\n"
        "    # llm-scaler era-3 (DFlash2): layer-class hook.\n"
        "    decoder_layer_cls = DFlashQwen3DecoderLayer\n\n"
        "    def __init__(\n",
        "qwen3_dflash.py: decoder_layer_cls hook",
    )
    t = edit(
        t,
        "                DFlashQwen3DecoderLayer(\n                    current_vllm_config,\n",
        "                self.decoder_layer_cls(\n                    current_vllm_config,\n",
        "qwen3_dflash.py: layer construction via hook",
    )
    t = edit(
        t,
        "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
        "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n",
        "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
        "    # llm-scaler era-3 (DFlash2): model-class hook.\n"
        "    model_cls = DFlashQwen3Model\n\n"
        "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n",
        "qwen3_dflash.py: model_cls hook",
    )
    t = edit(
        t,
        "        self.model = DFlashQwen3Model(\n            vllm_config=vllm_config,\n            prefix=\"model\",\n",
        "        self.model = self.model_cls(\n            vllm_config=vllm_config,\n            prefix=\"model\",\n",
        "qwen3_dflash.py: model construction via hook",
    )
    (W / "patched/vllm/model_executor/models/qwen3_dflash.py").parent.mkdir(
        parents=True, exist_ok=True
    )
    (W / "patched/vllm/model_executor/models/qwen3_dflash.py").write_text(t)

    # ---- registry.py: DFlash2DraftModel entry ----
    p = SP / "vllm/model_executor/models/registry.py"
    t = p.read_text()
    t = edit(
        t,
        "    \"DFlashDraftModel\": (\"qwen3_dflash\", \"DFlashQwen3ForCausalLM\"),\n",
        "    \"DFlashDraftModel\": (\"qwen3_dflash\", \"DFlashQwen3ForCausalLM\"),\n"
        "    \"DFlash2DraftModel\": (\"qwen3_dflash2\", \"DFlash2Qwen3ForCausalLM\"),\n",
        "registry.py: DFlash2DraftModel",
    )
    (W / "patched/vllm/model_executor/models/registry.py").parent.mkdir(
        parents=True, exist_ok=True
    )
    (W / "patched/vllm/model_executor/models/registry.py").write_text(t)

    # ---- gpu_model_runner.py: DFlash2Proposer selection ----
    p = SP / "vllm/v1/worker/gpu_model_runner.py"
    t = p.read_text()
    t = edit(
        t,
        "            elif self.speculative_config.use_dflash():\n"
        "                self.drafter = DFlashProposer("
        "self.vllm_config, self.device, self)\n"
        "                self.use_aux_hidden_state_outputs = True\n",
        "            elif self.speculative_config.use_dflash():\n"
        "                draft_archs = (\n"
        "                    getattr(\n"
        "                        self.speculative_config.draft_model_config,\n"
        "                        \"architectures\",\n"
        "                        None,\n"
        "                    )\n"
        "                    or []\n"
        "                )\n"
        "                if any(\"DFlash2\" in str(a) for a in draft_archs):\n"
        "                    from vllm.v1.spec_decode.dflash2 import (\n"
        "                        DFlash2Proposer,\n"
        "                    )\n\n"
        "                    self.drafter = DFlash2Proposer(\n"
        "                        self.vllm_config, self.device, self\n"
        "                    )\n"
        "                else:\n"
        "                    self.drafter = DFlashProposer(\n"
        "                        self.vllm_config, self.device, self\n"
        "                    )\n"
        "                self.use_aux_hidden_state_outputs = True\n",
        "gpu_model_runner.py: DFlash2Proposer selection",
    )
    if os.environ.get("DFLASH2_VERBATIM_LIDS") == "1":
        # A/B lever: upstream DFlash2 checkpoints train target_layer_ids
        # against verbatim layer indices; the fork's +1 conversion is
        # DFlash-v1 semantics. Measured a wash (0.9% vs 1.0% acceptance),
        # so it stays opt-in to keep DFlash-v1 lanes on stock behavior.
        t = edit(
            t,
            "                # Add 1 to convert DFlash's aux layer id semantics.\n"
            "                layer_ids = [i + 1 for i in"
            " dflash_config.get(\"target_layer_ids\", [])]\n",
            "                # llm-scaler era-3 (DFlash2 A/B): upstream DFlash2"
            " checkpoints\n"
            "                # train target_layer_ids against verbatim layer"
            " indices; the\n"
            "                # +1 conversion is DFlash-v1 semantics.\n"
            "                layer_ids = [i for i in"
            " dflash_config.get(\"target_layer_ids\", [])]\n",
            "gpu_model_runner.py: DFlash2 verbatim target_layer_ids",
        )
    (W / "patched/vllm/v1/worker/gpu_model_runner.py").parent.mkdir(
        parents=True, exist_ok=True
    )
    (W / "patched/vllm/v1/worker/gpu_model_runner.py").write_text(t)

    # ---- byte-compile patched + new modules ----
    import py_compile

    for rel in EDITED:
        py_compile.compile(str(W / "patched" / rel), doraise=True)
    py_compile.compile(str(W / "new/qwen3_dflash2.py"), doraise=True)
    py_compile.compile(str(W / "new/dflash2_proposer.py"), doraise=True)
    print("  ok: py_compile all modules")

    # ---- import smoke test on a scratch overlay (does not touch SP) ----
    # Overlay the patched+new files on top of a venv copy via sys.path
    # shim: copy patched files into SP of THIS ephemeral helper container
    # (it is disposable), then import.
    for rel in EDITED:
        dst = SP / rel
        shutil.copy2(W / "patched" / rel, dst)
    shutil.copy2(
        W / "new/qwen3_dflash2.py",
        SP / "vllm/model_executor/models/qwen3_dflash2.py",
    )
    shutil.copy2(
        W / "new/dflash2_proposer.py",
        SP / "vllm/v1/spec_decode/dflash2.py",
    )
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm.model_executor.models.qwen3_dflash2 as m; "
            "import vllm.v1.spec_decode.dflash2 as p; "
            "assert hasattr(m.DFlash2Qwen3ForCausalLM, 'model_cls'); "
            "assert m.DFlash2Qwen3ForCausalLM.model_cls "
            "is m.DFlash2Qwen3Model; "
            "print('IMPORT_OK', m.DFlash2Qwen3ForCausalLM.__name__, "
            "p.DFlash2Proposer.__name__)",
        ],
        capture_output=True,
        text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-3000:])
        print("EDIT_FAIL import smoke test")
        return 4
    print("DFLASH2_EDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
