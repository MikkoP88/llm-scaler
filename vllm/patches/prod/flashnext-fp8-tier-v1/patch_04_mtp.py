#!/usr/bin/env python3
"""FN8_04 (flashnext-fp8-tier-v1 P4): teach this vLLM build the qwen4_exp
in-checkpoint MTP head, so --speculative-config '{"method":"mtp",...}' boots.

Boot-13 law: this build's SpeculativeConfig auto-detect raises
`NotImplementedError: Unsupported speculative method: 'mtp'` (speculative.py
~686) because qwen4_exp is not in MTPModelTypes and has no draft-model
registry entry.  The checkpoint DOES ship the head (3101 mtp.* tensors).

Edits (all tagged FN8_04, idempotent, --restore reverses):
  1. INSTALL  files/qwen4_exp_mtp.py -> models/qwen4_exp_mtp.py
     (new module: stem + 1 full-attention MoE decoder layer + mixer for
     compute_logits; forward returns the pre-mixer (T, hc*H) hyper stream,
     DeepseekV4 MTP law).
  2. models/registry.py: draft entry "Qwen4ExpMTP".
  3. config/speculative.py: MTPModelTypes += "qwen4_exp_mtp" and an
     hf_config_override branch (model_type -> qwen4_exp_mtp, n_predict from
     text_config.mtp_num_hidden_layers, architectures=["Qwen4ExpMTP"]).
  4. v1/spec_decode/llm_base_proposer.py: hyper-stream draft detection —
     draft buffer hidden_size *= hc_count (qwen4 analogue of dsv4 hc_mult).
  5. models/qwen4_exp.py: Qwen4ExpModel allocates _mtp_hidden_buffer
     (max_num_batched_tokens x hc_count*hidden, spec-gated) and stashes the
     pre-mixer stream in forward(); ForConditionalGeneration exposes
     get_mtp_target_hidden_states() — the existing model_runner duck-type
     (gpu/model_runner.py:525/1286) slices [:T] into propose().

Laws: every in-place edit py_compile-checked on a temp file before atomic
os.replace; md5 printed after each write; backups at <file>.fn8_04.bak.
Order: apply AFTER patch_00b (edits live post-00b qwen4_exp.py text).
"""
import hashlib
import importlib.util
import os
import py_compile
import shutil
import sys
import tempfile

TAG = "FN8_04"
MODULE_FILENAME = "qwen4_exp_mtp.py"  # payload, ./files/ then ./

VLLM_ROOT = None
spec = importlib.util.find_spec("vllm")
if spec is not None and spec.origin:
    # spec.origin == .../site-packages/vllm/__init__.py
    PKG = os.path.dirname(spec.origin)
else:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def backup(path):
    bak = path + ".fn8_04.bak"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
        print(f"{TAG} backup {bak} md5={md5(bak)}")
    return bak


def edit_in_place(rel, anchor, replacement, already, expect=1):
    """Replace `anchor` with `replacement` in PKG/rel; anchor must occur
    exactly `expect` times.  Idempotent: skip when `already` (a unique
    substring of the replacement that cannot pre-exist) is present."""
    path = os.path.join(PKG, rel)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if already in src:
        print(f"{TAG} [{rel}] edit already present, skip")
        return
    n = src.count(anchor)
    if n != expect:
        print(f"{TAG} FAIL [{rel}]: anchor occurs {n}x (want {expect})")
        sys.exit(2)
    backup(path)
    out = src.replace(anchor, replacement)
    # compile-check BEFORE touching the file
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(out)
        tmp = tf.name
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"{TAG} FAIL [{rel}]: compile error -> {e}")
        sys.exit(3)
    finally:
        os.unlink(tmp)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"{TAG} [{rel}] edited, md5={md5(path)}")


def restore():
    ok = True
    for rel in TO_RESTORE:
        path = os.path.join(PKG, rel)
        bak = path + ".fn8_04.bak"
        if os.path.exists(bak):
            shutil.copyfile(bak, path)
            print(f"{TAG} restored {rel} md5={md5(path)}")
        else:
            print(f"{TAG} no backup for {rel}")
            ok = False
    payload = os.path.join(PKG, "model_executor", "models", MODULE_FILENAME)
    if os.path.exists(payload):
        os.unlink(payload)
        print(f"{TAG} removed {MODULE_FILENAME}")
    sys.exit(0 if ok else 4)


TO_RESTORE = [
    "model_executor/models/registry.py",
    "config/speculative.py",
    "v1/spec_decode/llm_base_proposer.py",
    "model_executor/models/qwen4_exp.py",
]

if "--restore" in sys.argv:
    restore()

here = os.path.dirname(os.path.abspath(__file__))

# ---- 1. install the draft-model payload ----------------------------------
payload_src = None
for cand in (
    os.path.join(here, "files", MODULE_FILENAME),
    os.path.join(here, MODULE_FILENAME),
):
    if os.path.exists(cand):
        payload_src = cand
        break
if payload_src is None:
    print(f"{TAG} FAIL: payload {MODULE_FILENAME} not found (./files/ or ./)")
    sys.exit(5)

dst = os.path.join(PKG, "model_executor", "models", MODULE_FILENAME)
py_compile.compile(payload_src, doraise=True)
shutil.copyfile(payload_src, dst)
print(f"{TAG} installed {MODULE_FILENAME} md5={md5(dst)}")

# ---- 2. registry draft entry ----------------------------------------------
edit_in_place(
    "model_executor/models/registry.py",
    '    "Qwen3NextMTP": ("qwen3_next_mtp", "Qwen3NextMTP"),\n',
    '    "Qwen3NextMTP": ("qwen3_next_mtp", "Qwen3NextMTP"),\n'
    '    "Qwen4ExpMTP": ("qwen4_exp_mtp", "Qwen4ExpMTP"),  # FN8_04\n',
    already='"Qwen4ExpMTP": ("qwen4_exp_mtp"',
)

# ---- 3a. MTPModelTypes membership ------------------------------------------
edit_in_place(
    "config/speculative.py",
    '    "qwen3_5_mtp",\n    "longcat_flash_mtp",\n',
    '    "qwen3_5_mtp",\n'
    '    "qwen4_exp_mtp",  # FN8_04\n'
    '    "longcat_flash_mtp",\n',
    already='    "qwen4_exp_mtp",  # FN8_04',
)

# ---- 3b. hf_config_override qwen4_exp branch -------------------------------
edit_in_place(
    "config/speculative.py",
    '        if hf_config.model_type == "qwen3_next":\n'
    '            hf_config.model_type = "qwen3_next_mtp"\n'
    '        if hf_config.model_type == "qwen3_next_mtp":\n'
    '            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)\n'
    '            hf_config.update(\n'
    '                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}\n'
    '            )\n',
    '        if hf_config.model_type == "qwen3_next":\n'
    '            hf_config.model_type = "qwen3_next_mtp"\n'
    '        if hf_config.model_type == "qwen3_next_mtp":\n'
    '            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)\n'
    '            hf_config.update(\n'
    '                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}\n'
    '            )\n'
    '\n'
    '        # FN8_04: qwen4_exp (flash-next) in-checkpoint MTP head.  n_predict\n'
    '        # lives on the text config (mtp_num_hidden_layers); the draft\n'
    '        # rides the target\'s pre-mixer hyper stream (hc_count * hidden).\n'
    '        if hf_config.model_type == "qwen4_exp":\n'
    '            hf_config.model_type = "qwen4_exp_mtp"\n'
    '        if hf_config.model_type == "qwen4_exp_mtp":\n'
    '            tc = getattr(hf_config, "text_config", hf_config)\n'
    '            n_predict = getattr(tc, "mtp_num_hidden_layers", None)\n'
    '            hf_config.update(\n'
    '                {"n_predict": n_predict, "architectures": ["Qwen4ExpMTP"]}\n'
    '            )\n',
    already='if hf_config.model_type == "qwen4_exp_mtp":',
)

# ---- 4. proposer hyper-stream draft detection -------------------------------
edit_in_place(
    "v1/spec_decode/llm_base_proposer.py",
    '        if hasattr(draft_hf_config, "compress_ratios") and hasattr(\n'
    '            draft_hf_config, "hc_mult"\n'
    '        ):\n'
    '            self.hidden_size = self.hidden_size * draft_hf_config.hc_mult\n',
    '        if hasattr(draft_hf_config, "compress_ratios") and hasattr(\n'
    '            draft_hf_config, "hc_mult"\n'
    '        ):\n'
    '            self.hidden_size = self.hidden_size * draft_hf_config.hc_mult\n'
    '        elif getattr(draft_hf_config, "model_type", "") == "qwen4_exp_mtp":\n'
    '            # FN8_04: qwen4_exp MTP rides the target\'s pre-mixer hyper\n'
    '            # stream (hc_count * hidden), the qwen4 analogue of hc_mult.\n'
    '            tc = getattr(draft_hf_config, "text_config", draft_hf_config)\n'
    '            self.hidden_size = self.hidden_size * getattr(tc, "hc_count", 4)\n',
    already='== "qwen4_exp_mtp":\n            # FN8_04',
)

# ---- 5a. Qwen4ExpModel: spec-gated pre-mixer buffer -------------------------
edit_in_place(
    "model_executor/models/qwen4_exp.py",
    '        # Final hyper-connection mixer; its hc_norm IS the final norm\n'
    '        # (reference: no separate model.norm).\n'
    '        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)\n',
    '        # Final hyper-connection mixer; its hc_norm IS the final norm\n'
    '        # (reference: no separate model.norm).\n'
    '        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)\n'
    '\n'
    '        # FN8_04 (P4 MTP): stash of the PRE-mixer hyper stream for the\n'
    '        # speculative draft (DeepseekV4 buffer law: stable address\n'
    '        # outside the cudagraph pool; refreshed by copy_ in forward).\n'
    '        # Allocated only when an mtp draft rides it.\n'
    '        self._mtp_hidden_buffer = None\n'
    '        if (\n'
    '            vllm_config.speculative_config is not None\n'
    '            and vllm_config.speculative_config.method == "mtp"\n'
    '        ):\n'
    '            self._mtp_hidden_buffer = torch.empty(\n'
    '                vllm_config.scheduler_config.max_num_batched_tokens,\n'
    '                self.hc_count * config.hidden_size,\n'
    '                dtype=vllm_config.model_config.dtype,\n'
    '                device=next(self.parameters()).device,\n'
    '            )\n',
    already='self._mtp_hidden_buffer = None',
)

# ---- 5b. Qwen4ExpModel.forward: stash before the mixer ----------------------
edit_in_place(
    "model_executor/models/qwen4_exp.py",
    '        hidden_states = self.hyper_connection_mixer(hidden_states)\n'
    '        return hidden_states\n',
    '        # FN8_04: publish the pre-mixer stream for the MTP draft before\n'
    '        # the mixer collapses it (buffer is None unless spec mtp is on).\n'
    '        if self._mtp_hidden_buffer is not None:\n'
    '            self._mtp_hidden_buffer[: hidden_states.shape[0]].copy_(hidden_states)\n'
    '        hidden_states = self.hyper_connection_mixer(hidden_states)\n'
    '        return hidden_states\n',
    already='publish the pre-mixer stream',
)

# ---- 5c. ForConditionalGeneration: target-hidden-states accessor ------------
edit_in_place(
    "model_executor/models/qwen4_exp.py",
    '    # -- MoE plumbing (delegate to the causal LM which owns .model.layers) --\n',
    '    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:\n'
    '        """Pre-mixer hyper-stream buffer (max_num_batched_tokens,\n'
    '        hc_count * hidden_size) stashed by Qwen4ExpModel.forward for the\n'
    '        MTP draft (FN8_04).  model_runner slices [:T] into propose()."""\n'
    '        return getattr(self.language_model.model, "_mtp_hidden_buffer", None)\n'
    '\n'
    '    # -- MoE plumbing (delegate to the causal LM which owns .model.layers) --\n',
    already='def get_mtp_target_hidden_states',
)

print(f"{TAG} ALL_EDITS_OK")
