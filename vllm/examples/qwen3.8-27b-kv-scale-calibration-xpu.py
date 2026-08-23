# SPDX-License-Identifier: Apache-2.0
"""Calibrate FP8 KV-cache scales for Qwen3.8-27B on Intel XPU (2x Arc Pro B70).

Measures per-layer post-RoPE key and value activation maxima with forward
hooks over a small calibration set, then registers k_scale/v_scale buffers
(amax / 448, the fp8 e4m3 maximum) and saves a copy of the checkpoint.

Serve the result with `--quantization fp8 --kv-cache-dtype fp8_e4m3`; vLLM
prefers checkpoint scales over the default 1.0 and the "Using KV cache
scaling factor 1.0" warning disappears.

Requirements (host Python or container with XPU torch):
    pip install transformers accelerate datasets
The model loads in bf16 (~55 GB) sharded over both GPUs; saving needs ~64 GB
of system RAM to gather the state dict. Vision and GDN layers are skipped —
only text attention layers get scales, matching what vLLM quantizes.
"""

import gc
import importlib
import json
import math

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.8-27B"  # or a local path, e.g. "/models/qwen3.8-27b"
OUTPUT_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-KV-Scales"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
MAX_MEMORY_GB_PER_GPU = 28
FP8_E4M3_MAX = 448.0
DATASET_ID = "HuggingFaceH4/ultrachat_200k"

TEXT_LAYER_EXCLUDE = ("visual", "vision", "tower")


def load_model():
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    arch = getattr(cfg, "architectures", [""])[0]
    if arch.endswith("ForConditionalGeneration"):
        from transformers import AutoModelForImageTextToText

        auto_cls = AutoModelForImageTextToText
    else:
        from transformers import AutoModelForCausalLM

        auto_cls = AutoModelForCausalLM
    max_memory = {
        i: f"{MAX_MEMORY_GB_PER_GPU}GiB" for i in range(torch.xpu.device_count())
    }
    return auto_cls.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )


def find_text_attn_modules(model):
    attns = {}
    for name, mod in model.named_modules():
        if "Attention" not in type(mod).__name__:
            continue
        if any(part in name for part in TEXT_LAYER_EXCLUDE):
            continue
        if not (hasattr(mod, "k_proj") and hasattr(mod, "v_proj")):
            continue
        attns[name] = mod
    return attns


class KVScaleTracker:
    def __init__(self, attns):
        self.k_amax = {name: 0.0 for name in attns}
        self.v_amax = {name: 0.0 for name in attns}
        self.current = {"name": None}
        self._attns = attns
        self._hooks = []
        self._patched_module = None
        self._orig_apply_rotary = None

    def install(self):
        # Post-RoPE K: patch apply_rotary_pos_emb in the module that defines
        # the attention class, attributing each call to the layer whose
        # forward is currently running (batch 1, sequential execution).
        first = next(iter(self._attns.values()))
        module = importlib.import_module(type(first).__module__)
        assert hasattr(module, "apply_rotary_pos_emb"), (
            f"{module.__name__} does not define apply_rotary_pos_emb; this "
            "attention implementation cannot be hooked for post-RoPE keys."
        )
        orig = module.apply_rotary_pos_emb
        self._patched_module = module
        self._orig_apply_rotary = orig
        tracker = self

        def patched(q, k, cos, sin, *args, **kwargs):
            q2, k2 = orig(q, k, cos, sin, *args, **kwargs)
            name = tracker.current["name"]
            if name is not None:
                m = k2.detach().abs().amax().item()
                if m > tracker.k_amax[name]:
                    tracker.k_amax[name] = m
            return q2, k2

        module.apply_rotary_pos_emb = patched

        for name, mod in self._attns.items():
            self._hooks.append(
                mod.register_forward_pre_hook(self._make_pre_hook(self, name))
            )
            self._hooks.append(
                mod.v_proj.register_forward_hook(self._make_v_hook(name))
            )

    @staticmethod
    def _make_pre_hook(tracker, name):
        def hook(_mod, _args):
            tracker.current["name"] = name

        return hook

    def _make_v_hook(self, name):
        tracker = self

        def hook(_mod, _inp, out):
            m = out.detach().abs().amax().item()
            if m > tracker.v_amax[name]:
                tracker.v_amax[name] = m

        return hook

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        if self._patched_module is not None:
            self._patched_module.apply_rotary_pos_emb = self._orig_apply_rotary
            self._patched_module = None
            self._orig_apply_rotary = None


def calibrate(model, tokenizer):
    attns = find_text_attn_modules(model)
    assert attns, "no text attention modules found"
    print(f"Calibrating {len(attns)} text attention layers")

    tracker = KVScaleTracker(attns)
    tracker.install()

    ds = load_dataset(
        DATASET_ID, split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]"
    ).shuffle(seed=42)
    device = next(model.parameters()).device

    try:
        for i, example in enumerate(ds):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
            enc = tokenizer(
                text,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                add_special_tokens=False,
                return_tensors="pt",
            )
            with torch.inference_mode():
                model(enc.input_ids.to(device))
            if (i + 1) % 32 == 0:
                print(f"  {i + 1}/{NUM_CALIBRATION_SAMPLES} prompts processed")
    finally:
        tracker.remove()

    return attns, tracker


def register_scales(attns, tracker):
    report = {}
    for name, mod in attns.items():
        k = tracker.k_amax[name] / FP8_E4M3_MAX
        v = tracker.v_amax[name] / FP8_E4M3_MAX
        assert math.isfinite(k) and k > 0.0, f"bad k scale for {name}: {k}"
        assert math.isfinite(v) and v > 0.0, f"bad v scale for {name}: {v}"
        mod.register_buffer(
            "k_scale", torch.tensor(k, dtype=torch.float32), persistent=True
        )
        mod.register_buffer(
            "v_scale", torch.tensor(v, dtype=torch.float32), persistent=True
        )
        report[name] = {
            "k_amax": tracker.k_amax[name],
            "v_amax": tracker.v_amax[name],
            "k_scale": k,
            "v_scale": v,
        }
    return report


def main():
    assert torch.xpu.is_available(), "torch.xpu is not available on this host"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = load_model()
    model.eval()

    attns, tracker = calibrate(model, tokenizer)
    report = register_scales(attns, tracker)
    print(
        f"k_scale range: {min(r['k_scale'] for r in report.values()):.4g}"
        f" .. {max(r['k_scale'] for r in report.values()):.4g}"
    )

    gc.collect()
    torch.xpu.empty_cache()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(f"{OUTPUT_DIR}/kv_scale_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved calibrated checkpoint to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
