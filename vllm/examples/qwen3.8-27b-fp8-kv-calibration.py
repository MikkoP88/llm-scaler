# SPDX-License-Identifier: Apache-2.0
"""Calibrate Qwen3.8-27B FP8 KV-cache scales with llm-compressor.

Produces a checkpoint with `quantization_config.kv_cache_scheme` and per-layer
k_scale/v_scale tensors so vLLM uses calibrated (not default 1.0) KV scales.

Run on a CUDA machine with enough memory to hold the model in bf16
(~55 GB weights + activations): single GPU >= 64-80 GB, or pass
device="cuda:0,1" for multi-GPU oneshot. Output is portable; copy the
SAVE_DIR to the inference server afterwards.

Adapted from the llm-compressor KV cache quantization example:
https://docs.vllm.ai/projects/llm-compressor/en/0.10.0/examples/quantization_kv_cache/
"""

import gc

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.8-27B"  # or a local path, e.g. "/models/qwen3.8-27b"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-KV"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096

# Static W8A8 FP8 + calibrated FP8 KV cache (official recipe).
RECIPE = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""


def load_model():
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    arch = getattr(cfg, "architectures", [""])[0]
    if arch.endswith("ForConditionalGeneration"):
        # Qwen3.5/3.8 vision-language variants (e.g. Qwen3_5ForConditionalGeneration)
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype="auto", trust_remote_code=True
        )
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype="auto", trust_remote_code=True
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = load_model()

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]"
    )
    ds = ds.shuffle(seed=42)

    def process_and_tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(
            text,
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

    from llmcompressor import oneshot

    oneshot(
        model=model,
        dataset=ds,
        recipe=RECIPE,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Sanity check before saving.
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))

    gc.collect()
    torch.cuda.empty_cache()
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Saved calibrated model to {SAVE_DIR}")


if __name__ == "__main__":
    main()
