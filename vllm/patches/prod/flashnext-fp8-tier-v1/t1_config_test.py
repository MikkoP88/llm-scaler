#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 P0 round-1 dry-run (throwaway container fn8-t1).

Verifies, without any GPU:
  1. get_config resolves /models/flash-next-fp8 through OUR vendored classes
  2. the model registry resolves Qwen4ExpForConditionalGeneration
  3. the model module imports (heavy vllm import chain)
  4. checkpoint-side facts match the port's expectations
"""
import json
import os
import sys

import torch

MODEL = "/models/flash-next-fp8"

# --- 1. config resolution -------------------------------------------------
from vllm.transformers_utils.config import get_config

c = get_config(MODEL, trust_remote_code=False)
print("CONFIG_CLASS", type(c).__name__)
t = c.text_config
print("TEXT_CLASS", type(t).__name__, t.model_type)
print("layers", t.num_hidden_layers, "hidden", t.hidden_size,
      "experts", t.num_experts, "topk", t.num_experts_per_tok)
print("ple_layer_ids", t.ple_layer_ids, "conv_states", t.number_of_conv_states)
print("layer_types_first9", t.layer_types[:9])
print("hc", t.hc_count, t.hc_lowrank, "ngram", t.ngram_size,
      "heads_per_ngram", t.heads_per_ngram, "ple_dim", t.ple_embed_dim)
print("vocab", t.vocab_size, "eos", t.eos_token_id, "tie", t.tie_word_embeddings)
print("gdn", t.linear_num_key_heads, t.linear_num_value_heads,
      t.linear_key_head_dim, t.linear_value_head_dim, "convk", t.linear_conv_kernel_dim)
print("ple_conv_k", t.ple_conv_kernel_size, "split_parts", t.split_ngram_parts,
      "ngram_base", t.ngram_vocab_size_base, "seed", t.seed)

assert type(c).__name__ == "Qwen4ExpConfig"
assert type(t).__name__ == "Qwen4ExpTextConfig"
assert t.model_type == "qwen4_exp_text"
assert all(t.layer_types[i] == "linear_attention" for i in t.ple_layer_ids if i - 1 < len(t.layer_types))
assert all(lt in ("linear_attention", "full_attention") for lt in t.layer_types), "indexer leaked through"

# --- 2. registry ------------------------------------------------------------
from vllm.model_executor.models.registry import _VLLM_MODELS

assert "Qwen4ExpForConditionalGeneration" in _VLLM_MODELS
print("REGISTRY_ENTRY", _VLLM_MODELS["Qwen4ExpForConditionalGeneration"])
arch = ("Qwen4ExpForConditionalGeneration",)

# --- 3. module import --------------------------------------------------------
import vllm.model_executor.models.qwen4_exp as q4

print("MODULE_OK", q4.__file__)
cls = q4.Qwen4ExpForConditionalGeneration
assert cls.is_hybrid is True, "IsHybrid protocol flag missing"
assert hasattr(cls, "get_mamba_state_dtype_from_config"), "mamba state plumbing missing"

# --- 4. checkpoint cross-facts ----------------------------------------------
with open(MODEL + "/model.safetensors.index.json") as f:
    idx = json.load(f)
wm = idx["weight_map"]
ple = sorted(k for k in wm if "ngram_embedding" in k and "scale" not in k)
ple_scales = sorted(k for k in wm if "ngram_embedding" in k and "scale" in k)
bufs = sorted(k for k in wm if k.endswith(("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")))
print("CKPT ngram shards", len(ple), "scales", len(ple_scales), "buffers", len(bufs))
assert len(ple_scales) == 1, f"expected exactly the per-tensor weight_scale, got {ple_scales}"
assert len(bufs) == 3, f"expected 3 hash-buffer tensors, got {len(bufs)}"
vis = [k for k in wm if k.startswith("model.visual.")]
mtp = [k for k in wm if k.startswith("mtp.")]
print("CKPT visual", len(vis), "mtp", len(mtp), "total", len(wm), idx["metadata"])

# --- 5. hash-buffer math vs checkpoint (no table alloc: SKIP_TABLE mode) ----
os.environ.setdefault("VLLM_QW4_PLE_SKIP_TABLE", "1")
emb = q4.Qwen4ExpNGramEmbedding(t, t.ple_embed_dim, layer_idx=1, ple_layer_index=0)
assert emb.weight.shape[0] == 0, "SKIP_TABLE must not allocate the 51GB table"
assert emb.ngram_heads == (t.ngram_size - 1) * t.heads_per_ngram == 16
print("PLE_ID_MATH total_vocab", emb.total_vocab_size,
      "padded", emb.padded_vocab_size, "head_dim", emb.head_dim_per_ngram)

from safetensors import safe_open

buf_names = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
ckpt_bufs = {}
for name in bufs:
    short = name.rsplit(".", 1)[-1]
    with safe_open(MODEL + "/" + wm[name], framework="pt") as f:
        ckpt_bufs[short] = f.get_tensor(name)
    got = getattr(emb, short)
    same = got.shape == ckpt_bufs[short].shape and torch.equal(got.to(ckpt_bufs[short].dtype).cpu(), ckpt_bufs[short])
    print("BUF", short, tuple(ckpt_bufs[short].shape), "equal", same)
    assert same, f"hash-buffer {short} diverges from checkpoint"

assert emb.total_vocab_size <= 128 * 2500012 <= emb.padded_vocab_size, (
    f"shard rows 320,001,536 do not cover computed vocab {emb.total_vocab_size}"
)
print("ROWS_COVERED", emb.total_vocab_size, "<= 320001536 <=", emb.padded_vocab_size)

# --- 6. PLE layer construct (pure nn, CPU-safe) ------------------------------
ple_layer = q4.Qwen4ExpPLELayer(t, layer_idx=1, ple_layer_index=0)
n_params = sum(p.numel() for p in ple_layer.parameters())
print("PLE_LAYER params", n_params)
assert n_params > 0

print("T1_CONFIG_TEST_OK")
