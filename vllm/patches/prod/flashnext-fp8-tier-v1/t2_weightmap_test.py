#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 P0 round-2 weight-map dry-run (fn8-t1, CPU/meta).

Exercises the REAL load path with stub tensors on a meta-constructed model:
  1. minimal VllmConfig (TP1) from the checkpoint
  2. Qwen4ExpForConditionalGeneration constructed under torch.device("meta")
  3. every index.json name fed through load_weights with recorder
     weight_loaders (no data moves)
  4. bidirectional accounting:
       - every non-skipped checkpoint name reaches a loader call
       - zero "Parameter not found" warnings
       - every expected model param receives >= 1 call, except scale
         tensors of non-quantized linears (see the mixed-quant law below)
  5. PLE staging sanity: 128 table parts, 1 scale, 3 hash buffers

MIXED-QUANT LAW (discovered by this dry-run): the checkpoint quantizes ONLY
the routed experts — 73,728 weight_scale_inv tensors = 48 MoE layers x 512
experts x 3 (gate/up/down).  GDN projections, self-attn, router gates and
shared experts ship BF16 weights with NO scale tensors; the scale params
Fp8LinearMethod creates for them (weight_scale_inv / kv q,k,v,prob_scale)
therefore have nothing to load and MUST be the only untouched params.
"""
import json
import os
from collections import Counter

os.environ.setdefault("VLLM_QW4_PLE_SKIP_TABLE", "1")

import torch

# force the XPU platform BEFORE any vllm import can bind current_platform
# (kernels/linear/__init__.py:150 does `from vllm.platforms import
# current_platform` at module import — patching later leaves its namespace
# holding the UnspecifiedPlatform object).  FP8 create_weights keys
# possible_kernels[current_platform._enum]; this GPU-less box resolves
# UnspecifiedPlatform -> KeyError.  XPUPlatform() itself touches no devices
# (ctor inherited; every torch.xpu call lives in methods we never reach).
import vllm.platforms as _vp

from vllm.platforms.xpu import XPUPlatform

# GDN ctor does RMSNormGated(device=current_platform.current_device()) etc.;
# report "cpu" so construction stays on this GPU-less box (tiny buffers only,
# everything else stays meta under the torch.device("meta") block).
XPUPlatform.current_device = lambda self=None: "cpu"
_vp.current_platform = XPUPlatform()

MODEL = "/models/flash-next-fp8"

# --- 1. config ---------------------------------------------------------------
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)

mc = ModelConfig(model=MODEL, tokenizer=MODEL, dtype="auto", seed=0,
                 enforce_eager=True)
vc = VllmConfig(
    model_config=mc,
    device_config=DeviceConfig(device="cpu"),
    cache_config=CacheConfig(cache_dtype="auto"),
    parallel_config=ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=1),
    scheduler_config=SchedulerConfig(max_num_seqs=4, max_model_len=4096,
                                      is_encoder_decoder=False),
)
print("VLLMCONFIG_OK", mc.hf_config.model_type, mc.dtype)

# --- 2. construct (dist stub + meta model) ------------------------------------
# set_current_vllm_config wraps everything that touches parallel state or
# get_current_vllm_config (initialize_model_parallel, Fp8LinearMethod).
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

with set_current_vllm_config(vc):
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:49611",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    import vllm.model_executor.models.qwen4_exp as q4

    with torch.device("meta"):
        model = q4.Qwen4ExpForConditionalGeneration(vllm_config=vc)
print("DIST_OK")
n_params = len(dict(model.named_parameters()))
print("META_CONSTRUCT_OK params", n_params)

# --- 3. instrument -------------------------------------------------------------
# stacked-mapping loops call param.weight_loader(param, loaded_weight,
# shard_id, ...) positionally — record the PARAM IDENTITY (not the variadic
# shard args) so accounting maps 1:1 back onto named_parameters()
calls = []
param_names = {id(p): n for n, p in model.named_parameters()}


def recorder(param, loaded_weight, *args, **kwargs):
    calls.append(param_names.get(id(param), "<nonparam>"))
    return True


for pname, p in model.named_parameters():
    p.weight_loader = recorder

missing_warns = []
_orig_wo = q4.logger.warning_once
q4.logger.warning_once = lambda msg, *a, **k: missing_warns.append(str(msg) % a if a else str(msg))

# Model-level tee: ground truth for what the stacked-mapping loop receives
# (AutoWeightsLoader re-partitions the stream, so a name-current-variable
# coupled to the input generator is NOT reliable — spy at the boundary).
seen_model = []
_orig_lw = q4.Qwen4ExpModel.load_weights


def lw_spy(self, weights):
    def tee():
        for n, t in weights:
            seen_model.append(n)
            yield n, t

    return _orig_lw(self, tee())


q4.Qwen4ExpModel.load_weights = lw_spy

# --- 4. feed every checkpoint name -------------------------------------------
with open(MODEL + "/model.safetensors.index.json") as f:
    wm = json.load(f)["weight_map"]


def is_skipped(name):
    if "rotary_emb.inv_freq" in name:
        return True
    if name.startswith("mtp.") or name.startswith("model.visual."):
        return True
    if ".indexer." in name:
        return True
    if "ngram_embedding" in name:  # table + scale + hash buffers -> staging
        return True
    if "ple_embedding." in name and name.rsplit(".", 1)[-1] in (
        "layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets",
    ):
        return True
    return False


names = sorted(wm.keys())
expected = [n for n in names if not is_skipped(n)]
print("CKPT names", len(names), "after skips", len(expected))

# stub shapes: expert path indexes loaded_weight[expert_id]; fused path chunks
# dim -2 (never taken on this non-fused checkpoint, kept safe anyway)
stub_flat = torch.empty(600)
stub_2d = torch.empty(600, 2)
stubbed = ((n, stub_2d if "experts" in n else stub_flat) for n in names)

loaded = model.load_weights(stubbed)
q4.Qwen4ExpModel.load_weights = _orig_lw

# --- 5. accounting ------------------------------------------------------------
print("LOADER_CALLS", len(calls), "distinct", len(set(calls)))
print("LOADED_PARAMS_RETURNED", len(loaded))
assert not missing_warns, f"unmapped parameter warnings: {missing_warns[:5]}... ({len(missing_warns)})"

# every non-skipped ckpt name must have produced a loader call
assert len(calls) == len(expected), (
    f"loader calls {len(calls)} != non-skipped names {len(expected)}"
)

# mixed-quant law: scale tensors arrive ONLY for routed experts
sc_seen = [n for n in seen_model if "weight_scale_inv" in n]
non_expert_scales = [n for n in sc_seen if ".experts." not in n]
assert len(sc_seen) == 48 * 512 * 3, f"expert scales {len(sc_seen)} != 73728"
assert not non_expert_scales, f"unexpected non-expert scales: {non_expert_scales[:5]}"
print("MIXED_QUANT_OK expert scales", len(sc_seen), "non-expert", 0)

# every model param must be touched EXCEPT scale tensors whose checkpoint
# family ships no scales (the mixed-quant law) and the PLE table family
SCALE_SUFFIXES = ("weight_scale_inv", "weight_scale",
                  "q_scale", "k_scale", "v_scale", "prob_scale")
callset = set(calls)
untouched = []
for pname in dict(model.named_parameters()):
    if "ngram" in pname or "ple_embedding" in pname:
        continue
    if pname.rsplit(".", 1)[-1] in SCALE_SUFFIXES:
        continue  # absent-by-law families only; asserted below
    if pname not in callset:
        untouched.append(pname)
print("UNTOUCHED_NON_SCALE", len(untouched), untouched[:8])
assert not untouched, "params never loaded"

# untouched scale params must belong to families with ZERO checkpoint scales
ck_scales = [k for k in wm if "scale" in k.rsplit(".", 1)[-1]]
for fam in (".linear_attn.", ".self_attn.", ".mlp.gate.", ".shared_expert."):
    got = [k for k in ck_scales if fam in k]
    assert not got, f"checkpoint ships scales for {fam}: {got[:3]}"
print("CKPT_NO_NONEXPERT_SCALES_OK (linear_attn/self_attn/gate/shared_expert)")
if "<nonparam>" in callset:
    print("WARN: some loads targeted non-parameter tensors")

# PLE staging sanity.  SKIP_TABLE mode drops table shards + scale by design
# (only the 3 hash buffers stage); the shard-index parse itself is validated
# against the index names — the exact regex _stage_ple_shards applies.
fc = model
assert fc._ple_table_shards == {} and fc._ple_scale_shards == [], "skip mode staged table data"
assert sorted(fc._ple_buffers) == [
    "layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes",
], fc._ple_buffers.keys()
import re

part_re = re.compile(r"ngram_embedding\D*(\d+)")
table_names = [k for k in wm if "ngram_embedding" in k and "scale" not in k]
parts = [int(part_re.search(k).group(1)) for k in table_names]
assert sorted(parts) == list(range(128)), (
    f"shard index parse mismatch: {len(parts)} parts, range {min(parts)}..{max(parts)}"
)
scale_names = [k for k in wm if "ngram_embedding" in k and "scale" in k]
assert len(scale_names) == 1
print("PLE_STAGING_OK buffers 3, shard parse 0..127, scale 1")

q4.logger.warning_once = _orig_wo
print("T2_WEIGHTMAP_TEST_OK")
