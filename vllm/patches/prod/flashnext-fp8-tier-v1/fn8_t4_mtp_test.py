#!/usr/bin/env python3
"""fn8 P4 CPU gate: patch_04 qwen4_exp MTP wiring (run INSIDE fn8-p2):

    docker exec fn8-p2 python3 /stage/fn8_t4_mtp_test.py

1. WIRING  — MTPModelTypes membership, registry draft entry resolvable,
   hf_config_override converts the REAL checkpoint config (model_type,
   architectures, n_predict), proposer/source qwen4 hyper-stream edits in
   place, model_runner duck-tap intact.
2. STEM    — the draft stem (real Qwen4ExpRMSNorm + nn.Linear modules) vs
   the naive per-branch Strata loop (mtp.cpp:695-714), exact-close; plus
   the mixer collapse (T, hc*H) -> (T, H) that compute_logits relies on.
3. REMAP   — draft weight-name remap on the exact checkpoint name grammar
   (mtp.* -> model.*, shared embed/lm_head kept, target/visual dropped).
4. COMPILE — py_compile of every touched file.
"""
import os
import py_compile
import sys
from types import SimpleNamespace
from typing import get_args

import torch
from torch import nn

BASE = "/models/flash-next-fp8"

# ---- 1. WIRING --------------------------------------------------------------
import vllm.config.speculative as sc  # noqa: E402

assert "qwen4_exp_mtp" in get_args(sc.MTPModelTypes), "MTPModelTypes missing entry"
print("TYPES_OK qwen4_exp_mtp in MTPModelTypes")

import vllm.model_executor.models.qwen4_exp_mtp as qm  # noqa: E402

assert hasattr(qm, "Qwen4ExpMTP") and hasattr(qm, "Qwen4ExpMultiTokenPredictor")
print("MODULE_OK qwen4_exp_mtp imports, classes present")

import vllm  # noqa: E402

pkg = os.path.dirname(vllm.__file__)
reg_src = open(os.path.join(pkg, "model_executor", "models", "registry.py"),
               encoding="utf-8").read()
assert '"Qwen4ExpMTP": ("qwen4_exp_mtp", "Qwen4ExpMTP")' in reg_src
print("REGISTRY_OK draft entry present")

# real checkpoint config -> draft conversion (vLLM-registered class; plain
# AutoConfig does not know qwen4_exp in this transformers build)
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpConfig  # noqa: E402

hf_config = Qwen4ExpConfig.from_pretrained(BASE)
assert hf_config.model_type == "qwen4_exp", hf_config.model_type
out_cfg = sc.SpeculativeConfig.hf_config_override(hf_config)
assert out_cfg.model_type == "qwen4_exp_mtp", out_cfg.model_type
assert out_cfg.architectures == ["Qwen4ExpMTP"], out_cfg.architectures
assert out_cfg.n_predict == 1, out_cfg.n_predict
tc = getattr(out_cfg, "text_config", out_cfg)
assert tc.hc_count == 4 and tc.hidden_size == 2560
assert getattr(tc, "mtp_num_hidden_layers", None) == 1
print("OVERRIDE_OK qwen4_exp -> qwen4_exp_mtp, n_predict=1, hc 4x2560")

# proposer detection edit (source-level: the ctor needs a live engine)
prop_src = open(
    os.path.join(pkg, "v1", "spec_decode", "llm_base_proposer.py"),
    encoding="utf-8",
).read()
assert '== "qwen4_exp_mtp":' in prop_src, "proposer hc_count elif missing"
assert 'getattr(tc, "hc_count", 4)' in prop_src
draft_hidden = tc.hidden_size * tc.hc_count
assert draft_hidden == 10240
print("PROPOSER_OK hc_count elif wired, draft hidden 2560*4=10240")

# target-side stash + accessor + runner duck-tap
qe_src = open(os.path.join(pkg, "model_executor", "models", "qwen4_exp.py"),
              encoding="utf-8").read()
assert qe_src.count("_mtp_hidden_buffer") >= 3, "stash not wired"
assert "def get_mtp_target_hidden_states" in qe_src
mr_src = open(os.path.join(pkg, "v1", "worker", "gpu", "model_runner.py"),
              encoding="utf-8").read()
assert "get_mtp_target_hidden_states" in mr_src
print("TARGET_STASH_OK buffer+accessor wired, runner duck-tap intact")

# ---- 2. STEM PARITY (real modules, naive Strata loop) ----------------------
from vllm.model_executor.models.qwen4_exp import (  # noqa: E402
    Qwen4ExpGatedResidual,
    Qwen4ExpRMSNorm,
)

torch.manual_seed(0)
T, H, HC = 5, 64, 4
emb_n = Qwen4ExpRMSNorm(H, eps=1e-6)
fc_e = nn.Linear(H, H, bias=False)
hid_n = Qwen4ExpRMSNorm(HC * H, group_size=H, eps=1e-6)
fc_h = nn.Linear(H, H, bias=False)
embeds = torch.randn(T, H)
hyper = torch.randn(T, HC * H)

got = qm.qwen4_exp_mtp_stem(emb_n, fc_e, hid_n, fc_h, HC, embeds, hyper)
assert got.shape == (T, HC * H), got.shape

e2 = fc_e(emb_n(embeds))
hn = hid_n(hyper)
ref = torch.empty(T, HC, H)
for c in range(HC):
    ref[:, c, :] = fc_h(hn[:, c * H:(c + 1) * H]) + e2
torch.testing.assert_close(got, ref.flatten(1))
print("STEM_OK broadcast-add law matches naive Strata loop exactly")

mixer = Qwen4ExpGatedResidual(
    SimpleNamespace(hc_count=HC, hidden_size=H, hc_lowrank=16, rms_norm_eps=1e-6),
    use_combine=False,
)
collapsed = mixer(got)
assert collapsed.shape == (T, H), collapsed.shape
print("MIXER_OK (T,4H)->(T,H) collapse, compute_logits path valid")

# ---- 3. WEIGHT REMAP (exact checkpoint name grammar) ------------------------
names_in = [
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.indexer.0.theta",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.gate.weight",
    "mtp.layers.0.mlp.experts.511.gate_proj.weight",
    "mtp.layers.0.mlp.experts.511.gate_proj.weight_scale_inv",
    "model.language_model.embed_tokens.weight",
    "lm_head.weight",
    # drops: target decoder, target mixer, visual tower
    "model.layers.3.self_attn.q_proj.weight",
    "model.layers.3.mlp.experts.7.down_proj.weight",
    "model.language_model.hyper_connection_mixer.hc_norm.weight",
    "model.visual.patch_embed.proj.weight",
]
w = torch.zeros(1)
pairs = list(qm.fn8_qwen4_exp_mtp_remap((n, w) for n in names_in))
kept = [n for n, _ in pairs]
assert len(kept) == 15, kept
for n in kept:
    assert n.startswith("model.") or n == "lm_head.weight", n
    assert not n.startswith("model.visual"), n
    assert "language_model" not in n, n
assert "mtp.layers.0.mlp.experts.511.gate_proj.weight" not in kept
assert "model.layers.0.mlp.experts.511.gate_proj.weight" in kept
assert "model.layers.0.mlp.experts.511.gate_proj.weight_scale_inv" in kept
assert "model.layers.3.self_attn.q_proj.weight" not in kept
assert "model.layers.3.mlp.experts.7.down_proj.weight" not in kept
assert kept.count("model.layers.0.self_attn.q_proj.weight") == 1
assert "model.embed_tokens.weight" in kept and "lm_head.weight" in kept
# target layers were dropped, draft layer kept exactly once
assert sum(1 for n in kept if n.startswith("model.layers.")) == 7, kept
print("REMAP_OK 15 kept (mtp.->model., embed+lm_head shared), 4 dropped")

# ---- 4. COMPILE --------------------------------------------------------------
for rel in (
    "model_executor/models/qwen4_exp_mtp.py",
    "model_executor/models/qwen4_exp.py",
    "model_executor/models/registry.py",
    "config/speculative.py",
    "v1/spec_decode/llm_base_proposer.py",
):
    py_compile.compile(os.path.join(pkg, rel), doraise=True)
print("COMPILE_OK all touched files")

print("FN8_T4_MTP_ALL_OK")
sys.exit(0)
