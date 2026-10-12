# SPDX-License-Identifier: Apache-2.0
"""Inference-only Qwen4-Exp MTP draft model (FN8_04, flashnext-fp8-tier-v1 P4).

One full-attention MoE decoder layer riding the target's hyper-connection
stream.  Weight/stem ground truth: /models/flash-next-fp8 ``mtp.*`` tensors
(3101 entries incl. 512 routed experts) and Strata src/core/mtp.cpp:695-714
(the vendor runtime):

    e2  = fc_embedding(pre_fc_norm_embedding(embed))     # (T, H)
    hn  = pre_fc_norm_hidden(hyper)       # grouped RMSNorm, per-branch stats
    h2  = fc_hidden(hn)                   # per-branch fc, (T, hc, H)
    R   = h2 + e2[:, None, :]             # broadcast add on EVERY branch
    out = decoder_layer(R)                # (T, hc*H) hyper stream

forward() returns the raw pre-mixer hyper stream (DeepseekV4 MTP law): the
proposer uses it as the next step's input and samples via compute_logits(),
which applies this mixer (hc_norm doubles as the final norm — the reference
has no separate model.norm) before lm_head.  The target side stashes its own
pre-mixer stream in Qwen4ExpModel._mtp_hidden_buffer (FN8_04 edit in
qwen4_exp.py), read by model_runner's get_mtp_target_hidden_states() plumbing.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen3_next import QwenNextMixtureOfExperts
from vllm.model_executor.models.qwen4_exp import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpGatedResidual,
    Qwen4ExpRMSNorm,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


def qwen4_exp_mtp_stem(
    pre_fc_norm_embedding: nn.Module,
    fc_embedding: nn.Module,
    pre_fc_norm_hidden: nn.Module,
    fc_hidden: nn.Module,
    hc_count: int,
    inputs_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """MTP stem, module-free so the CPU gate can parity-check it against the
    naive per-branch Strata loop (fn8_t4_mtp_test.py)."""
    e2 = fc_embedding(pre_fc_norm_embedding(inputs_embeds))
    hn = pre_fc_norm_hidden(hidden_states)
    T, H = e2.shape[0], e2.shape[-1]
    h2 = fc_hidden(hn.reshape(-1, H)).view(T, hc_count, H)
    x = h2 + e2.unsqueeze(1)  # broadcast into every hyper branch
    return x.reshape(T, hc_count * H)


def fn8_qwen4_exp_mtp_remap(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Checkpoint names -> draft tree.

    The draft loads the SAME checkpoint stream as the target, so: ``mtp.*``
    is the draft proper (``mtp.`` -> ``model.``), the target embedding and
    lm_head are shared by name, and everything else (48 target decoder
    layers, the visual tower, the target mixer) is dropped.  Exact top-level
    name set ground-truthed from model.safetensors.index.json (fn8_mtp_keys).
    """
    for name, weight in weights:
        if name.startswith("mtp."):
            yield name.replace("mtp.", "model.", 1), weight
        elif name == "model.language_model.embed_tokens.weight":
            yield "model.embed_tokens.weight", weight
        elif name == "lm_head.weight":
            yield name, weight


class Qwen4ExpMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1) or 1
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, config.hidden_size
        )
        # Stem norms/fcs: plain nn.Linear — hyper machinery is replicated on
        # every TP rank and plain names match the checkpoint 1:1 (the
        # Qwen4ExpGatedResidual mixer law, qwen4_exp.py).
        self.pre_fc_norm_embedding = Qwen4ExpRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen4ExpRMSNorm(
            self.hc_count * config.hidden_size,
            group_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.fc_embedding = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.fc_hidden = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.layers = nn.ModuleList(
            Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )
        # Kept for compute_logits: hc_norm doubles as the draft's final norm.
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Target law (Qwen4ExpModel.get_expert_mapping): the helper walks
        # self.modules() so the draft layer's FusedMoE produces its own
        # (param_name, ckpt_name, expert_id, shard_id) tuples.
        num_experts = getattr(self.config, "num_experts", 0)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hyper = qwen4_exp_mtp_stem(
            self.pre_fc_norm_embedding,
            self.fc_embedding,
            self.pre_fc_norm_hidden,
            self.fc_hidden,
            self.hc_count,
            inputs_embeds,
            hidden_states,
        )
        step = spec_step_idx % self.num_mtp_layers
        return self.layers[step](hyper, positions=positions, **kwargs)


class Qwen4ExpMTP(nn.Module, QwenNextMixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        tc = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )
        self.lm_head = ParallelLMHead(
            tc.vocab_size, tc.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
        )
        self.logits_processor = LogitsProcessor(tc.vocab_size)
        self.set_moe_parameters()

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.model.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        # Mixer deferred to sampling time (dsv4 law): forward() returns the
        # raw (T, hc*H) stream; only logits need the (T, H) collapse.
        mixed = self.model.hyper_connection_mixer(hidden_states)
        return self.logits_processor(self.lm_head, mixed)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft tokens without full-vocab logits (TP MTP law:
        fork's use_local_argmax_reduction / VLLM_XPU_MTP_LOCAL_ARGMAX
        path).  forward() returns the raw pre-mixer hyper stream (dsv4
        law) — collapse it here exactly like compute_logits before the
        local argmax."""
        mixed = self.model.hyper_connection_mixer(hidden_states)
        return self.logits_processor.get_top_tokens(self.lm_head, mixed)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        # Target-loop law (Qwen4ExpModel.load_weights, proven under
        # TP2+EP2): fused-expert models own their loading.  AutoWeightsLoader
        # would delegate experts to FusedMoE.load_weights, which requires the
        # externally-set expert_mapping the drafter path never installs
        # (boot-14 ValueError).  Adaptations: names arrive pre-remapped by
        # fn8_qwen4_exp_mtp_remap(); the fused-gate_up_proj branch is dropped
        # — this checkpoint ships per-expert gate/up/down tensors (ground
        # truth: index dump, model+mtb layers alike); no PP (single stage).
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # GDN
            ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            ("in_proj_qkvz", "in_proj_z", 3),
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.model.get_expert_mapping()
        for name, loaded_weight in fn8_qwen4_exp_mtp_remap(weights):
            if "rotary_emb.inv_freq" in name:
                continue
            # QSA indexer is not ported (target design law); skip its tensors.
            if ".indexer." in name:
                continue
            # Host-resident PLE n-gram table: target loads it via a dedicated
            # path; the draft layer has no PLE (ple_layer_ids check).
            if "ngram_embedding" in name:
                continue
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            stacked_hit = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "mlp.experts" in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name.endswith(".bias") and new_name not in params_dict:
                    continue
                if new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                param.weight_loader(param, loaded_weight, shard_id)
                stacked_hit = True
                break
            if stacked_hit:
                loaded_params.add(name)
                continue

            is_expert_weight = False
            expert_hit = False
            for param_name, weight_name, expert_id, shard_id in (
                expert_params_mapping
            ):
                if weight_name not in name:
                    continue
                is_expert_weight = True
                new_name = name.replace(weight_name, param_name)
                if (
                    new_name.endswith(".bias") or new_name.endswith("_bias")
                ) and new_name not in params_dict:
                    continue
                if new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                success = param.weight_loader(
                    param,
                    loaded_weight,
                    new_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    expert_hit = True
                    break
            if expert_hit:
                loaded_params.add(name)
                continue
            if is_expert_weight:
                # Expert weight not mapped on this EP rank — skip silently.
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                logger.warning_once(
                    f"Parameter {name} not found in params_dict, skip loading"
                )
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
