# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen4-Exp (Qwen3.8-Flash-Next) model configuration — llm-scaler vendored port.

Ported from the upstream transformers ``configuration_qwen4_exp.py`` (v5 style,
class-annotation defaults) to this fork's classic ``__init__`` style so that
``AutoConfig`` resolves it through the ``_CONFIG_REGISTRY`` /
``configs.__init__._CLASS_TO_MODULE`` shims on the installed transformers.

Source of truth for the defaults: ``Qwen/Qwen3.8-Flash-Next-FP8`` config.json
(text_config.model_type "qwen4_exp_text", architectures
["Qwen4ExpForConditionalGeneration"]).

Deliberate divergences from the upstream reference config:

* ``layer_types`` — the reference remaps ``full_attention`` to
  ``indexed_attention`` (QSA indexer).  This port has NO indexer: dense full
  attention everywhere.  An explicit ``indexed_attention`` entry in a provided
  ``layer_types`` list is mapped back to ``full_attention`` so a stock
  transformers-style config still loads.
* The vision config is parsed for round-trip fidelity only; NO vision model is
  registered in vllm for this architecture (text-only serving).
* ``number_of_conv_states`` (3 when PLE is enabled) is kept as an attribute:
  the fork's hybrid/mamba pool allocates a uniform 4-slot state per linear
  layer ([0] GDN conv, [1] GDN temporal, [2] PLE short-conv, [3] PLE token
  history) regardless, so this is informational for validation.
"""

from transformers.configuration_utils import PretrainedConfig


class Qwen4ExpTextConfig(PretrainedConfig):
    model_type = "qwen4_exp_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2560,
        num_hidden_layers=48,
        num_attention_heads=24,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        # GDN linear attention (16 K-heads + 48 V-heads x 128)
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        # MoE
        intermediate_size=None,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        num_experts_per_tok=10,
        num_experts=512,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        norm_topk_prob=True,
        # hyper-connections
        hc_count=4,
        hc_lowrank=320,
        # PLE (Per-Layer Embedding); ple_layer_ids is ONE-indexed (2 == layer
        # index 1), matching upstream reference semantics.
        ple_layer_ids=None,
        ple_embed_dim=None,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20000000,
        make_ngram_vocab_size_divisible_by=128,
        seed=1234,
        split_ngram_parts=128,
        # QSA indexer fields — parsed for fidelity, NOT used by this port
        indexer_budget=2048,
        indexer_compress_ratio=4,
        indexer_head_dim=128,
        indexer_kv_heads=1,
        indexer_n_heads=4,
        # misc
        mamba_ssm_dtype=None,
        output_gate_type="sigmoid",
        layer_types=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.25)

        # layer_types: same default pattern as qwen3_5/qwen3_next. An explicit
        # list may arrive from a stock transformers config in which case any
        # "indexed_attention" entries are mapped back to "full_attention"
        # (this port ships no QSA indexer).
        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.get("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention"
                if bool((i + 1) % interval_pattern)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                "full_attention" if lt == "indexed_attention" else lt
                for lt in self.layer_types
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types ({len(self.layer_types)}) must match "
                f"num_hidden_layers ({self.num_hidden_layers})"
            )
        if hasattr(self, "validate_layer_type"):
            # Transformers v5
            kwargs["ignore_keys_at_rope_validation"] = {
                "mrope_section",
                "mrope_interleaved",
            }
            self.validate_layer_type()
        else:
            # Transformers v4
            from transformers.configuration_utils import layer_type_validation

            layer_type_validation(self.layer_types, self.num_hidden_layers)

        # linear attention part
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.norm_topk_prob = norm_topk_prob

        # hyper-connections
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank

        # PLE
        self.ple_layer_ids = sorted(set(ple_layer_ids)) if ple_layer_ids else []
        self.ple_embed_dim = ple_embed_dim if ple_embed_dim is not None else hidden_size
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.seed = seed
        self.split_ngram_parts = split_ngram_parts
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if ngram_heads == 0 or self.ple_embed_dim % ngram_heads != 0:
            raise ValueError(
                f"ple_embed_dim ({self.ple_embed_dim}) must be divisible by "
                f"(ngram_size - 1) * heads_per_ngram ({ngram_heads})"
            )
        for ple_id in self.ple_layer_ids:
            # one-indexed ids, PLE only on linear-attention layers
            if not isinstance(ple_id, int) or ple_id < 1:
                raise ValueError(f"ple_layer_ids entries must be 1-indexed ints, got {ple_id}")
            if self.layer_types[ple_id - 1] != "linear_attention":
                raise ValueError(
                    f"PLE layer {ple_id} must sit on a linear_attention layer"
                )
        # reference semantics: 3 conv states when PLE (GDN conv + PLE conv +
        # token history), else 1 (GDN conv only)
        self.number_of_conv_states = 3 if self.ple_layer_ids else 1

        # indexer (parsed, unused by this port)
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio
        self.indexer_head_dim = indexer_head_dim
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_n_heads = indexer_n_heads

        # training-time field; the fork takes mamba cache dtypes from
        # cache_config (mamba_cache_dtype / mamba_ssm_cache_dtype), not here
        self.mamba_ssm_dtype = mamba_ssm_dtype
        self.output_gate_type = output_gate_type

        super().__init__(**kwargs)
        # Set these AFTER super().__init__() because transformers v4's
        # PretrainedConfig.__init__ has these as explicit params with different
        # defaults (e.g. tie_word_embeddings=True) that would overwrite our values.
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings


class Qwen4ExpVisionConfig(PretrainedConfig):
    model_type = "qwen4_exp_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=2560,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class Qwen4ExpConfig(PretrainedConfig):
    model_type = "qwen4_exp"
    sub_configs = {
        "vision_config": Qwen4ExpVisionConfig,
        "text_config": Qwen4ExpTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        # The checkpoint ships vision_config with model_type "qwen4_exp" (same
        # as the top level); the reference rewrites it to "qwen4_exp_vision".
        if isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            vision_config.pop("model_type", None)
        elif vision_config is not None:
            vision_config = dict(vision_config.to_diff_dict())
            vision_config.pop("model_type", None)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.pop("model_type", None)
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs)
        # Set after super().__init__() to avoid v4 PretrainedConfig overwrite
        self.tie_word_embeddings = tie_word_embeddings


__all__ = ["Qwen4ExpConfig", "Qwen4ExpTextConfig", "Qwen4ExpVisionConfig"]
