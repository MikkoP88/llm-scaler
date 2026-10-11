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
"""Qwen4-Exp (Qwen3.8-Flash-Next) — text-only inference port for the
llm-scaler vLLM XPU fork (flashnext-fp8-tier-v1, P0).

Port map (reference ``modeling_qwen4_exp.py`` -> this file):

* ``Qwen4ExpTextRMSNorm``            -> ``Qwen4ExpRMSNorm`` (group Gemma-style)
* ``Qwen4ExpTextGatedResidual``      -> ``Qwen4ExpGatedResidual`` (hyper-connections)
* ``Qwen4ExpTextNGramEmbedding``     -> ``Qwen4ExpNGramEmbedding`` (HOST-resident
  FP8 n-gram table, ~45B params / ~51 GB; per-token gathered rows are moved
  H2D and dequantized on device)
* ``Qwen4ExpTextPLELayer``           -> ``Qwen4ExpPLELayer`` (pure-tensor API;
  the DecoderLayer handles flattened-batch <-> padded-view reshaping and the
  mamba-state pool read/write)
* ``Qwen4ExpTextGatedDeltaNet``      -> fork ``GatedDeltaNetAttention`` reused
  as-is via ``Qwen4ExpGatedDeltaNet`` (4-slot state)
* full-attention layers               -> fork ``Qwen3NextAttention`` as-is
* MoE                                 -> fork ``Qwen3NextSparseMoeBlock`` as-is

Design laws (P0):

1. STATE SLOTS — every linear-attention layer gets a uniform 4-slot mamba
   state (copy funcs are MODEL-level, so all layers must share one layout):
   [0] GDN conv   (kernel-1+num_spec, conv_dim/tp)   <- fork GDN reads [0]/[1]
   [1] GDN temporal (v_heads/tp, v_dim, k_dim)          positionally, untouched
   [2] PLE short conv ((ple_k-1)*dilation + num_spec, hc_hidden) — NOT TP-sharded
   [3] PLE token history (ngram_size-1 + num_spec, 1) int64 — NOT TP-sharded
   Slots [2]/[3] sit idle (zero) on the 35 non-PLE linear layers (~6.4 MB/req
   waste, accepted for v1).

2. TP2 — all reducing blocks (o_proj/out_proj/down_proj, ESIMD MoE paths)
   return FULL hidden on both ranks; hyper-connections and PLE are
   elementwise/small-linear over the full (tokens, hc_count*hidden) tensor and
   are replicated on both ranks.  The HF tp_plan rowwise_split_input for
   hc/PLE is deliberately NOT adopted in v1.

3. layer_types — no QSA indexer in this port: ``full_attention`` everywhere a
   full-attention layer sits (dense attention).

4. PLE batch handling — vLLM flattens tokens; this port rebuilds a padded
   (num_reqs, max_len) view driven by ``query_start_loc`` from the forward
   context, runs the reference-shaped vectorized math, and scatters back.
   Speculative decoding with PLE states is NOT supported yet (P4) and fails
   loud.

5. Eager-first: no @support_torch_compile in P0 (data-dependent padded views
   in the PLE path); the lane runs enforce_eager.

Env knobs:
* ``VLLM_QW4_PLE_SKIP_TABLE=1`` — dev-only: allocate the n-gram table as
  zeros and skip checkpoint shard loading (construction/dry-run testing
  without the 51 GB download).
* ``VLLM_QW4_PLE_PIN=1`` — pin the host table after load (default off; pin
  only when the deployment's host memory budget allows it).
"""

import math
import os
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig

logger = init_logger(__name__)


def _ple_env_flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


# ---------------------------------------------------------------------------
# Norms (reference Qwen4ExpTextRMSNorm, byte-exact)
# ---------------------------------------------------------------------------
class Qwen4ExpRMSNorm(nn.Module):
    """Gemma-style (1 + w) RMSNorm with optional per-group reduction."""

    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen4Exp is (x * w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Hyper-connections (reference Qwen4ExpTextGatedResidual, byte-exact math).
# Replicated on every TP rank (see design law 2); plain nn.Linear matches the
# checkpoint parameter names 1:1 and AutoWeightsLoader default-loads them.
# ---------------------------------------------------------------------------
class Qwen4ExpGatedResidual(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = nn.Linear(hc_hidden_size, self.hc_count, bias=False) if use_combine else None

    def forward(self, hyper_input: torch.Tensor):
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                f"Expected {self.hc_count * self.hidden_size} hyper-connection features, "
                f"got {hyper_input.shape[-1]}."
            )
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (
            input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, hyper_input, injection_weights


# ---------------------------------------------------------------------------
# N-gram hash machinery (reference, byte-exact)
# ---------------------------------------------------------------------------
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed: int) -> torch.Tensor:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _shift_right_ignore_eos(token_ids: torch.Tensor, shift: int, eos_token_id: int) -> torch.Tensor:
    """Reference Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos, verbatim."""
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
    segment_start = previous_eos + 1
    position_in_segment = positions.unsqueeze(0) - segment_start
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))


# ---------------------------------------------------------------------------
# Host-resident n-gram embedding table
# ---------------------------------------------------------------------------
class Qwen4ExpNGramEmbedding(nn.Module):
    """PLE n-gram embedding with a HOST-resident table.

    The table (~320M rows x 160 dims, FP8, ~51 GB) never moves to the device:
    per token only ``ngram_heads`` rows (16 x 160 B = 2.56 KB) are gathered on
    host and streamed H2D, then dequantized on device.

    LAW (2026-10-11, leg-3 OOM): the table and its scales are PLAIN
    ATTRIBUTES (``_host_table`` / ``_host_scale_inv`` / ``_host_scale``), NOT
    registered parameters.  As Parameters they were visible to the weight
    offloader's per-layer walk (offloader/uva.py): ``p.data.to("cpu")`` is a
    no-op returning self, the 47.69 GiB table was counted against the offload
    budget, the layer got the functional_call wrapper, and the wrapper's
    ``state_dict()`` copy H2D'd the whole table per forward -> XPU OOM
    (single 47.69 GiB alloc, card is 31.89 GiB).  Plain attributes are
    invisible to ``parameters()`` / ``state_dict()`` / ``_apply``, so nothing
    but this module's own gather ever touches the table.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        layer_idx: int,
        ple_layer_index: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.head_dim_per_ngram = head_dim_per_ngram
        self.embedding_dim = embedding_dim
        self.seed = config.seed
        eos = config.eos_token_id
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos

        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size

        self.register_buffer(
            "layer_multipliers",
            _build_layer_multipliers(self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed),
            persistent=False,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(self.head_vocab_sizes, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(self.head_offsets, dtype=torch.long),
            persistent=False,
        )
        ngram_vocab_divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(self.total_vocab_size / ngram_vocab_divisor) * ngram_vocab_divisor
        self.padded_vocab_size = padded_vocab_size

        skip_table = _ple_env_flag("VLLM_QW4_PLE_SKIP_TABLE")
        if skip_table:
            logger.warning_once(
                "Qwen4ExpNGramEmbedding: VLLM_QW4_PLE_SKIP_TABLE=1 — table is a "
                "0-row placeholder (vocab would be %d x %d). Dev/dry-run only, "
                "outputs are garbage.",
                padded_vocab_size,
                head_dim_per_ngram,
            )
        # Host-resident from the start: explicit device="cpu" overrides any
        # meta-device init context. float8 storage is byte-sized.  SKIP_TABLE
        # allocates a 0-row table (dev mode must not reserve the ~51 GB).
        # PLAIN TENSORS, not Parameters — see the class docstring LAW: any
        # registered form is visible to the offloader's state_dict copy.
        self._host_table = torch.zeros(
            0 if skip_table else padded_vocab_size, head_dim_per_ngram,
            dtype=torch.float8_e4m3fn, device="cpu",
        )
        # FP8 scales: layout (per-tensor vs weight_block_size [128, 128]) is
        # finalized in process_weights_after_loading from the checkpoint's
        # scale tensor shape; kept host-resident too.
        self._host_scale_inv = torch.zeros(0, dtype=torch.float32, device="cpu")
        # Per-tensor scale variant (some checkpoints ship "weight_scale").
        self._host_scale = torch.zeros(0, dtype=torch.float32, device="cpu")
        # column block split derived from the scale layout at load time;
        # None => passthrough (unquantized table)
        self._col_block_size: int | None = None

    def compute_ngram_ids(self, token_history: torch.Tensor) -> torch.Tensor:
        """Reference Qwen4ExpTextNGramEmbedding.forward id-math, verbatim.

        ``token_history``: (R, context_len + L) int64 on device (padded view).
        Returns ngram_ids (R, L, ngram_heads) int64 on device.
        """
        seq_len = token_history.shape[1] - self.context_len
        shifted_tokens = [
            _shift_right_ignore_eos(token_history, shift, self.eos_token_id)
            for shift in range(self.ngram_size)
        ]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        return torch.cat(blocks, dim=-1)[:, -seq_len:]

    def _dequant_rows(self, rows: torch.Tensor, row_ids: torch.Tensor) -> torch.Tensor:
        """Dequantize gathered FP8 rows on device.

        Supported checkpoint layouts (selected at load time):
        * no scales                     -> passthrough cast (unquantized table)
        * scalar / (1,) ``weight_scale``-> per-tensor multiply
        * (ceil(V/128), K) block scales (weight_block_size [128, 128]) ->
          per-row-block x per-col-block multiply
        """
        out = rows.to(torch.float32)
        if self._col_block_size is None:
            if self._host_scale_inv.numel() == 1:
                out = out * self._host_scale_inv.to(out.device).to(torch.float32)
            elif self._host_scale.numel() == 1:
                out = out * self._host_scale.to(out.device).to(torch.float32)
            # else: no scales at all — pure values cast up (unexpected for a
            # modules_to_convert table; logged at load time)
            return out.to(torch.float16)
        block = self._col_block_size
        scale = self._host_scale_inv.to(out.device).to(torch.float32)  # (V/128, K)
        scale_rows = scale[row_ids // block]  # (T, K)
        width = rows.shape[-1]
        for k, start in enumerate(range(0, width, block)):
            end = min(start + block, width)
            out[..., start:end] = out[..., start:end] * scale_rows[..., k, None]
        return out.to(torch.float16)

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather host table rows for (R, L, heads) ids -> (R, L, embedding_dim)
        fp16 on the ids' device."""
        id_device = ngram_ids.device
        flat_ids = ngram_ids.reshape(-1)
        num_rows = flat_ids.numel()
        if num_rows == 0:
            return torch.zeros(
                *ngram_ids.shape[:-1], self.embedding_dim,
                dtype=torch.float16, device=id_device,
            )
        # P0: synchronous D2H of the ids, host gather, H2D of the rows.
        # (2.56 KB/token; optimization — pinned staging + async copies — is a
        # later-phase item, see PHASES.md.)
        ids_cpu = flat_ids.to("cpu", non_blocking=False).long()
        if self._host_table.dtype == torch.float8_e4m3fn:
            rows = self._host_table.view(torch.uint8).index_select(0, ids_cpu).view(torch.float8_e4m3fn)
        else:
            rows = self._host_table.index_select(0, ids_cpu)
        rows_dev = rows.to(id_device, non_blocking=False)
        emb = self._dequant_rows(rows_dev, ids_cpu.to(id_device))
        # (R, L, heads) ids -> (R, L, heads*head_dim) == ple_embed_dim,
        # already the reference flatten(-2) result.
        return emb.view(*ngram_ids.shape[:-1], self.embedding_dim)


# ---------------------------------------------------------------------------
# PLE layer (reference Qwen4ExpTextPLELayer, byte-exact math)
# Pure-tensor interface: the caller owns the mamba-pool read/write and the
# flattened <-> padded reshaping.
# ---------------------------------------------------------------------------
class Qwen4ExpPLELayer(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        ple_embed_dim = config.ple_embed_dim
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(config, ple_embed_dim, layer_idx, ple_layer_index)
        conv_kernel_size = config.ple_conv_kernel_size
        conv_dilation = config.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        self.key_proj = nn.Linear(ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_query = Qwen4ExpRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_conv = Qwen4ExpRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=conv_kernel_size,
            groups=hc_hidden_size,
            dilation=conv_dilation,
            bias=False,
        )

    @staticmethod
    def _tail_gather(x: torch.Tensor, lengths: torch.Tensor, state_len: int, old_state: torch.Tensor | None = None) -> torch.Tensor:
        """Per-request last-``state_len`` time-slices of padded ``x`` (R, L, D).

        ``lengths``: (R,) true sequence lengths.  For L >= state_len the
        result is x[:, L-S:L]; for L < S it is cat(old_state[:S-L], x[:, :L])
        (reference update_conv_state semantics).  ``old_state`` is required
        when any length < state_len.
        """
        R = x.shape[0]
        js = torch.arange(state_len, device=x.device)
        xi = js[None, :] - state_len + lengths[:, None]  # (R, S)
        from_old = xi < 0
        xi_safe = xi.clamp(min=0, max=x.shape[1] - 1)
        gathered = x.gather(1, xi_safe.unsqueeze(-1).expand(R, state_len, x.shape[-1]))
        if bool(from_old.any().item()):
            if old_state is None:
                raise ValueError("_tail_gather needs old_state for short sequences")
            gathered = torch.where(from_old.unsqueeze(-1), old_state, gathered)
        return gathered

    def _short_conv(
        self,
        hidden_states: torch.Tensor,  # (R, dim, L) — already transposed
        conv_state: torch.Tensor,     # (R, short_conv_state_len, dim) SD layout
    ) -> torch.Tensor:
        """Reference _short_conv output math (padded columns are garbage and
        never read: the conv is backward-looking, so valid columns see only
        the true history).  State extraction happens separately via
        ``_tail_gather`` (padding must not leak into states)."""
        seq_len = hidden_states.shape[-1]
        state_len = self.short_conv_state_len
        # We always pad and slice due to the dilation in the conv, to make
        # sure we have enough states (reference comment).
        padded = torch.cat([conv_state.transpose(1, 2), hidden_states], dim=-1)
        padded = padded[..., -(state_len + seq_len):]
        return F.silu(self.conv1d(padded))

    def forward(
        self,
        hidden_states: torch.Tensor,  # (R, L, hc_hidden) fp16 (padded view)
        token_history: torch.Tensor,  # (R, context_len + maxL) int64 (padded view)
        lengths: torch.Tensor,        # (R,) true lengths
        conv_state: torch.Tensor,     # (R, short_conv_state_len, hc_hidden)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (ple_out (R, maxL, hc_hidden), new_conv_state (R, S, hc_hidden),
        new_context (R, context_len)).  Padded output columns are garbage."""
        ngram_ids = self.ple_embedding.compute_ngram_ids(token_history)
        embeddings = self.ple_embedding(ngram_ids)  # (R, L, ple_embed_dim)

        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)

        conv_out = self._short_conv(gated_value_normed.transpose(1, 2), conv_state)
        output = gated_value + conv_out.transpose(1, 2)

        # State extraction from TRUE tails (padding must never leak in).
        new_conv_state = self._tail_gather(
            gated_value_normed, lengths, self.short_conv_state_len, old_state=conv_state
        )
        ctx = self.ple_embedding.context_len
        # new history = last `ctx` of cat(prev_ctx, ids) == positions
        # lengths..lengths+ctx-1 of the padded token_history.
        js = torch.arange(ctx, device=token_history.device)
        new_context = token_history.gather(1, lengths[:, None] + js[None, :])
        return output, new_conv_state, new_context


# ---------------------------------------------------------------------------
# GDN with the uniform 4-slot state (design law 1)
# ---------------------------------------------------------------------------
class Qwen4ExpGatedDeltaNet(GatedDeltaNetAttention):
    """GatedDeltaNetAttention with PLE slots appended after the native two.

    The fork GDN core reads ``kv_cache[0]`` (conv) and ``kv_cache[1]``
    (temporal) positionally and is untouched; the DecoderLayer consumes
    ``kv_cache[2]`` (PLE short conv) and ``kv_cache[3]`` (PLE token history).
    """

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__(config=config, vllm_config=vllm_config, prefix=prefix, **kwargs)
        text_cfg = config
        self._ple_conv_state_len = (text_cfg.ple_conv_kernel_size - 1) * text_cfg.ngram_size
        self._ple_history_len = text_cfg.ngram_size - 1
        self._hc_hidden_size = text_cfg.hidden_size * text_cfg.hc_count

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        conv_d, temporal_d = super().get_state_dtype()
        return (conv_d, temporal_d, conv_d, torch.int64)

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        conv_s, temporal_s = super().get_state_shape()
        return (
            conv_s,
            temporal_s,
            (self._ple_conv_state_len + self.num_spec, self._hc_hidden_size),
            (self._ple_history_len + self.num_spec, 1),
        )


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------
class Qwen4ExpDecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        # Bypass Qwen3NextDecoderLayer.__init__ entirely (qwen3_5 bypass
        # pattern): qwen4_exp layers have no input/post_attention layernorm
        # and no residual — hyper-connections replace them — so neither the
        # parent init nor the parent forward applies.  ESIMD layer-level
        # fusion flags are intentionally NOT set up here (those fused paths
        # live in the parent forward we do not call; the ESIMD kernels inside
        # GDN/Attention/MoE blocks remain active).
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpGatedDeltaNet(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        self.mlp = Qwen3NextSparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )

        ple_layer_index = config.ple_layer_ids.index(self.layer_idx + 1) if (self.layer_idx + 1) in config.ple_layer_ids else None
        self.ple = (
            Qwen4ExpPLELayer(config, self.layer_idx, ple_layer_index)
            if ple_layer_index is not None
            else None
        )
        self.attn_hyper_connection = Qwen4ExpGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(config)
        self.layer_scale = False

    # -- PLE flattened <-> padded plumbing ---------------------------------
    def _ple_metadata(self):
        """(query_start_loc, block_ids, has_initial_state) for the PLE layer."""
        raw = get_forward_context().attn_metadata
        md = raw.get(self.linear_attn.prefix) if isinstance(raw, dict) else raw
        if md is None:
            raise RuntimeError(
                f"Qwen4Exp PLE: no attention metadata for {self.linear_attn.prefix}"
            )
        if getattr(md, "spec_sequence_masks", None) is not None:
            raise NotImplementedError(
                "Qwen4Exp PLE does not support speculative decoding yet (P4); "
                "run without a speculative config."
            )
        qsl = getattr(md, "non_spec_query_start_loc", None)
        if qsl is None:
            qsl = getattr(md, "query_start_loc", None)
        block_ids = getattr(md, "non_spec_state_indices_tensor", None)
        if block_ids is None:
            block_ids = getattr(md, "state_indices_tensor", None)
        if qsl is None or block_ids is None:
            raise RuntimeError(
                "Qwen4Exp PLE: missing query_start_loc/state indices in "
                f"attention metadata ({type(md)})"
            )
        has_initial_state = getattr(md, "has_initial_state", None)
        return qsl, block_ids, has_initial_state

    def _ple_forward_flat(
        self,
        hidden_states: torch.Tensor,  # (T, hc_hidden)
        input_ids: torch.Tensor,      # (T,)
    ) -> torch.Tensor:
        ple = self.ple
        # The XPU runner feeds int32 input_ids (lane bandwidth law); every
        # PLE history buffer is int64 (leg-8, 2026-10-11: index-put dtype
        # mismatch on ids_pad).  Coerce once — no-op when already int64.
        input_ids = input_ids.to(torch.long)
        kv_cache = getattr(self.linear_attn, "kv_cache", None)
        fwd_ctx = get_forward_context()
        attn_md = getattr(fwd_ctx, "attn_metadata", None)
        if attn_md is None:
            # Fork profile law (leg-6, 2026-10-11): profile_run ->
            # _dummy_run(is_profile=True) executes the whole model with
            # attn_metadata=None and no wired mamba pools; the native GDN
            # core no-ops on exactly this condition
            # (_xpu_ops._gdn_attention_core_xpu_impl early-returns before
            # any kv_cache touch).  PLE mirrors that idiom: stateless
            # synthetic single-sequence run keeps activation memory
            # representative while touching zero mamba slots.
            return self._ple_forward_stateless(hidden_states, input_ids)
        if kv_cache is None or len(kv_cache) < 4:
            raise RuntimeError("Qwen4Exp PLE: 4-slot mamba state not wired to the layer")

        qsl, block_ids_t, has_initial_state = self._ple_metadata()
        device = hidden_states.device
        eos = ple.ple_embedding.eos_token_id
        qsl = qsl.to(torch.long)
        num_reqs = qsl.shape[0] - 1
        lens = (qsl[1:] - qsl[:-1])
        max_len = int(lens.max().item())
        block_ids = block_ids_t[:num_reqs].to(torch.long)

        conv_pool = kv_cache[2]   # (num_blocks, state_len(+spec), hc_hidden)
        hist_pool = kv_cache[3]   # (num_blocks, context_len(+spec), 1) int64

        # --- flatten -> padded ---
        T = hidden_states.shape[0]
        tok_pos = torch.arange(T, device=device, dtype=torch.long)
        req_of_tok = torch.searchsorted(qsl[1:], tok_pos, right=True)
        local_of_tok = tok_pos - qsl[req_of_tok]

        hidden_pad = hidden_states.new_zeros(num_reqs, max_len, hidden_states.shape[-1])
        hidden_pad[req_of_tok, local_of_tok] = hidden_states
        ids_pad = torch.full((num_reqs, max_len), eos, dtype=torch.long, device=device)
        ids_pad[req_of_tok, local_of_tok] = input_ids

        # --- read states (fresh requests: history inits to EOS, not zeros) ---
        prev_ctx = hist_pool[block_ids, :ple.ple_embedding.context_len, 0].clone()
        if has_initial_state is not None and not bool(
            torch.all(has_initial_state[:num_reqs]).item()
        ):
            fresh = ~has_initial_state[:num_reqs].to(torch.bool)
            if bool(fresh.any().item()):
                prev_ctx[fresh] = eos
        conv_state = conv_pool[block_ids, :ple.short_conv_state_len].clone()

        token_history = torch.cat([prev_ctx, ids_pad], dim=-1)
        out_pad, new_conv_state, new_ctx = ple(hidden_pad, token_history, lens, conv_state)

        # --- write states back ---
        conv_pool[block_ids, :ple.short_conv_state_len] = new_conv_state
        hist_pool[block_ids, :ple.ple_embedding.context_len, 0] = new_ctx

        # --- padded -> flatten ---
        return out_pad[req_of_tok, local_of_tok]

    def _ple_forward_stateless(
        self,
        hidden_states: torch.Tensor,  # (T, hc_hidden)
        input_ids: torch.Tensor,      # (T,)
    ) -> torch.Tensor:
        """Profile/dummy-run path (leg-6 law): no pools exist yet — run all T
        tokens as ONE synthetic sequence with a zero conv state and an
        EOS-seeded history.  Exercises the full PLE math (host-table gather,
        conv, projections) so activation profiling stays representative;
        reads/writes no mamba slots."""
        ple = self.ple
        device = hidden_states.device
        eos = ple.ple_embedding.eos_token_id
        ctx_len = ple.ple_embedding.context_len
        T = hidden_states.shape[0]
        prev_ctx = torch.full((1, ctx_len), eos, dtype=torch.long, device=device)
        conv_state = hidden_states.new_zeros(
            (1, ple.short_conv_state_len, hidden_states.shape[-1])
        )
        token_history = torch.cat(
            [prev_ctx, input_ids.to(torch.long).unsqueeze(0)], dim=-1
        )
        lens = torch.tensor([T], dtype=torch.long, device=device)
        out_pad, _, _ = ple(
            hidden_states.unsqueeze(0), token_history, lens, conv_state
        )
        return out_pad.squeeze(0)

    # -- main forward -------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,     # (T, hc_count*hidden)
        positions: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.ple is not None:
            if input_ids is None:
                raise RuntimeError("Qwen4Exp PLE layer called without input_ids")
            hidden_states = hidden_states + self._ple_forward_flat(hidden_states, input_ids)

        mixed, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
        out = torch.empty_like(mixed)
        if self.layer_type == "linear_attention":
            if mixed.shape[0] == 1:
                self.linear_attn.forward_xpu(mixed, out)
            else:
                self.linear_attn(hidden_states=mixed, output=out)
        else:
            self.self_attn(positions=positions, output=out, hidden_states=mixed)
        injection = out.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)

        mixed, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        out = self.mlp(mixed)
        injection = out.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)
        return hidden_states


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------
class Qwen4ExpModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.hc_count = config.hc_count

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        # No residual flows between layers (hyper-connections carry the
        # stream); PP boundary carries the post-embed 2560-wide hidden.
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

        # Final hyper-connection mixer; its hc_norm IS the final norm
        # (reference: no separate model.norm).
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        num_experts = getattr(self.config, "num_experts", 0)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # (T, hidden) -> (T, hc_count*hidden): each 2560-block identical,
        # matching reference inputs_embeds.repeat(1, 1, hc_count).
        hidden_states = inputs_embeds.repeat(1, self.hc_count)

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states = layer(hidden_states, positions=positions, input_ids=input_ids, **kwargs)

        hidden_states = self.hyper_connection_mixer(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        fused_expert_params_mapping = [
            (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 0, "w1"),
            (f"experts.{base_layer}w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # QSA indexer is not ported (dense full attention, design law 3);
            # its tensors (index_qk_proj, q/k_layernorm) are not loaded.
            if ".indexer." in name:
                continue

            # The host-resident PLE n-gram table is loaded by the wrapper's
            # dedicated shard-assembly path (never through this loop).
            if "ngram_embedding" in name:
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # qwen4_exp checkpoint ships fused experts (no transpose)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (
                            name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ---------------------------------------------------------------------------
# ForCausalLM (language_model)
# ---------------------------------------------------------------------------
class Qwen4ExpForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4Exp currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen4ExpModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # MoE hyperparameters (QwenNextMixtureOfExperts)
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)


# ---------------------------------------------------------------------------
# ForConditionalGeneration — the registered checkpoint architecture
# (text-only: "model.visual.*" is skipped at load; no vision model exists)
# ---------------------------------------------------------------------------
class Qwen4ExpForConditionalGeneration(
    nn.Module,
    IsHybrid,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    # Checkpoint names ("model.language_model.layers...",
    # "lm_head.weight", "model.visual...") -> this tree
    # ("language_model.model.layers...", "language_model.lm_head...").
    # "model.visual.*" is dropped via skip_prefixes (no vision port).
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        config: Qwen4ExpConfig = vllm_config.model_config.hf_config

        self.config = config
        self.language_model = Qwen4ExpForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        # PLE table shard staging: part_idx -> cpu tensor (streamed concat in
        # process_weights_after_loading keeps peak host mem at ~1x table).
        # Scale tensors carry no shard index in their names -> plain list.
        self._ple_table_shards: dict[int, torch.Tensor] = {}
        self._ple_scale_shards: list[torch.Tensor] = []
        self._ple_assembled = False
        # Checkpoint-side copies of the deterministic hash buffers, staged for
        # an assert-equal gate against the values computed at init (catches
        # any divergence in the prime/multiplier math for free).
        self._ple_buffers: dict[str, torch.Tensor] = {}

    # -- forward plumbing ----------------------------------------------------
    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.model.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # VllmModel protocol surface (ModelConfig runner validation reads it)
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    # -- MoE plumbing (delegate to the causal LM which owns .model.layers) --
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def set_moe_parameters(self):
        self.language_model.set_moe_parameters()
        for attr in (
            "expert_weights", "moe_layers", "num_moe_layers", "num_expert_groups",
            "num_shared_experts", "num_logical_experts", "num_physical_experts",
            "num_local_physical_experts", "num_routed_experts", "num_redundant_experts",
        ):
            if hasattr(self.language_model, attr):
                setattr(self, attr, getattr(self.language_model, attr))

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None:
        self.language_model.update_physical_experts_metadata(
            num_physical_experts, num_local_physical_experts
        )

    # -- mamba state plumbing (uniform 4-slot, design law 1) -----------------
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]:
        conv_d, temporal_d = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )
        return (conv_d, temporal_d, conv_d, torch.int64)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        cfg = vllm_config.model_config.hf_text_config
        tp = get_tensor_model_parallel_world_size()
        spec = vllm_config.speculative_config
        num_spec = spec.num_speculative_tokens if spec is not None else 0
        conv_s, temporal_s = MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            cfg.linear_conv_kernel_dim,
            num_spec,
        )
        ple_len = (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size
        hc_hidden = cfg.hidden_size * cfg.hc_count
        return (
            conv_s,
            temporal_s,
            (ple_len + num_spec, hc_hidden),
            (cfg.ngram_size - 1 + num_spec, 1),
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc]:
        # slot0 GDN conv | slot1 GDN temporal | slot2 PLE conv | slot3 history
        return (
            get_conv_copy_spec,
            get_temporal_copy_spec,
            get_conv_copy_spec,
            get_conv_copy_spec,
        )

    # -- weight loading -------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = self._stage_ple_shards(weights, skip_table=_ple_env_flag("VLLM_QW4_PLE_SKIP_TABLE"))
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["model.visual.", "mtp."],
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        # The fork's loader never calls process_weights_after_loading on the
        # top-level wrapper (leg-4, 2026-10-11: no "Qwen4Exp PLE:" lines in the
        # serve log — table stayed zeros past load).  Assemble here, where the
        # call is guaranteed.
        self._assemble_ple_table()
        return loaded

    def _stage_ple_shards(
        self, weights: Iterable[tuple[str, torch.Tensor]], skip_table: bool = False
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Intercept PLE artifacts before the mapper; stream everything else
        through.  Hash buffers are always staged (verification gate); table
        shards and scales are staged unless ``skip_table`` (dev mode drops
        them entirely).  Shard index = the first number that follows
        "ngram_embedding" in the checkpoint tensor name."""
        import re

        part_re = re.compile(r"ngram_embedding\D*(\d+)")
        ple_buffers = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
        for name, tensor in weights:
            last = name.rsplit(".", 1)[-1]
            if "ple_embedding." in name and last in ple_buffers:
                self._ple_buffers[last] = tensor.to("cpu", copy=True)
                continue
            if "ngram_embedding" not in name:
                yield name, tensor
                continue
            if "scale" in name:
                # per-tensor scale ("ngram_embedding.weight_scale", BF16 [1]
                # in this checkpoint); no shard index in the name
                if not skip_table:
                    self._ple_scale_shards.append(tensor.to("cpu", copy=True))
                continue
            if skip_table:
                continue
            m = part_re.search(name)
            if m is None:
                raise RuntimeError(f"Qwen4Exp PLE: cannot parse shard index from {name!r}")
            part = int(m.group(1))
            self._ple_table_shards[part] = tensor.to("cpu", copy=True)

    def process_weights_after_loading(self, weights: Iterable[tuple[str, torch.Tensor]] | None = None) -> None:
        """Assemble the host-resident PLE table from staged shards (kept as
        the standard hook name; the real entry is load_weights)."""
        self._assemble_ple_table()

    def _assemble_ple_table(self) -> None:
        if self._ple_assembled:
            return
        self._ple_assembled = True
        ple_modules = [
            layer.ple.ple_embedding
            for layer in self.language_model.model.layers
            if getattr(layer, "ple", None) is not None
        ]
        if not ple_modules:
            return
        assert len(ple_modules) == 1, f"expected exactly one PLE layer, got {len(ple_modules)}"
        emb = ple_modules[0]

        # Hash-buffer gate: the checkpoint ships its own copies of the
        # deterministic multipliers / vocab sizes / offsets (verified
        # against the computed ones regardless of table staging).
        if self._ple_buffers:
            for buf_name, loaded in self._ple_buffers.items():
                have = getattr(emb, buf_name, None)
                if have is None:
                    continue
                loaded = loaded.to(have.device).to(have.dtype)
                if not torch.equal(loaded, have):
                    bad = int((loaded != have).sum().item())
                    raise RuntimeError(
                        f"Qwen4Exp PLE: checkpoint buffer {buf_name} diverges "
                        f"from computed values at {bad} positions — "
                        "hash/prime math mismatch"
                    )
            logger.info(
                "Qwen4Exp PLE: checkpoint hash buffers verified equal (%s)",
                sorted(self._ple_buffers),
            )
            self._ple_buffers.clear()

        if _ple_env_flag("VLLM_QW4_PLE_SKIP_TABLE") or not self._ple_table_shards:
            if not _ple_env_flag("VLLM_QW4_PLE_SKIP_TABLE"):
                logger.warning(
                    "Qwen4Exp PLE: no n-gram shards staged — table left as zeros "
                    "(was VLLM_QW4_PLE_SKIP_TABLE intended?)"
                )
            return

        parts = sorted(self._ple_table_shards.keys())
        expected = getattr(self.language_model.model.config, "split_ngram_parts", None)
        if expected is not None and len(parts) != expected:
            logger.warning(
                "Qwen4Exp PLE: staged %d shards, config split_ngram_parts=%d "
                "(continuing with what loaded)", len(parts), expected,
            )
        first = self._ple_table_shards[parts[0]]
        if first.dtype == torch.float8_e4m3fn or first.dtype == torch.uint8:
            table_dtype = torch.float8_e4m3fn
        else:
            # Unquantized checkpoint table (bf16/fp16): re-type the host
            # tensor so the concat and the gather both see the real dtype.
            table_dtype = first.dtype
            emb._host_table = torch.empty(
                tuple(emb._host_table.shape), dtype=table_dtype, device="cpu"
            )
        rows_per_part = first.shape[0]
        total_rows = rows_per_part * len(parts)
        if total_rows < emb.total_vocab_size:
            raise RuntimeError(
                f"Qwen4Exp PLE: shards cover {total_rows} rows < table vocab "
                f"{emb.total_vocab_size}"
            )

        logger.info(
            "Qwen4Exp PLE: assembling host table from %d shards (%d rows/shard, "
            "%s) — ~%.1f GB",
            len(parts), rows_per_part, table_dtype,
            total_rows * emb.head_dim_per_ngram * (1 if table_dtype == torch.float8_e4m3fn else 2) / 2**30,
        )
        view = emb._host_table.view(torch.uint8) if table_dtype == torch.float8_e4m3fn else emb._host_table
        for part in parts:
            shard = self._ple_table_shards.pop(part)
            src = shard.view(torch.uint8) if table_dtype == torch.float8_e4m3fn else shard
            lo = part * rows_per_part
            hi = min(lo + rows_per_part, view.shape[0])
            view[lo:hi].copy_(src[: hi - lo].to(view.dtype))
            del shard
        self._ple_table_shards.clear()

        # Scales: finalize layout from whatever landed.
        if table_dtype != torch.float8_e4m3fn:
            emb._col_block_size = None
        elif len(self._ple_scale_shards) == 0:
            logger.warning_once(
                "Qwen4Exp PLE: FP8 table loaded without scales — passthrough cast"
            )
            emb._col_block_size = None
        else:
            shards = self._ple_scale_shards
            scale = shards[0] if len(shards) == 1 else torch.cat(shards, dim=0)
            scale = scale.to(torch.float32)
            if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
                # per-tensor multiply scale (checkpoint "weight_scale")
                emb._host_scale = scale.reshape(())
                emb._col_block_size = None
            elif scale.ndim == 2:
                # block [128, 128] scales: (rows/128, col_blocks)
                if emb._host_table.shape[0] % 128:
                    raise RuntimeError(
                        f"Qwen4Exp PLE: block-scale rows {scale.shape[0]} do not "
                        "divide by 128 as expected"
                    )
                emb._host_scale_inv = scale
                emb._col_block_size = 128
            else:
                raise RuntimeError(f"Qwen4Exp PLE: unsupported scale shape {tuple(scale.shape)}")
            self._ple_scale_shards = []

        if _ple_env_flag("VLLM_QW4_PLE_PIN"):
            logger.info("Qwen4Exp PLE: pinning host table (%.1f GB)",
                        emb._host_table.numel() / 2**30)
            emb._host_table = emb._host_table.pin_memory()
        logger.info("Qwen4Exp PLE: host table ready (%s layout)",
                    "block128" if emb._col_block_size else "per-tensor/passthrough")
