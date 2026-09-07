# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for vLLM.

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K and store K+V into combined cache slot.
Decode:  Compute TQ attention scores from compressed cache,
         unpack FP16 values, softmax + weighted sum.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""

import functools
import math
import os
import time
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention,
    triton_turboquant_mq_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

logger = init_logger(__name__)

# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
_CONTINUATION_DECODE_THRESHOLD = 128

# llm-scaler v19: multi-query verify fast path. Speculative verify steps
# (and tiny continuation chunks) with 1 < q_len ≤ 8 run ONE multi-query
# kernel call that scores all query rows per KV tile, instead of q_len
# synthetic single-token decode calls that each re-scan the full
# compressed context (~q_len x KV bandwidth). Roll back with
# VLLM_TQ_MQ_VERIFY=0 (synthetic-decode path is kept).
_TQ_MQ_VERIFY = os.getenv("VLLM_TQ_MQ_VERIFY", "1") != "0"

# llm-scaler v47 diag: host-time counters for the drafter-forward cost hunt.
# SPECTIMING (v20) showed dflash dforward host=73ms vs device=30ms per step —
# host-bound. These counters split the TQ-side host time per attention call
# (steady-state decode calls are DRAFT layers only; the target verify path is
# graph-replayed so its python does not run per step). Zero cost when
# VLLM_TQ_TIME is unset (one dict-lookup bool check per call).
_TQ_TIME = os.getenv("VLLM_TQ_TIME", "0") == "1"
_TQT: dict[str, float] = {
    "fwd": 0.0, "store": 0.0, "mq": 0.0, "fa": 0.0, "nocpu": 0.0,
    "n": 0.0, "lfwd": 0.0, "lstore": 0.0, "lmq": 0.0, "lfa": 0.0,
    "ln": 0.0, "lnocpu": 0.0,
}


def _tqt_flush() -> None:
    n = int(_TQT["n"])
    if n and n % 200 == 0:
        d = max(n - _TQT["ln"], 1.0)
        logger.info(
            "TQTIME calls=%d fwd=%.2f store=%.2f mq=%.2f fa=%.2f nocpu=%d (ms/call over %d calls)",
            n,
            (_TQT["fwd"] - _TQT["lfwd"]) / d,
            (_TQT["store"] - _TQT["lstore"]) / d,
            (_TQT["mq"] - _TQT["lmq"]) / d,
            (_TQT["fa"] - _TQT["lfa"]) / d,
            int(_TQT["nocpu"] - _TQT["lnocpu"]),
            int(d),
        )
        _TQT["lfwd"] = _TQT["fwd"]
        _TQT["lstore"] = _TQT["store"]
        _TQT["lmq"] = _TQT["mq"]
        _TQT["lfa"] = _TQT["fa"]
        _TQT["lnocpu"] = _TQT["nocpu"]
        _TQT["ln"] = float(n)


def _tqt_profile(impl, *args) -> None:
    """llm-scaler v47 diag: one-off cProfile snapshot of a single
    _forward_impl call, dumped to the log. Pure-read wrt engine state
    (the MQ/store kernels are idempotent; output buffers are not
    consumed by this extra call)."""
    import cProfile
    import io as _io
    import pstats

    _pr = cProfile.Profile()
    _pr.enable()
    impl._forward_impl(*args)
    _pr.disable()
    _s = _io.StringIO()
    pstats.Stats(_pr, stream=_s).sort_stats("tottime").print_stats(25)
    logger.info("TQPROF snapshot:\n%s", _s.getvalue())
# One-time marker for the v21 non-causal draft path (host-side log only,
# never reads device memory in the hot path).
_TQ_NC_MQ_LOGGED = False
_TQ_MQ_MAX_Q = max(2, int(os.getenv("VLLM_TQ_MQ_MAX_Q", "8") or 8))

# llm-scaler v19b: spec-verify XPU graphs must capture the continuation
# path (KV-cache-reading kernels), never the raw-KV flash fast path.
# Capture dummies have seq_len == q_len, so without this the fast path
# fires at capture and every replay attends ONLY the in-batch verify
# tokens — the target never sees the cached context (#TQ-blind-verify:
# near-zero acceptance + "blank message" hallucinations). Decode graphs
# (q_len == 1) are unaffected. Roll back with VLLM_TQ_VERIFY_GRAPH_FIX=0.
_TQ_VERIFY_GRAPH_FIX = os.getenv("VLLM_TQ_VERIFY_GRAPH_FIX", "1") != "0"

# llm-scaler v52 (SPLITS G1): MQ-verify / DFlash-drafter split count
# knob. The MQ kernels historically ran at the hard-coded kernel default
# (32 splits) while q=1 decode scales with the ctx-tier ladder (up to
# 256 under full-decode graphs) — hypothesis was that verify at depth
# was split-starved. REFUTED BY MEASUREMENT (2026-09-06, mtp4/tq4nc
# ctxscan, v1.2.5t1): S=64 = 72.7/41.9/26.2/17.8 tps @2k/16k/32k/65k
# and S=16 = 74.5/32.9/20.4/14.1 vs S=32 = 72.9/47.3/29.0/21.6 — 32 is
# the optimum on BOTH sides (warps=1 wide-Q-tile kernel extracts its
# parallelism per-split, unlike the decode kernel's BLOCK_KV=4 tiles +
# per-row-skip stage2 at 256). Default stays 32 (v1.2.4-identical);
# VLLM_TQ_MQ_SPLITS>0 pins a value for future experiments only.
try:
    _TQ_MQ_SPLITS = max(0, int(os.getenv("VLLM_TQ_MQ_SPLITS", "0") or 0))
except ValueError:
    _TQ_MQ_SPLITS = 0

# llm-scaler v53 (v125-audit L10, 2026-09-07): env-gated KV-splits tier
# usage logging for the df7 128k-cliff bisect (REPORT D10). Default off;
# VLLM_V53_TQTIER=1 enables periodic _effective_kv_splits() decision logs.
try:
    _V53_TQTIER = os.getenv("VLLM_V53_TQTIER", "0") == "1"
except ValueError:
    _V53_TQTIER = False

# Shared grow-only dequant buffers for _continuation_prefill when the
# workspace manager is locked (PIECEWISE graphs). Full-attention layers
# execute sequentially, so a single (k, v) pair shared across all layers
# and steps is safe and uses ~1/num_layers the memory of per-layer buffers.
_DEQUANT_BUFS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), cached per (d, device).

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    # Normalize device string so "cuda" and "cuda:0" hit the same cache entry.
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Standard attention backends use (2, num_blocks, block_size, num_kv_heads,
        head_dim) with a leading 2 to separate K and V. TurboQuant packs K+V
        into a single interleaved slot per head per position, so the cache is:

            (num_blocks, block_size, num_kv_heads, slot_size_aligned)

        Each slot = [key_packed | value_packed | padding].
        This is safe because TQ has its own get_kv_cache_shape override and
        never shares cache tensors with other backends. Layers that fall back
        to native dtype via kv_cache_dtype_skip_layers get their own
        standard-shaped cache allocation.

        head_size is the model's real head_dim. slot_size_aligned is computed
        from the TQ config to ensure correct cache allocation for all head dims.
        """
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        # llm-scaler v21c: the probe result is dtype-independent for TQ
        # (num_blocks is always dim 0), but the base implementation derives
        # its sentinel shape through get_kv_cache_shape, which parses
        # cache_dtype_str. In a mixed-dtype runner (v21 draft-KV policy,
        # e.g. an fp8 target pool next to a k8v4 DFlash draft pool) the
        # runner passes ITS dtype here, which is not a TQ preset, and pool
        # init died with "Unknown TurboQuant cache dtype: 'fp8_e4m3'" in
        # _update_hybrid_attention_mamba_layout. Probe with a valid preset;
        # the returned dim is identical across presets.
        if not cache_dtype_str.startswith("turboquant_"):
            cache_dtype_str = "turboquant_k8v4"
        return super().get_kv_cache_block_dim(
            block_size, num_kv_heads, head_size, cache_dtype_str
        )

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("turboquant_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    seq_lens: torch.Tensor  # (num_reqs,) — total context length per request
    slot_mapping: torch.Tensor  # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — cu_seqlens for queries
    num_actual_tokens: int = 0  # actual tokens (excluding padding)
    max_query_len: int = 0  # longest query in batch
    max_seq_len: int = 0  # longest context in batch
    is_prefill: bool = False
    num_decodes: int = 0  # number of decode requests (first in batch)
    num_decode_tokens: int = 0  # tokens from decode requests
    # CPU-resident copies used by the prefill path for per-request iteration
    # without per-step D2H syncs.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    # llm-scaler v21: causal mask flag. True = standard causal attention
    # (target prefill/decode/spec-verify). False = DFlash draft steps, whose
    # full-attention draft layers must attend NON-causally (every query row
    # sees the whole stored context incl. the current chunk). Honored by the
    # _prefill_attention continuation paths; the pure-decode path is mask-
    # invariant (q_len == 1). DFlash asserts metadata.causal is False for
    # non-SWA draft layers, so the field must exist and be reported.
    causal: bool = True


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # llm-scaler v19b: multi-token (spec-verify) captures must not take
        # the raw-KV flash fast path in _prefill_attention. The dummy batch
        # has seq_len == q_len == num_spec_tokens, so max_query_len ==
        # max_seq_len would bake flash attention over ONLY the in-batch
        # verify tokens into the graph; replays then never read the KV
        # cache (#TQ-blind-verify). Force the continuation path instead —
        # its kernels derive the attention extent from the dynamic
        # seq_lens/block_table buffers, which are refreshed at replay.
        # Keep seq_lens small so capture-time kernel work stays trivial.
        if _TQ_VERIFY_GRAPH_FIX and attn_metadata.max_query_len > 1:
            attn_metadata.max_seq_len = attn_metadata.max_query_len + 1
            if attn_metadata.seq_lens_cpu is not None:
                attn_metadata.seq_lens_cpu.fill_(attn_metadata.max_query_len + 1)
            attn_metadata.seq_lens.fill_(attn_metadata.max_query_len)
        else:
            # Set seq_lens to 1 so CUDA graph capture is fast
            # (real seq_lens are filled at replay time).
            attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build TurboQuantMetadata from common attention metadata."""
        cam = common_attn_metadata

        # With reorder_batch_threshold=1, the model runner guarantees
        # decodes come first in the batch. split_decodes_and_prefills
        # finds the boundary (operates on CPU tensors — no GPU sync).
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
            causal=getattr(cam, "causal", True),
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> TurboQuantMetadata:
        """llm-scaler v21: metadata for the DFlash draft step (TQ draft KV).

        The draft propose batch is heterogeneous: per-request query length =
        accepted+1 in 1..k+1, and unlike the target runner the drafter does
        not reorder_batch, so decode-style (q_len == 1) requests are NOT
        guaranteed to precede prefill-style (q_len > 1) ones. The mixed path
        in TurboQuantAttentionImpl.forward splits on that decodes-first
        ordering, so for a genuinely mixed batch force the pure-prefill path:
        its per-request continuation loop handles any query-length mix
        (q_len == 1 rows take the synthetic-decode path, q_len > 1 rows the
        multi-query kernel; both read the dynamic seq_lens/block_table
        buffers). Pure-decode and pure-prefill batches keep their natural
        fast paths untouched.
        """
        attn_metadata = self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )
        if (
            attn_metadata.num_decodes > 0
            and attn_metadata.num_decode_tokens < attn_metadata.num_actual_tokens
        ):
            attn_metadata.num_decodes = 0
            attn_metadata.num_decode_tokens = 0
            attn_metadata.is_prefill = True
        return attn_metadata


class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
    """TurboQuant attention implementation.

    Vectorized PyTorch: batch quantize/store, vectorized bit-unpack
    decode with einsum scores and value gather.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        # llm-scaler v41: keep the read-side sliding window. Stock
        # dropped it here, so a sliding-window draft (DFlash2,
        # sliding_window=2048) ramped causal attention over the
        # FULL prefix instead of the trained trailing window
        # (acceptance 72% @2k -> 9% @74k). None for full-attention
        # layers / MTP / DFlash-v1 -> all hooks below stay inert.
        self.sliding_window = sliding_window

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Pre-compute kernel constants from config (avoid repeated arithmetic)
        cfg = self.tq_config
        self._mse_bytes = (
            math.ceil(head_size * cfg.key_mse_bits / 8)
            if not cfg.key_fp8
            else head_size
        )
        self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
        self._n_centroids = cfg.n_centroids if not cfg.key_fp8 else 1

        # Detect flash-attn version (FA2/3/4) for prefill paths.
        self.fa_version = get_flash_attn_version(head_size=head_size)

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

        # Adaptive split-KV scaling for long contexts. Under PIECEWISE
        # XPU graphs (and eager mode) attention is a splitting op that
        # runs OUTSIDE the captured pieces (see platforms/xpu.py
        # splitting_ops), so the launch grid may vary per batch. Under
        # full-decode capture modes the attention kernel is captured and
        # the grid must stay constant -- but both TQ stages derive each
        # split's [start, end) range from the seq_lens DEVICE tensor
        # (same cdiv formula in stage1 and stage2) and skip inactive
        # splits, so a fixed oversized grid is graph-safe: at replay the
        # fixed split count simply re-partitions whatever the live
        # context length is. Size the fixed grid to the adaptive tier
        # the model's max context would select.
        from vllm.config import CUDAGraphMode

        adaptive_env = os.environ.get("VLLM_TQ_ADAPTIVE_KV_SPLITS", "1")
        try:
            self._tq_max_kv_splits = max(
                self.max_num_kv_splits,
                int(os.environ.get("VLLM_TQ_MAX_KV_SPLITS", "256")),
            )
            self._tq_max_kv_splits = min(self._tq_max_kv_splits, 1024)
        except ValueError:
            self._tq_max_kv_splits = 256
        graph_mode = vllm_config.compilation_config.cudagraph_mode
        full_capture = graph_mode in (
            CUDAGraphMode.FULL_DECODE_ONLY,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        )
        self._tq_adaptive_splits = adaptive_env == "1" and not full_capture

        if full_capture:
            try:
                env_g = int(os.environ.get("VLLM_TQ_GRAPH_KV_SPLITS", "0") or 0)
            except ValueError:
                env_g = 0
            if env_g > 0:
                self._tq_graph_kv_splits = max(
                    1, min(env_g, self._tq_max_kv_splits)
                )
            else:
                max_len = vllm_config.model_config.max_model_len
                if max_len <= 16384:
                    tier = 1
                elif max_len <= 49152:
                    tier = 2
                elif max_len <= 131072:
                    tier = 4
                else:
                    tier = 8
                self._tq_graph_kv_splits = min(
                    self.max_num_kv_splits * tier, self._tq_max_kv_splits
                )
            logger.info(
                "TurboQuant graph-fixed KV splits: grid=%d (device-side "
                "split masking; VLLM_TQ_GRAPH_KV_SPLITS overrides).",
                self._tq_graph_kv_splits,
            )
        else:
            self._tq_graph_kv_splits = self.max_num_kv_splits

        if adaptive_env == "1" and not self._tq_adaptive_splits:
            logger.info(
                "TurboQuant adaptive KV splits ladder inactive under "
                "cudagraph_mode=%s; using graph-fixed split grid=%d.",
                graph_mode,
                self._tq_graph_kv_splits,
            )
        elif self._tq_adaptive_splits:
            logger.info(
                "TurboQuant adaptive KV splits enabled: base=%d ceiling=%d "
                "(VLLM_TQ_MAX_KV_SPLITS, VLLM_TQ_ADAPTIVE_KV_SPLITS).",
                self.max_num_kv_splits,
                self._tq_max_kv_splits,
            )
        # llm-scaler v52 (SPLITS G1): one-time MQ split-parallelism notice.
        if _TQ_MQ_SPLITS > 0:
            logger.info(
                "TurboQuant MQ verify KV splits: pinned=%d (VLLM_TQ_MQ_SPLITS "
                "EXPERIMENTAL — 32 is the measured optimum).",
                max(1, min(_TQ_MQ_SPLITS, self._tq_max_kv_splits)),
            )
        else:
            logger.info(
                "TurboQuant MQ verify KV splits: 32 (measured optimum; "
                "ladder-follow refuted 2026-09-06).",
            )

    def _effective_kv_splits(self, max_seq_len: int) -> int:
        """Split-KV parallelism for this step.

        Eager/PIECEWISE (attention outside the captured pieces): scale
        the split count with the live context length -- with fixed splits
        each program walks seq_len/splits KV positions, and at 40k+
        contexts that serial walk dominates decode time.

        Full-decode capture: return the constant graph grid; the kernels
        re-partition the live context device-side at replay, so deep
        contexts still get full split parallelism without a dynamic grid.
        """
        if not self._tq_adaptive_splits or max_seq_len <= 0:
            return self._tq_graph_kv_splits
        base = self.max_num_kv_splits
        if max_seq_len <= 16384:
            splits = base
        elif max_seq_len <= 49152:
            splits = base * 2
        elif max_seq_len <= 131072:
            splits = base * 4
        else:
            splits = base * 8
        eff = min(splits, self._tq_max_kv_splits)
        # llm-scaler v53 (v125-audit L10, 2026-09-07): env-gated tier-usage
        # log for the df7 128k anomaly bisect (REPORT D10: 128k decode runs
        # at ~half the 227k rate on df7+tq4). Static analysis already
        # weakens the tier-boundary suspect: with base=256/cap=1024 the
        # 49k-131k (4x) and >131k (8x->capped) tiers collapse to the SAME
        # effective splits, so if this log shows equal splits at 136k vs
        # 243k the cliff must live elsewhere (dflash draft window vs the
        # 2048-token mamba blocks, or a graph-capture bucket gap).
        # VLLM_V53_TQTIER=1 enables; logs at most every 512th call.
        if _V53_TQTIER:
            self._v53_tier_calls = getattr(self, "_v53_tier_calls", 0) + 1
            if self._v53_tier_calls == 1 or self._v53_tier_calls % 512 == 0:
                logger.info(
                    "V53_TQTIER #%d: max_seq_len=%d base=%d tier_splits=%d "
                    "eff=%d graph_kv=%d",
                    self._v53_tier_calls, max_seq_len, base, splits, eff,
                    self._tq_graph_kv_splits)
        return eff

    def _mq_kv_splits(self, max_seq_len: int) -> int:
        """llm-scaler v52 (SPLITS G1): split count for MQ-verify /
        DFlash-drafter kernel calls (and the synthetic-decode rollback).

        Default 32 = the historical kernel default, MEASURED OPTIMAL
        (2026-09-06: ladder-follow/256-style scaling REFUTED — S=64 lost
        11-18% and S=16 lost 30-35% at 16k-65k on mtp4/tq4nc; see the
        module-top comment). VLLM_TQ_MQ_SPLITS>0 pins an experimental
        value (graph-safe: stage1 re-partitions the live context from
        the q0_seq_lens device tensor)."""
        if _TQ_MQ_SPLITS > 0:
            return max(1, min(_TQ_MQ_SPLITS, self._tq_max_kv_splits))
        return 32

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = True,
    ) -> torch.Tensor:
        # fa_utils.get_flash_attn_version() returns None on backends that
        # should not pass an explicit fa_version kwarg.
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=causal,
            )
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=causal,
            fa_version=self.fa_version,
        )

    def _ensure_on_device(self, layer, device):
        """One-time derivation of TQ buffers (rotation matrix, midpoints).

        The Hadamard rotation is shared across all layers: random sign
        flips do not improve Lloyd-Max quantization quality because the
        quantizer is symmetric around zero (sign-flipping a coordinate
        maps it to the mirror centroid with identical distortion).
        """
        if not hasattr(layer, "_tq_cached"):
            D = self.head_size

            # Pure Hadamard: orthonormal + symmetric (H = H^T), enabling
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            # fp16 copy for rotation in continuation prefill path
            layer._tq_Pi_half = H.to(torch.float16)

            # Centroids for Lloyd-Max quantization.
            layer._tq_centroids = get_centroids(D, self.tq_config.centroid_bits).to(
                device=device, dtype=torch.float32
            )

            c_sorted, _ = layer._tq_centroids.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # llm-scaler v47 diag: host-time wrapper (VLLM_TQ_TIME=1).
        if not _TQ_TIME:
            return self._do_kv_cache_update_impl(
                layer, key, value, kv_cache, slot_mapping
            )
        _t0 = time.perf_counter()
        try:
            return self._do_kv_cache_update_impl(
                layer, key, value, kv_cache, slot_mapping
            )
        finally:
            _TQT["store"] += (time.perf_counter() - _t0) * 1e3

    def _do_kv_cache_update_impl(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache.

        Called as a separate custom op (unified_kv_cache_update) BEFORE
        the attention forward, matching FlashAttention's split pattern.
        slot_mapping is already sliced to num_actual_tokens by the caller.
        """
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        # llm-scaler v45 diag: store-kernel fence for the DFlash2 x TQ
        # concurrent wedge (fires before forward; closes the "died before
        # attention" hole).
        _st_dbg = (
            os.environ.get("DFLASH_TQ_DBG", "0") == "1"
            and N <= 128
            and not torch.xpu.is_current_stream_capturing()
        )
        if _st_dbg:
            _slots = slot_mapping.tolist()
            torch.xpu.synchronize()
            logger.warning(
                "DFLASH_TQDBG store L=%s N=%d kvptr=0x%x cap=%d "
                "slots[:8]=%s slots[-8:]=%s smin=%d smax=%d (pre)",
                getattr(layer, "layer_id", -1),
                N,
                kv_cache.data_ptr(),
                kv_cache.shape[0] * kv_cache.shape[1],
                _slots[:8],
                _slots[-8:],
                min(_slots),
                max(_slots),
            )
        self._store_kv(k, v, kv_cache, slot_mapping, layer)
        if _st_dbg:
            torch.xpu.synchronize()
            logger.warning(
                "DFLASH_TQDBG store post OK L=%s N=%d",
                getattr(layer, "layer_id", -1),
                N,
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # llm-scaler v47 diag: host-time wrapper (VLLM_TQ_TIME=1).
        if not _TQ_TIME:
            return self._forward_impl(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )
        _t0 = time.perf_counter()
        try:
            _ret = self._forward_impl(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )
        finally:
            _TQT["fwd"] += (time.perf_counter() - _t0) * 1e3
            _TQT["n"] += 1.0
        if int(_TQT["n"]) % 600 == 0:
            try:
                _tqt_profile(
                    self, layer, query, key, value, kv_cache, attn_metadata,
                    output, output_scale, output_block_scale,
                )
            except Exception:
                logger.exception("TQPROF failed")
        _tqt_flush()
        return _ret

    def _forward_impl(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        # Get TQ buffers, ensure on device (one-time migration).
        # Use Any-typed alias for dynamic _tq_* attrs set by _ensure_on_device.
        tq_layer: Any = layer
        device = q.device
        self._ensure_on_device(tq_layer, device)
        Pi = tq_layer._tq_Pi
        PiT = tq_layer._tq_PiT
        centroids = tq_layer._tq_centroids

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        # llm-scaler v45 diag: forward-entry probe for the DFlash2 x TQ
        # concurrent wedge. Pure-decode batches excluded (is_prefill False);
        # verify steps and small/medium prefills (incl. the death batch
        # [verify-8 + prefill-58]) are dumped with their full composition.
        _tq_dbg = (
            os.environ.get("DFLASH_TQ_DBG", "0") == "1"
            and attn_metadata.is_prefill
            and N <= 128
            and not torch.xpu.is_current_stream_capturing()
        )
        if _tq_dbg:
            _qslc = (
                attn_metadata.query_start_loc_cpu.tolist()
                if getattr(attn_metadata, "query_start_loc_cpu", None) is not None
                else attn_metadata.query_start_loc.tolist()
            )
            _slc = (
                attn_metadata.seq_lens_cpu.tolist()
                if getattr(attn_metadata, "seq_lens_cpu", None) is not None
                else attn_metadata.seq_lens.tolist()
            )
            logger.warning(
                "DFLASH_TQDBG fwd L=%s N=%d ndec=%d ndectok=%d mq=%s ms=%s "
                "causal=%s sw=%s qsl=%s sl=%s bt=%s kvc=%s kvptr=0x%x cu0=%s",
                getattr(layer, "layer_id", -1),
                N,
                num_decodes,
                num_decode_tokens,
                attn_metadata.max_query_len,
                attn_metadata.max_seq_len,
                getattr(attn_metadata, "causal", True),
                self.sliding_window,
                _qslc,
                _slc,
                tuple(attn_metadata.block_table.shape),
                tuple(kv_cache.shape[:2]),
                kv_cache.data_ptr(),
                (
                    self._cu_2.tolist()[0]
                    if hasattr(self, "_cu_2")
                    else None
                ),
            )

        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q,
                k,
                v,
                kv_cache,
                attn_metadata,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            attn_out = torch.zeros(
                N, self.num_heads, self.head_size, device=device, dtype=q.dtype
            )

            # --- Decode portion (first num_decodes requests) ---
            # Use full-batch max_seq_len as safe upper bound (no GPU sync).
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, Pi, centroids, PiT, layer
            )

            # --- Prefill portion (remaining requests) ---
            # CRITICAL: use prefill-specific max_seq_len so flash_attn's
            # fast path (max_query_len == max_seq_len) triggers for
            # first-chunk prefills. Using full-batch max_seq_len breaks
            # this because decode requests inflate max_seq_len.
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            # Use the CPU-resident `seq_lens` upper-bound from the metadata
            # (populated in the builder) to compute the prefill sub-batch
            # max without a GPU→CPU sync.
            if attn_metadata.seq_lens_cpu is not None:
                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())
            else:
                prefill_max_seq = attn_metadata.max_seq_len
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_qsl_cpu = None
            if attn_metadata.query_start_loc_cpu is not None:
                prefill_qsl_cpu = (
                    attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
            prefill_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
                query_start_loc_cpu=prefill_qsl_cpu,
                seq_lens_cpu=attn_metadata.seq_lens_cpu[num_decodes:]
                if attn_metadata.seq_lens_cpu is not None
                else None,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )

        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache (vectorized)                         #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        """Quantize + store via fused Triton kernel."""
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
        )

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        # llm-scaler v21: causal mask flag from metadata (False only for
        # DFlash draft steps — non-causal over the stored context).
        causal = getattr(attn_metadata, "causal", True)

        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                causal=causal,
            )

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Prefer the CPU-resident copies from the metadata if populated —
        # otherwise `.tolist()` on GPU tensors forces a synchronizing copy.
        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
            if _TQ_TIME:
                _TQT["nocpu"] += 1.0
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        # Pre-allocate cu_seqlens for single-request flash_attn calls
        # to avoid per-request host→device tensor creation.
        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if _HAS_FLASH_ATTN:
                    # llm-scaler v46: self-heal cu_seqlens[0]. The buffer is
                    # allocated once (torch.zeros) and element 0 was never
                    # rewritten after creation, so any stray device-side
                    # write that lands on it persists FOREVER — the next
                    # flash_attn call then reads a wild batch-start offset
                    # (observed: cu[0] = -1096007830 in the DFlash2 x TQ
                    # concurrent wedge) and its index math runs ~hundreds of
                    # GB out of bounds, wedging both GPUs (ccs resets,
                    # DEVICE_LOST). cu[0] MUST be 0 by contract; rewrite it
                    # stream-ordered immediately before every call.
                    if not getattr(self, "_cu_heal_logged", False):
                        self._cu_heal_logged = True
                        logger.info(
                            "TurboQuant cu_seqlens self-heal active (v46):"
                            " element 0 rewritten before every flash_attn"
                            " call (DFlash2 x TQ wedge fix)."
                        )
                    self._cu_2[0:1] = 0
                    self._cu_2[1:2] = q_len
                    cu = self._cu_2
                    _fa_dbg = (
                        os.environ.get("DFLASH_TQ_DBG", "0") == "1"
                        and q_len <= 128
                        and not torch.xpu.is_current_stream_capturing()
                    )
                    if _fa_dbg:
                        torch.xpu.synchronize()
                        logger.warning(
                            "DFLASH_TQDBG fa L=%s i=%d qlen=%d cu=%s "
                            "qptr=0x%x kptr=0x%x vptr=0x%x qfin=%s kfin=%s "
                            "vfin=%s (pre)",
                            getattr(layer, "layer_id", -1),
                            i,
                            q_len,
                            cu.tolist(),
                            q_seq.data_ptr(),
                            k_seq.data_ptr(),
                            v_seq.data_ptr(),
                            bool(torch.isfinite(q_seq.float()).all().item()),
                            bool(torch.isfinite(k_seq.float()).all().item()),
                            bool(torch.isfinite(v_seq.float()).all().item()),
                        )
                    _fa_t0 = time.perf_counter() if _TQ_TIME else 0.0
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                    if _TQ_TIME:
                        _TQT["fa"] += (time.perf_counter() - _fa_t0) * 1e3
                    if _fa_dbg:
                        torch.xpu.synchronize()
                        logger.warning(
                            "DFLASH_TQDBG fa post OK L=%s i=%d",
                            getattr(layer, "layer_id", -1),
                            i,
                        )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to TQ cache
                # by do_kv_cache_update. Use decode kernel directly to
                # avoid O(cached_len) full-dequant per continuation.
                # For large continuations, fall back to _continuation_prefill.
                cached_len = seq_len - q_len
                if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    # Fast path: treat each query as a decode request
                    # with incremental seq_lens for causal masking.
                    # llm-scaler v19b: derive per-row causal limits from the
                    # DYNAMIC seq_lens buffer (seq_lens[i] - (q_len-1) + row)
                    # instead of slicing a static arange with capture-time
                    # CPU scalars. Inside a captured graph the subtraction
                    # re-executes at replay with the refreshed seq_lens, so
                    # the attention extent is always the real context length.
                    if _TQ_MQ_VERIFY and 1 < q_len <= _TQ_MQ_MAX_Q:
                        # llm-scaler v19: single multi-query kernel call —
                        # one pass over the compressed KV for all q_len
                        # rows (spec verify / tiny continuation chunk).
                        # Causal: row-0 limit is cached_len + 1 (the kernel
                        # derives the remaining rows' ramp internally).
                        # v21: non-causal DFlash draft steps route here too
                        # (q0 = FULL seq_len, every row attends the stored
                        # context) — replaces the q_len full-context
                        # rescans of the synthetic-decode path below.
                        if not causal:
                            global _TQ_NC_MQ_LOGGED
                            if not _TQ_NC_MQ_LOGGED:
                                _TQ_NC_MQ_LOGGED = True
                                logger.info(
                                    "TurboQuant non-causal MQ draft attention"
                                    " active (v21): DFlash draft steps use the"
                                    " multi-query kernel (q_len=%d).",
                                    q_len,
                                )
                        mq_bt = attn_metadata.block_table[i : i + 1]
                        if causal:
                            mq_q0 = (
                                attn_metadata.seq_lens[i : i + 1]
                                - (q_len - 1)
                            )
                        else:
                            mq_q0 = attn_metadata.seq_lens[i : i + 1]
                        mq_win = None
                        if causal and self.sliding_window is not None:
                            # llm-scaler v41: enforce the read-side
                            # sliding window for sliding drafts
                            # (DFlash2). Rebase the block table so
                            # local position 0 = the window's
                            # block-aligned start, shift q0 by the
                            # same base, pass the (possibly
                            # negative) local window start for the
                            # kernel's per-row lower bound. Device
                            # ops only — replay-safe under XPU
                            # graphs; short sequences resolve to
                            # off=0 / negative win == exact causal
                            # semantics.
                            #
                            # llm-scaler v49 (DFlash2 perf): the v41
                            # rebase ran its ~10-op device chain
                            # (arange/clamp/gather/sub/mul/cast) PER
                            # REQUEST PER LAYER — 5 draft layers x B
                            # requests x every draft sub-step, each
                            # tiny launch inflating to ~0.5 ms under
                            # load (TQTIME fwd 7.9 ms/call vs 0.45 on
                            # the windowless MTP lane = the whole
                            # propose gap). The rebase is
                            # layer-independent: compute it ONCE per
                            # metadata instance, VECTORIZED over all
                            # requests (~6 ops total), and cache it on
                            # the metadata object for the remaining
                            # draft layers. Cache is bypassed while a
                            # graph is capturing so captured regions
                            # keep emitting their own (baked, live-
                            # buffer-reading) ops — replay semantics
                            # unchanged.
                            if not getattr(
                                self, "_sw_mq_logged", False
                            ):
                                self._sw_mq_logged = True
                                logger.info(
                                    "TurboQuant sliding-window MQ"
                                    " draft attention active (v41,"
                                    " v49 batched rebase):"
                                    " w=%d, q_len=%d.",
                                    self.sliding_window,
                                    q_len,
                                )
                            _bs = kv_cache.shape[1]
                            _capturing = False
                            try:
                                _capturing = (
                                    torch.xpu.is_current_stream_capturing()
                                )
                            except Exception:
                                pass
                            _sw_entry = None
                            if not _capturing:
                                _cache = getattr(
                                    attn_metadata, "_v49_sw_rebase", None
                                )
                                if _cache is not None and _cache[0] == (
                                    self.sliding_window,
                                    _bs,
                                ):
                                    _sw_entry = _cache[1].get(q_len)
                            if _sw_entry is None:
                                _bt_all = attn_metadata.block_table
                                _nreq = _bt_all.shape[0]
                                _q0_all = (
                                    attn_metadata.seq_lens[:_nreq]
                                    - (q_len - 1)
                                )
                                _ws_all = _q0_all - self.sliding_window
                                _off_all = (
                                    torch.clamp_min(_ws_all, 0) // _bs
                                )
                                _idx_all = (
                                    torch.arange(
                                        _bt_all.shape[1],
                                        device=_bt_all.device,
                                        dtype=torch.int64,
                                    ).unsqueeze(0)
                                    + _off_all.to(torch.int64).unsqueeze(1)
                                ).clamp_max(_bt_all.shape[1] - 1)
                                _sw_entry = (
                                    torch.gather(_bt_all, 1, _idx_all),
                                    _q0_all - _off_all * _bs,
                                    (
                                        _ws_all - _off_all * _bs
                                    ).to(torch.int32),
                                )
                                if not _capturing:
                                    try:
                                        _cache = getattr(
                                            attn_metadata,
                                            "_v49_sw_rebase",
                                            None,
                                        )
                                        if _cache is None or _cache[0] != (
                                            self.sliding_window,
                                            _bs,
                                        ):
                                            _cache = (
                                                (
                                                    self.sliding_window,
                                                    _bs,
                                                ),
                                                {},
                                            )
                                            attn_metadata._v49_sw_rebase = (
                                                _cache
                                            )
                                        _cache[1][q_len] = _sw_entry
                                    except Exception:
                                        pass
                            mq_bt = _sw_entry[0][i : i + 1]
                            mq_q0 = _sw_entry[1][i : i + 1]
                            mq_win = _sw_entry[2][i : i + 1]
                        _mq_dbg = (
                            os.environ.get("DFLASH_TQ_DBG", "0") == "1"
                            and not torch.xpu.is_current_stream_capturing()
                        )
                        if _mq_dbg:
                            _btraw = attn_metadata.block_table[i : i + 1].tolist()[0]
                            _btused = mq_bt.tolist()[0]
                            logger.warning(
                                "DFLASH_TQDBG mq L=%s i=%d qlen=%d slen=%d "
                                "cached=%d causal=%s sw=%s q0=%s win=%s "
                                "btraw[:20]=%s nzraw=%d btused[:20]=%s "
                                "nblocks=%d bs=%d",
                                getattr(layer, "layer_id", -1),
                                i,
                                q_len,
                                seq_len,
                                cached_len,
                                causal,
                                self.sliding_window,
                                mq_q0.tolist(),
                                mq_win.tolist() if mq_win is not None else None,
                                _btraw[:20],
                                sum(1 for b in _btraw if b != 0),
                                _btused[:20],
                                kv_cache.shape[0],
                                kv_cache.shape[1],
                            )
                            torch.xpu.synchronize()
                            logger.warning(
                                "DFLASH_TQDBG mq pre-sync OK L=%s i=%d",
                                getattr(layer, "layer_id", -1),
                                i,
                            )
                        _mq_t0 = time.perf_counter() if _TQ_TIME else 0.0
                        out = triton_turboquant_mq_decode_attention(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=mq_bt,
                            q0_seq_lens=mq_q0,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                            non_causal=(not causal),
                            win_start=mq_win,
                            max_num_kv_splits=self._mq_kv_splits(
                                getattr(attn_metadata, "max_seq_len", 0)
                            ),
                        )
                        if _TQ_TIME:
                            _TQT["mq"] += (time.perf_counter() - _mq_t0) * 1e3
                        if _mq_dbg:
                            torch.xpu.synchronize()
                            logger.warning(
                                "DFLASH_TQDBG mq post-sync OK L=%s i=%d",
                                getattr(layer, "layer_id", -1),
                                i,
                            )
                    else:
                        # Synthetic decode: one single-token decode call
                        # per query row (v18 behavior; VLLM_TQ_MQ_VERIFY=0).
                        # v21: also the NON-CAUSAL path for DFlash draft
                        # steps — every row attends to the FULL stored
                        # seq_len (the chunk's K/V were already stored by
                        # do_kv_cache_update via the drafter's precompute),
                        # instead of the causal ramp seq_len - (q_len-1) + r.
                        synth_bt_row = attn_metadata.block_table[i : i + 1]
                        if causal:
                            synth_dyn = (
                                attn_metadata.seq_lens[i : i + 1]
                                + torch.arange(
                                    q_len,
                                    device=attn_metadata.seq_lens.device,
                                    dtype=attn_metadata.seq_lens.dtype,
                                )
                                - (q_len - 1)
                            )
                        else:
                            synth_dyn = attn_metadata.seq_lens[i : i + 1].repeat(
                                q_len
                            )
                        if causal and self.sliding_window is not None:
                            # llm-scaler v41: block-aligned window
                            # rebase for the synthetic-decode
                            # fallback (no per-row head cut in the
                            # single-query kernel: up to
                            # block_size-1 stale tokens beyond the
                            # window start may leak in; debug-env
                            # path only — the default MQ path above
                            # is exact).
                            if not getattr(
                                self, "_sw_synth_logged", False
                            ):
                                self._sw_synth_logged = True
                                logger.info(
                                    "TurboQuant sliding-window"
                                    " synthetic decode (v41):"
                                    " block-aligned rebase,"
                                    " <= block_size-1 stale head"
                                    " tokens (q_len=%d, w=%d).",
                                    q_len,
                                    self.sliding_window,
                                )
                            _bs = kv_cache.shape[1]
                            _ws = synth_dyn[:1] - self.sliding_window
                            _off = torch.clamp_min(_ws, 0) // _bs
                            _idx = (
                                torch.arange(
                                    synth_bt_row.shape[1],
                                    device=synth_bt_row.device,
                                    dtype=torch.int64,
                                )
                                + _off.to(torch.int64)
                            ).clamp_max(
                                synth_bt_row.shape[1] - 1
                            ).unsqueeze(0)
                            synth_bt_row = torch.gather(
                                synth_bt_row, 1, _idx
                            )
                            synth_dyn = synth_dyn - _off * _bs
                        synth_bt = synth_bt_row.expand(q_len, -1)
                        out = triton_turboquant_decode_attention(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_dyn,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(self.tq_config.effective_value_quant_bits),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                            max_num_kv_splits=self._mq_kv_splits(
                                getattr(attn_metadata, "max_seq_len", 0)
                            ),
                        )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    out = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Dequants previously cached K/V, concatenates with the current
        chunk's raw K/V, then runs flash_attn with causal masking.
        """
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        mse_bytes = self._mse_bytes
        val_data_bytes = self._val_data_bytes

        # Dequant cached K/V from TQ cache
        # Allocate slightly over to align to block_size for the grid.
        # Reuse cached buffers to avoid per-call allocation (~16MB at 8K).
        alloc_len = math.ceil(cached_len / block_size) * block_size
        buf_shape = (1, Hk, alloc_len, D)
        # Use WorkspaceManager for dequant buffers.
        # Shared across all layers — saves 60× memory at long context.
        # Required for CUDA Graph capture (per-layer growth incompatible with CG).
        try:
            k_buf, v_buf = current_workspace_manager().get_simultaneous(
                (buf_shape, torch.float16),
                (buf_shape, torch.float16),
            )
        except AssertionError:
            # The workspace was locked (at capture time) before any TQ
            # attention ran, so it was never sized for these buffers. In
            # PIECEWISE graph mode, dummy runs skip attention entirely and
            # attention executes eagerly outside the captured pieces, so a
            # grow-only buffer shared across layers is safe (it is never
            # baked into a graph and layers execute sequentially).
            dev = device.index if device.index is not None else 0
            entry = _DEQUANT_BUFS.get(dev)
            if (
                entry is None
                or entry[0].shape[2] < alloc_len
                or entry[0].shape[1] != Hk
            ):
                entry = (
                    torch.empty(buf_shape, dtype=torch.float16, device=device),
                    torch.empty(buf_shape, dtype=torch.float16, device=device),
                )
                _DEQUANT_BUFS[dev] = entry
            k_buf, v_buf = entry
        # Skip .zero_() — kernel writes all positions up to cached_len,
        # and we only read [:cached_len] afterwards.
        k_cached = k_buf[:, :, :alloc_len, :]
        v_cached = v_buf[:, :, :alloc_len, :]

        grid = (alloc_len, 1 * Hk)
        _tq_full_dequant_kv[grid](
            kv_cache,
            block_table,
            centroids,
            k_cached,
            v_cached,
            k_cached.stride(0),
            k_cached.stride(1),
            k_cached.stride(2),
            v_cached.stride(0),
            v_cached.stride(1),
            v_cached.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=Hk,
            MSE_BYTES=mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=self.tq_config.key_mse_bits,
            KEY_FP8=1 if self.tq_config.key_fp8 else 0,
            BLOCK_D=BLOCK_D,
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(device.index or 0),
            num_warps=4,
        )

        # Inverse-rotate MSE keys back to original space
        if not self.tq_config.key_fp8:
            # fp16 matmul for rotation (2× less bandwidth, uses fp16 tensor cores)
            Pi_half = layer._tq_Pi_half
            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)
            k_flat = k_flat @ Pi_half
            k_cached_trim = k_flat.reshape(Hk, cached_len, D).transpose(
                0, 1
            )  # (cached_len, Hk, D) — already fp16
        else:
            k_cached_trim = k_cached[0, :, :cached_len, :].transpose(
                0, 1
            )  # (cached_len, Hk, D)

        # Skip .contiguous() — the copy into k_full/v_full handles layout
        v_cached_trim = v_cached[0, :, :cached_len, :].transpose(0, 1)

        # Concatenate cached + current chunk K/V (match query dtype)
        # Pre-allocate full K/V buffer, copy into slices (no cat alloc)
        qdtype = query.dtype
        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk

        # Attention: q_len queries attending to seq_len K/V with causal mask
        if _HAS_FLASH_ATTN:
            # Reuse pre-allocated cu_seqlens (avoid host→device transfer)
            if not hasattr(self, "_cu_2_q"):
                self._cu_2_q = torch.zeros(2, device=device, dtype=torch.int32)
                self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
            # Assigning to slice uses fill_ which avoids cpu/gpu sync.
            # v46: self-heal element 0 of both cu buffers (see first-chunk
            # path above — same reasoning).
            self._cu_2_q[0:1] = 0
            self._cu_2_q[1:2] = q_len
            self._cu_2_k[0:1] = 0
            self._cu_2_k[1:2] = seq_len
            cu_seqlens_q = self._cu_2_q
            cu_seqlens_k = self._cu_2_k
            return self._flash_attn_varlen(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )
        else:
            # SDPA fallback: expand KV for GQA, build causal mask
            q_t = query.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)
            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            # Build causal mask: query position p can attend to K position j
            # where j <= cached_len + p (p is 0-indexed within chunk)
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos  # (q_len, seq_len)
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=mask,
                scale=self.scale,
                enable_gqa=(Hk < Hq),
            )  # (1, Hq, q_len, D)
            return out[0].transpose(0, 1)  # (q_len, Hq, D)

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # Acquire shared decode scratch buffers from WorkspaceManager.
        # Layers execute sequentially so one set of buffers is sufficient.
        # Falls back to kernel-internal allocation if workspace unavailable.
        B = query.shape[0]
        D = self.head_size
        # max_seq_len is a host-side int on the metadata (no D2H sync).
        # Attention runs eagerly (splitting op) under PIECEWISE XPU
        # graphs, so a per-batch split count is graph-safe.
        S = self._effective_kv_splits(getattr(attn_metadata, "max_seq_len", 0))
        Hq = self.num_heads
        mid_o_buf = output_buf = lse_buf = None
        if is_workspace_manager_initialized():
            try:
                # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.
                mid_o_buf, output_buf, lse_buf = (
                    current_workspace_manager().get_simultaneous(
                        ((B, Hq, S, D + 1), torch.float32),
                        ((B, Hq, D), query.dtype),
                        ((B, Hq), torch.float32),
                    )
                )
            except AssertionError:
                # Workspace was locked (at graph capture time) before any TQ
                # decode ran, so it was never sized for these buffers. In
                # PIECEWISE graph mode attention executes eagerly outside the
                # captured pieces, so falling back to kernel-internal
                # allocation (per-call torch.empty via the caching
                # allocator) is graph-safe.
                mid_o_buf = output_buf = lse_buf = None

        result = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=layer,
            max_num_kv_splits=S,
        )
        return result
