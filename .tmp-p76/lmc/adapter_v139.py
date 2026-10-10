# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
import os
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from lmcache import utils
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import (
    validate_and_set_config_value as _validate_and_set_config_value,
)
from lmcache.v1.metadata import LMCacheMetadata as LMCacheEngineMetadata
# lmcache 0.5.5 does not re-export the concrete vLLM connectors at package
# level (unreleased main does); import them from their defining submodule.
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lookup_client import LookupClientFactory
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupServer,
)
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer

try:
    from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher
except ImportError:
    # Backwards compatibility for lmcache <= 0.3.10-post1
    from lmcache.v1.plugin.plugin_launcher import (
        PluginLauncher as RuntimePluginLauncher,
    )

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.platforms import current_platform
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    lmcache_get_or_create_config,
    mla_enabled,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tp_group
from vllm.sampling_params import SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_kv_cache_torch_dtype
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.version import __version__ as VLLM_VERSION

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> dict | None:
    request_configs = None
    if (
        sampling_params.extra_args is not None
        and "kv_transfer_params" in sampling_params.extra_args
    ):
        kv_transfer_params = sampling_params.extra_args.get("kv_transfer_params")
        if kv_transfer_params is None:
            return None
        assert isinstance(kv_transfer_params, dict)
        for k, v in kv_transfer_params.items():
            if k.startswith("lmcache."):
                if request_configs is None:
                    request_configs = {}
                request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far, per kv-cache group.
    # NOTE: allocated blocks could be more than the number of tokens.
    # Index 0 is the full-attention group (single-group models only have it).
    allocated_block_ids: list[list[int]]

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: DisaggSpec | None = None

    # Multimodal hashes and positions
    mm_hashes: list[str] | None = None
    mm_positions: list["PlaceholderRange"] | None = None

    # The configs of the request, includes tags and other configs
    request_configs: dict | None = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # tuple[list[int], ...] (one list per kv-cache group, hybrid HMA).
        # Keep every group so hybrid connectors can address the mamba pool.
        if not isinstance(new_request.block_ids[0], list):
            block_ids_per_group = [new_request.block_ids.copy()]
        else:
            block_ids_per_group = [list(g) for g in new_request.block_ids]

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        if new_request.sampling_params:
            request_configs = extract_request_configs(new_request.sampling_params)
        else:
            request_configs = None

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        assert new_request.prompt_token_ids is not None
        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=block_ids_per_group,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: tuple[list[int], ...] | None | list[int],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """

        self.token_ids.extend(new_token_ids)

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db
            per_group_new: list[list[int]] = [[] for _ in self.allocated_block_ids]
        elif isinstance(new_block_ids, tuple):
            # Per-kv-cache-group new blocks (hybrid HMA path).
            per_group_new = [list(g) if g is not None else [] for g in new_block_ids]
            if len(per_group_new) < len(self.allocated_block_ids):
                per_group_new += [
                    [] for _ in range(len(self.allocated_block_ids) - len(per_group_new))
                ]
        elif isinstance(new_block_ids, list) and new_block_ids and isinstance(
            new_block_ids[0], list
        ):
            per_group_new = [list(g) for g in new_block_ids]
        elif isinstance(new_block_ids, list):
            per_group_new = [new_block_ids]
        elif len(new_block_ids) == 0:
            per_group_new = [[] for _ in self.allocated_block_ids]
        else:
            raise ValueError(
                f"Unsupported new_block_ids type {type(new_block_ids)}: "
                f"should be None[list[int], ...], tuple or list[int]."
            )
        for g_idx, group_new in enumerate(per_group_new):
            if g_idx < len(self.allocated_block_ids):
                self.allocated_block_ids[g_idx].extend(group_new)
            else:
                self.allocated_block_ids.append(list(group_new))

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor
    # Slot mapping for the recurrent-state (mamba) group on hybrid models.
    # Synthesized as page_id * tokens_per_page + offset so that
    # slot // tokens_per_page == page_id into the mamba pool.
    mamba_slot_mapping: torch.Tensor | None = None

    # llm-scaler LMC fix (2026-10-09): mamba page addressing is per-GROUP.
    # The recurrent pool is one storage with globally-interleaved page ids;
    # each mamba group's block table holds DIFFERENT page ids for the same
    # token chunk, and a single token-indexed slot mapping can only address
    # one group's pages. The per-chunk page ids for every mamba group travel
    # here and MambaPageConnector fans each chunk out to all groups.
    mamba_chunk_pages: list[list[int]] | None = None

    # Running-state (tail) page id per group for load-bearing requests: the
    # GDN kernel reads its initial state from the tail block, which
    # preprocess_mamba's PRE-COPY fills from the boundary page BEFORE
    # start_load_kv writes that boundary page. The post-retrieve
    # boundary->tail page mirror needs these ids.
    mamba_tail_pages: list[int] | None = None

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: SaveSpec | None = None
    # load_spec
    load_spec: LoadSpec | None = None
    # disagg spec
    disagg_spec: DisaggSpec | None = None
    # the configs of the request
    request_configs: dict | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: LoadSpec | None = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
        mamba_group_idx: int | None = None,
        mamba_block_size: int | None = None,
        attn_group_idx: int = 0,
        mamba_group_idxs: list[int] | None = None,
        mamba_tail_idx: int | None = None,
    ) -> "ReqMeta | None":
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len == tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a
        # trqansfer. Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        num_tokens_to_save = (
            (input_token_len // lmcache_chunk_size * lmcache_chunk_size)
            if not is_last_prefill or discard_partial_chunks
            else input_token_len
        )

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = input_token_ids[:num_tokens_to_save]

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            token_ids_tensor = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids_tensor, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids_tensor.tolist()

        # TEMP (LMCDbgGroups): ground truth for the group-selection bug —
        # mamba_gidxs=[0,1,2] on this fork means allocated_block_ids[0] is a
        # MAMBA table, not the attn table the sm below assumes. Log every
        # group's table (capped) to pin each request's real layout. Remove
        # after diagnosis.
        if os.environ.get("LMC_DBG_GROUPS") == "1":
            _gs = tracker.allocated_block_ids or []
            logger.info(
                "LMCDbgGroups %s n_groups=%d ntok=%d tables=%s",
                tracker.req_id,
                len(_gs),
                len(token_ids),
                [list(map(int, _g))[:48] for _g in _gs],
            )

        # llm-scaler LMC fix (2026-10-09): index the attn group's table —
        # allocated_block_ids[0] is a MAMBA table on this fork's hybrid
        # layout (mamba groups 0-2, attn group 3), which addressed the
        # store/load slots at mamba-page ids inside the attn pool.
        attn_block_ids = (
            tracker.allocated_block_ids[attn_group_idx]
            if tracker.allocated_block_ids
            else []
        )
        num_blocks = len(attn_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!"
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(attn_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long

        # Recurrent-state (mamba) group slot mapping for hybrid models:
        # page-granular addressing synthesized as page_id * page_tokens + off.
        mamba_slot_mapping = None
        if (
            mamba_group_idx is not None
            and mamba_block_size is not None
            and mamba_group_idx < len(tracker.allocated_block_ids)
        ):
            mamba_ids = tracker.allocated_block_ids[mamba_group_idx]
            m_blocks = len(mamba_ids)
            m_block_ids = torch.tensor(mamba_ids, dtype=torch.long)
            m_offsets = torch.arange(0, mamba_block_size, dtype=torch.long)
            mamba_slot_mapping = (
                m_offsets.reshape((1, mamba_block_size))
                + m_block_ids.reshape((m_blocks, 1)) * mamba_block_size
            ).flatten()[: len(token_ids)]

        # llm-scaler LMC fix (2026-10-09): per-group mamba page ids per chunk
        # (see ReqMeta.mamba_chunk_pages) and, for load-bearing requests, the
        # running-state tail pages. mamba_tail_idx mirrors preprocess_mamba's
        # curr_state_idx formula — cdiv(computed + scheduled, block) - 1; the
        # num_speculative_blocks term cancels, so the caller passes the
        # end-of-step token count directly.
        mamba_chunk_pages = None
        mamba_tail_pages = None
        if mamba_group_idxs and mamba_block_size is not None:
            tables = tracker.allocated_block_ids
            if tables:
                n_chunks = len(token_ids) // mamba_block_size
                mamba_chunk_pages = [
                    [int(tables[g][c]) for g in mamba_group_idxs]
                    for c in range(n_chunks)
                ]
                if mamba_tail_idx is not None:
                    min_len = min(len(tables[g]) for g in mamba_group_idxs)
                    if mamba_tail_idx < min_len:
                        mamba_tail_pages = [
                            int(tables[g][mamba_tail_idx])
                            for g in mamba_group_idxs
                        ]
                    else:
                        logger.error(
                            "mamba_tail_idx=%d beyond table len=%d for %s; "
                            "skipping tail mirror",
                            mamba_tail_idx,
                            min_len,
                            tracker.req_id,
                        )

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.lmcache_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            mamba_slot_mapping=mamba_slot_mapping,
            mamba_chunk_pages=mamba_chunk_pages,
            mamba_tail_pages=mamba_tail_pages,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
        )


def need_gpu_interim_buffer(lmcache_config: LMCacheEngineConfig):
    return not lmcache_config.enable_pd


def _calculate_mtp_layers(vllm_config, model_config):
    num_mtp_layers = 0
    if vllm_config is not None and vllm_config.speculative_config is not None:
        logger.info(
            "vllm_config.speculative_config: %s", vllm_config.speculative_config
        )
        # TODO(baoloongmao): Support other MTP methods
        if vllm_config.speculative_config.method == "deepseek_mtp":
            num_mtp_layers = getattr(
                model_config.hf_config, "num_nextn_predict_layers", 0
            )

        elif vllm_config.speculative_config.use_eagle():
            try:
                draft_model_config = vllm_config.speculative_config.draft_model_config
                num_mtp_layers = draft_model_config.get_num_layers(
                    vllm_config.parallel_config
                )
                logger.info("EAGLE detected %d extra layer(s)", num_mtp_layers)
            except Exception:
                logger.info(
                    "EAGLE detected, but failed to get the number of extra layers"
                    "falling back to 1"
                )
                num_mtp_layers = 1
    return num_mtp_layers


def _get_rpc_identity(vllm_config: "VllmConfig") -> tuple[str | None, dict | None]:
    """Extract the (engine_id, kv_connector_extra_config) pair used to build
    matching zmq lookup socket paths on the scheduler and every TP worker.
    ``engine_id`` set in ``--kv-transfer-config`` keeps all roles on the same
    socket; when unset, both sides deterministically use None.
    """
    ktc = getattr(vllm_config, "kv_transfer_config", None)
    if ktc is None:
        return None, None
    engine_id = getattr(ktc, "engine_id", None)
    extra = getattr(ktc, "kv_connector_extra_config", None)
    return engine_id, extra


def _init_lmcache_engine(
    lmcache_config: LMCacheEngineConfig,
    vllm_config: "VllmConfig",
    num_layers_override: int | None = None,
) -> LMCacheEngine:
    """Initialize the LMCache engine by the given model config and parallel
    config. This function will check the environment variable
    `LMCACHE_CONFIG_FILE` to load the configuration file. If that environment
    variable is not set, this function will return None.

    :param lmcache_config: The LMCache configuration.
    :type lmcache_config: LMCacheEngineConfig
    :param vllm_config: The vLLM configuration.
    :type vllm_config: VllmConfig
    :param num_layers_override: On hybrid models, the number of layers in
        the full-attention kv-cache groups (the engine only addresses the
        attention tensors, not the recurrent-state pool).

    :return: The initialized LMCache engine
    :rtype: LMCacheEngine
    """
    if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
        return curr_engine

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    assert isinstance(lmcache_config, LMCacheEngineConfig), (
        "LMCache v1 configuration is should be passed."
    )

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)

    use_mla = mla_enabled(model_config)
    if use_mla and (
        lmcache_config.remote_serde != "naive"
        and lmcache_config.remote_serde is not None
    ):
        raise ValueError("MLA only works with naive serde mode..")

    # construct kv shape (for mem pool)
    num_mtp_layers = _calculate_mtp_layers(vllm_config, model_config)
    if num_layers_override is not None:
        # Hybrid: only the full-attention group's layers flow through this
        # engine; any draft/eagle attention layers are already included in
        # the group's layer_names by the kv cache config.
        num_layer = num_layers_override
    else:
        num_layer = model_config.get_num_layers(parallel_config) + num_mtp_layers
    chunk_size = lmcache_config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)
    logger.info(
        "use mla: %s, kv shape: %s, num_mtp_layers: %s",
        use_mla,
        kv_shape,
        num_mtp_layers,
    )

    # Change current device.
    num_gpus = torch.accelerator.device_count()
    local_rank = parallel_config.rank % num_gpus
    torch.accelerator.set_device_index(local_rank)
    device = torch.device(f"{current_platform.device_type}:{local_rank}")
    rpc_engine_id, kv_connector_extra = _get_rpc_identity(vllm_config)
    metadata = LMCacheEngineMetadata(
        model_name=model_config.model,
        world_size=parallel_config.world_size,
        local_world_size=parallel_config.world_size,
        worker_id=parallel_config.rank,
        local_worker_id=parallel_config.rank,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=use_mla,
        role="vllm",
        served_model_name=model_config.served_model_name,
        chunk_size=chunk_size,
        engine_id=rpc_engine_id,
        kv_connector_extra_config=kv_connector_extra,
    )

    use_gpu = need_gpu_interim_buffer(lmcache_config)
    vllm_gpu_connector: (
        VLLMBufferLayerwiseGPUConnector
        | VLLMPagedMemGPUConnectorV2
        | VLLMPagedMemLayerwiseGPUConnector
    )

    if use_mla and lmcache_config.use_layerwise:
        raise ValueError("layerwise MLA connector is not supported yet")

    # When use_mla is True, num_kv_head is 1
    hidden_dim_size = num_kv_head * head_size
    if lmcache_config.use_layerwise:
        if lmcache_config.enable_blending:
            # Use layerwise connector for blending
            vllm_gpu_connector = VLLMBufferLayerwiseGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        else:
            vllm_gpu_connector = VLLMPagedMemLayerwiseGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
    else:
        if current_platform.is_xpu():
            # XPU-native paged connector (torch.xpu streams, no cuda asserts)
            from lmcache.v1.gpu_connector.xpu_connectors import (
                VLLMPagedMemXPUConnectorV2,
            )

            vllm_gpu_connector = VLLMPagedMemXPUConnectorV2(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
                use_mla=use_mla,
            )
        else:
            vllm_gpu_connector = VLLMPagedMemGPUConnectorV2(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
                use_mla=use_mla,
            )
    tpg = get_tp_group()
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME,
        lmcache_config,
        metadata,
        vllm_gpu_connector,
        tpg.broadcast,
        tpg.broadcast_object,
    )

    return engine


class MambaPageConnector(GPUConnectorInterface):
    """Byte-opaque, page-granular GPU connector for recurrent (mamba/GDN)
    kv-cache groups on hybrid models.

    In mamba "align" cache mode the recurrent-state pool allocates exactly
    one page per full-attention block, so a chunk of `tokens_per_page`
    prompt tokens maps 1:1 onto one pool page. Recurrent state is only
    valid on page boundaries, hence `chunk_size == tokens_per_page` is
    enforced by the adapter. The connector receives a single
    ``[num_pages, page_size_bytes]`` uint8 view of the mamba pool and copies
    whole pages as raw bytes; the synthesized slot mapping is
    ``page_id * tokens_per_page + offset`` so that
    ``slot // tokens_per_page == page_id``.

    Pure-torch implementation (index_copy_/index_select) so it runs on any
    accelerator device, including XPU.
    """

    def __init__(
        self,
        page_size_bytes: int,
        tokens_per_page: int,
        chunk_size: int,
        num_layers: int = 1,
        num_groups: int = 1,
    ):
        assert chunk_size == tokens_per_page, (
            "MambaPageConnector requires chunk_size == tokens_per_page "
            f"({chunk_size} != {tokens_per_page}); recurrent state is only "
            "valid on page boundaries."
        )
        self.page_size_bytes = page_size_bytes
        self.tokens_per_page = tokens_per_page
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        # llm-scaler LMC fix (2026-10-09): number of mamba kv-cache groups
        # sharing the pool storage. Each group owns a DIFFERENT page id for
        # the same token chunk, so one MemoryObj carries num_groups pages
        # and _page_ids must return num_groups ids per chunk.
        self.num_groups = num_groups
        self.kvcaches: list[torch.Tensor] | None = None
        self.device: torch.device | None = None
        # Per-chunk page table installed by the adapter around each
        # store/retrieve: {chunk_start_token: [page_id per group]}. Keyed by
        # the engine's token-space chunk start, which is the same (start,
        # end) handed to to_gpu/from_gpu. Wins over the slot-mapping
        # derivation, which can only address one group's pages.
        self._lmc_chunk_pages: dict[int, list[int]] | None = None

    def get_shape(self, num_tokens: int) -> torch.Size:
        # MemoryObj layout: [1, 1, num_tokens, bytes_per_token] uint8 so a
        # full chunk flattens to exactly num_layers pool pages (one per
        # mamba layer across all groups). v136: boundary objects are
        # registry/seed BYTES, not pool reads, but the sizing law is the
        # same: one page per layer.
        return torch.Size(
            (
                1,
                1,
                num_tokens,
                self.page_size_bytes * self.num_layers // self.tokens_per_page,
            )
        )

    def initialize_kvcaches_ptr(self, **kwargs) -> None:
        if "kvcaches" in kwargs:
            kvs = kwargs["kvcaches"]
            # kvcaches may arrive per-layer (shared storage); dedup to the
            # single pool view.
            if len(kvs) > 1:
                seen: dict[int, torch.Tensor] = {}
                for t in kvs:
                    if isinstance(t, list):
                        t = t[0]
                    seen.setdefault(t.untyped_storage().data_ptr(), t)
                kvs = list(seen.values())
            self.kvcaches = kvs
            self.device = kvs[0].device

    def _page_ids(
        self, slot_mapping: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        # llm-scaler LMC fix (2026-10-09): the token-indexed slot mapping can
        # only carry one page per chunk, but every mamba group owns a
        # different page for the same chunk. When the adapter installed the
        # per-chunk page table (keyed by the engine's token-space chunk
        # start), it wins; the slot-mapping derivation stays as fallback.
        pages = (
            self._lmc_chunk_pages.get(int(start))
            if self._lmc_chunk_pages
            else None
        )
        if pages is not None:
            return torch.tensor(
                pages, dtype=torch.long, device=self.kvcaches[0].device
            )
        tpp = self.tokens_per_page
        # start/end are chunk-aligned by the engine (chunk == tpp), so every
        # tpp-th slot identifies one page.
        return (slot_mapping[start:end:1][::tpp] // tpp).to(torch.long).to(
            self.kvcaches[0].device
        )

    def to_gpu(self, memory_obj: "MemoryObj", start: int, end: int, **kwargs):
        """CPU MemoryObj -> mamba pool pages."""
        slot_mapping = kwargs["slot_mapping"]
        ids = self._page_ids(slot_mapping, start, end)
        pool = self.kvcaches[0]  # [num_pages, page_size_bytes] uint8
        src = (
            memory_obj.tensor.view(torch.uint8)
            .reshape(-1, self.page_size_bytes)
            .to(pool.device, non_blocking=False)
        )
        assert src.shape[0] == ids.shape[0], (
            f"Mamba page count mismatch: {src.shape[0]} objs vs {ids.shape[0]} pages"
        )
        pool.index_copy_(0, ids, src)

    def from_gpu(self, memory_obj: "MemoryObj", start: int, end: int, **kwargs):
        """Mamba pool pages -> CPU MemoryObj."""
        slot_mapping = kwargs["slot_mapping"]
        ids = self._page_ids(slot_mapping, start, end)
        pool = self.kvcaches[0]
        gathered = pool.index_select(0, ids)
        memory_obj.tensor.view(torch.uint8).reshape(
            -1, self.page_size_bytes
        ).copy_(gathered.to(memory_obj.tensor.device, non_blocking=False))

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        # llm-scaler v136: store path. `lmc_seed_bytes` carries the frozen
        # CPU boundary snapshots (one flat uint8 tensor per chunk, layout
        # group-major / layer / conv-ssm) taken by mamba_utils at the copy
        # that materialized each boundary. Align-mode nulls pool boundary
        # pages, so the object is filled from these bytes instead of from
        # the pool -- immune to page-freeing races by construction.
        seeds = kwargs.pop("lmc_seed_bytes", None)
        for i, (memory_obj, start, end) in enumerate(
            zip(memory_objs, starts, ends, strict=False)
        ):
            if seeds is not None:
                dst = memory_obj.tensor.view(torch.uint8).reshape(-1)
                src = seeds[i].to(dst.device, non_blocking=False)
                if src.numel() != dst.numel():
                    raise RuntimeError(
                        "Mamba store seed size mismatch: "
                        f"{src.numel()} vs obj {dst.numel()}"
                    )
                dst.copy_(src)
            else:
                self.from_gpu(memory_obj, start, end, **kwargs)

    def batched_to_gpu(self, memory_objs, starts=None, ends=None, **kwargs):
        # llm-scaler v136: load path. `lmc_seed` is a holder dict the
        # adapter reads after retrieve(); the deepest hit chunk's object
        # (the last one, chunk order is prefix order) is parked as CPU
        # bytes for preprocess_mamba's seed write. No pool pages are
        # touched.
        holder = kwargs.pop("lmc_seed", None)
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            if holder is not None:
                holder["bytes"] = (
                    memory_obj.tensor.view(torch.uint8).reshape(-1).clone()
                )
                holder["tokens"] = int(end)
            else:
                self.to_gpu(memory_obj, start, end, **kwargs)


def _init_mamba_engine(
    lmcache_config: LMCacheEngineConfig,
    vllm_config: "VllmConfig",
    page_size_bytes: int,
    tokens_per_page: int,
    num_groups: int = 1,
    num_layers: int = 1,
) -> LMCacheEngine:
    """Initialize the LMCache engine backing the recurrent-state (mamba) kv
    cache group of a hybrid model.

    Uses a dedicated engine name so the storage pool is separate from the
    attention engine. The CPU pool is sized down from the attention engine's
    pool by the page/chunk byte ratio (with a floor), because the recurrent
    state is far smaller per token than the attention KV: this keeps the
    mamba horizon >= the attention horizon so joint prefix hits are always
    bounded by the attention engine (the lookup authority).

    llm-scaler v136: one stored chunk = the boundary state at that chunk's
    END -- one page per mamba LAYER across ALL groups (num_layers total),
    snapshotted by mamba_utils._lmc_snap_boundary. Object sizing therefore
    follows the total layer count, not the group count.
    """
    engine_name = f"{ENGINE_NAME}-mamba"
    if curr_engine := LMCacheEngineBuilder.get(engine_name):
        return curr_engine

    import copy as _copy

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    mamba_config = _copy.deepcopy(lmcache_config)
    # No disk tier for the recurrent-state engine in v1; CPU only.
    mamba_config.local_disk = None
    mamba_config.max_local_disk_size = 0
    # v136: the mamba lookup pair (this engine's server + the scheduler's
    # client) runs the SYNC zmq flavor; async loading never applies to the
    # seed-bytes path and the two flavors must match on both ends.
    mamba_config.enable_async_loading = False

    attn_chunk_bytes = None  # reserved for explicit ratio overrides
    ratio = mamba_config.get_extra_config_value("mamba_pool_ratio", None)
    if ratio is None:
        # default: page is ~1MB vs ~16MB attn chunk -> mamba pool 1/8 of the
        # attention pool keeps a >=2x longer horizon.
        ratio = 8
    if mamba_config.max_local_cpu_size > 0:
        mamba_mb = max(
            1, int(mamba_config.max_local_cpu_size / max(1, int(ratio)))
        )
        logger.info(
            "Mamba engine CPU pool scaled %d GB -> %d GB (ratio=%s, "
            "page_bytes=%d, tokens_per_page=%d layers=%d)",
            mamba_config.max_local_cpu_size,
            mamba_mb,
            ratio,
            page_size_bytes,
            tokens_per_page,
            num_layers,
        )
        mamba_config.max_local_cpu_size = mamba_mb

    num_gpus = torch.accelerator.device_count()
    local_rank = parallel_config.rank % num_gpus
    torch.accelerator.set_device_index(local_rank)

    chunk_size = tokens_per_page
    # llm-scaler v136: one stored chunk = one page per mamba LAYER (all
    # groups), so the MemoryObj must hold num_layers pages of bytes.
    kv_shape = (
        1,
        1,
        chunk_size,
        max(1, page_size_bytes * num_layers // chunk_size),
        1,
    )
    rpc_engine_id, kv_connector_extra = _get_rpc_identity(vllm_config)
    metadata = LMCacheEngineMetadata(
        model_name=model_config.model,
        world_size=parallel_config.world_size,
        local_world_size=parallel_config.world_size,
        worker_id=parallel_config.rank,
        local_worker_id=parallel_config.rank,
        kv_dtype=torch.uint8,
        kv_shape=kv_shape,
        use_mla=False,
        role="vllm",
        served_model_name=model_config.served_model_name,
        chunk_size=chunk_size,
        engine_id=(
            f"{rpc_engine_id}-mamba" if rpc_engine_id is not None else None
        ),
        kv_connector_extra_config=kv_connector_extra,
    )
    connector = MambaPageConnector(
        page_size_bytes,
        tokens_per_page,
        chunk_size,
        num_layers=num_layers,
        num_groups=num_groups,
    )
    tpg = get_tp_group()
    engine = LMCacheEngineBuilder.get_or_create(
        engine_name,
        mamba_config,
        metadata,
        connector,
        tpg.broadcast,
        tpg.broadcast_object,
    )
    return engine


def _lmc_dbg_md5_u8(t: "torch.Tensor") -> str:
    """TEMP (LMCDbgBytes): md5[:8] of a tensor's raw bytes (uint8 view)."""
    import hashlib

    return hashlib.md5(
        t.detach().view(torch.uint8).reshape(-1).cpu().numpy().tobytes()
    ).hexdigest()[:8]


def _lmc_dbg_patch_connectors() -> None:
    """TEMP (LMCDbgBytes): wrap attn (V2/V3 XPU) and mamba connector
    to_gpu/from_gpu to answer content-vs-placement for the load no-op
    (2026-10-09: retrieved span's pool pages stay all-zero across retrieve
    while the store-side poll proves the source pages held real KV).

    Logs, per transfer: span, engine format params, the memory_obj layer-1
    byte digest (what the cache actually holds) and the target pool page
    digests before/after (where/if the write lands).
    Remove after diagnosis."""
    if os.environ.get("LMC_DBG_TOGPU") != "1":
        return

    def _wrap(cls, name, pool_pages_fn, obj_slice_fn):
        orig = getattr(cls, name)

        def wrapped(self, memory_obj, start, end, **kwargs):
            gather = None
            sm = kwargs.get("slot_mapping")
            kv = getattr(self, "kvcaches", None) or []
            if sm is not None and len(kv) > 0:
                # Fresh gather per digest: the page rows are snapshotted at
                # read time, so pre/post must re-run the gather.
                gather = lambda: pool_pages_fn(self, kv, sm, start, end)  # noqa: E731
            try:
                pre = ""
                if gather is not None:
                    g = gather()
                    if g is not None:
                        pre = " pre=" + _lmc_dbg_md5_u8(g)
                objinfo = ""
                obj = memory_obj.tensor
                if obj is not None:
                    sl = obj_slice_fn(obj)
                    objinfo = (
                        f" obj={tuple(obj.shape)}/{obj.dtype}"
                        f"/{obj.device.type} md5={_lmc_dbg_md5_u8(sl)}"
                    )
                logger.info(
                    "LMCDbgBytes %s(%d,%d) fmt=%s bs=%s pbs=%s vc=%s%s%s",
                    name,
                    start,
                    end,
                    getattr(self, "engine_kv_format", "?"),
                    getattr(self, "block_size", "?"),
                    getattr(self, "page_buffer_size", "?"),
                    kwargs.get("vllm_cached_tokens", 0),
                    objinfo,
                    pre,
                )
            except Exception:
                logger.exception("LMCDbgBytes %s wrap pre-log failed", name)
            r = orig(self, memory_obj, start, end, **kwargs)
            try:
                objpost = ""
                obj = getattr(memory_obj, "tensor", None)
                if obj is not None:
                    objpost = " objpost=" + _lmc_dbg_md5_u8(obj_slice_fn(obj))
                if gather is not None:
                    g = gather()
                    if g is not None:
                        torch.xpu.synchronize()
                        logger.info(
                            "LMCDbgBytes %s(%d,%d) post=%s%s",
                            name,
                            start,
                            end,
                            _lmc_dbg_md5_u8(g),
                            objpost,
                        )
                elif objpost:
                    logger.info(
                        "LMCDbgBytes %s(%d,%d)%s",
                        name,
                        start,
                        end,
                        objpost,
                    )
            except Exception:
                logger.exception("LMCDbgBytes %s wrap post-log failed", name)
            return r

        setattr(cls, name, wrapped)

    def _attn_pages(self, kv, sm, start, end):
        # Same page math as the pool: slot // tokens_per_page, page-slice of
        # the layer-1 tensor. The connector's kvcaches may be the registered
        # K/V-major views (2, NB, tpp, ...) OR, after the first transfer
        # normalized them, contiguous (NB, 2, tpp, ...) tensors — slice the
        # matching axis either way (both orders expose the same page bytes:
        # K plane then V plane).
        t = kv[1] if len(kv) > 1 else kv[0]
        s = sm[start:end].to(t.device)
        s = s[s >= 0]
        if len(s) == 0:
            return None
        tpp = kv[0].shape[2]
        kv_first = t.dim() == 5 and t.shape[0] == 2
        ps = sorted(set((s // tpp).cpu().tolist()))
        torch.xpu.synchronize()
        if len(ps) == 1:
            rows = t[:, int(ps[0])] if kv_first else t[int(ps[0])]
            return rows.reshape(-1)
        return torch.cat(
            [
                (t[:, int(p)] if kv_first else t[int(p)]).reshape(-1)
                for p in ps
            ]
        )

    def _mamba_pages(self, kv, sm, start, end):
        # Mirror MambaPageConnector._page_ids, then keep the gathered rows
        # on device so the post-call digest reads the same physical pages.
        tpp = self.tokens_per_page
        ids = (sm[start:end:1][::tpp] // tpp).to(torch.long).to(kv[0].device)
        if len(ids) == 0:
            return None
        torch.xpu.synchronize()
        return kv[0].index_select(0, ids).reshape(-1)

    _wrap(MambaPageConnector, "to_gpu", _mamba_pages, lambda o: o)
    _wrap(MambaPageConnector, "from_gpu", _mamba_pages, lambda o: o)
    try:
        from lmcache.v1.gpu_connector import xpu_connectors as _xc

        for _cls in (_xc.VLLMPagedMemXPUConnectorV2, _xc.VLLMPagedMemXPUConnectorV3):
            _wrap(
                _cls,
                "to_gpu",
                _attn_pages,
                lambda o: (o[:, 1] if o.dim() == 4 else o),
            )
            _wrap(
                _cls,
                "from_gpu",
                _attn_pages,
                lambda o: (o[:, 1] if o.dim() == 4 else o),
            )
        logger.info("LMCDbgBytes to_gpu/from_gpu patches installed (V2+V3+mamba)")
    except Exception:
        logger.exception("LMCDbgBytes XPU connector patch failed")


_lmc_dbg_patch_connectors()

# TEMP (LMCDbgZero): worker-side adapter reference for the KVBlockZeroer
# wrap below (needs the live attn tensors to hash the vllm-prefix pages).
# Remove after diagnosis.
_LMC_DBG_ADAPTER_REF = None


def _lmc_dbg_patch_zeroer() -> None:
    """TEMP (LMCDbgZero): wrap KVBlockZeroer.zero_block_ids to log the block
    ids handed in, the zeroer's computed page geometry (page_size_el,
    blk_size, n_segs), and attn pages 0/1/22 digests before/after the wipe.
    Ground truth for the connector-only attn wipe: req-2's vllm-prefix page
    0 reads all-zero at load time (b2d1236c) while real at req-1 store time
    (fe92d11b), and the zeroer is the only attn-pool writer in the window.
    Remove after diagnosis."""
    if os.environ.get("LMC_DBG_ZERO") != "1":
        return
    try:
        from vllm.v1.worker.utils import KVBlockZeroer
    except Exception:
        logger.exception("LMCDbgZero: KVBlockZeroer import failed")
        return
    if getattr(KVBlockZeroer, "_lmc_zero_wrapped", False):
        return
    KVBlockZeroer._lmc_zero_wrapped = True
    _orig = KVBlockZeroer.zero_block_ids

    def _lmc_p012() -> str:
        import hashlib

        ad = _LMC_DBG_ADAPTER_REF
        ts = getattr(ad, "_attn_kvcaches", None) if ad is not None else None
        if not ts:
            return "n/a"
        t = ts[1] if len(ts) > 1 else ts[0]
        torch.xpu.synchronize()
        p0 = hashlib.md5(
            t[:, 0].reshape(-1).cpu().numpy().tobytes()
        ).hexdigest()[:8]
        comb = hashlib.md5(
            torch.cat([t[:, int(p)].reshape(-1) for p in (0, 1, 22)])
            .cpu()
            .numpy()
            .tobytes()
        ).hexdigest()[:8]
        return f"p0={p0} c={comb}"

    def _wrapped(self, block_ids):
        try:
            pre = _lmc_p012()
        except Exception:
            pre = "err"
        _orig(self, block_ids)
        try:
            meta = getattr(self, "_meta", None)
            if meta is not None:
                _, pse, bsz, nseg = meta
                minfo = f"pse={pse} blk={bsz} nseg={nseg}"
            else:
                minfo = "meta=None"
            ids_s = str(block_ids[:24])
            if len(block_ids) > 24:
                ids_s += f"...(+{len(block_ids) - 24})"
            logger.info(
                "LMCDbgZero n=%d ids=%s %s pre=%s post=%s",
                len(block_ids),
                ids_s,
                minfo,
                pre,
                _lmc_p012(),
            )
        except Exception:
            logger.exception("LMCDbgZero post-log failed (non-fatal)")

    KVBlockZeroer.zero_block_ids = _wrapped
    logger.info("LMCDbgZero: KVBlockZeroer.zero_block_ids wrapped")


_lmc_dbg_patch_zeroer()


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)
    lookup_requests_in_step: list[str] = field(default_factory=list)
    # llm-scaler v136: mamba boundary-lookup ids ("m-<req_id>") taken this
    # scheduling round. The worker unpins them in wait_for_save after the
    # seed retrieve — the exact mirror of the attn uuid4 flow above, so
    # lookups that never lead to a load cannot leak pins.
    mamba_lookup_ids: list[str] = field(default_factory=list)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        assert vllm_config.kv_transfer_config is not None
        self._parent = parent
        self._vllm_config = vllm_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        # Put the leading with "lmcache." and matched configs from
        # vllm extra_config to the config
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if _validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            "Updated config %s from vLLM extra config: %s",
                            config_key,
                            value,
                        )

        self.config = config

        # ---- Hybrid (full-attention + recurrent/mamba) detection ----
        # HMA gives this connector per-kv-cache-group block ids. The mamba
        # group needs its own byte-opaque LMCache engine; the attention
        # engine stays the lookup authority (joint hits are bounded by it
        # because the mamba pool is sized to a strictly longer horizon).
        # KVConnectorBase_V1 stores the config as _kv_cache_config on this
        # fork; fall back to the public name for upstream parity.
        kv_cache_config = getattr(self._parent, "_kv_cache_config", None) or getattr(
            self._parent, "kv_cache_config", None
        )
        self._fa_group_idx = 0
        self._mamba_group_idx: int | None = None
        self._attn_layer_names: list[str] = []
        self._mamba_layer_names: list[str] = []
        self._mamba_page_size_bytes: int | None = None
        self._mamba_tokens_per_page: int | None = None
        self._mamba_group_ids: list[int] = []
        self._attn_kvcaches: list[torch.Tensor] = []
        self._mamba_kvcaches: list[torch.Tensor] = []
        self.mamba_engine: LMCacheEngine | None = None
        # llm-scaler v136: scheduler-side mamba lookup client (joint clamp)
        # and worker-side mamba lookup server — a second zmq lookup pair
        # under engine_id f"{rpc_engine_id}-mamba".
        self._mamba_lookup_client = None
        self.mamba_lookup_server = None
        if kv_cache_config is not None and getattr(
            kv_cache_config, "has_mamba_layers", False
        ):
            from vllm.v1.kv_cache_interface import (
                FullAttentionSpec,
                MambaSpec,
                SlidingWindowSpec,
            )

            attn_names: list[str] = []
            mamba_names: list[str] = []
            mamba_group_ids: list[int] = []
            fa_group_ids: list[int] = []
            for g_idx, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                if isinstance(spec, MambaSpec):
                    # Uniform-page-size grouping may split the recurrent
                    # layers across several groups with identical specs.
                    # Merge them into one logical mamba engine; all groups
                    # must share the same page geometry.
                    if self._mamba_page_size_bytes is not None and (
                        self._mamba_page_size_bytes != spec.page_size_bytes
                        or self._mamba_tokens_per_page != spec.block_size
                    ):
                        raise ValueError(
                            "Mamba groups with differing page geometry are "
                            "not supported"
                        )
                    mamba_group_ids.append(g_idx)
                    mamba_names.extend(group.layer_names)
                    self._mamba_page_size_bytes = spec.page_size_bytes
                    self._mamba_tokens_per_page = spec.block_size
                elif isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                    attn_names.extend(group.layer_names)
                    fa_group_ids.append(g_idx)
            self._mamba_group_ids = mamba_group_ids
            self._mamba_layer_names = mamba_names
            if mamba_group_ids:
                self._mamba_group_idx = mamba_group_ids[0]
            self._attn_layer_names = attn_names
            # llm-scaler LMC fix (2026-10-09): the full-attention group is
            # NOT necessarily group 0 — this fork's hybrid layout orders the
            # mamba groups first (fa_gidxs=[3], mamba_gidxs=[0,1,2]). The
            # attn slot mapping must index the ATTN group's table; group 0
            # is a mamba table, and slots derived from it point the store
            # and load at mamba-page locations in the attn pool.
            if fa_group_ids:
                self._fa_group_idx = fa_group_ids[0]
            if self._mamba_group_idx is not None:
                # Recurrent state is only valid on page boundaries: the
                # chunk size MUST equal the (aligned) tokens per page.
                if config.chunk_size != self._mamba_tokens_per_page:
                    logger.info(
                        "Hybrid model: forcing lmcache chunk_size %d -> %d "
                        "(mamba tokens_per_page)",
                        config.chunk_size,
                        self._mamba_tokens_per_page,
                    )
                    config.chunk_size = self._mamba_tokens_per_page
                if config.use_layerwise or config.enable_blending:
                    logger.warning(
                        "Hybrid models do not support layerwise/blending "
                        "LMCache paths yet; disabling for correctness."
                    )
                    config.use_layerwise = False
                    config.enable_blending = False
                logger.info(
                    "LMCache hybrid mode: attn layers=%d mamba layers=%d "
                    "(fa_gidx=%d fa_gidxs=%s mamba_gidxs=%s page_bytes=%d "
                    "tpp=%d)",
                    len(self._attn_layer_names),
                    len(self._mamba_layer_names),
                    self._fa_group_idx,
                    fa_group_ids,
                    self._mamba_group_ids,
                    self._mamba_page_size_bytes,
                    self._mamba_tokens_per_page,
                )

        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[Generator[torch.Tensor | None, None, None]] = []
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        if role == KVConnectorRole.SCHEDULER:
            # 0.5.5 factory convention: (config, metadata, engine=None). The
            # scheduler metadata only needs the rpc identity (engine_id) and
            # world topology for the zmq lookup transport; kv_shape is not
            # consumed on this side but is filled in for consistency.
            model_config = vllm_config.model_config
            parallel_config = vllm_config.parallel_config
            kv_dtype = get_kv_cache_torch_dtype(
                vllm_config.cache_config.cache_dtype, model_config.dtype
            )
            if self._mamba_group_idx is not None:
                sched_num_layer = len(self._attn_layer_names)
            else:
                sched_num_layer = model_config.get_num_layers(parallel_config) + _calculate_mtp_layers(vllm_config, model_config)
            rpc_engine_id, kv_connector_extra = _get_rpc_identity(vllm_config)
            scheduler_metadata = LMCacheEngineMetadata(
                model_name=model_config.model,
                world_size=parallel_config.world_size,
                local_world_size=parallel_config.world_size,
                worker_id=parallel_config.rank,
                local_worker_id=parallel_config.rank,
                kv_dtype=kv_dtype,
                kv_shape=(
                    sched_num_layer,
                    2,
                    config.chunk_size,
                    model_config.get_num_kv_heads(parallel_config),
                    model_config.get_head_size(),
                ),
                use_mla=mla_enabled(model_config),
                role="vllm",
                served_model_name=model_config.served_model_name,
                chunk_size=config.chunk_size,
                engine_id=rpc_engine_id,
                kv_connector_extra_config=kv_connector_extra,
            )
            self.lookup_client = LookupClientFactory.create_lookup_client(
                config, scheduler_metadata, None
            )
            # llm-scaler v136: scheduler-side half of the mamba lookup
            # pair. Same zmq topology as the attn pair, addressed by the
            # mamba engine_id; sync flavor + tpp chunking to mirror the
            # worker's mamba engine exactly (chain hashes must match).
            if self._mamba_group_idx is not None:
                import copy as _copy

                mamba_sched_cfg = _copy.deepcopy(config)
                mamba_sched_cfg.enable_async_loading = False
                mamba_sched_cfg.chunk_size = self._mamba_tokens_per_page
                mamba_sched_metadata = LMCacheEngineMetadata(
                    model_name=model_config.model,
                    world_size=parallel_config.world_size,
                    local_world_size=parallel_config.world_size,
                    worker_id=parallel_config.rank,
                    local_worker_id=parallel_config.rank,
                    kv_dtype=torch.uint8,
                    kv_shape=(
                        1,
                        1,
                        self._mamba_tokens_per_page,
                        max(
                            1,
                            self._mamba_page_size_bytes
                            * len(self._mamba_layer_names)
                            // self._mamba_tokens_per_page,
                        ),
                        1,
                    ),
                    use_mla=False,
                    role="vllm",
                    served_model_name=model_config.served_model_name,
                    chunk_size=self._mamba_tokens_per_page,
                    engine_id=f"{rpc_engine_id}-mamba",
                    kv_connector_extra_config=kv_connector_extra,
                )
                self._mamba_lookup_client = (
                    LookupClientFactory.create_lookup_client(
                        mamba_sched_cfg, mamba_sched_metadata, None
                    )
                )
            self._unfinished_requests: dict[str, Request] = {}
            self._lookup_requests_in_step: list[str] = []
            # llm-scaler v136: ids of mamba boundary lookups fired this
            # round; drained into LMCacheConnectorMetadata for the worker's
            # unpin mirror.
            self._mamba_lookup_ids: list[str] = []
            self.lmcache_engine = None
        else:
            self.lmcache_engine = _init_lmcache_engine(
                config,
                vllm_config,
                num_layers_override=(
                    # `is not None`: the mamba group index can be 0, which is
                    # falsy but must still select the hybrid path.
                    len(self._attn_layer_names)
                    if self._mamba_group_idx is not None
                    else None
                ),
            )
            if self._mamba_group_idx is not None:
                assert self._mamba_page_size_bytes is not None
                self.mamba_engine = _init_mamba_engine(
                    config,
                    vllm_config,
                    self._mamba_page_size_bytes,
                    self._mamba_tokens_per_page,
                    num_groups=len(self._mamba_group_ids),
                    num_layers=len(self._mamba_layer_names),
                )

            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending

            if self.enable_blending:
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                )

            # Create lookup server using factory (0.5.5 convention:
            # engine, engine.metadata)
            assert self.lmcache_engine is not None
            self.lookup_server = LookupClientFactory.create_lookup_server(
                self.lmcache_engine, self.lmcache_engine.metadata
            )
            # llm-scaler v136: dedicated sync lookup server for the mamba
            # engine; the scheduler-side joint clamp queries this pair
            # under engine_id f"{rpc_engine_id}-mamba".
            if self.mamba_engine is not None:
                self.mamba_lookup_server = (
                    LookupClientFactory.create_lookup_server(
                        self.mamba_engine, self.mamba_engine.metadata
                    )
                )

            # 0.5.5 convention: (engine, worker_id)
            self.offload_server = ZMQOffloadServer(
                self.lmcache_engine,
                get_tensor_model_parallel_rank(),
            )

            # In case of MLA, the lookup server is only created on worker 0
            if self.async_loading and self.lookup_server is not None:
                assert isinstance(self.lookup_server, LMCacheAsyncLookupServer)
                self.lmcache_engine.post_init(async_lookup_server=self.lookup_server)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = vllm_config.cache_config.block_size

        # request_id -> (vllm cached tokens, lmcache cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}

        self.kv_cache_manager: KVCacheManager | None = None

        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}

        # llm-scaler v133 (async-gate fix, 2026-10-09): route external
        # loads through the fork scheduler's WAITING_FOR_REMOTE_KS
        # handshake — the load runs in a dedicated no-forward engine step
        # (kv_connector_no_forward -> start_load_kv), get_finished reports
        # completion, and only then does the scheduler schedule the
        # forward. This kills the load/forward race where the forward
        # consumed zero pages before the (byte-perfect) load landed one
        # step later. Scheduler-side: pending gate set + staged load metas.
        # Worker-side: completed-load req ids + load-error blocks.
        self._lmc_async_pending: set[str] = set()
        self._lmc_async_load_metas: dict[str, ReqMeta] = {}
        self._lmc_async_loaded: set[str] = set()
        self._lmc_load_error_blocks: set[int] = set()

        # Whether to discard partial chunks
        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size
        self._save_decode_cache = config.save_decode_cache

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))

        self._requests_priority: dict[str, int] = {}

        # TODO(baoloongmao): Internal api server & plugin framework support
        # dp > 1
        if (
            vllm_config.parallel_config.data_parallel_size_local == 1
            or vllm_config.parallel_config.data_parallel_rank_local == 0
        ):
            # Start internal API server if enabled
            # The enabled check is in the InternalAPIServer constructor.
            # 0.5.5's InternalAPIServer expects an LMCacheManager rather than
            # the connector; these are auxiliary services, so any mismatch
            # degrades to disabled instead of failing the engine.
            self.api_server = None
            self.plugin_launcher = None
            try:
                self.api_server = InternalAPIServer(self)
                self.api_server.start()
            except Exception:
                logger.warning(
                    "InternalAPIServer unavailable on this lmcache version; "
                    "continuing without it",
                    exc_info=True,
                )
                self.api_server = None
            try:
                # Launch plugins
                self.plugin_launcher = RuntimePluginLauncher(
                    self.config,
                    role,
                    self.worker_count,
                    -1
                    if self.lmcache_engine is None  # scheduler side
                    else self.lmcache_engine.metadata.worker_id,
                )
                self.plugin_launcher.launch_plugins()
            except Exception:
                logger.warning(
                    "RuntimePluginLauncher unavailable; continuing without it",
                    exc_info=True,
                )
                self.plugin_launcher = None
        else:
            self.api_server = None  # type: ignore[assignment]
            self.plugin_launcher = None  # type: ignore[assignment]
        logger.info(
            "LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            utils.get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": vllm_config.model_config.get_vocab_size(),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache

    ####################
    # Worker side APIs
    ####################
    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        if self.lmcache_engine is not None:
            if self._mamba_group_idx is not None:
                # Hybrid: attention tensors in group order (layer count must
                # match the engine's kv_shape[0]).
                attn_kvs = [
                    kv_caches[n]
                    for n in self._attn_layer_names
                    if n in kv_caches
                ]
                missing = [
                    n for n in self._attn_layer_names if n not in kv_caches
                ]
                if missing:
                    logger.warning(
                        "Attn layer tensors missing from kv_caches: %s", missing
                    )
                logger.info(
                    "Hybrid register_kv_caches: %d attn tensors "
                    "(engine expects %d layers)",
                    len(attn_kvs),
                    self.lmcache_engine.metadata.kv_shape[0],
                )
                self._attn_kvcaches = attn_kvs
                global _LMC_DBG_ADAPTER_REF
                _LMC_DBG_ADAPTER_REF = self
                self.lmcache_engine.post_init(kvcaches=attn_kvs)
                # post_init only builds the StorageManager; the gpu
                # connector's pool pointers must be wired explicitly or
                # every store/load dies on kvcaches=None.
                self.lmcache_engine.gpu_connector.initialize_kvcaches_ptr(
                    kvcaches=attn_kvs
                )

                # Mamba pool: all recurrent layers share one storage; build a
                # [num_pages, page_size_bytes] uint8 view over the raw pool.
                assert self.mamba_engine is not None
                mamba_tensor = None
                for n in self._mamba_layer_names:
                    if n in kv_caches:
                        mamba_tensor = kv_caches[n]
                        break
                if isinstance(mamba_tensor, list):
                    mamba_tensor = mamba_tensor[0]
                assert mamba_tensor is not None, (
                    "No mamba layer tensor found in kv_caches"
                )
                num_blocks = getattr(
                    getattr(self._parent, "kv_cache_config", None),
                    "num_blocks",
                    None,
                )
                storage = mamba_tensor.untyped_storage()
                raw = torch.empty(
                    0, dtype=torch.int8, device=mamba_tensor.device
                ).set_(storage, 0, (storage.nbytes(),))
                if num_blocks is None or num_blocks <= 0:
                    num_blocks = storage.nbytes() // self._mamba_page_size_bytes
                page_bytes = storage.nbytes() // num_blocks
                # llm-scaler v136b: the dtype-true per-layer boundary bytes
                # come from the layer tensors THEMSELVES. MambaSpec's
                # page_size_bytes (and the padded pool storage) stay at the
                # fp16 value regardless of --mamba-ssm-cache-dtype — under
                # fp8_e4m3 the real state page is 464896 B vs 524288 — so
                # every MemoryObj sized from the spec constant mismatches
                # the boundary snapshots. Measure one block of one layer
                # exactly the way _lmc_snap_boundary gathers (sum over the
                # layer's state tensors of the first block's raw bytes) and
                # re-point every consumer BEFORE any store/retrieve
                # allocates.
                _mt = kv_caches.get(self._mamba_layer_names[0])
                _mtl = _mt if isinstance(_mt, list) else [_mt]
                try:
                    layer_page_bytes = 0
                    for _t in _mtl:
                        _pg0 = _t[0] if _t.dim() > 1 else _t
                        layer_page_bytes += (
                            _pg0.reshape(-1).view(torch.uint8).numel()
                        )
                except Exception:
                    layer_page_bytes = self._mamba_page_size_bytes
                if (
                    layer_page_bytes > 0
                    and layer_page_bytes != self._mamba_page_size_bytes
                ):
                    logger.info(
                        "Mamba dtype page override: spec page_size_bytes %d "
                        "-> true %d (pool padded %d, storage %d / %d blocks)",
                        self._mamba_page_size_bytes,
                        layer_page_bytes,
                        page_bytes,
                        storage.nbytes(),
                        num_blocks,
                    )
                    self._mamba_page_size_bytes = layer_page_bytes
                    self.mamba_engine.metadata.kv_shape = (
                        1,
                        1,
                        self._mamba_tokens_per_page,
                        max(
                            1,
                            layer_page_bytes
                            * len(self._mamba_layer_names)
                            // self._mamba_tokens_per_page,
                        ),
                        1,
                    )
                    self.mamba_engine.gpu_connector.page_size_bytes = (
                        layer_page_bytes
                    )
                logger.info(
                    "Mamba pool: storage %d bytes, num_blocks %d, "
                    "page_bytes %d (spec page_size_bytes %d, tpp %d)",
                    storage.nbytes(),
                    num_blocks,
                    page_bytes,
                    self._mamba_page_size_bytes,
                    self._mamba_tokens_per_page,
                )
                pool_view = raw.view(torch.uint8).view(num_blocks, page_bytes)
                self._mamba_kvcaches = [pool_view]
                self.mamba_engine.post_init(kvcaches=[pool_view])
                self.mamba_engine.gpu_connector.initialize_kvcaches_ptr(
                    kvcaches=[pool_view]
                )
            else:
                kvcaches = list(self.kv_caches.values())
                self.lmcache_engine.post_init(kvcaches=kvcaches)
                self.lmcache_engine.gpu_connector.initialize_kvcaches_ptr(
                    kvcaches=kvcaches
                )
        if len(self._attn_kvcaches) == 0 and self.lmcache_engine is not None:
            self._attn_kvcaches = list(self.kv_caches.values())

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        # TEMP (LMCDbgEntry): p0 liveness at start_load_kv entry — brackets
        # everything between the scheduler zero call (last wfs/zero digest)
        # and the load staging (mamba PRE-COPY, post_init). Remove after
        # diagnosis.
        if (
            os.environ.get("LMC_DBG_WATCH") == "1"
            and getattr(self, "_attn_kvcaches", None)
        ):
            try:
                import hashlib

                _t = self._attn_kvcaches[1]
                _r = _t[:, 0].reshape(-1)
                torch.xpu.synchronize()
                logger.info(
                    "LMCDbgEntry p0=%s",
                    hashlib.md5(_r.cpu().numpy().tobytes()).hexdigest()[:8],
                )
            except Exception:
                logger.exception("LMCDbgEntry hash failed (non-fatal)")

        self.current_layer = 0

        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        if len(self._attn_kvcaches) > 0:
            kvcaches = self._attn_kvcaches
        else:
            kvcaches = list(self.kv_caches.values())
        device = kvcaches[0].device

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # llm-scaler v133 (async-gate fix, 2026-10-09): the fork's
            # WAITING_FOR_REMOTE_KS handshake drives loads through
            # kv_connector_no_forward, whose pre_forward wraps this call in
            # set_forward_context(None, ...) — attn_metadata is ALWAYS None
            # on that step, yet the load path never reads it (kvcaches are
            # the pre-registered pools, slot mappings come from the ReqMeta).
            # Only bail when no request actually carries a load spec.
            if not any(r.load_spec is not None for r in metadata.requests):
                logger.debug(
                    "In connector.start_load_kv, but the attn_metadata is None"
                )
                return

        assert self.lmcache_engine is not None

        self.lmcache_engine.post_init(kvcaches=kvcaches)

        self.layerwise_retrievers = []

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.to(device)
            assert len(tokens) == len(slot_mapping)

            # TEMP (LMCDbgTail): where do the COMPUTED tail tokens' slots
            # point? Loaded requests answered as if the prompt ended at the
            # external boundary -> tail KV invisible to decode. Log the head
            # and tail rows of the computed span (attn + mamba) to compare
            # against the decode-time block table.
            try:
                _lmc = request.load_spec.lmcache_cached_tokens
                _n = len(tokens)
                if _lmc < _n:
                    _t = slot_mapping[_lmc : min(_lmc + 4, _n)].tolist()
                    _te = slot_mapping[max(_lmc, _n - 4) : _n].tolist()
                    _ms = request.mamba_slot_mapping
                    _mt = (
                        _ms[_lmc : min(_lmc + 4, _n)].tolist()
                        if _ms is not None
                        else None
                    )
                    _mte = (
                        _ms[max(_lmc, _n - 4) : _n].tolist()
                        if _ms is not None
                        else None
                    )
                    logger.info(
                        "LMCDbgTail req=%s n=%d lmc=%d vllm=%d is_last=%s "
                        "attn_head=%s attn_tail=%s m_head=%s m_tail=%s",
                        request.req_id,
                        _n,
                        _lmc,
                        request.load_spec.vllm_cached_tokens,
                        request.is_last_prefill,
                        _t,
                        _te,
                        _mt,
                        _mte,
                    )
            except Exception:
                logger.exception("LMCDbgTail log failed (non-fatal)")

            self._stats_monitor.update_interval_vllm_hit_tokens(
                request.load_spec.vllm_cached_tokens
            )
            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if self.use_layerwise:
                sync = idx == last_idx
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending
                    # compatible
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                # TEMP (LMCDbgBytes): light single-layer hash of rows
                # PRE-to_gpu — separates the live-pool state from the
                # bytes lmcache is about to write.
                self._lmc_dbg_hash_light(
                    f"preload req={request.req_id} "
                    f"vllm={request.load_spec.vllm_cached_tokens} "
                    f"lmc={lmcache_cached_tokens}",
                    kvcaches,
                    slot_mapping[:lmcache_cached_tokens],
                )
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!"
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    # llm-scaler v133 (async-gate fix): with the load now
                    # decoupled from the forward, a shortfall MUST fail
                    # loudly — promoting the request would schedule a
                    # forward over missing pages (silent poison). Matches
                    # the mamba-side shortfall hard-error.
                    raise RuntimeError(
                        f"Attention engine retrieve shortfall for req "
                        f"{request.req_id}: {num_retrieved_tokens} < "
                        f"{num_expected_tokens}"
                    )

                # TEMP (LMCDbgBytes): light single-layer hash of the attn
                # pool rows post-load (did to_gpu write, and what).
                self._lmc_dbg_hash_light(
                    f"load req={request.req_id}",
                    kvcaches,
                    slot_mapping[:lmcache_cached_tokens],
                )

        # Recurrent-state (mamba) engine retrieve. llm-scaler v136: the
        # retrieve parks the deepest hit chunk's boundary-state BYTES as a
        # CPU seed (no pool pages are written — align mode would null
        # them and the GDN kernel reads its initial state from the
        # running-state block anyway). preprocess_mamba writes the seed
        # into the carry-from page at the request's first scheduled step;
        # the normal PRE-COPY + v135 repair carry it from there.
        if self.mamba_engine is not None and self._mamba_kvcaches:
            from vllm.v1.worker.mamba_utils import _LMC_SEED_OVERRIDE

            _seed_bytes_expected = (
                self._mamba_page_size_bytes * len(self._mamba_layer_names)
            )
            for request in metadata.requests:
                if request.load_spec is None or not request.load_spec.can_load:
                    continue
                if request.mamba_slot_mapping is None:
                    # llm-scaler v133 (async-gate fix): hard error — a
                    # silent skip would leave the request in
                    # WAITING_FOR_REMOTE_KS forever (unreported load).
                    raise RuntimeError(
                        f"Req {request.req_id} has load_spec but no "
                        f"mamba_slot_mapping"
                    )
                tokens = request.token_ids
                m_slot_mapping = request.mamba_slot_mapping.to(device)
                lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
                # llm-scaler v136: the only mamba object a resume needs is
                # the boundary state AT the joint depth B (deepest banked
                # boundary <= the attn hit). Retrieve the FINAL chunk only:
                # a prefix-shaped mask would fetch nothing, because early
                # chunk keys are legitimately absent (intra-chunk
                # boundaries never materialize). The connector parks the
                # object's bytes as the seed; no pool pages are written.
                b_tokens = lmcache_cached_tokens
                m_token_mask = torch.zeros(b_tokens, dtype=torch.bool)
                m_token_mask[
                    b_tokens - self._lmcache_chunk_size : b_tokens
                ] = True
                seed_holder: dict = {}
                m_ret = self.mamba_engine.retrieve(
                    tokens[:b_tokens],
                    m_token_mask,
                    kvcaches=self._mamba_kvcaches,
                    slot_mapping=m_slot_mapping[:b_tokens],
                    request_configs=None,
                    req_id=request.req_id,
                    lmc_seed=seed_holder,
                )
                m_num_retrieved = int(m_ret.sum().item())
                seed = seed_holder.get("bytes")
                seed_end = int(seed_holder.get("tokens", 0) or 0)
                if (
                    m_num_retrieved != self._lmcache_chunk_size
                    or seed is None
                    or seed_end != b_tokens
                ):
                    logger.error(
                        "MAMBA SEED RETRIEVE FAILED: req %s got %d tokens "
                        "(want %d), seed=%s, seed_end=%d, B=%d",
                        request.req_id,
                        m_num_retrieved,
                        self._lmcache_chunk_size,
                        seed is not None,
                        seed_end,
                        b_tokens,
                    )
                    raise RuntimeError(
                        f"Mamba seed retrieve failed for req "
                        f"{request.req_id}: {m_num_retrieved}/"
                        f"{self._lmcache_chunk_size} tokens, "
                        f"seed_end={seed_end}, B={b_tokens}"
                    )
                if seed.numel() != _seed_bytes_expected:
                    raise RuntimeError(
                        f"Mamba seed size {seed.numel()} != expected "
                        f"{_seed_bytes_expected} for req {request.req_id}"
                    )
                _LMC_SEED_OVERRIDE[request.req_id] = (b_tokens, seed)
                logger.info(
                    "LMC mamba seed parked req=%s B=%d bytes=%d",
                    request.req_id,
                    b_tokens,
                    seed.numel(),
                )

                # llm-scaler v133 (async-gate fix): commit point — attn
                # retrieve + mamba seed have both landed for this request.
                # get_finished reports it so the scheduler promotes the
                # WAITING_FOR_REMOTE_KS request and only then schedules
                # the forward.
                if request.load_spec is not None and request.load_spec.can_load:
                    self._lmc_async_loaded.add(request.req_id)
                    logger.info(
                        "LMC async load complete req=%s", request.req_id
                    )
        else:
            # llm-scaler v133 (async-gate fix): non-hybrid deployments —
            # the attn retrieve alone completes the load.
            for request in metadata.requests:
                if request.load_spec is not None and request.load_spec.can_load:
                    self._lmc_async_loaded.add(request.req_id)
                    logger.info(
                        "LMC async load complete req=%s (attn-only)",
                        request.req_id,
                    )

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.layerwise_retrievers:
            logger.debug("Waiting for layer %s to be loaded", self.current_layer)

        # Wait for the layer to be loaded
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info("Retrieved %s tokens", num_retrieved_tokens)

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
        """
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        if len(self._attn_kvcaches) > 0:
            kvcaches = self._attn_kvcaches
        else:
            kvcaches = list(self.kv_caches.values())
        device = kvcaches[0].device
        if self.current_layer == 0:
            self.layerwise_storers = []

            is_first = True

            for idx, request in enumerate(connector_metadata.requests):
                save_spec = request.save_spec
                if save_spec is None or not save_spec.can_save:
                    continue

                token_ids = request.token_ids
                assert isinstance(token_ids, list)

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.to(device)

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.info(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                )
                self.layerwise_storers.append(layerwise_storer)
                if is_first:
                    is_first = False

        for layerwise_storer in self.layerwise_storers:
            next(layerwise_storer)

        self.current_layer += 1

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        # TEMP DEBUG (LWSDbg): store-path forensics, remove after diagnosis.
        _n_can_save = sum(
            1
            for r in connector_metadata.requests
            if r.save_spec is not None and r.save_spec.can_save
        )
        logger.info(
            "LWSDbg wait_for_save: nreqs=%d can_save=%d kvcaches=%d role=%s",
            len(connector_metadata.requests),
            _n_can_save,
            len(self.kv_caches),
            self.kv_role,
        )

        # TEMP (LMCDbgWatch): vllm-prefix page liveness probe. Hash attn
        # layer-7 pages 0/1/22 on every wait_for_save call (fires every
        # scheduler step) and log only when the digest CHANGES — pins the
        # exact step where the reused prefix page (page 0) flips real->zero
        # between req-1 store and req-2 load. Remove after diagnosis.
        if (
            os.environ.get("LMC_DBG_WATCH") == "1"
            and self.kv_role == "kv_both"
            and getattr(self, "_attn_kvcaches", None)
        ):
            try:
                import hashlib

                _t = self._attn_kvcaches[1]
                torch.xpu.synchronize()
                _ds = []
                for _p in (0, 1, 22):
                    _ds.append(
                        hashlib.md5(
                            _t[:, int(_p)].reshape(-1).cpu().numpy().tobytes()
                        ).hexdigest()[:8]
                    )
                _d = "/".join(_ds)
                if _d != getattr(self, "_lmc_watch_last", None):
                    self._lmc_watch_last = _d
                    logger.info(
                        "LMCDbgWatch wfs p0=%s p1=%s p22=%s", _ds[0], _ds[1], _ds[2]
                    )
            except Exception:
                logger.exception("LMCDbgWatch wfs hash failed (non-fatal)")

        # TEMP (LMCDbgBytes): poll stashed rows at EVERY step (even steps
        # where the request has no metadata) to pin down when each chunk's
        # KV bytes actually finalize in the live pool.
        _stash = getattr(self, "_lmc_dbg_stash", None)
        if (
            _stash
            and os.environ.get("LMC_DBG_POLL") == "1"
            and len(self.kv_caches) > 0
        ):
            if len(self._attn_kvcaches) > 0:
                _kv = self._attn_kvcaches
            else:
                _kv = list(self.kv_caches.values())
            _dev = _kv[0].device
            for _rid, (_a_sm, _m_sm) in list(_stash.items()):
                _n = self._lmc_dbg_polls.get(_rid, 0) + 1
                self._lmc_dbg_polls[_rid] = _n
                # n=1 (first post-store wfs) is banked; the decisive datum
                # is n=2: are the LAST step's chunk rows final one step on?
                if _n != 2:
                    continue
                _cs = self._lmcache_chunk_size
                self._lmc_dbg_hash_light(
                    f"pollc1 req={_rid} n={_n}",
                    _kv,
                    None if _a_sm is None else _a_sm[_cs : 2 * _cs].to(_dev),
                )

        # 0.5.5 lookup_unpin takes one lookup_id at a time
        for lookup_id in connector_metadata.lookup_requests_in_step:
            self.lmcache_engine.lookup_unpin(lookup_id)
        # llm-scaler v136: mirror for the mamba boundary pins ("m-<req_id>"
        # is deterministic). start_load's seed retrieve ran earlier in this
        # same step, so the pinned boundary object survived lookup -> load;
        # ids whose request never loaded are released here too, so pins
        # cannot accumulate across warm requests.
        if self.mamba_engine is not None:
            for lookup_id in (
                getattr(connector_metadata, "mamba_lookup_ids", None) or []
            ):
                self.mamba_engine.lookup_unpin(lookup_id)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
            for layerwise_storer in self.layerwise_storers:
                next(layerwise_storer)
            return

        # llm-scaler LMC dbg: bisect sentinel — skip ALL stores (attn lag
        # flush + from_gpu gathers + mamba bank). Lookups still run.
        if os.path.exists("/tmp/lmc_store_off"):
            return

        assert len(self.kv_caches) > 0
        if len(self._attn_kvcaches) > 0:
            kvcaches = self._attn_kvcaches
        else:
            kvcaches = list(self.kv_caches.values())
        device = kvcaches[0].device

        assert self.lmcache_engine is not None

        # llm-scaler LMC fix (2026-10-09): one-step store lag. The store-side
        # from_gpu gathers raced the KV writes: under async scheduling the
        # span for prefill chunk k is handed to wait_for_save before the
        # unified_kv_cache_update / mamba-state kernels for that chunk are
        # submitted, so the pool pages still read all-zero (chunk-1 zeros at
        # every gather, deterministic across boots 8/9/10; a device-wide
        # sync at the gather did NOT help; chunk 0 real because the first
        # step runs pre-overlap). Cached zeros -> zero KV reloaded -> vllm
        # skips computing those tokens -> diverged output. Fix: snapshot each
        # store here, execute it on the NEXT wait_for_save call (which fires
        # every scheduler step, decode included) after a device sync, so the
        # writing step's kernels are enqueued AND complete. Lookups racing
        # the lag miss cleanly (recompute), never read partial bytes.
        _lag = os.environ.get("LMC_STORE_LAG", "1") != "0" and (
            self.kv_role == "kv_both"
        )
        if _lag:
            self._lmc_flush_pending()

        # Recurrent-state (mamba) store. llm-scaler v136: bank the frozen
        # boundary snapshots from mamba_utils._LMC_BOUNDARY_REGISTRY — one
        # object per materialized 512-token boundary, keyed by the stock
        # chain hash of tokens[0:B] (so a later resume at ANY banked depth
        # strict-prefix-hits). Objects are filled from the CPU seed bytes
        # (batched_from_gpu lmc_seed_bytes branch), never from pool pages:
        # align mode nulls boundary pages, and the bytes were frozen at
        # materialization time so no lag is needed. Ordering law kept: a
        # failed mamba store skips the request's attn store (both absent
        # = clean miss, never an attn hit without its mamba state).
        mamba_store_failed: set[str] = set()
        if self.mamba_engine is not None and self._mamba_kvcaches:
            from vllm.v1.worker.mamba_utils import _LMC_BOUNDARY_REGISTRY

            # llm-scaler v139: token-id stash for the finish/lag flush
            # below. A request's final wfs row can be dropped by the
            # async-overlap empty-row class (P84/v60c) or its deepest
            # boundary can materialize one step AFTER its last wfs, so at
            # flush time the registry entry is the only witness and the
            # token_ids must come from here. Keeps the LAST cumulative span
            # seen per request; FIFO-bounded.
            _stash = getattr(self, "_lmc_token_stash", None)
            if _stash is None:
                _stash = self._lmc_token_stash = {}
            for _r in connector_metadata.requests:
                _tok = getattr(_r, "token_ids", None)
                if _tok:
                    _stash[_r.req_id] = list(_tok)
            while len(_stash) > 64:
                _stash.pop(next(iter(_stash)))

            for request in connector_metadata.requests:
                save_spec = request.save_spec
                if (
                    save_spec is None or not save_spec.can_save
                ) and self.kv_role != "kv_producer":
                    continue
                assert save_spec is not None

                token_ids = request.token_ids
                m_skip = save_spec.skip_leading_tokens
                if self.kv_role == "kv_producer":
                    assert request.disagg_spec is not None
                    m_skip = min(
                        m_skip, request.disagg_spec.num_transferred_tokens
                    )
                # chunk_size == mamba tokens_per_page, so chunk alignment IS
                # page alignment here
                m_skip = (
                    m_skip // self._lmcache_chunk_size * self._lmcache_chunk_size
                )

                if not request.is_last_prefill:
                    token_ids = token_ids[
                        : len(token_ids)
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    ]

                boundaries = _LMC_BOUNDARY_REGISTRY.pop(request.req_id, None)
                # TEMP v136 debug: which silent-skip point eats the store
                logger.info(
                    "LWSDbg mamba-save: req=%s m_skip=%d last=%s ntok=%d "
                    "reg_reqs=%d bnd=%s",
                    request.req_id,
                    m_skip,
                    request.is_last_prefill,
                    len(token_ids),
                    len(_LMC_BOUNDARY_REGISTRY),
                    [b for b, _ in boundaries] if boundaries else None,
                )
                if not boundaries:
                    continue
                try:
                    deepest = boundaries[-1][0]
                    if deepest > len(token_ids):
                        boundaries = [
                            (b, s) for (b, s) in boundaries if b <= len(token_ids)
                        ]
                        if not boundaries:
                            continue
                        deepest = boundaries[-1][0]
                    hash_map: dict[int, object] = {}
                    for s, e, h in self.mamba_engine.token_database.process_tokens(
                        tokens=list(token_ids[:deepest]), make_key=False
                    ):
                        hash_map[int(e)] = h
                    hashes = []
                    offsets = []
                    seeds = []
                    for b, cpu_bytes in boundaries:
                        # NO m_skip filter: attn's skip_leading semantics
                        # ("already stored by an earlier chunk") do NOT
                        # transfer to mamba — skip_leading == num_computed
                        # == the boundary itself at the wfs that carries
                        # it, so b <= m_skip would discard EVERY boundary.
                        # Each registry entry was materialized fresh by
                        # this request and banked exactly once; duplicate
                        # stores after preemption re-materialization are
                        # idempotent overwrites.
                        if b not in hash_map:
                            continue
                        hashes.append(hash_map[b])
                        offsets.append(self._lmcache_chunk_size)
                        seeds.append(cpu_bytes)
                    if not hashes:
                        continue
                    # store() tensorizes slot_mapping itself when hashes
                    # are given; a plain list avoids torch.tensor(tensor).
                    dummy_slots = [0] * sum(offsets)
                    # tokens=None is REQUIRED: with tokens given,
                    # process_tokens takes the tokens branch and chunks the
                    # whole range (every 512 chunk), while hashes+offsets
                    # here cover ONLY the materialized boundaries — the
                    # mismatch indexes seeds out of range. tokens=None
                    # activates the hashes branch: one (start,end) per
                    # (hash,offset) pair, aligned with seeds.
                    self.mamba_engine.store(
                        None,
                        hashes=hashes,
                        offsets=offsets,
                        kvcaches=self._mamba_kvcaches,
                        slot_mapping=dummy_slots,
                        lmc_seed_bytes=seeds,
                        # request_configs stays out of the mamba keys:
                        # boundary states are prefix property, shared
                        # across requests (None on both sides of the pair).
                        request_configs=None,
                    )
                    logger.info(
                        "LMC mamba store req=%s boundaries=%d deepest=%d",
                        request.req_id,
                        len(hashes),
                        boundaries[-1][0],
                    )
                except Exception:
                    logger.exception(
                        "Mamba engine store failed for req %s; skipping "
                        "its attn store to keep the two engines "
                        "consistent",
                        request.req_id,
                    )
                    mamba_store_failed.add(request.req_id)

            # llm-scaler v139: finish/lag flush. Boundary snapshots whose
            # consuming wfs row never arrived (overlap-drop) or whose
            # materialization lagged past the request's last wfs are still
            # sitting in the registry — store them now from the stashed
            # token_ids so a later resume reaches the deepest boundary.
            # Early flush of a mid-flight request is harmless: bytes were
            # frozen at materialization, the key is a strict-prefix hash,
            # and re-stores are idempotent overwrites.
            try:
                _live = {_r.req_id for _r in connector_metadata.requests}
                for _rid in list(_LMC_BOUNDARY_REGISTRY.keys()):
                    if _rid in _live:
                        continue
                    _tok = _stash.get(_rid)
                    if not _tok:
                        continue
                    _bnd = _LMC_BOUNDARY_REGISTRY.pop(_rid)
                    _deepest = _bnd[-1][0]
                    if _deepest > len(_tok):
                        _bnd = [(b, s) for (b, s) in _bnd if b <= len(_tok)]
                        if not _bnd:
                            continue
                        _deepest = _bnd[-1][0]
                    _hash_map: dict[int, object] = {}
                    for _s, _e, _h in (
                        self.mamba_engine.token_database.process_tokens(
                            tokens=list(_tok[:_deepest]), make_key=False
                        )
                    ):
                        _hash_map[int(_e)] = _h
                    _hashes, _offsets, _seeds = [], [], []
                    for _b, _cpu_bytes in _bnd:
                        if _b not in _hash_map:
                            continue
                        _hashes.append(_hash_map[_b])
                        _offsets.append(self._lmcache_chunk_size)
                        _seeds.append(_cpu_bytes)
                    if not _hashes:
                        continue
                    self.mamba_engine.store(
                        None,
                        hashes=_hashes,
                        offsets=_offsets,
                        kvcaches=self._mamba_kvcaches,
                        slot_mapping=[0] * sum(_offsets),
                        lmc_seed_bytes=_seeds,
                        request_configs=None,
                    )
                    logger.info(
                        "LMC mamba store(final) req=%s boundaries=%d deepest=%d",
                        _rid,
                        len(_hashes),
                        _bnd[-1][0],
                    )
            except Exception:
                logger.exception(
                    "LMC mamba finish-flush failed (non-fatal)"
                )

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue
            if request.req_id in mamba_store_failed:
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)
            assert save_spec is not None

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.to(device)

            skip_leading_tokens = save_spec.skip_leading_tokens
            if self.kv_role == "kv_producer":
                assert request.disagg_spec is not None
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.info(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                token_len = len(token_ids)
                aligned_token_len = (
                    token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                )
                token_ids = token_ids[:aligned_token_len]
                store_mask = store_mask[:aligned_token_len]
                slot_mapping = slot_mapping[:aligned_token_len]

            if _lag:
                # kv_both-only path: disagg_spec is None here (lag is gated
                # on kv_role == "kv_both"), so the immediate branch keeps the
                # transfer_spec semantics for producer roles unchanged.
                if not hasattr(self, "_lmc_pending_stores"):
                    self._lmc_pending_stores = []
                _tok = (
                    token_ids.detach().clone()
                    if torch.is_tensor(token_ids)
                    else list(token_ids)
                )
                self._lmc_pending_stores.append(
                    (
                        "a",
                        request.req_id,
                        _tok,
                        store_mask,
                        slot_mapping,
                        skip_leading_tokens,
                        request.request_configs,
                        None,
                    )
                )
            else:
                self.lmcache_engine.store(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    transfer_spec=request.disagg_spec,
                    request_configs=request.request_configs,
                )

            # TEMP (LMCDbgBytes): store-time digests are BANKED (cross-boot
            # bit-stable); the heavy per-store hash was removed after it
            # OOR-crashed the worker (UR_RESULT_ERROR_OUT_OF_RESOURCES
            # 2026-10-09). Only the stash below remains, for the n=2 poll.
            _aligned = (
                len(token_ids)
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            _m_sm = request.mamba_slot_mapping
            if _m_sm is not None:
                _m_sm = _m_sm[:_aligned].to(slot_mapping.device)

            # TEMP (LMCDbgBytes): stash this request's chunk-aligned slot
            # maps (CPU copies) so every LATER wait_for_save can re-hash
            # the same rows — incl. steps where the request has already
            # left the connector metadata (final prefill chunk + decode).
            if os.environ.get("LMC_DBG_POLL") == "1":
                if not hasattr(self, "_lmc_dbg_stash"):
                    self._lmc_dbg_stash = {}
                    self._lmc_dbg_polls = {}
                self._lmc_dbg_stash[request.req_id] = (
                    slot_mapping[:_aligned].to("cpu", copy=True),
                    None if _m_sm is None else _m_sm.to("cpu", copy=True),
                )
                self._lmc_dbg_polls.setdefault(request.req_id, 0)

            # NOTE(Jiayi): We assume all tokens are saved
            save_spec.skip_leading_tokens = len(token_ids)
            if request.disagg_spec:
                request.disagg_spec.num_transferred_tokens = len(token_ids)

    def _lmc_mamba_page_map(
        self, request: "ReqMeta"
    ) -> dict[int, list[int]] | None:
        """llm-scaler LMC fix (2026-10-09): per-chunk per-group mamba page
        ids as a {chunk_start_token: [page_id per group]} map, in the same
        token coordinate space the engine hands to to_gpu/from_gpu."""
        pages = getattr(request, "mamba_chunk_pages", None)
        if not pages:
            return None
        cs = self._lmcache_chunk_size
        return {i * cs: p for i, p in enumerate(pages)}

    def _lmc_set_mamba_page_map(self, request: "ReqMeta") -> None:
        m = self._lmc_mamba_page_map(request)
        if m is not None:
            self.mamba_engine.gpu_connector._lmc_chunk_pages = m

    def _lmc_clear_mamba_page_map(self) -> None:
        if self.mamba_engine is not None:
            self.mamba_engine.gpu_connector._lmc_chunk_pages = None

    def _lmc_flush_pending(self) -> None:
        """llm-scaler LMC fix (2026-10-09): execute the one-step-lagged
        stores captured by the previous wait_for_save call. The lag + sync
        guarantees the writing step's KV kernels are enqueued and complete
        before the D2H gathers, killing the cached-zeros divergence.
        Mamba-first ordering law preserved: every pending mamba store runs
        before any attn store, and a failed mamba store drops that
        request's pending attn store so the two engines stay consistent.
        """
        pending = getattr(self, "_lmc_pending_stores", None)
        if not pending:
            return
        self._lmc_pending_stores = []
        logger.info("LMC store lag: flushing %d pending store(s)", len(pending))
        # One wfs call has passed since these spans were computed; the
        # writing step's kernels are enqueued. Complete them before reading.
        torch.xpu.synchronize()
        failed: set[str] = set()
        if self.mamba_engine is not None and self._mamba_kvcaches:
            for kind, rid, token_ids, mask, slot, offset, cfg, pmap in pending:
                if kind != "m":
                    continue
                if pmap is not None:
                    self.mamba_engine.gpu_connector._lmc_chunk_pages = pmap
                try:
                    self.mamba_engine.store(
                        token_ids,
                        mask=mask,
                        kvcaches=self._mamba_kvcaches,
                        slot_mapping=slot,
                        offset=offset,
                        request_configs=cfg,
                    )
                except Exception:
                    logger.exception(
                        "Lagged mamba store failed for req %s; dropping its "
                        "pending attn store (consistency law)",
                        rid,
                    )
                    failed.add(rid)
                finally:
                    self._lmc_clear_mamba_page_map()
        else:
            # No mamba engine: every pending mamba closure is vacuous (the
            # snapshot loop only emits them when the engine exists), so
            # nothing to fail.
            pass
        if len(self._attn_kvcaches) > 0:
            kvcaches = self._attn_kvcaches
        else:
            kvcaches = list(self.kv_caches.values())
        for kind, rid, token_ids, mask, slot, offset, cfg, _pmap in pending:
            if kind != "a" or rid in failed:
                continue
            try:
                self.lmcache_engine.store(
                    token_ids,
                    mask=mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot,
                    offset=offset,
                    transfer_spec=None,
                    request_configs=cfg,
                )
            except Exception:
                logger.exception(
                    "Lagged attn store failed for req %s "
                    "(cached chunk absent; later lookups recompute)",
                    rid,
                )

    def _lmc_dbg_hash_light(
        self,
        tag: str,
        kvcaches: list[torch.Tensor] | None,
        slot_mapping: torch.Tensor | None,
        m_kvcaches: list[torch.Tensor] | None = None,
        m_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """TEMP (LMCDbgBytes): single-layer md5 of the pool rows backing a
        token span. The heavy 17-layer variant OOR-crashed the worker
        (UR_RESULT_ERROR_OUT_OF_RESOURCES + 603s step-watchdog kill,
        2026-10-09); this one reads ONE representative target layer
        (index 1 = layers.7.self_attn, tensor map banked) plus optionally
        the mamba pages, ~2MB D2H, so per-step overhead stays trivially
        under the RPC budget. Digest layout matches the heavy per-layer
        digest (index_select rows -> uint8 view -> md5[:8]) so banked
        values remain comparable. Remove after diagnosis.
        """
        import hashlib

        if os.environ.get("LMC_DBG_BYTES") not in ("1", "2"):
            return
        # One-shot: dump shapes (name list banked earlier) so the gather's
        # dim semantics are pinned: dim0 vs dim1, K/V stacking, dtypes.
        if not getattr(self, "_lmc_dbg_dumped", False):
            self._lmc_dbg_dumped = True
            try:
                _shapes = [
                    (
                        tuple(t.shape),
                        str(t.dtype),
                        f"contig={t.is_contiguous()}",
                        tuple(t.stride()),
                    )
                    for t in (kvcaches or [])[:2]
                ]
                _mshapes = (
                    [(tuple(t.shape), str(t.dtype)) for t in m_kvcaches[:2]]
                    if m_kvcaches
                    else None
                )
                logger.info(
                    "LMCDbgBytes shapes attn_shapes=%s mamba_shapes=%s",
                    _shapes,
                    _mshapes,
                )
            except Exception:
                logger.exception("LMCDbgBytes shape dump failed (non-fatal)")
        try:
            parts: list[str] = []
            if kvcaches and slot_mapping is not None and len(slot_mapping) > 0:
                t = kvcaches[1] if len(kvcaches) > 1 else kvcaches[0]
                sm = slot_mapping.to(t.device)
                sm = sm[sm >= 0]
                if len(sm) > 0:
                    # Page-slice gather: the fork's attn layout is
                    # [2(K/V), NB pages, 1024 tok/page, NH, HS] (a K/V-major
                    # permuted view over blocks-first memory). slot =
                    # page*1024 + off, so a chunk-aligned span is exactly
                    # t[:, page] per page — basic indexing, one reshape-copy,
                    # one D2H. Raw slot ids or sm//shape[1] on dim0 both
                    # mis-gather (OOM 2026-10-09; contaminated digests).
                    tpp = kvcaches[0].shape[2]
                    pages = torch.unique(sm // tpp).cpu().tolist()
                    torch.xpu.synchronize()
                    if len(pages) == 1:
                        rows = t[:, int(pages[0])].reshape(-1)
                    else:
                        rows = torch.cat(
                            [t[:, int(p)].reshape(-1) for p in pages]
                        )
                    buf = rows.cpu()
                    _pp = ",".join(
                        f"{int(p)}:"
                        + hashlib.md5(
                            t[:, int(p)].reshape(-1).cpu().numpy().tobytes()
                        ).hexdigest()[:8]
                        for p in pages
                    )
                    parts.append(
                        "L1="
                        + hashlib.md5(buf.numpy().tobytes()).hexdigest()[:8]
                        + f" pages={pages} pp=[{_pp}]"
                    )
            if m_kvcaches and m_slot_mapping is not None and len(m_slot_mapping) > 0:
                p = m_kvcaches[0]
                msm = m_slot_mapping.to(p.device)
                msm = msm[msm >= 0]
                tpp = self._lmcache_chunk_size
                ids = torch.unique(msm[::tpp] // tpp)
                if len(ids) > 0:
                    torch.xpu.synchronize()
                    rows = torch.index_select(p, 0, ids)
                    buf = rows.reshape(-1).view(torch.uint8).cpu()
                    parts.append(
                        "M0="
                        + hashlib.md5(buf.numpy().tobytes()).hexdigest()[:8]
                        + f" pages={len(ids)}"
                    )
            if parts:
                logger.info("LMCDbgBytes light %s %s", tag, " ".join(parts))
        except Exception:
            logger.exception("LMCDbgBytes light hash failed (non-fatal)")

    def _lmc_dbg_hash(
        self,
        tag: str,
        kvcaches: list[torch.Tensor] | None,
        slot_mapping: torch.Tensor | None,
        m_kvcaches: list[torch.Tensor] | None = None,
        m_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """TEMP (LMCDbgBytes): per-chunk md5 of the GPU pool bytes backing a
        token span, so store-side and load-side calls over the same chunk are
        directly comparable. A store-vs-later-store mismatch means the pool
        mutated after the store (timing bug); a store-vs-load mismatch means
        the CPU round trip corrupted bytes. LMC_DBG_BYTES=2 additionally logs
        per-layer digests to localize attn corruption (draft vs target
        layers). Remove after diagnosis.
        """
        import hashlib

        level = os.environ.get("LMC_DBG_BYTES")
        if level != "1" and level != "2":
            return
        # One-shot: dump the attn tensor list (name/shape/data_ptr) so hash
        # indices can be mapped to real layers and storage aliasing checked.
        if not getattr(self, "_lmc_dbg_dumped", False):
            self._lmc_dbg_dumped = True
            try:
                _names = list(getattr(self, "kv_caches", {}).keys())
                _ptrs = [t.data_ptr() for t in (kvcaches or [])][:20]
                _shapes = [
                    (tuple(t.shape), str(t.dtype)) for t in (kvcaches or [])[:4]
                ]
                _mshapes = (
                    [(tuple(t.shape), str(t.dtype)) for t in m_kvcaches[:2]]
                    if m_kvcaches
                    else None
                )
                logger.info(
                    "LMCDbgBytes tensors kvcaches_dict=%s attn_ptrs=%s "
                    "attn_shapes=%s mamba_shapes=%s",
                    _names,
                    _ptrs,
                    _shapes,
                    _mshapes,
                )
            except Exception:
                logger.exception("LMCDbgBytes tensor dump failed (non-fatal)")
        cs = self._lmcache_chunk_size
        _chunks = os.environ.get("LMC_DBG_CHUNKS")
        _chunk_set = (
            {int(x) for x in _chunks.split(",")} if _chunks else None
        )
        try:
            if kvcaches and slot_mapping is not None:
                sm = slot_mapping
                bs = kvcaches[0].shape[1]
                parts = []
                for c0 in range(0, len(sm), cs):
                    if _chunk_set is not None and (c0 // cs) not in _chunk_set:
                        continue
                    sm_c = sm[c0 : c0 + cs]
                    if len(sm_c) == 0:
                        continue
                    if bool((sm_c < 0).any()):
                        # prefix-cached tokens carry -1 slots; block 0 would
                        # pollute the digest
                        parts.append(f"c{c0 // cs}=neg")
                        continue
                    blocks = torch.unique(sm_c // bs)
                    gathered = [torch.index_select(t, 0, blocks) for t in kvcaches]
                    snap = torch.stack(gathered).view(torch.uint8)
                    torch.xpu.synchronize()
                    blob = snap.to("cpu").contiguous().numpy().tobytes()
                    torch.xpu.synchronize()
                    h = hashlib.md5()
                    h.update(blob)
                    parts.append(f"c{c0 // cs}={h.hexdigest()[:10]}")
                    if level == "2" and len(gathered) > 0:
                        n_el = gathered[0].numel()
                        per_layer = " ".join(
                            hashlib.md5(
                                blob[i * n_el : (i + 1) * n_el]
                            ).hexdigest()[:6]
                            for i in range(len(gathered))
                        )
                        parts.append(f"[c{c0 // cs}L] {per_layer}")
                if parts:
                    logger.info("LMCDbgBytes %s attn %s", tag, " ".join(parts))
            if m_kvcaches and m_slot_mapping is not None:
                msm = m_slot_mapping
                pool = m_kvcaches[0]
                parts = []
                for c0 in range(0, len(msm), cs):
                    if _chunk_set is not None and (c0 // cs) not in _chunk_set:
                        continue
                    sm_c = msm[c0 : c0 + cs]
                    if len(sm_c) == 0:
                        continue
                    if bool((sm_c < 0).any()):
                        parts.append(f"p{c0 // cs}=neg")
                        continue
                    pages = torch.unique(sm_c // cs)
                    snap = torch.index_select(pool, 0, pages).view(torch.uint8)
                    torch.xpu.synchronize()
                    blob = snap.to("cpu").contiguous().numpy().tobytes()
                    torch.xpu.synchronize()
                    h = hashlib.md5()
                    h.update(blob)
                    parts.append(f"p{c0 // cs}={h.hexdigest()[:10]}")
                if parts:
                    logger.info("LMCDbgBytes %s mamba %s", tag, " ".join(parts))
        except Exception:
            logger.exception("LMCDbgBytes %s hash failed (non-fatal)", tag)

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        # TEMP (LMCDbgBytes): drop poll stashes for finished requests.
        _stash = getattr(self, "_lmc_dbg_stash", None)
        if _stash:
            for _rid in finished_req_ids:
                _stash.pop(_rid, None)
                self._lmc_dbg_polls.pop(_rid, None)
        # llm-scaler v133 (async-gate fix): report loads completed by
        # start_load_kv as finished_recving — the scheduler promotes the
        # WAITING_FOR_REMOTE_KS requests and schedules the forward. Loads
        # are inline in the no-forward step, so completion is immediate.
        finished_recving = self._lmc_async_loaded or None
        self._lmc_async_loaded = set()
        return None, finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        """llm-scaler v133 (async-gate fix): blocks whose external load
        failed; the scheduler invalidates them and recomputes the affected
        tokens instead of promoting a poisoned prefix."""
        errs = self._lmc_load_error_blocks
        self._lmc_load_error_blocks = set()
        return errs

    ###################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int | None:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        # llm-scaler LMC dbg (2026-10-09): A/B discriminator — when the
        # sentinel file exists the connector reports zero external hit so
        # the fork takes its pure-local resume path (own prefix cache +
        # PRE-COPY mamba state restore) on the SAME warm engine state.
        # Toggle via touch/rm inside the container; strip before bake.
        if os.path.exists("/tmp/lmc_off"):
            logger.info(
                "LMC lookup disabled (lmc_off): req=%s num_computed=%d",
                request.request_id,
                num_computed_tokens,
            )
            self._lmc_async_pending.discard(request.request_id)
            return 0

        self._requests_priority[request.request_id] = request.priority

        token_ids = request.prompt_token_ids

        # If the request has multimodal hashes, apply them to the token ids
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            # TODO(Jiayi): Optimize this
            token_ids_tensor = torch.tensor(request.prompt_token_ids)
            apply_mm_hashes_to_token_ids(token_ids_tensor, mm_hashes, mm_positions)
            token_ids = token_ids_tensor.tolist()

        if request.sampling_params:
            request_configs = extract_request_configs(request.sampling_params)
        else:
            request_configs = None

        if self.skip_last_n_tokens > 0:
            assert token_ids is not None
            token_ids = token_ids[: -self.skip_last_n_tokens]
        lookup_id = request.request_id if self.async_loading else str(uuid.uuid4())

        self._lookup_requests_in_step.append(lookup_id)

        num_external_hit_tokens = self.lookup_client.lookup(
            token_ids,
            lookup_id=lookup_id,
            request_configs=request_configs,
        )

        # llm-scaler v136: joint (attn ∩ mamba) hit over the dedicated
        # mamba zmq lookup pair. The mamba engine banks a boundary state
        # per 512-token chunk (registry snapshots), so a hit here means a
        # resumeable recurrent state exists at that depth. The worker
        # books exactly the joint depth as computed; a mamba miss clamps
        # the promise to 0 and the request recomputes cleanly (the old
        # null-page shortfall crash is unreachable by construction).
        if (
            num_external_hit_tokens is not None
            and num_external_hit_tokens > 0
            and self._mamba_lookup_client is not None
        ):
            # llm-scaler v136: the mamba engine answers with the deepest
            # BANKED boundary (scheduler chunk ends are the only
            # materializable states, so early chunk keys can be absent),
            # NOT a consecutive-prefix hit. Bound the scan by the attn hit
            # (the joint depth can never exceed it) and keep
            # request_configs=None so the key space matches the store side,
            # which banks boundaries with None.
            m_hit = self._mamba_lookup_client.lookup(
                token_ids[:num_external_hit_tokens],
                lookup_id=f"m-{request.request_id}",
                request_configs=None,
            )
            self._mamba_lookup_ids.append(f"m-{request.request_id}")
            if m_hit is None:
                m_hit = 0
            m_hit = m_hit // self._lmcache_chunk_size * self._lmcache_chunk_size
            if m_hit < num_external_hit_tokens:
                logger.info(
                    "LMC joint lookup: req=%s attn_hit=%d mamba_hit=%d -> %d",
                    request.request_id,
                    num_external_hit_tokens,
                    m_hit,
                    m_hit,
                )
                num_external_hit_tokens = m_hit

        if num_external_hit_tokens is None:
            logger.info(
                "Reqid: %s, Total tokens %d, LMCache hit tokens: None.",
                request.request_id,
                request.num_tokens,
            )
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        logger.info(
            "Reqid: %s, Total tokens %d, LMCache hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        if need_to_allocate <= 0:
            self._lmc_async_pending.discard(request.request_id)
            return 0

        # llm-scaler v133 (async-gate fix): a real allocation means a real
        # load — gate this request through the async handshake (load in a
        # dedicated no-forward step) instead of racing the forward.
        self._lmc_async_pending.add(request.request_id)
        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(
        self,
        request: "Request",
        num_external_tokens: int,
        blocks: "KVCacheBlocks | None" = None,
    ):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        self.lookup_client.clear_lookup_status(request.request_id)
        # llm-scaler v136: the mamba client's lookup_id is deterministic
        # (f"m-{request_id}"), so its cached status must be cleared too.
        if self._mamba_lookup_client is not None:
            self._mamba_lookup_client.clear_lookup_status(
                f"m-{request.request_id}"
            )

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        # Only check for non-prompt-hit case
        if (
            self.load_specs[request.request_id].lmcache_cached_tokens
            != request.num_tokens
        ):
            assert (
                num_external_tokens > 0
                and num_external_tokens
                == self.load_specs[request.request_id].lmcache_cached_tokens
                - self.load_specs[request.request_id].vllm_cached_tokens
            ), (
                f"Mismatch in number of tokens: {num_external_tokens} vs "
                f"{self.load_specs[request.request_id].lmcache_cached_tokens} -"
                f" {self.load_specs[request.request_id].vllm_cached_tokens}"
                f" for request {request.request_id}"
            )

        self.load_specs[request.request_id].can_load = True

        # llm-scaler v133 (async-gate fix): the scheduler will skip this
        # request (WAITING_FOR_REMOTE_KS, zero scheduled tokens), so it
        # never reaches scheduled_new_reqs in build_connector_meta. Build
        # the load ReqMeta HERE — consuming the load spec — and stage it
        # for the next build_connector_meta, which drains it into the
        # connector metadata of the dedicated no-forward load step.
        if request.request_id in self._lmc_async_pending:
            self._lmc_async_pending.discard(request.request_id)
            spec = self.load_specs.pop(request.request_id)
            num_tokens_to_compute = (
                spec.vllm_cached_tokens + num_external_tokens
            )
            # Build the tracker directly from the live Request + the
            # allocated KVCacheBlocks — RequestTracker.from_new_request
            # duck-types NewRequestData (.req_id/.block_ids), which the
            # scheduler-side Request does not provide.
            if blocks is None:
                raise RuntimeError(
                    f"Async load for req {request.request_id} has no "
                    f"allocated blocks; cannot build load metadata"
                )
            mm_hashes, mm_positions = extract_mm_features(request)
            request_tracker = RequestTracker(
                req_id=request.request_id,
                prompt_len=len(request.prompt_token_ids),
                token_ids=request.prompt_token_ids[
                    :num_tokens_to_compute
                ].copy(),
                allocated_block_ids=[
                    list(g) for g in blocks.get_block_ids()
                ],
                num_saved_tokens=spec.lmcache_cached_tokens,
                disagg_spec=tmp_disagg_tracker.pop(request.request_id, None),
                mm_hashes=mm_hashes,
                mm_positions=mm_positions,
                # No stores on the load step — the post-promotion step
                # rebuilds a pure-store meta from the real block table.
                skip_save=True,
                request_configs=(
                    extract_request_configs(request.sampling_params)
                    if request.sampling_params
                    else None
                ),
            )
            self._request_trackers[request.request_id] = request_tracker
            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
                mamba_group_idx=self._mamba_group_idx,
                mamba_block_size=self._mamba_tokens_per_page,
                attn_group_idx=self._fa_group_idx,
                mamba_group_idxs=self._mamba_group_ids,
                # Tail (running-state) page for the load's boundary->tail
                # mirror; boundary = the loaded region's end (vllm cached
                # + external), same formula as the scheduled branch.
                mamba_tail_idx=(
                    (num_tokens_to_compute + self._mamba_tokens_per_page - 1)
                    // self._mamba_tokens_per_page
                    - 1
                    if (spec.can_load and self._mamba_group_idx is not None)
                    else None
                ),
            )
            if req_meta is not None:
                self._lmc_async_load_metas[request.request_id] = req_meta

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        # set and update lookup requests for unpin
        meta.lookup_requests_in_step = self._lookup_requests_in_step
        self._lookup_requests_in_step = []
        # llm-scaler v136: mamba boundary-lookup ids for the worker's unpin
        # mirror (deterministic "m-<req_id>").
        meta.mamba_lookup_ids = self._mamba_lookup_ids
        self._mamba_lookup_ids = []

        # llm-scaler v133 (async-gate fix): drain load metas staged by
        # update_state_after_alloc for requests sitting in
        # WAITING_FOR_REMOTE_KS — they are absent from scheduled_new_reqs
        # by design. The worker's no-forward step consumes them.
        if self._lmc_async_load_metas:
            for _rid, _meta in self._lmc_async_load_metas.items():
                meta.add_request(_meta)
                logger.info(
                    "LMC async load meta staged for req %s (no-forward step)",
                    _rid,
                )
            self._lmc_async_load_metas = {}

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
                mamba_group_idx=self._mamba_group_idx,
                mamba_block_size=self._mamba_tokens_per_page,
                attn_group_idx=self._fa_group_idx,
                mamba_group_idxs=self._mamba_group_ids,
                # llm-scaler LMC fix: tail (running-state) page for the
                # load's boundary->tail mirror. Only loads need it; index
                # mirrors preprocess_mamba's curr_state_idx (spec cancels).
                mamba_tail_idx=(
                    (num_tokens_to_compute + self._mamba_tokens_per_page - 1)
                    // self._mamba_tokens_per_page
                    - 1
                    if (
                        load_spec is not None
                        and load_spec.can_load
                        and self._mamba_group_idx is not None
                    )
                    else None
                ),
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                request_tracker = self._request_trackers[req.req_id]
                request_tracker.update(req.new_token_ids, req.new_block_ids)

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=None,
                    discard_partial_chunks=self._discard_partial_chunks,
                    mamba_group_idx=self._mamba_group_idx,
                    mamba_block_size=self._mamba_tokens_per_page,
                    attn_group_idx=self._fa_group_idx,
                    mamba_group_idxs=self._mamba_group_ids,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if cached_request := self._unfinished_requests.get(req_id):
                num_current_tokens = len(request_tracker.token_ids)
                new_token_ids = cached_request.all_token_ids[
                    num_current_tokens : num_current_tokens + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            request_tracker.update(new_token_ids, new_block_ids)

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=None,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
                mamba_group_idx=self._mamba_group_idx,
                mamba_block_size=self._mamba_tokens_per_page,
                attn_group_idx=self._fa_group_idx,
                mamba_group_idxs=self._mamba_group_ids,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        # TEMP DEBUG (LWSDbg): store-path forensics, remove after diagnosis.
        logger.info(
            "LWSDbg build_meta: new=%d cached=%d meta_reqs=%d",
            len(scheduler_output.scheduled_new_reqs),
            len(cached_reqs.req_ids),
            len(meta.requests),
        )
        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        return False, return_params
