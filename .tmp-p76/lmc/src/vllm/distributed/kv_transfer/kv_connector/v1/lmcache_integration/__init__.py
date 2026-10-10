# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from . import vllm_v1_adapter

try:
    from .multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        LoadStoreOp,
        ParallelStrategy,
    )
except ImportError:
    # The multi-process adapter targets unreleased lmcache main APIs
    # (e.g. CudaIPCWrapper); with released lmcache (0.5.5) only the
    # in-process LMCacheConnectorV1 path is supported. LMCacheMPConnector
    # will fail at its own import with a clear error if requested.
    LMCacheMPSchedulerAdapter = None
    LMCacheMPWorkerAdapter = None
    LoadStoreOp = None
    ParallelStrategy = None

__all__ = [
    "vllm_v1_adapter",
    "multi_process_adapter",
    "LMCacheMPSchedulerAdapter",
    "LMCacheMPWorkerAdapter",
    "LoadStoreOp",
    "ParallelStrategy",
]
