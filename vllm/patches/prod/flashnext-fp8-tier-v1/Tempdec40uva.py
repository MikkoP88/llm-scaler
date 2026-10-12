# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Generator

import torch
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

# FN8_02B (flashnext-fp8-tier-v1 P2): tiered-offload support.  Same
# master gate as patch_02a; inert unless FN8_TIERED=1.
import os as _fn8_os

_FN8_TIERED = _fn8_os.environ.get("FN8_TIERED", "") == "1"
_FN8_TIER_OFFLOAD_PARAMS = {
    seg for seg in _fn8_os.environ.get(
        "FN8_TIER_OFFLOAD_PARAMS",
        "w13_weight,w2_weight,w13_weight_scale_inv,w2_weight_scale_inv",
    ).split(",") if seg
}


class UVAOffloader(BaseOffloader):
    """Offloader using Unified Virtual Addressing (UVA) for zero-copy access.

    This offloader moves parameters to pinned CPU memory and creates CUDA views
    using UVA. The GPU can then directly access the CPU memory without explicit
    transfers, at the cost of PCIe bandwidth (slower than GPU memory).

    When UVA is disabled via env var, falls back to a functional_call-based
    approach that moves parameters on-demand.

    Args:
        cpu_offload_max_bytes: Maximum bytes to offload to CPU.
        cpu_offload_params: Set of parameter name segments to selectively
            offload. If empty, all parameters are eligible up to the byte limit.
    """

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
    ):
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0
        self.cpu_offload_params = cpu_offload_params or set()
        if _FN8_TIERED and not self.cpu_offload_params:
            # FN8_02B: page ONLY routed-expert tensors; the tiered store
            # (patch_02a) owns their placement from the first apply on.
            self.cpu_offload_params = set(_FN8_TIER_OFFLOAD_PARAMS)

        self.pin_memory = should_pin_memory()
        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading."""
        modules = [self._maybe_offload_to_cpu(module) for module in modules_generator]
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return modules

    def _maybe_offload_to_cpu(self, module: nn.Module) -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            if self.cpu_offload_params:
                # Check if parameter belongs to the offloading set
                # Add dots here to ensure we match full segments only
                # e.g., "experts.w2_weight" matches "mlp.experts.w2_weight"
                # but not "mlp.experts.w2_weight_scale"
                should_offload = any(
                    f".{param}." in f".{name}." for param in self.cpu_offload_params
                )
                if not should_offload:
                    continue

            cpu_data = p.data.to(device="cpu")
            if self.pin_memory:
                cpu_data = cpu_data.pin_memory()

            if not self.uva_offloading:
                p.data = cpu_data
            else:
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True

            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        # FN8_02B: modules whose subtree contains a whitelisted routed-expert
        # tensor never get the per-step wholesale-copy closure under tiering
        # — the store does row-granular placement instead.  Granularity law:
        # wrap_modules receives whole decoder layers (get_layers layer_fn
        # generator), NOT FusedMoE leaves — detection must be containment
        # over recursive named_parameters, not direct attributes (the
        # hasattr form missed the layer and its state_dict materialized the
        # 47.69 GiB CPU-side ngram table at the dummy run = TP1 boot OOM).
        _fn8_owns_moe = (
            hasattr(module, "w13_weight")
            and hasattr(module, "w2_weight")
        )
        _fn8_no_wrap = (
            _FN8_TIERED
            and any(
                any(f".{seg}." in f".{n}." for seg in _FN8_TIER_OFFLOAD_PARAMS)
                for n, _ in module.named_parameters()
            )
        )
        if (_fn8_owns_moe and offloaded_parameters):
            module._fn8_tier_no_wrap = True
        if (offloaded_parameters and not self.uva_offloading
                and not _fn8_no_wrap):
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device,
                    # it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }

                # set `tie_weights=False` as tied weights in original model
                # become untied when calling .to(device) individually
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    tie_weights=False,
                )
                # FN8_00D (flashnext-fp8-tier-v1 P0): 00c law extended to
                # the forward path — each wrapped module's H2D copies hoard in
                # the XPU caching allocator (expandable_segments never returns
                # freed blocks); across layers the segment growth stalls in
                # xe_vm_bind_ioctl (P82 family; leg-5 profile wedge 2026-10-11:
                # D-state wchan xe_vm_bind_ioctl, xpu-smi/py-spy hung).  Empty
                # the cache per module boundary so every layer's copies start
                # from actually-free VRAM.
                try:
                    torch.xpu.empty_cache()
                except Exception:
                    pass
                module.forward = forward
                return output

            module.forward = forward

        return module
