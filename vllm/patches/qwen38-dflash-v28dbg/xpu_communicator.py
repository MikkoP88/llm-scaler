# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from .base_device_communicator import DeviceCommunicatorBase

import fr as _fr  # llm-scaler v28dbg flight recorder (site-packages root)

logger = init_logger(__name__)
_xpu_graph_replay_seen = False
_xpu_allreduce_retry_seq = 0
_XPU_ALLREDUCE_RETRY_SHAPES = "25,5120;25,2048"
# The XPU all-reduce collective only returns NaN on large (prefill-sized)
# buffers. The repair guard arms on the reduce's token count (rows), which is
# model-agnostic: it spares the num_tokens==1 decode reduce on any hidden size
# while catching prefill/embedding reduces of >= this many tokens.
_XPU_ALLREDUCE_GUARD_MIN_ROWS = int(
    os.environ.get("VLLM_XPU_ALLREDUCE_GUARD_MIN_ROWS", "128")
)


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def mark_xpu_graph_replay_seen() -> None:
    global _xpu_graph_replay_seen
    _xpu_graph_replay_seen = True


def _xpu_retry_shape_allowed(tensor: torch.Tensor) -> bool:
    shape_spec = os.environ.get(
        "VLLM_XPU_ALLREDUCE_RETRY_SHAPE",
        _XPU_ALLREDUCE_RETRY_SHAPES,
    )
    if not shape_spec:
        return False
    shape = ",".join(str(dim) for dim in tensor.shape)
    return shape in {item.strip() for item in shape_spec.split(";")}


def _xpu_retry_context_allowed() -> tuple[bool, object | None, object | None]:
    if not is_forward_context_available():
        return False, None, None

    forward_context = get_forward_context()
    batch_descriptor = forward_context.batch_descriptor
    runtime_mode = forward_context.cudagraph_runtime_mode
    allowed = (
        batch_descriptor is not None
        and getattr(batch_descriptor, "num_tokens", 0) > 1
        and str(runtime_mode) == "NONE"
    )
    return allowed, batch_descriptor, runtime_mode


def _xpu_tensor_has_nan(tensor: torch.Tensor) -> bool:
    try:
        return bool(torch.isnan(tensor).any().item())
    except Exception:
        logger.exception("Failed to inspect XPU all_reduce tensor for NaN")
        return False


def _xpu_global_any_nan(
    local_nan: bool,
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> bool:
    flag = torch.empty((), device=tensor.device, dtype=torch.int32)
    flag.fill_(1 if local_nan else 0)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    return bool(flag.item())


def _xpu_allgather_allreduce(
    output: torch.Tensor,
    device_group: ProcessGroup,
    world_size: int,
    buffer_cache: dict[
        tuple[torch.device, torch.dtype, tuple[int, ...]], list[torch.Tensor]
    ],
) -> None:
    key = (output.device, output.dtype, tuple(output.shape))
    gathered = buffer_cache.get(key)
    if gathered is None:
        gathered = [torch.empty_like(output) for _ in range(world_size)]
        buffer_cache[key] = gathered

    dist.all_gather(gathered, output, group=device_group)
    output.copy_(gathered[0])
    for tensor in gathered[1:]:
        output.add_(tensor)


class XpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

            else:  # type: ignore[has-type]
                logger.warning(
                    "`%s` all2all manager is not supported on XPU. "
                    "Falling back to AgRs manager for XPU, "
                    "which is the Default backend",
                    self.all2all_backend,  # type: ignore[has-type]
                )
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")
        self._xpu_allgather_fallback_buffers: dict[
            tuple[torch.device, torch.dtype, tuple[int, ...]], list[torch.Tensor]
        ] = {}

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # llm-scaler v28dbg flight recorder: entry/exit around the whole
        # collective so a hang localizes to it. The is_compiling() guard is
        # load-bearing: dynamo traces through here during profile_run and
        # faults on the f-string tensor method (verified boot crash).
        if not torch.compiler.is_compiling():
            _fr.log(f"AR begin numel={input_.numel()} via_env={_env_enabled('VLLM_XPU_ALLREDUCE_VIA_ALLGATHER', False)}")
        try:
            return self._all_reduce_impl(input_)
        finally:
            if not torch.compiler.is_compiling():
                _fr.log("AR end")

    def _all_reduce_impl(self, input_: torch.Tensor) -> torch.Tensor:
        if os.environ.get("SKIP_ALL_REDUCE") == "1":
            return input_
        # ALLREDUCE_BOUNCE_FIX_v1: only bounce off the custom op when XPU
        # graphs are actually enabled. With graphs off, inductor lowers the
        # AR pieces with static memory planning and the op's aliased return
        # corrupts TP output under torch.compile (KNOWN_ISSUES #04; adv
        # images v4-v9, both dtypes). NOTE: the alias return is LOAD-BEARING
        # under PIECEWISE XPU graphs - the op runs eagerly between captured
        # pieces and must write the stable activation buffer, not a fresh
        # clone (a clone was tried in adv:v11 and corrupted graphs-on
        # serving; reverted).
        if torch.compiler.is_compiling() and os.environ.get(
            "VLLM_XPU_ENABLE_XPU_GRAPH", "0"
        ) not in ("", "0"):
            # Bounce the collective off the registered `vllm::all_reduce`
            # custom op while tracing, so dynamo emits a single fx node
            # instead of inlining this function. platforms/xpu.py registers
            # that op as a splitting op, which keeps the oneCCL collective
            # OUT of captured XPU graph pieces: a replayed oneCCL kernel can
            # wedge both TP ranks mid-decode (py-spy shows both workers
            # parked inside torch/xpu/graphs.py replay) and is also the
            # documented source of intermittent NaN all-reduce output. The
            # op dispatches back to this method (with is_compiling() False)
            # when the piecewise wrapper runs it eagerly between pieces.
            return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
        # VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 replaces the oneCCL *reduce*
        # kernel with allgather + local add (the same primitive the NaN
        # repair fallback below already uses, which has been stable). This is
        # a bisect lever + workaround for the mid-generation TP wedge: all
        # three observed hang modes (XPU-graph replay, eager event sync,
        # _to_list D2H sync) die at the first blocking queue touch after the
        # device queue stalls, and oneCCL all_reduce is the only TP=2 op that
        # runs ~80x/step across ranks with a documented history of both
        # intermittent NaN output and wedging when replayed from a graph.
        # Routing through allgather swaps the flaky reduce kernel out while
        # keeping the same numerics (sum). Costs an extra clone per call, so
        # it is opt-in.
        if (
            _env_enabled("VLLM_XPU_ALLREDUCE_VIA_ALLGATHER", False)
            and self.world_size > 1
            and not torch.xpu.is_current_stream_capturing()
        ):
            output = input_.clone()
            _xpu_allgather_allreduce(
                output,
                self.device_group,
                self.world_size,
                self._xpu_allgather_fallback_buffers,
            )
            return output
        global _xpu_allreduce_retry_seq
        output = input_
        retry_armed = False
        batch_descriptor = None
        runtime_mode = None
        # The XPU all-reduce collective can intermittently return NaN from finite
        # input on large (prefill-sized) buffers; the NaN then corrupts the
        # token's hidden state and cascades to garbage ("!!!!") decode output.
        # VLLM_XPU_ALLREDUCE_RETRY_ON_NAN gates the detect/retry/allgather-repair
        # guard. Previously arming *also* required `_xpu_graph_replay_seen`, which
        # is only set in XPU-graph mode, so under --enforce-eager the guard never
        # armed and these NaNs went unrepaired. That graph requirement is now
        # dropped. To avoid taxing every all-reduce (clone + NaN scan + an extra
        # scalar all-reduce) on the hot decode path, arming is still gated: a
        # tensor is armed only when it covers enough tokens to be at risk (row
        # gate), or the forward context marks it as a multi-token/prefill reduce,
        # or its shape is in the explicit retry list. The row gate is what covers
        # the pre-forward embedding all-reduce (batch_descriptor is None, so
        # context gating alone misses it) while sparing the num_tokens==1 decode
        # reduces on any hidden size.
        if (
            _env_enabled("VLLM_XPU_ALLREDUCE_RETRY_ON_NAN", False)
            and not torch.compiler.is_compiling()
            and output.is_floating_point()
            and output.numel() > 0
            and not torch.xpu.is_current_stream_capturing()
        ):
            context_allowed, batch_descriptor, runtime_mode = (
                _xpu_retry_context_allowed()
            )
            big = (
                output.dim() >= 1
                and output.shape[0] >= _XPU_ALLREDUCE_GUARD_MIN_ROWS
            )
            retry_armed = (
                big or context_allowed or _xpu_retry_shape_allowed(output)
            )
        retry_input = output.clone() if retry_armed else None

        dist.all_reduce(output, group=self.device_group)
        post_nan = _xpu_tensor_has_nan(output) if retry_armed else False
        global_post_nan = (
            _xpu_global_any_nan(post_nan, output, self.device_group)
            if retry_armed
            else False
        )
        retry_count = 0
        fallback_used = False

        if global_post_nan and retry_input is not None:
            max_retries = int(os.environ.get("VLLM_XPU_ALLREDUCE_RETRY_MAX", "3"))
            while global_post_nan and retry_count < max_retries:
                retry_count += 1
                output.copy_(retry_input)
                if _env_enabled("VLLM_XPU_ALLREDUCE_RETRY_SYNC", True):
                    torch.xpu.synchronize()
                dist.all_reduce(output, group=self.device_group)
                post_nan = _xpu_tensor_has_nan(output)
                global_post_nan = _xpu_global_any_nan(
                    post_nan, output, self.device_group)

            if (
                global_post_nan
                and _env_enabled(
                    "VLLM_XPU_ALLREDUCE_FALLBACK_AFTER_RETRY", True)
            ):
                output.copy_(retry_input)
                if _env_enabled("VLLM_XPU_ALLREDUCE_RETRY_SYNC", True):
                    torch.xpu.synchronize()
                _xpu_allgather_allreduce(
                    output,
                    self.device_group,
                    self.world_size,
                    self._xpu_allgather_fallback_buffers,
                )
                fallback_used = True
                post_nan = _xpu_tensor_has_nan(output)
                global_post_nan = _xpu_global_any_nan(
                    post_nan, output, self.device_group)

            _xpu_allreduce_retry_seq += 1
            logger.error(
                "XPU_ALLREDUCE_RETRY seq=%d retries=%d fallback_used=%s "
                "local_final_post_nan=%s global_final_post_nan=%s "
                "batch_descriptor=%s runtime_mode=%s shape=%s",
                _xpu_allreduce_retry_seq,
                retry_count,
                fallback_used,
                post_nan,
                global_post_nan,
                batch_descriptor,
                runtime_mode,
                tuple(output.shape),
            )
        return output

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        if sizes is not None and sizes.count(sizes[0]) != len(sizes):
            # if inputs shape in different ranks is not the same using reduce_scatter
            input_splits = list(input_tensor.split(sizes, dim=0))
            dist.reduce_scatter(output, input_splits, group=self.device_group)
        else:
            dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if not torch.compiler.is_compiling():
            _fr.log(f"AG begin dim={dim} sizes={sizes}")  # llm-scaler v28dbg
        try:
            return self._all_gatherv_impl(input_, dim, sizes)
        finally:
            if not torch.compiler.is_compiling():
                _fr.log("AG end")

    def _all_gatherv_impl(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )

            if sizes is not None:
                all_gather_list = []
                for size in sizes:
                    all_gather_list.append(
                        torch.empty(
                            (size,) + input_.shape[1:],
                            dtype=input_.dtype,
                            device=input_.device,
                        )
                    )
                dist.all_gather(all_gather_list, input_, group=self.device_group)
                output_tensor = torch.cat(all_gather_list, dim=0)
            else:
                dist.all_gather([output_tensor], input_, group=self.device_group)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        return output_list

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (self.world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        if self.rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> None:
        dist.broadcast(input_, src=src, group=self.device_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )
