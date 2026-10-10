#!/usr/bin/env python3
"""llm-scaler P83b: VIA-POOL -- kill the TP allreduce shape-ratchet.

SHAPE-RATCHET LAW (measured, P83 legs 1-2): the via-allgather all-reduce
path cached its buffers per EXACT tensor shape --

  _xpu_allgather_allreduce: gathered = [empty_like(output)] * world_size,
  keyed (device, dtype, tuple(shape)); never evicted
  _all_reduce_impl_direct: per-shape stable output dict (v29, rows<=512)

Ragged chunked-prefill shapes differ at every chunk boundary, so a deep
storm mints dozens of distinct shapes and pins (world_size+1) x shape
tensors forever. Argsnap curve (leg 2, rank 537): allocated climbed
monotonically 23.2 -> 29.9 GB over ~58k GDN calls in 8 minutes -- pure
retention, the dominant driver of the P82 "reserved 19->35.5 GB within one
storm" climb and of the death-time 142 MiB first-seen-shape allocation
(both legs died on exactly 142.00 MiB at reserved==cap).

Fix: ONE buffer set per (device, dtype, hidden), sized to the row-max seen
so far, every call served as prefix-row views (contiguous, storage offset
0). Zero hot-path allocations after warmup; footprint bounded at
(1+world_size) x MAX_ROWS x hidden x dtype (~192 MiB at 8192x4096 fp16
TP2) instead of unbounded-per-shape. data_ptr guards let the pooled-output
caller alias the pool buffer directly while legacy callers stage in/out.

Knob: VLLM_XPU_VIA_POOL_MAX_ROWS (default 8192 = max_num_batched_tokens;
0 = fully inert, legacy per-shape path everywhere).

Sites (vllm/distributed/device_communicators/xpu_communicator.py):
  B: pool state + getter after the _XPU_VIA_STABLEBUF_MAX_ROWS constant
  A: pooled branch at the head of _xpu_allgather_allreduce (legacy body
     kept verbatim as the oversize/non-contiguous fallback)
  C: prefill rows in (STABLEBUF cap, pool cap] stage through the pool
     instead of clone-per-distinct-shape; decode path (<=512) untouched

Apply AFTER P80/P82/P83 patchers (disjoint files). Idempotent.

Usage (in container):
  /opt/venv/bin/python /tmp/patch_p83b_via_pool.py image
"""
import hashlib
import os
import py_compile
import sys

MARK = "llm-scaler P83b"

POOL_BLOCK = '''

# llm-scaler P83b: VIA-POOL (shape-ratchet kill). The via-allgather AR path
# cached its buffers per EXACT tensor shape; ragged chunked-prefill shapes
# (every chunk boundary differs) pinned world_size+1 tensors per distinct
# shape, never evicted -- measured +6 GB of LIVE allocator growth within one
# deep storm (argsnap a=23.2->29.9 GB monotone over 58k GDN calls, P83 legs
# 1-2), driving the P82 reserved 19->35.5 GB climb and the death-time
# 142 MiB first-seen-shape allocation. The pool keeps ONE buffer set per
# (device, dtype, hidden), sized to the row-max seen so far, and serves
# every call as a prefix-row view (contiguous, storage offset 0). Zero
# hot-path allocations after warmup; footprint bounded at
# (1+world_size) * MAX_ROWS * hidden * dtype. VLLM_XPU_VIA_POOL_MAX_ROWS
# bounds the pooled row range (default 8192 = max_num_batched_tokens;
# 0 = fully inert, legacy per-shape path).
_XPU_VIA_POOL_MAX_ROWS = int(
    os.environ.get("VLLM_XPU_VIA_POOL_MAX_ROWS", "8192") or 0
)  # llm-scaler P83b

_xpu_via_pool: dict = {}


def _xpu_via_pool_get(
    device, dtype, rows: int, hidden: int, world_size: int
) -> dict:
    ent = _xpu_via_pool.get((device, dtype, hidden))
    if (
        ent is not None
        and ent["rows"] >= rows
        and ent.get("world") == world_size
    ):
        return ent
    if ent is None:
        ent = {"rows": 0}
        _xpu_via_pool[(device, dtype, hidden)] = ent
    ent["out"] = torch.empty(rows, hidden, device=device, dtype=dtype)
    ent["g"] = [
        torch.empty(rows, hidden, device=device, dtype=dtype)
        for _ in range(world_size)
    ]
    ent["rows"] = rows
    ent["world"] = world_size
    _total_mb = (
        (ent["out"].numel() + sum(g.numel() for g in ent["g"]))
        * ent["out"].element_size()
    ) >> 20
    logger.info(
        "P83b via-pool grow: rows=%d hidden=%d world=%d total=%d MB",
        rows, hidden, world_size, _total_mb,
    )  # llm-scaler P83b
    return ent
'''

GATHER_OLD = '''    key = (output.device, output.dtype, tuple(output.shape))
    gathered = buffer_cache.get(key)
    if gathered is None:
        gathered = [torch.empty_like(output) for _ in range(world_size)]
        buffer_cache[key] = gathered

    dist.all_gather(gathered, output, group=device_group)
    output.copy_(gathered[0])
    for tensor in gathered[1:]:
        output.add_(tensor)
'''

GATHER_NEW = '''    # llm-scaler P83b: pooled gathered buffers (SHAPE-RATCHET LAW -- the
    # per-exact-shape cache below pinned world_size+1 tensors per distinct
    # ragged chunk shape and never evicted: +6 GB live growth in one deep
    # storm, the P82 reserved-climb driver). Prefix-row views are
    # contiguous at storage offset 0; the data_ptr guards let a
    # pooled-output caller alias the pool buffer directly (no staging
    # copies) while legacy callers stage in/out. Oversize or ragged inputs
    # keep the legacy path below.
    if (
        _XPU_VIA_POOL_MAX_ROWS > 0
        and output.dim() == 2
        and output.is_contiguous()
        and 0 < output.shape[0] <= _XPU_VIA_POOL_MAX_ROWS
    ):
        _ent = _xpu_via_pool_get(
            output.device, output.dtype, output.shape[0], output.shape[1],
            world_size,
        )
        _oview = _ent["out"][: output.shape[0]]
        _gviews = [g[: output.shape[0]] for g in _ent["g"]]
        if output.data_ptr() != _oview.data_ptr():
            _oview.copy_(output)
        dist.all_gather(_gviews, _oview, group=device_group)
        _oview.copy_(_gviews[0])
        for _gt in _gviews[1:]:
            _oview.add_(_gt)
        if output.data_ptr() != _oview.data_ptr():
            output.copy_(_oview)
        return
    key = (output.device, output.dtype, tuple(output.shape))
    gathered = buffer_cache.get(key)
    if gathered is None:
        gathered = [torch.empty_like(output) for _ in range(world_size)]
        buffer_cache[key] = gathered

    dist.all_gather(gathered, output, group=device_group)
    output.copy_(gathered[0])
    for tensor in gathered[1:]:
        output.add_(tensor)
'''

STABLE_OLD = '''            rows = input_.shape[0] if input_.dim() >= 1 else 0
            use_stable = (
                _env_enabled("VLLM_XPU_VIA_STABLEBUF", True)
                and input_.is_contiguous()
                and 0 < rows <= _XPU_VIA_STABLEBUF_MAX_ROWS
            )
            if use_stable:
                key = (input_.device, input_.dtype, tuple(input_.shape))
                output = self._xpu_via_out_buffers.get(key)
                if output is None:
                    output = torch.empty_like(input_)
                    self._xpu_via_out_buffers[key] = output
                output.copy_(input_)
            else:
                output = input_.clone()
'''

STABLE_NEW = '''            rows = input_.shape[0] if input_.dim() >= 1 else 0
            use_stable = (
                _env_enabled("VLLM_XPU_VIA_STABLEBUF", True)
                and input_.is_contiguous()
                and 0 < rows <= _XPU_VIA_STABLEBUF_MAX_ROWS
            )
            if use_stable:
                key = (input_.device, input_.dtype, tuple(input_.shape))
                output = self._xpu_via_out_buffers.get(key)
                if output is None:
                    output = torch.empty_like(input_)
                    self._xpu_via_out_buffers[key] = output
                output.copy_(input_)
            elif (
                _env_enabled("VLLM_XPU_VIA_STABLEBUF", True)
                and _XPU_VIA_POOL_MAX_ROWS > 0
                and input_.dim() == 2
                and input_.is_contiguous()
                and 0 < rows <= _XPU_VIA_POOL_MAX_ROWS
            ):
                # llm-scaler P83b: prefill rows between the v29 stable-buf
                # cap and the pool cap stage through the shared pool buffer
                # instead of clone-per-distinct-shape -- the output-side
                # half of the SHAPE-RATCHET LAW. Stream-ordered reuse is
                # the v29 pattern (same-stream enqueue ordering).
                _ent = _xpu_via_pool_get(
                    input_.device, input_.dtype, rows, input_.shape[1],
                    self.world_size,
                )
                output = _ent["out"][:rows]
                output.copy_(input_)
            else:
                output = input_.clone()
'''

# unique anchors, verified count==1 at patch time
POOL_ANCHOR = (
    "_XPU_VIA_STABLEBUF_MAX_ROWS = int(\n"
    '    os.environ.get("VLLM_XPU_VIA_STABLEBUF_MAX_ROWS", "512")\n'
    ")\n"
)
GATHER_ANCHOR = GATHER_OLD
STABLE_ANCHOR = STABLE_OLD


def _ins_after(src, anchor, block, tag):
    n = src.count(anchor)
    assert n == 1, "%s: anchor count %d != 1" % (tag, n)
    return src.replace(anchor, anchor + block, 1)


def _replace(src, old, new, tag):
    n = src.count(old)
    assert n == 1, "%s: anchor count %d != 1" % (tag, n)
    return src.replace(old, new, 1)


def _patch_ins(path, anchor, block, tag):
    src = open(path).read()
    if MARK in src:
        print("SKIP(already) %s" % path)
        return
    src = _ins_after(src, anchor, block, tag)
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True, cfile="/tmp/p83bc_%s.pyc" % tag)
    n = src.count(MARK)
    print("PATCHED %s md5 %s markers=%d"
          % (path, hashlib.md5(open(path, "rb").read()).hexdigest(), n))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode != "image":
        print("usage: patch_p83b_via_pool.py image")
        sys.exit(2)
    path = os.path.join(
        "/opt/venv/lib/python3.12/site-packages/vllm",
        "distributed/device_communicators/xpu_communicator.py",
    )
    src = open(path).read()
    if MARK in src:
        print("SKIP(already) %s" % path)
        return
    src = _ins_after(src, POOL_ANCHOR, POOL_BLOCK, "pool_state")
    src = _replace(src, GATHER_ANCHOR, GATHER_NEW, "gather_pool")
    src = _replace(src, STABLE_ANCHOR, STABLE_NEW, "stable_pool")
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True, cfile="/tmp/p83bc_all.pyc")
    n = src.count(MARK)
    print("PATCHED %s md5 %s markers=%d"
          % (path, hashlib.md5(open(path, "rb").read()).hexdigest(), n))


if __name__ == "__main__":
    main()
