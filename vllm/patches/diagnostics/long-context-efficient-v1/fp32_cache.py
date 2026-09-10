"""Inference-only immutable-value snapshots; cache invalidates on tensor mutation.

Preserves conversion of the *loaded* BF16/FP16 values, not checkpoint values
before rounding. No collectives, host tensor copies, or numeric approximations.
Unversioned inference tensors and trainable parameters safely use the old path.
"""
import torch


def fp32_snapshot(owner, name, source):
    if source.requires_grad or source.dtype == torch.float32:
        return source.float()
    try:
        version = source._version
    except RuntimeError:
        # Inference tensors may have no mutation counter: don't cache them.
        return source.float()
    metadata = (version, source.device, source.dtype, tuple(source.shape),
                tuple(source.stride()), source.data_ptr())
    cache = getattr(owner, '_lce1_fp32_snapshots', None)
    if cache is None:
        cache = {}
        owner._lce1_fp32_snapshots = cache
    entry = cache.get(name)
    if entry is None or entry[0] is not source or entry[1] != metadata:
        converted = source.float()
        cache[name] = (source, metadata, converted)
        return converted
    return entry[2]


def clear_fp32_snapshots(owner):
    """Explicit invalidation for model unload / graph rebuild lifecycles."""
    getattr(owner, '_lce1_fp32_snapshots', {}).clear()
