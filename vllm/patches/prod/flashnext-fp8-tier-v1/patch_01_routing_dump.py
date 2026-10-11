#!/usr/bin/env python3
"""flashnext-fp8-tier-v1 patch_01 — routing histogram tap (P1 profiler).

Makes the fork's routed-experts capturer usable on XPU and adds an env-gated
per-(layer, expert) activation histogram:

1. XPU D2H fallback: `_RoutedExpertsCapturerReal` builds its async pipeline
   on ``torch.cuda.Stream``/``torch.cuda.Event`` — unavailable on XPU.  When
   creation fails, the capturer now falls back to a synchronous D2H into the
   existing pinned staging buffer + immediate scatter.  Slower (one sync copy
   per step), but this is a profiling mode: correctness over overlap.
2. Histogram tap: when ``V1X_ROUTING_DUMP=<file>`` is set, every scatter
   aggregates its (N, L, K) int16 routing ids into per-(layer, expert)
   activation counts and flushes them to JSON every 500 scatters and at
   atexit (atomic tmp+rename).  Disabled by default; one env read at import
   and a None check per scatter otherwise.

Profiling-mode companions (no patch needed, verified 2026-10-11):
- serve with ``--enable-return-routed-experts`` (EngineArgs field exists;
  initializes the capturer + binds buffers to all FusedMoE layers).
- ``DISABLE_ESIMD_MOE=1`` — the ESIMD fused MoE paths (moe_forward_full*,
  up to 12 tokens for the fp8-block variant) do topk INSIDE the kernel and
  bypass moe_runner, so the capture buffer is never written on the decode
  fast path.  esimd_utils documents this env as restoring the upstream
  modular path 1:1 (toggle-equivalence acceptance basis) — the modular path
  routes through ``_apply_quant_method`` which writes the buffer.

Idempotent (FN8_01 marker), backs up .q4dbak, --restore undoes.
"""
import sys

TARGET = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/"
          "layers/fused_moe/routed_experts_capturer.py")
MARKER = "FN8_01"

ANCHOR_IMPORTS = (
    "import contextlib\n"
    "import logging\n"
)
NEW_IMPORTS = (
    "import atexit\n"
    "import contextlib\n"
    "import json\n"
    "import logging\n"
    "import os\n"
)

ANCHOR_MB = "_MB = 1024 * 1024\n"

HISTOGRAM_BLOCK = '''
# ---------------------------------------------------------------------------
# FN8_01 (flashnext-fp8-tier-v1 P1): env-gated routing histogram tap.
# V1X_ROUTING_DUMP=<file> enables: every _scatter_to_host() aggregates the
# (N, L, K) int16 routing ids into per-(layer, expert) activation counts and
# periodically flushes them as JSON (atomic tmp+rename).  Off by default;
# one env read at import and a None check per scatter when disabled.
# ---------------------------------------------------------------------------

_V1X_DUMP_PATH = os.environ.get("V1X_ROUTING_DUMP", "")
_V1X_NUM_EXPERTS = int(os.environ.get("V1X_ROUTING_EXPERTS", "512"))


class _V1XRoutingHistogram:
    FLUSH_EVERY = 500  # scatters between flushes

    def __init__(self, path: str) -> None:
        self.path = path
        self.counts = None
        self.layer_tokens = None
        self.scatters = 0
        self.tapped_tokens = 0

    def tap(self, host_values: np.ndarray) -> None:
        n_tokens, n_layers, _k = host_values.shape
        if self.counts is None:
            self.counts = np.zeros(
                (n_layers, _V1X_NUM_EXPERTS), dtype=np.int64
            )
            self.layer_tokens = np.zeros(n_layers, dtype=np.int64)
        n_experts = self.counts.shape[1]
        if host_values.size:
            max_id = int(host_values.max())
            if max_id >= n_experts:
                grown = max_id + 1
                self.counts = np.pad(
                    self.counts, ((0, 0), (0, grown - n_experts))
                )
                n_experts = grown
        for l in range(n_layers):
            ids = host_values[:, l, :].ravel().astype(np.int64, copy=False)
            bc = np.bincount(ids, minlength=n_experts)
            self.counts[l] += bc
        self.layer_tokens += n_tokens
        self.scatters += 1
        self.tapped_tokens += n_tokens
        if self.scatters % self.FLUSH_EVERY == 0:
            self.flush()

    def flush(self) -> None:
        if self.counts is None:
            return
        payload = {
            "v": 1,
            "scatters": self.scatters,
            "tapped_tokens": self.tapped_tokens,
            "num_layers": int(self.counts.shape[0]),
            "num_experts": int(self.counts.shape[1]),
            "layer_tokens": self.layer_tokens.tolist(),
            "counts": self.counts.tolist(),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self.path)


_V1X_HIST = _V1XRoutingHistogram(_V1X_DUMP_PATH) if _V1X_DUMP_PATH else None
if _V1X_HIST is not None:
    atexit.register(_V1X_HIST.flush)


def _v1x_tap_routing_histogram(host_values: np.ndarray) -> None:
    if _V1X_HIST is not None:
        try:
            _V1X_HIST.tap(host_values)
        except Exception:
            # A profiling tap must never take down the engine.
            pass

''' + ANCHOR_MB

ANCHOR_INIT_STREAM = (
    "            self._device_staging = torch.empty_like(self.device_cache.buffer)\n"
    "            self._copy_stream = torch.cuda.Stream(device=device)\n"
    "            self._copy_event = torch.cuda.Event()\n"
)
NEW_INIT_STREAM = (
    "            self._device_staging = torch.empty_like(self.device_cache.buffer)\n"
    "            # FN8_01: XPU has no torch.cuda streams.  Fall back to a\n"
    "            # synchronous D2H + immediate scatter in\n"
    "            # sync_fwd_experts_buffer_DtoH (profiling mode: correctness\n"
    "            # over async overlap).\n"
    "            try:\n"
    "                self._copy_stream = torch.cuda.Stream(device=device)\n"
    "                self._copy_event = torch.cuda.Event()\n"
    "            except Exception:\n"
    "                self._copy_stream = None\n"
    "                self._copy_event = None\n"
)

ANCHOR_SYNC_HEAD = (
    "        if self.host_cache is None:\n"
    "            return\n"
    "\n"
    "        # 1. Finalize previous async copy -- the copy had an entire\n"
)
NEW_SYNC_HEAD = (
    "        if self.host_cache is None:\n"
    "            return\n"
    "\n"
    "        # FN8_01: synchronous XPU fallback (no torch.cuda stream).\n"
    "        # One blocking D2H into the pinned staging buffer, then the\n"
    "        # regular scatter path (which also feeds the histogram tap).\n"
    "        if self._copy_stream is None:\n"
    "            total_tokens = sum(num_scheduled_tokens.values())\n"
    "            if total_tokens == 0:\n"
    "                return\n"
    "            self._pinned_staging[:, :total_tokens, :].copy_(\n"
    "                self.device_cache.buffer[:, :total_tokens, :]\n"
    "            )\n"
    "            self._pending_positions = positions.numpy().copy()\n"
    "            self._pending_num_scheduled = num_scheduled_tokens\n"
    "            self._pending_total_tokens = total_tokens\n"
    "            self._scatter_to_host()\n"
    "            return\n"
    "\n"
    "        # 1. Finalize previous async copy -- the copy had an entire\n"
)

ANCHOR_SCATTER = (
    "        host_values = (\n"
    "            self._pinned_staging[:, : self._pending_total_tokens, :]\n"
    "            .numpy()\n"
    "            .transpose(1, 0, 2)\n"
    "        )\n"
)
NEW_SCATTER = ANCHOR_SCATTER + (
    "        # FN8_01: histogram tap (no-op unless V1X_ROUTING_DUMP is set)\n"
    "        _v1x_tap_routing_histogram(host_values)\n"
)


def md5(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        bak = TARGET + ".q4dbak"
        import shutil
        shutil.copyfile(bak, TARGET)
        print("FN8_01 RESTORED", TARGET, md5(TARGET))
        return

    with open(TARGET) as f:
        src = f.read()

    if MARKER in src:
        print("FN8_01 already applied, nothing to do", md5(TARGET))
        return

    for name, anchor, count_expected in (
        ("imports", ANCHOR_IMPORTS, 1),
        ("_MB", ANCHOR_MB, 1),
        ("init-stream", ANCHOR_INIT_STREAM, 1),
        ("sync-head", ANCHOR_SYNC_HEAD, 1),
        ("scatter", ANCHOR_SCATTER, 1),
    ):
        n = src.count(anchor)
        assert n == count_expected, f"anchor {name} count {n} != {count_expected}"

    import shutil
    shutil.copyfile(TARGET, TARGET + ".q4dbak")
    before = md5(TARGET)

    src = src.replace(ANCHOR_IMPORTS, NEW_IMPORTS)
    src = src.replace(ANCHOR_MB, HISTOGRAM_BLOCK)
    src = src.replace(ANCHOR_INIT_STREAM, NEW_INIT_STREAM)
    src = src.replace(ANCHOR_SYNC_HEAD, NEW_SYNC_HEAD)
    src = src.replace(ANCHOR_SCATTER, NEW_SCATTER)
    assert src.count(MARKER) >= 4, "marker missing after edit?!"

    with open(TARGET, "w") as f:
        f.write(src)
    print(f"FN8_01 backup written: {TARGET}.q4dbak (md5 {before})")
    print(f"FN8_01 PATCHED: {TARGET} md5 {before} -> {md5(TARGET)}")


if __name__ == "__main__":
    main()
