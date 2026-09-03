"""llm-scaler v20 (Phase A2): env-gated spec step segment timing.

VLLM_SPEC_TIMING=1 enables. Hot-path overhead when disabled is one function
call returning a shared null context. When enabled, each named segment gets:
  - host wall time (perf_counter, accumulated per step)
  - device time (torch.xpu.Event pairs, pooled in a ring of FLUSH slots;
    read ONLY at flush with a single stream synchronize every FLUSH steps,
    so no per-step sync is introduced).

Segments (see gpu_model_runner.py / dflash.py / llm_base_proposer.py hooks):
  step_wall  - wall between successive spec propose entries (= 1000/steps-per-s)
  tforward   - target model forward (the verify step replay for spec decode)
  tlogits    - target compute_logits over the k+1 verify rows
  propose    - drafter.propose total (host wall; includes all glue)
  precompute - dflash context-KV precompute (fused KV GEMM + loops)
  dforward   - drafter model forward under set_forward_context
  greedy     - dflash block head + markov sampling loop

Flush log line (logger.info, also greppable as "SPECTIMING"):
  SPECTIMING steps=200 step_wall_ms=95.1 | tforward h=5.2 d=30.5 | ...
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

_ENABLED = os.getenv("VLLM_SPEC_TIMING", "0") == "1"
_FLUSH = int(os.getenv("VLLM_SPEC_TIMING_FLUSH", "200"))


class _NullCtx:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NULL_CTX = _NullCtx()

if not _ENABLED:
    # Disabled: zero-cost path.
    def spec_seg(name: str):
        return _NULL_CTX

    def spec_step() -> None:
        pass

else:
    import torch

    from vllm.logger import init_logger

    _logger = init_logger(__name__)

    class _SpecTimer:
        def __init__(self) -> None:
            self.flush = _FLUSH
            self.n = 0
            self.names: list[str] = []
            self.evs: dict[str, list] = {}
            self.host_acc: dict[str, float] = {}
            self.dev_mean: dict[str, float] = {}
            self.host_mean: dict[str, float] = {}
            self._last_step_t = time.perf_counter()
            self.step_wall_acc = 0.0
            self.step_wall_mean = 0.0

        def _register(self, name: str) -> None:
            if name in self.evs:
                return
            self.names.append(name)
            self.evs[name] = [
                (
                    torch.xpu.Event(enable_timing=True),
                    torch.xpu.Event(enable_timing=True),
                )
                for _ in range(self.flush)
            ]
            self.host_acc[name] = 0.0
            self.dev_mean[name] = 0.0
            self.host_mean[name] = 0.0

        @contextmanager
        def seg(self, name: str):
            self._register(name)
            slot = self.n % self.flush
            ev_start, ev_end = self.evs[name][slot]
            t0 = time.perf_counter()
            ev_start.record()
            try:
                yield
            finally:
                ev_end.record()
                self.host_acc[name] += (time.perf_counter() - t0) * 1e3

        def step(self) -> None:
            now = time.perf_counter()
            self.step_wall_acc += (now - self._last_step_t) * 1e3
            self._last_step_t = now
            self.n += 1
            if self.n % self.flush == 0:
                self._flush()

        def _flush(self) -> None:
            torch.xpu.synchronize()
            for name in self.names:
                total = 0.0
                for ev_start, ev_end in self.evs[name]:
                    try:
                        total += ev_start.elapsed_time(ev_end)
                    except Exception:
                        pass
                self.dev_mean[name] = total / self.flush
                self.host_mean[name] = self.host_acc[name] / self.flush
                self.host_acc[name] = 0.0
            self.step_wall_mean = self.step_wall_acc / self.flush
            self.step_wall_acc = 0.0
            parts = [f"steps={self.flush}", f"step_wall_ms={self.step_wall_mean:.1f}"]
            for name in self.names:
                parts.append(
                    f"{name} h={self.host_mean[name]:.1f} d={self.dev_mean[name]:.1f}"
                )
            _logger.info("SPECTIMING %s", " | ".join(parts))

    _TIMER = _SpecTimer()

    def spec_seg(name: str):
        return _TIMER.seg(name)

    def spec_step() -> None:
        _TIMER.step()
