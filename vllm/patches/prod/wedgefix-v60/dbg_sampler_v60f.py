#!/usr/bin/env python3
"""llm-scaler v60f TEMP debug — log the logits-row state ENTERING the
custom XPU sampling op, plus k/p, first N calls.

Pinned so far (2026-09-21):
  - only all--inf rows make the op return 0 (offline repro3);
  - chat_template_kwargs={"enable_thinking": False} + json_schema ->
    deterministic 500 (A/B/C/D discriminator probe);
  - thinking_budget_state._apply_forcing_to_logits bumps
    logits[row, think_end_id]=1e9 (would sample </think>, NOT 0).
Gap: what exactly does the op see? This logs per-call row stats:
  nrows, k, p, per-row (isinf_all, isnan_any, min, max, argmax)
TEMP ONLY — reverted before bake. Gate: VLLM_V60F_DBG=1, first 40 calls.
"""
import py_compile
import shutil
import sys

SP = "/opt/venv/lib/python3.12/site-packages/vllm"
F = f"{SP}/v1/sample/ops/topk_topp_sampler.py"

OLD = """\
        torch.ops.vllm.xpu_topk_topp_sampler(
            random_sampled, logits_to_return, logits, k, p, self.logprobs_mode, seeds
        )
"""

NEW = """\
        # llm-scaler v60f TEMP debug: row state entering the op
        import os as _os60f
        if _os60f.environ.get("VLLM_V60F_DBG", "1") == "1":
            self._v60f_n = getattr(self, "_v60f_n", 0) + 1
            if self._v60f_n <= 40:
                try:
                    _inf_all = torch.isinf(logits).all(dim=-1)
                    _nan_any = torch.isnan(logits).any(dim=-1)
                    _mn = logits.min(dim=-1).values
                    _mx = logits.max(dim=-1).values
                    _am = logits.argmax(dim=-1)
                    logger.warning(
                        "llm-scaler v60f PREOP #%d rows=%d k=%s p=%s "
                        "inf_all=%s nan_any=%s min=%s max=%s amax=%s "
                        "seeds=%s",
                        self._v60f_n, int(logits.shape[0]),
                        (k.tolist() if k is not None else None),
                        (p.tolist() if p is not None else None),
                        _inf_all.tolist(), _nan_any.tolist(),
                        [round(float(x), 1) for x in _mn.tolist()[:8]],
                        [round(float(x), 1) for x in _mx.tolist()[:8]],
                        _am.tolist()[:8], seeds.tolist())
                except Exception as _e60f:
                    logger.warning("llm-scaler v60f PREOP-EXC %s", _e60f)
        torch.ops.vllm.xpu_topk_topp_sampler(
            random_sampled, logits_to_return, logits, k, p, self.logprobs_mode, seeds
        )
        # llm-scaler v60f TEMP debug: what came out
        if _os60f.environ.get("VLLM_V60F_DBG", "1") == "1":
            if getattr(self, "_v60f_n", 0) <= 40:
                try:
                    logger.warning(
                        "llm-scaler v60f POSTOP #%d out=%s",
                        self._v60f_n, random_sampled.tolist()[:8])
                except Exception:
                    pass
"""


def main():
    src = open(F, encoding="utf-8").read()
    if "v60f" in src:
        print("[v60f] ALREADY-APPLIED")
        return
    n = src.count(OLD)
    if n != 1:
        print(f"[v60f] ANCHOR-COUNT {n} (need 1) — SKIP")
        sys.exit(1)
    bak = F + ".v60fbak"
    if not shutil.os.path.exists(bak):
        shutil.copyfile(F, bak)
    with open(F, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    try:
        py_compile.compile(F, doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copyfile(bak, F)
        print(f"[v60f] COMPILE-FAIL restored: {e}")
        sys.exit(1)
    print("[v60f] APPLIED")


if __name__ == "__main__":
    main()
