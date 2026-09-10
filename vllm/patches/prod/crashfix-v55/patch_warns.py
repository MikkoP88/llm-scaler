#!/usr/bin/env python3
"""patch_warns.py — llm-scaler v55 (crashfix-v55), patch 2 of 2.

Warning hygiene so a healthy certified boot logs ZERO warnings:

1. envs.py `validate_environ`: this deployment's XPU/llm-scaler patch
   knobs are read via direct os.environ sites (not registered as formal
   envs), so every prod boot warned "Unknown vLLM environment variable"
   for e.g. VLLM_XPU_FP8_MQ / VLLM_XPU_ALLOW_E5M2_FP8_CKPT (baked into
   the prod images) — the certified v1.2.7 boot log shows exactly 3 such
   warnings. Fix: an explicit frozenset allowlist (2026-09-10 census of
   os.environ/os.getenv read sites vs the envs.py registry). Typos and
   dead vars still warn — e.g. VLLM_ALLOW_LONG_MODEL_LEN, which is read
   NOWHERE and is being dropped from the boot scripts.

2. scheduler.py v52l BOUNDARY-CLAMP / BOUNDARY-SPAN-CORNER /
   BOUNDARY-SKIP and mamba_utils.py v52f PRE-COPY / DEF-PP logs: these
   fire on ROUTINE self-correcting mamba block-boundary machinery
   (once per ~2048 tokens per request) and flooded the log at WARNING
   level during normal multi-stream traffic. Demoted to DEBUG (full
   text retained). The v52c DISCARD-GAP probe stays at WARNING and the
   v55 GAP-FENCE error stays — those are real anomaly signals.

Patcher protocol: exact anchors / counted regex hits, py_compile gate,
idempotent.
"""

from __future__ import annotations

import argparse
import py_compile
import re
import sys
from pathlib import Path

# Census 2026-09-10 (v1.2.7 image): grep -rhoE '(os\.environ(\.get)?[?(]|'
# 'os\.getenv()["'']VLLM_...' over site-packages/vllm minus the envs.py
# registry, minus CI-only/other-platform entries.
KNOWN_ENV = [
    "VLLM_ALLOW_TQ_SPEC",
    "VLLM_DFLASH_DRAFT_KV_DTYPE",
    "VLLM_DFLASH_TQ_DRAFT_KV",
    "VLLM_ESIMD_F8_SCALE_FIX",
    "VLLM_FP8MQ_BLOCK_KV",
    "VLLM_FP8MQ_SPLITS",
    "VLLM_FP8MQ_STAGE1_STAGES",
    "VLLM_FP8MQ_STAGE1_WARPS",
    "VLLM_INT4_GROUP_SIZE",
    "VLLM_SPEC_TIMING",
    "VLLM_SPEC_TIMING_FLUSH",
    "VLLM_TQ_ADAPTIVE_KV_SPLITS",
    "VLLM_TQ_BLOCK_KV",
    "VLLM_TQ_GRAPH_KV_SPLITS",
    "VLLM_TQ_MAX_KV_SPLITS",
    "VLLM_TQ_MQ_MAX_Q",
    "VLLM_TQ_MQ_SPLITS",
    "VLLM_TQ_MQ_STAGE1_STAGES",
    "VLLM_TQ_MQ_STAGE1_WARPS",
    "VLLM_TQ_MQ_VERIFY",
    "VLLM_TQ_STAGE1_STAGES",
    "VLLM_TQ_STAGE1_WARPS",
    "VLLM_TQ_TIME",
    "VLLM_TQ_VERIFY_GRAPH_FIX",
    "VLLM_V50_PLACEHOLDER_LIMIT",
    "VLLM_V51_F8_MODE",
    "VLLM_V52M_STRIKES",
    "VLLM_V52_F8_GRACE",
    "VLLM_V55_EVENT_TIMEOUT_S",
    "VLLM_V55_GAP_FENCE",
    "VLLM_XPU_ALLREDUCE_GUARD_MIN_ROWS",
    "VLLM_XPU_ALLREDUCE_RETRY_MAX",
    "VLLM_XPU_ALLOW_E5M2_FP8_CKPT",
    "VLLM_XPU_CAPTURE_STAB",
    "VLLM_XPU_DFLASH_TP1",
    "VLLM_XPU_DRAFTER_PG",
    "VLLM_XPU_FA2_PIN_SPLITS",
    "VLLM_XPU_FP8_FALLBACK_SHAPE",
    "VLLM_XPU_FP8_FANOUT",
    "VLLM_XPU_FP8_MQ",
    "VLLM_XPU_FP8_MQ_Q1",
    "VLLM_XPU_MAX_CAPTURE_SIZE",
    "VLLM_XPU_MQ3D_SEGS",
    "VLLM_XPU_MTP_EAGER_HEAD",
    "VLLM_XPU_MTP_LOCAL_ARGMAX",
    "VLLM_XPU_PREALLOC_CCL_ARENA",
    "VLLM_XPU_SINGLECARD_TRITON",
    "VLLM_XPU_SPEC_DRAFT_BARRIER",
    "VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX",
    "VLLM_XPU_SPEC_EAGER_GDN",
    "VLLM_XPU_SPEC_SAFE_WARMUP",
    "VLLM_XPU_STARTUP_PROBE_TIMEOUT_S",
    "VLLM_XPU_TQ_SAFE_WARMUP",
    "VLLM_XPU_TRITON_MQ3D",
    "VLLM_XPU_VIA_STABLEBUF",
    "VLLM_XPU_VIA_STABLEBUF_MAX_ROWS",
    "VLLM_XPU_WARMUP_TIMEOUT_S",
]

ENVS_OLD = """def validate_environ(hard_fail: bool) -> None:
    for env in os.environ:
        if env.startswith("VLLM_") and env not in environment_variables:
"""

ENVS_NEW = (
    "# llm-scaler v55: knobs read directly (os.environ sites) by this\n"
    "# deployment's XPU/llm-scaler patches but not registered as formal\n"
    "# envs — silence the unknown-variable warning for exactly these\n"
    "# (census 2026-09-10, v1.2.7 image). Typos and dead vars still warn\n"
    "# (e.g. VLLM_ALLOW_LONG_MODEL_LEN: read nowhere; dropped from the\n"
    "# boot scripts).\n"
    "LLM_SCALER_KNOWN_ENV = frozenset({\n"
    + "".join(f'    "{v}",\n' for v in KNOWN_ENV)
    + "})\n"
    "\n"
    "\n"
    "def validate_environ(hard_fail: bool) -> None:\n"
    "    for env in os.environ:\n"
    "        if (\n"
    "            env.startswith(\"VLLM_\")\n"
    "            and env not in environment_variables\n"
    "            and env not in LLM_SCALER_KNOWN_ENV\n"
    "        ):\n"
)

# Routine mamba boundary bookkeeping: WARNING -> DEBUG. The regex keeps
# the call shape and only rewrites the function name; the message text
# ("llm-scaler v52l ..." / "llm-scaler v52f ...") is preserved verbatim.
DEMOTE_RE = re.compile(
    r"logger\.warning\((\s*\n\s*\"llm-scaler v52[lf] )")


def patch_envs(path: Path) -> None:
    src = _read(path)
    if "LLM_SCALER_KNOWN_ENV" in src:
        print(f"  {path.name}: already patched, skip")
        return
    if src.count(ENVS_OLD) != 1:
        raise SystemExit("ANCHOR-FAIL envs.py validate_environ")
    src = src.replace(ENVS_OLD, ENVS_NEW, 1)
    _write(path, src)
    _compile_gate(path)
    print(f"  {path.name}: allowlist ({len(KNOWN_ENV)} knobs) installed")


def demote(path: Path, prefix: str, expected: int) -> None:
    src = _read(path)
    hits = DEMOTE_RE.findall(src)
    if not hits:
        if f'llm-scaler v52{"l" if prefix == "l" else "f"}' in src and (
                "logger.debug(" in src):
            print(f"  {path.name}: already demoted, skip")
            return
        raise SystemExit(f"ANCHOR-FAIL {path.name}: no v52{prefix} warnings")
    if len(hits) != expected:
        raise SystemExit(
            f"ANCHOR-FAIL {path.name}: expected {expected} v52{prefix} "
            f"warning sites, found {len(hits)}")
    src = DEMOTE_RE.sub(r"logger.debug(\1", src)
    _write(path, src)
    _compile_gate(path)
    print(f"  {path.name}: {expected} v52{prefix} warnings -> debug")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, src: str) -> None:
    p.write_text(src, encoding="utf-8", newline="\n")


def _compile_gate(p: Path) -> None:
    py_compile.compile(str(p), doraise=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sp", default="/opt/venv/lib/python3.12/site-packages",
        help="site-packages root containing the vllm package")
    args = ap.parse_args()
    root = Path(args.sp) / "vllm"
    targets = {
        root / "envs.py": ("envs",),
        root / "v1" / "core" / "sched" / "scheduler.py": ("demote", "l", 4),
        root / "v1" / "worker" / "mamba_utils.py": ("demote", "f", 3),
    }
    for path, spec in targets.items():
        if not path.is_file():
            raise SystemExit(f"target missing: {path}")
    print(f"v55 warn hygiene: {root}")
    patch_envs(root / "envs.py")
    demote(root / "v1" / "core" / "sched" / "scheduler.py", "l", 4)
    demote(root / "v1" / "worker" / "mamba_utils.py", "f", 3)
    print("V55_WARNS_OK")


if __name__ == "__main__":
    sys.exit(main())
