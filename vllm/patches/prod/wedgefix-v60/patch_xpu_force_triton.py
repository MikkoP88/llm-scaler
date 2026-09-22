#!/usr/bin/env python3
"""llm-scaler v65 test patcher: force TRITON_ATTN on XPU (TEST-ONLY — NEVER BAKE).

Inserts an env-gated (VLLM_V65_FORCE_TRITON=1, default off) branch into
XPUPlatform.get_attn_backend() just before the existing TRITON_ATTN handling,
so the whole engine (metadata builders, KV cache shape) consistently uses the
Triton attention backend. Purpose: A/B prefill-attention kernel efficiency
(FA2 vs Triton) for the v65 big-TTFT deep-kernel census.

Modes: --apply (backup .pre_v65ft) / --check / --revert.
"""

import pathlib
import shutil
import sys

TARGET = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py"
)
BACKUP = TARGET.with_suffix(".py.pre_v65ft")
ANCHOR = "        if selected_backend == AttentionBackendEnum.TRITON_ATTN:"
INSERT = '''        if (
            os.environ.get("VLLM_V65_FORCE_TRITON", "0") == "1"
            and selected_backend == AttentionBackendEnum.FLASH_ATTN
        ):
            logger.info_once("V65_TEST forcing TRITON_ATTN backend for A/B")
            return AttentionBackendEnum.TRITON_ATTN.get_path()
'''


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--apply"
    if mode == "--apply":
        text = TARGET.read_text()
        if "V65_TEST forcing TRITON_ATTN" in text:
            print("FT_ALREADY_APPLIED")
            return 0
        idx = text.find(ANCHOR)
        if idx < 0:
            print("FT_ANCHOR_NOT_FOUND")
            return 1
        shutil.copy2(TARGET, BACKUP)
        TARGET.write_text(text[:idx] + INSERT + text[idx:])
        import py_compile

        py_compile.compile(str(TARGET), doraise=True)
        print("FT_APPLIED_OK")
    elif mode == "--check":
        text = TARGET.read_text()
        print("ft_marks=%d" % text.count("V65_TEST forcing TRITON_ATTN"))
    elif mode == "--revert":
        if BACKUP.exists():
            shutil.copy2(BACKUP, TARGET)
            print("FT_REVERTED")
        else:
            print("FT_NO_BACKUP")
    else:
        print("usage: patch_xpu_force_triton.py --apply|--check|--revert")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
