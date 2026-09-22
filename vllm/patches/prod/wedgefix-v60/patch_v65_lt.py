#!/usr/bin/env python3
"""llm-scaler v65 LAYER-PROBE patcher (TEST-ONLY — NEVER BAKE).

Appends an env-gated (VLLM_V65_LAYER_LOG=1) instrumentation block to
vllm/v1/attention/backends/flash_attn.py that times, on PREFILL steps only:

  1. FlashAttentionImpl.forward  -> V65_ATTN lines (per full-attn layer,
     16 per chunk step), with q/kmax context and cumulative ms.
  2. torch.ops.vllm.gdn_attention_core_xpu -> V65_GDN lines (every 48th
     call = one prefill step boundary), cumulative ms.

Decode and XPU-graph capture are never timed: both wrappers gate on the
token count (> 1024), so capture-time calls (small shapes) passthrough
without any synchronize.

Modes: --apply (backup .pre_v65lt) / --check (count marks) / --revert.
"""

import pathlib
import shutil
import sys

TARGET = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py"
)
BACKUP = TARGET.with_suffix(".py.pre_v65lt")
MARK = "V65 LAYER-PROBE"

BLOCK = '''

# ---------------------------------------------------------------------------
# llm-scaler v65 LAYER-PROBE (test-only; NEVER bake) ------------------------
if os.environ.get("VLLM_V65_LAYER_LOG", "0") == "1":
    import time as _time65

    _pc65 = _time65.perf_counter
    _LT65 = {"attn_ms": 0.0, "attn_n": 0, "gdn_ms": 0.0, "gdn_n": 0}

    _orig_fa_fwd65 = FlashAttentionImpl.forward

    def _fa_fwd65(self, layer, query, key, value, kv_cache, attn_metadata,
                  output, output_scale=None, output_block_scale=None):
        _big = (
            attn_metadata is not None
            and getattr(attn_metadata, "max_query_len", 0) is not None
            and attn_metadata.max_query_len > 1024
        )
        if not _big:
            return _orig_fa_fwd65(self, layer, query, key, value, kv_cache,
                                  attn_metadata, output, output_scale,
                                  output_block_scale)
        torch.xpu.synchronize()
        _t0 = _pc65()
        _r = _orig_fa_fwd65(self, layer, query, key, value, kv_cache,
                            attn_metadata, output, output_scale,
                            output_block_scale)
        torch.xpu.synchronize()
        _dt = (_pc65() - _t0) * 1000.0
        _LT65["attn_ms"] += _dt
        _LT65["attn_n"] += 1
        logger.info(
            "V65_ATTN ms=%.1f q=%d kmax=%d cum_ms=%.0f n=%d",
            _dt, attn_metadata.max_query_len, attn_metadata.max_seq_len,
            _LT65["attn_ms"], _LT65["attn_n"],
        )
        return _r

    FlashAttentionImpl.forward = _fa_fwd65

    _ns65 = torch.ops.vllm
    _orig_gdn65 = getattr(_ns65, "gdn_attention_core_xpu", None)
    if _orig_gdn65 is not None:

        def _gdn65(core_attn_out, z, qkvz, ba, layer_name):
            _nt = core_attn_out.shape[0]
            if _nt <= 1024:
                return _orig_gdn65(core_attn_out, z, qkvz, ba, layer_name)
            torch.xpu.synchronize()
            _t0 = _pc65()
            _r = _orig_gdn65(core_attn_out, z, qkvz, ba, layer_name)
            torch.xpu.synchronize()
            _dt = (_pc65() - _t0) * 1000.0
            _LT65["gdn_ms"] += _dt
            _LT65["gdn_n"] += 1
            if _LT65["gdn_n"] % 48 == 0:
                logger.info("V65_GDN layer_ms=%.1f nt=%d cum_ms=%.0f n=%d",
                            _dt, _nt, _LT65["gdn_ms"], _LT65["gdn_n"])
            return _r

        try:
            setattr(_ns65, "gdn_attention_core_xpu", _gdn65)
            logger.info("V65_LT gdn op wrapped")
        except Exception as _e65:  # noqa: BLE001
            logger.info("V65_LT gdn wrap failed: %s", _e65)
    else:
        logger.info("V65_LT gdn op not found")
# --- end v65 LAYER-PROBE ---
'''


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--apply"
    if mode == "--apply":
        text = TARGET.read_text()
        if MARK in text:
            print("LT_ALREADY_APPLIED")
            return 0
        shutil.copy2(TARGET, BACKUP)
        TARGET.write_text(text + BLOCK)
        import py_compile

        py_compile.compile(str(TARGET), doraise=True)
        print("LT_APPLIED_OK")
    elif mode == "--check":
        text = TARGET.read_text()
        print("lt_marks=%d" % text.count(MARK))
        print("lt_gdn_wrap=%d" % text.count("V65_LT gdn op wrapped"))
        print("lt_block_end=%d" % text.count("end v65 LAYER-PROBE"))
    elif mode == "--revert":
        if BACKUP.exists():
            shutil.copy2(BACKUP, TARGET)
            print("LT_REVERTED")
        else:
            print("LT_NO_BACKUP")
    else:
        print("usage: patch_v65_lt.py --apply|--check|--revert")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
