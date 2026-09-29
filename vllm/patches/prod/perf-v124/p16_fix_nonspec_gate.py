#!/opt/venv/bin/python
"""llm-scaler v124 P16 — fix the non-spec pipeline so it boots.

Root cause (P16, 2026-09-28 13:01): with speculative-config removed, the
v31.1 gate in platforms/xpu.py (TORCH_COMPILE_DISABLE for spec+TP>1) does
NOT fire, so dynamo compiles the forward and dies on the v58 arstage
stage_in — ``torch._dynamo.exc.Unsupported: Unsupported hasattr call`` on
``hasattr(StreamVariable, xpu_stream)`` at vllm/_arstage.py:98, reached via
gdn_linear_attn -> out_proj -> tensor_model_parallel_all_reduce ->
xpu_communicator._all_reduce_impl -> stage_in. Engine init fails in
profile_run -> spec-off can never boot on a TP>1 lane.

Fix: extend the gate condition from (spec AND TP>1) to (TP>1). The fork has
no validated compiled path (v31.1 convicted compiled pieces x spec, #11);
the certified spec4 posture already runs compile-OFF + whole-step XPU graph
capture, so the no-spec posture becomes identical instead of trading
anything away. TP1 behavior unchanged. Idempotent; asserts before writing.
"""
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py"
MARK = "# llm-scaler v124 P16:"

OLD_COND = """\
        _unsafe_spec_tp_graph = (
            vllm_config.speculative_config is not None
            and parallel_config.tensor_parallel_size > 1
            and not _allow_unsafe_spec_tp_graph
        )"""

NEW_COND = """\
        # llm-scaler v124 P16: condition extended from spec+TP>1 to ALL TP>1.
        # Spec-OFF previously fell through to dynamo, which cannot trace the
        # v58 arstage stage_in (hasattr StreamVariable xpu_stream,
        # vllm/_arstage.py:98) -> "Unsupported hasattr call" -> engine init
        # death in profile_run. No-spec TP>1 must boot compile-OFF exactly
        # like the certified spec4 posture (whole-step XPU graphs only).
        _unsafe_spec_tp_graph = (
            parallel_config.tensor_parallel_size > 1
            and not _allow_unsafe_spec_tp_graph
        )"""

OLD_MSG = '''\
            logger.warning(
                "inductor compilation disabled for speculative decoding "
                "with TP=%d: compiled pieces livelock both devices at "
                ">=32k context, while whole-step XPU graph capture alone "
                "is clean and faster (KNOWN_ISSUES #11, v31.1). Set "
                "VLLM_XPU_ALLOW_UNSAFE_SPEC_TP_GRAPH=1 to reproduce the "
                "wedging compiled configuration.",
                parallel_config.tensor_parallel_size,
            )'''

NEW_MSG = '''\
            logger.warning(
                "inductor compilation disabled with TP=%d (v31.1 #11 + "
                "v124 P16: spec-off dynamo cannot trace arstage): "
                "whole-step XPU graph capture alone is clean and faster. "
                "Set VLLM_XPU_ALLOW_UNSAFE_SPEC_TP_GRAPH=1 to reproduce "
                "the compiled configuration.",
                parallel_config.tensor_parallel_size,
            )'''


def main() -> int:
    src = open(P).read()
    if MARK in src and OLD_COND not in src:
        print("ALREADY_PATCHED")
        return 0
    assert OLD_COND in src, "gate condition not found — xpu.py drifted"
    assert OLD_MSG in src, "gate message not found — xpu.py drifted"
    out = src.replace(OLD_COND, NEW_COND, 1).replace(OLD_MSG, NEW_MSG, 1)
    open(P, "w").write(out)
    chk = open(P).read()
    assert MARK in chk and OLD_COND not in chk and NEW_MSG.splitlines()[1].strip()[:30] in chk
    print("PATCHED_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
