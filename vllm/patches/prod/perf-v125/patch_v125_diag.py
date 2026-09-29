#!/usr/bin/env python3
"""patch_v125_diag.py (v125 P22-C diagnostic instrument).

WHY: the C1 dispatch exclusion is statically correct — the ONLY live
caller of _determine_batch_execution_and_padding is execute_model:4670
(patched, disable_full_nospec=spec_decode_metadata is None), the
dispatcher subtracts FULL from allowed modes, set_forward_context:4816
feeds that same cudagraph_mode, and CUDAGraphWrapper.__call__ falls to
runnable (eager) on mode mismatch — yet the fixval ns leg STILL
degenerated from pos1 with v60g x20 (fix16ns ABORT_V60G). One of the
assumptions is factually wrong at runtime. This instrument prints the
actual decisions:

  D1 (gpu_model_runner, dispatch site): first 40 dispatches — nospec
     flag, env flag, tokens, returned mode, padded descriptor (40
     covers the 27 boot captures + live prefills/decodes).
  D2 (cuda_graph.py, wrapper __call__): first 40 wrapper entries —
     context mode vs wrapper runtime mode + descriptor (replay happens
     only when they match FULL).

Run AFTER patch_v125_nospec_no_full_graph.py (anchors include its
marker text). Revert with the .pre_v125_diag backups.
"""
import py_compile
import shutil
import sys

RUNNER = ("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/"
          "gpu_model_runner.py")
CG = "/opt/venv/lib/python3.12/site-packages/vllm/compilation/cuda_graph.py"
DIAG = "v125DIAG"

D1_OLD = """\
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded,
            disable_full=(
                use_cascade_attn or has_encoder_output
                or (disable_full_nospec and self._v125_nospec_no_full)
            ),
        )
"""
D1_NEW = """\
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded,
            disable_full=(
                use_cascade_attn or has_encoder_output
                or (disable_full_nospec and self._v125_nospec_no_full)
            ),
        )
        _v125_n = getattr(self, "_v125_diag_n", 0)
        if _v125_n < 40:
            self._v125_diag_n = _v125_n + 1
            logger.warning(
                "%s dispatch#%d nospec=%s env_flag=%s ntok_pad=%s "
                "mode=%s desc=%s cascade=%s enc=%s",
                "v125DIAG", _v125_n + 1, disable_full_nospec,
                getattr(self, "_v125_nospec_no_full", "<unset>"),
                num_tokens_padded,
                cudagraph_mode, batch_descriptor, use_cascade_attn,
                has_encoder_output)
"""

D2_OLD = """\
        assert batch_descriptor is not None
        if batch_descriptor not in self.concrete_cudagraph_entries:
"""
D2_NEW = """\
        assert batch_descriptor is not None
        _v125_w = getattr(CUDAGraphWrapper, "_v125_diag_w", 0)
        if _v125_w < 40:
            CUDAGraphWrapper._v125_diag_w = _v125_w + 1
            logger.warning(
                "%s wrapper#%d ctx_mode=%s wrapper_mode=%s match=%s "
                "desc=%s entries=%d",
                "v125DIAG", _v125_w + 1, cudagraph_runtime_mode,
                self.runtime_mode,
                cudagraph_runtime_mode == self.runtime_mode,
                batch_descriptor, len(self.concrete_cudagraph_entries))
        if batch_descriptor not in self.concrete_cudagraph_entries:
"""


def patch(path, old, new, tag):
    # reentrant: a prior diag application is rolled back first so a
    # relaunch always installs the CURRENT instrument, not the old one
    try:
        shutil.copyfile(path + ".pre_v125_diag", path)
        print("V125_DIAG_RESTORED_PRE %s" % tag)
    except FileNotFoundError:
        pass
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if new in src:
        print("V125_DIAG_ALREADY %s" % tag)
        return True
    n = src.count(old)
    if n != 1:
        print("V125_DIAG_ABORT %s anchor count=%d (want 1)" % (tag, n))
        return False
    shutil.copyfile(path, path + ".pre_v125_diag")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(old, new, 1))
    print("V125_DIAG_OK %s" % tag)
    return True


def main() -> int:
    ok1 = patch(RUNNER, D1_OLD, D1_NEW, "dispatch")
    ok2 = patch(CG, D2_OLD, D2_NEW, "wrapper")
    if ok1:
        py_compile.compile(RUNNER, doraise=True)
    if ok2:
        py_compile.compile(CG, doraise=True)
    print("V125_DIAG_DONE ok=%s,%s" % (ok1, ok2))
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
