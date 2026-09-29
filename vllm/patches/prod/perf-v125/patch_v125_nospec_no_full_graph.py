#!/usr/bin/env python3
"""patch_v125_nospec_no_full_graph.py (v125 P22-C C1 ROOT FIX).

Defect (PHASES.md Run 7d, perf-v125): a decode batch with NO spec
metadata — every decode step of a non-spec serve, or a rare no-draft
step on the certified MTP lane — dispatches through the FULL XPU graph
(FULL_DECODE_ONLY, 85-size set). That graph's replay NEVER materializes
the hidden rows the deferred tlogits gather reads (wedgefix-v60 E
root-cause, gpu_model_runner v60g comment): hidden rows NaN -> logits
NaN -> v60g sanitizer substitutes uniform -> degenerate emission
('Unit!!!'/'!' lockups, exact ln(1/vocab) logprob signature). Proven
matrix: graphs+async non-spec = degenerate from pos1 (run 6, ctrl16ns);
eager+async non-spec = FULLY COHERENT bit-consistent with certified
spec-4 (p22c1_eager eager16ns); spec verify batches (metadata present)
= coherent for months. The first generated token is sampled from the
PREFILL forward (eager), which is why pos0 was always correct.

Fix: dispatch-level exclusion. execute_model already computes
spec_decode_metadata (:4626 _prepare_inputs) BEFORE calling
_determine_batch_execution_and_padding (:4647); that helper already
supports invalid_modes={FULL} via disable_full (cascade-attn /
encoder-decoder batches run exactly this eager fallback today). This
patch threads `disable_full_nospec=(spec_decode_metadata is None)` so
no-draft batches resolve to CUDAGraphMode.NONE = eager forward.
Spec verify batches keep graphs — the certified MTP-4 posture is
byte-identical in behavior. Boot-time capture is untouched (graphs are
still captured; they are simply never dispatched for no-draft batches).

Env: VLLM_V125_NOSPEC_NO_FULL_GRAPH=0 restores the pre-fix dispatch
(diagnosis only). v60g stays armed as tripwire — it must fire ZERO
times on a fixed posture.

Idempotent: re-running detects the marker and verifies all three edits.
Run inside the target container:
    docker cp patch_v125_nospec_no_full_graph.py lsv-test:/root/
    docker exec lsv-test python3 /root/patch_v125_nospec_no_full_graph.py
"""
import py_compile
import shutil
import sys

TARGET = ("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/"
          "gpu_model_runner.py")
BACKUP = TARGET + ".pre_v125_c1"
MARKER = "llm-scaler v125 (perf-v125 C1 root fix)"

EDIT1_OLD = """\
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
"""
EDIT1_NEW = """\
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
        # llm-scaler v125 (perf-v125 C1 root fix): see the dispatch-site
        # comment below — no-spec batches must not use the FULL graph.
        disable_full_nospec: bool = False,
    ) -> tuple[
"""

EDIT2_OLD = """\
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
        )
"""
EDIT2_NEW = """\
        # llm-scaler v125 (perf-v125 C1 root fix): a batch with NO spec
        # metadata (non-spec serve — every decode step — or a rare
        # no-draft step on the MTP lane) must NOT dispatch through the
        # FULL XPU graph: wedgefix-v60 E proved the nospec
        # FULL_DECODE_ONLY graph replay never materializes the hidden
        # rows the tlogits gather reads (NaN logits; the v60g sanitizer
        # then substitutes uniform = degenerate emission — run-6/run-7
        # non-spec legs, PHASES.md Run 7d). Eager is proven coherent
        # (p22c1_eager eager16ns, bit-consistent with certified spec-4);
        # spec verify batches keep graphs. Cascade-attn / encoder batches
        # already run this exact disable_full eager fallback.
        # VLLM_V125_NOSPEC_NO_FULL_GRAPH=0 restores the old dispatch.
        if getattr(self, "_v125_nospec_no_full", None) is None:
            self._v125_nospec_no_full = os.environ.get(
                "VLLM_V125_NOSPEC_NO_FULL_GRAPH", "1") == "1"
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded,
            disable_full=(
                use_cascade_attn or has_encoder_output
                or (disable_full_nospec and self._v125_nospec_no_full)
            ),
        )
"""

EDIT3_OLD = """\
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )
"""
EDIT3_NEW = """\
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                # llm-scaler v125 C1: no spec metadata in this batch ->
                # never dispatch it through the FULL nospec graph.
                disable_full_nospec=spec_decode_metadata is None,
            )
"""


def main() -> int:
    with open(TARGET, encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        ok = all(new in src for _, new in ((EDIT1_OLD, EDIT1_NEW),
                                           (EDIT2_OLD, EDIT2_NEW),
                                           (EDIT3_OLD, EDIT3_NEW)))
        print("V125_C1_ALREADY_PATCHED markers=%s" % ("ALL" if ok else "PARTIAL"))
        return 0 if ok else 1

    edits = ((EDIT1_OLD, EDIT1_NEW, "signature"),
             (EDIT2_OLD, EDIT2_NEW, "dispatch-site"),
             (EDIT3_OLD, EDIT3_NEW, "call-site"))
    for old, _, name in edits:
        n = src.count(old)
        if n != 1:
            print("V125_C1_ABORT anchor %s count=%d (want 1)" % (name, n))
            return 1

    try:
        shutil.copyfile(TARGET, BACKUP)
    except FileNotFoundError:
        pass  # already backed up by a prior round
    patched = src
    for old, new, name in edits:
        patched = patched.replace(old, new, 1)
        print("V125_C1_EDIT_OK %s" % name)

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(patched)
    py_compile.compile(TARGET, doraise=True)
    print("V125_C1_PATCH_OK backup=%s" % BACKUP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
