#!/usr/bin/env python3
"""patch_v125_d3.py (v125 P22-C1 D3 NaN-slice instrument).

Run 7g/7h state: spec-off decode is degenerate (NaN hidden rows -> v60g)
on EVERY non-enforce-eager boot — FULL graphs (run 6/7), C1 mode=NONE
runnable (round-3), and cudagraph_mode=NONE with zero graphs captured
(H12). enforce-eager is coherent. The model callable itself is identical
(raw model, no wrapper at mode NONE), so the poison is in what the boot
changes around the forward. The H12 log diff surfaced a concrete
candidate: enforce_eager forces compilation mode=NONE which appends
custom_ops='all' (op substitution ON, ir_op_priority rms_norm
['xpu_kernels','native']); non-eager boots keep mode=VLLM_COMPILE which
appends 'none' (native ops) even after TORCH_COMPILE_DISABLE kills
compilation.

D3 prints, at the tlogits gather + lm_head (execute_model), whether the
forward output is NaN wholesale or only the gathered rows, plus the
logits indices — for the first 80 forwards after boot (profile + live
prefill + live decodes; MT12+LIST40 need ~54).

Revert with the .pre_v125_d3 backup. Run on top of the C1/diag venv
(inert at mode NONE) or a clean one.
"""
import py_compile
import shutil
import sys

RUNNER = ("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/"
          "gpu_model_runner.py")

D3_OLD = """\
                with spec_seg("tlogits"):  # llm-scaler v20 A2
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)
"""
D3_NEW = """\
                with spec_seg("tlogits"):  # llm-scaler v20 A2
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)
                    # llm-scaler v125 D3: NaN slice at the gather + lm_head.
                    _v125d3 = getattr(self, "_v125_d3_n", 0)
                    if _v125d3 < 80:
                        self._v125_d3_n = _v125d3 + 1
                        try:
                            _hn = int(
                                torch.isnan(hidden_states.float())
                                .any(dim=-1).sum())
                            _sn = int(
                                torch.isnan(sample_hidden_states.float())
                                .any(dim=-1).sum())
                            _ln = int(
                                torch.isnan(logits.float())
                                .any(dim=-1).sum())
                            logger.warning(
                                "v125D3 #%d hs_nan=%d/%d shs_nan=%d/%d "
                                "logits_nan=%d/%d li_max=%d li_tail=%s",
                                _v125d3 + 1, _hn, hidden_states.shape[0],
                                _sn, sample_hidden_states.shape[0],
                                _ln, logits.shape[0],
                                int(logits_indices.max()),
                                logits_indices[-3:].tolist())
                        except Exception as _e:
                            logger.warning("v125D3 #%d exc=%s",
                                           _v125d3 + 1, type(_e).__name__)
"""


def main() -> int:
    try:
        shutil.copyfile(RUNNER + ".pre_v125_d3", RUNNER)
        print("V125_D3_RESTORED_PRE")
    except FileNotFoundError:
        pass
    with open(RUNNER, encoding="utf-8") as f:
        src = f.read()
    if "v125D3" in src:
        print("V125_D3_ALREADY")
        return 0
    n = src.count(D3_OLD)
    if n != 1:
        print("V125_D3_ABORT anchor count=%d (want 1)" % n)
        return 1
    shutil.copyfile(RUNNER, RUNNER + ".pre_v125_d3")
    with open(RUNNER, "w", encoding="utf-8") as f:
        f.write(src.replace(D3_OLD, D3_NEW, 1))
    py_compile.compile(RUNNER, doraise=True)
    print("V125_D3_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
