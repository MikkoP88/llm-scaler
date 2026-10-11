#!/bin/bash
# fn8 P1 patcher dry-run (run INSIDE container fn8-t2)
set -u
F=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
echo "== original md5:"
md5sum "$F"
echo "== apply:"
python3 /stage_patch01.py || { echo APPLY_FAILED; exit 1; }
echo "== py_compile patched:"
python3 -m py_compile "$F" && echo PYCOMPILE_OK || { echo PYCOMPILE_FAIL; exit 1; }
echo "== marker count: $(grep -c FN8_01 "$F")"
echo "== prof tool py_compile:"
python3 -m py_compile /stage_prof.py && echo PROF_PYCOMPILE_OK
echo "== restore:"
python3 /stage_patch01.py --restore
echo "== final md5 (must equal original):"
md5sum "$F"
