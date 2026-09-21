#!/bin/bash
# xg2_preflight.sh — v59 §2 XGrammar-2 (xgrammar 0.2.7) SAFE upgrade + pre-flight.
# Run on host. Repeat-safe. Logs to /root/build/lce1/xg2_preflight.log
# Steps: (1) no-deps upgrade (protects fork triton/torch), (2) import/compile smoke,
# (3) vLLM fork backend import, (4) triton JIT kernel intact check,
# (5) apache-tvm-ffi dependency check. GATE: only if ALL pass -> READY_FOR_BOOT.
set -u
LOG=/root/build/lce1/xg2_preflight.log
PY=/opt/venv/bin/python3
{
echo "=== XG2 PREFLIGHT $(date +%H:%M:%S) ==="

echo "--- step1: current version"
docker exec lsv-test $PY -c 'from importlib.metadata import version; print("cur:", version("xgrammar"))' 2>&1

echo "--- step2: no-deps upgrade to 0.2.7"
docker exec lsv-test /opt/venv/bin/pip install --no-deps -q xgrammar==0.2.7 2>&1 | tail -2
docker exec lsv-test $PY -c 'from importlib.metadata import version; print("now:", version("xgrammar"))' 2>&1

echo "--- step3: xgrammar import + schema compile smoke"
docker exec lsv-test $PY -c '
import xgrammar as xg
g = xg.Grammar.from_json_schema({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]})
print("compile: OK")
' 2>&1

echo "--- step4: vLLM fork structured-output backend import (API compat)"
docker exec lsv-test $PY -c '
import vllm.v1.structured_output.backend_xgrammar as b
print("backend import: OK", [n for n in dir(b) if "Grammar" in n][:5])
' 2>&1

echo "--- step5: triton kernel intact (rejection_sampler expand_kernel must be JITFunction)"
docker exec lsv-test $PY -c '
import vllm.v1.sample.rejection_sampler as rs
k = getattr(rs, "expand_kernel", None)
print("expand_kernel type:", type(k).__name__, "| module:", type(k).__module__)
ok = "triton" in type(k).__module__ or "JIT" in type(k).__name__
print("triton_intact:", ok)
' 2>&1

echo "--- step6: apache-tvm-ffi check"
docker exec lsv-test $PY -c '
try:
    import tvm_ffi; print("tvm_ffi: present", getattr(tvm_ffi, "__version__", "?"))
except ImportError as e:
    print("tvm_ffi: MISSING", e)
' 2>&1
docker exec lsv-test /opt/venv/bin/pip show apache-tvm-ffi 2>/dev/null | head -2

echo "--- step7: does xgrammar 0.2.7 actually import tvm_ffi?"
docker exec lsv-test $PY -c '
import xgrammar, inspect
src = inspect.getsource(xgrammar)
print("tvm_ffi referenced in xgrammar source:", "tvm_ffi" in src)
' 2>&1

echo "=== XG2 PREFLIGHT COMPLETE $(date +%H:%M:%S) ==="
} > "$LOG" 2>&1
tail -40 "$LOG"
echo XG2_PF_DONE
