#!/bin/bash
# p22a_repair2_v125.sh — finish the repair after p22a_repair_v125.sh
# aborted at step 5: p22b2_python_patch.py expects the BASE P22B patch
# (p22b_python_patch.py, defines _gdn_fp8_native_enabled + the ENGAGED
# gate) already applied — the pre-recreate container had it from run-1
# (17:13); the fresh v1.2.24 container has neither. Correct stack on a
# fresh image: p22b (base) -> p22b2 (decode-only split).
# Precondition: lane UP on b13f56543057 certified serve (repair steps
# 1-4 done), watchdog paused.
set -u
L=/root/build/lce1/p22a_repair2_v125.log
SP=/opt/venv/lib/python3.12/site-packages
{
echo "=== P22A REPAIR2 V125 start $(date +%F' '%T) ==="
IMG=$(docker inspect lsv-test --format '{{.Image}}')
echo "container image=$IMG"
[ "$IMG" = "sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e" ] || { echo "ABORT_NOT_V1224"; exit 1; }
curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health || { echo "ABORT_LANE_NOT_HEALTHY"; exit 1; }

echo "--- 1. base patch p22b_python_patch.py ---"
docker cp /root/build/p22b_python_patch.py lsv-test:/root/p22b_python_patch.py
B1=$(docker exec lsv-test /opt/venv/bin/python /root/p22b_python_patch.py)
echo "$B1"
echo "$B1" | grep -qE "P22B_PYTHON_PATCHED|ALREADY_APPLIED" || { echo "ABORT_B1_PATCH"; exit 1; }

echo "--- 2. split patch p22b2_python_patch.py ---"
docker cp /root/build/p22b2_python_patch.py lsv-test:/root/p22b2_python_patch.py
B2=$(docker exec lsv-test /opt/venv/bin/python /root/p22b2_python_patch.py)
echo "$B2"
echo "$B2" | grep -qE "P22B2_PYTHON_PATCHED|ALREADY_APPLIED" || { echo "ABORT_B2_PATCH"; exit 1; }

echo "--- 3. syntax + helper import gates ---"
docker exec lsv-test /opt/venv/bin/python -c "import ast; ast.parse(open('$SP/vllm/_xpu_ops.py').read()); print('SYNTAX_OK _xpu_ops.py')" \
  || { echo "ABORT_SYNTAX"; exit 1; }
docker exec lsv-test /opt/venv/bin/python -c "from vllm._xpu_ops import _gdn_fp8_native_enabled; print('helper import OK')" \
  || { echo "ABORT_HELPER_IMPORT"; exit 1; }
docker exec lsv-test grep -c "v125 P22B2" $SP/vllm/_xpu_ops.py

echo "--- 4. math probe through the certified serve ---"
PR=$(python3 - <<'PYEOF'
import json, urllib.request
body = json.dumps({"model": "qwen3.8-27b-fp8", "max_tokens": 40, "temperature": 0,
                   "messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}]}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
try:
    r = json.load(urllib.request.urlopen(req, timeout=180))
    txt = r["choices"][0]["message"].get("content", "")
    print("PROBE_MATH http-ok contains391=%s text=%r" % ("391" in txt, txt[:60]))
except Exception as e:
    print("PROBE_MATH_FAIL %s: %s" % (type(e).__name__, e))
PYEOF
)
echo "$PR"
echo "$PR" | grep -q 'contains391=True' || { echo "ABORT_PROBE_NOT_391"; exit 1; }

echo "--- 5. re-arm watchdog ---"
rm -f /root/build/lane_watchdog.paused
sleep 65
tail -2 /root/build/lane_watchdog.log
echo "=== P22A REPAIR2 V125 DONE $(date +%F' '%T) ==="
echo "P22A_REPAIR2_V125_DONE"
} > "$L" 2>&1
tail -25 "$L"
echo P22A_REPAIR2_SCRIPT_DONE
