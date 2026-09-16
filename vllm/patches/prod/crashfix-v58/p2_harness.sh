#!/bin/bash
# p2_harness.sh — P2 (copy-tax recovery) harness setup on the host:
#   1. validate patch anchors against the LIVE standing container (--check)
#   2. build repro_bootP2V.sh = repro_bootNV.sh + patch_v58_p2 step
#   3. run cbench on the standing 26.14 container (baseline A/B numbers)
set -e
cd /root/build

echo ====P2_CHECK_ANCHORS
docker cp /root/build/patch_v58_p2.py lsv-test:/root/patch_v58_p2.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v58_p2.py --check

echo ====P2_MAKE_BOOT
python3 - <<'PYEOF'
src = open("/root/build/repro_bootNV.sh").read()
anchor = 'docker exec lsv-test /opt/venv/bin/python3 /root/patch_v58_p1.py | tee /root/build/lce1/v58p1_apply_boot${MODE}.log\n'
ins = (
    "docker cp /root/build/patch_v58_p2.py lsv-test:/root/patch_v58_p2.py\n"
    "docker exec lsv-test /opt/venv/bin/python3 /root/patch_v58_p2.py | tee /root/build/lce1/v58p2_apply_boot${MODE}.log\n"
    'docker exec lsv-test grep -c "llm-scaler v58 P2" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py\n'
)
assert src.count(anchor) == 1, f"NV anchor count={src.count(anchor)}"
out = src.replace(anchor, anchor + ins)
open("/root/build/repro_bootP2V.sh", "w").write(out)
print("repro_bootP2V.sh written, lines:", out.count("\n"))
PYEOF
chmod +x /root/build/repro_bootP2V.sh

echo ====P2_CBENCH_2614_STANDING
docker cp /root/build/p2_cbench.py lsv-test:/root/p2_cbench.py
docker exec lsv-test /opt/venv/bin/python3 /root/p2_cbench.py base2614 2>&1 | grep -E "CBENCH|Error" | head -20
echo ====P2_HARNESS_DONE
