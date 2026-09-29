#!/bin/sh
# build_ship_v1224_files.sh — generate the v1.2.24 ship-chain scripts from the
# certified v1223 pair (sed version rename) + surgical v124 insertions:
#   gates_v1224_lane.sh  = gates_v1223_lane.sh + v124 delta gates (P16 x2,
#                          P19.5a markers, mamba fp16 serve default, runtime
#                          fp8 dtype resolution) before the lsv-gate teardown
#   ship_v1224.sh        = ship_v1223.sh + generalized watchdog repoint (the
#                          v1223 pattern's source repro_bootV1221.sh no longer
#                          exists in lane_watchdog.sh — it points at
#                          repro_bootV1223_prod.sh since the v1223 ship)
#   cc_battery_v1224.sh  = cc_battery_fix_v1223.sh (python-built JSON bodies
#                          + Authorization: Bearer sk-dummy — the v1223 fixes
#                          carry over verbatim)
set -eu
cd /root/build

# --- 1. gates_v1224_lane.sh ---
sed -e 's/v1223/v1224/g' -e 's/V1223/V1224/g' -e 's/v1\.2\.23/v1.2.24/g' \
    gates_v1223_lane.sh > gates_v1224_lane.sh
python3 - <<'PYEOF'
p = '/root/build/gates_v1224_lane.sh'
src = open(p).read()
anchor = "docker rm -f lsv-gate >/dev/null 2>&1 || true"
add = '''echo "--- v124 deltas: P16 gate fix + P19.5a fp8 plumbing (inert on fp16 serve) ---"
SP=/opt/venv/lib/python3.12/site-packages
ck v124_p16_markers 2 "$(docker exec lsv-gate sh -c "grep -c 'v124 P16' $SP/vllm/platforms/xpu.py" | tr -d '[:space:]')"
ck v124_p195_cache_literal 1 "$(docker exec lsv-gate sh -c "grep -c 'v124 P19.5a' $SP/vllm/config/cache.py" | tr -d '[:space:]')"
ck v124_p195_mamba_resolver 1 "$(docker exec lsv-gate sh -c "grep -c 'v124 P19.5a' $SP/vllm/model_executor/layers/mamba/mamba_utils.py" | tr -d '[:space:]')"
ck v124_p195_sycl_bridge 1 "$(docker exec lsv-gate sh -c "grep -c 'v124 P19.5a' $SP/vllm/_xpu_ops.py" | tr -d '[:space:]')"
ck v124_serve_mamba_fp16_default 1 "$(docker exec lsv-gate grep -c -- '--mamba-ssm-cache-dtype float16' /root/serve_user.sh | tr -d '[:space:]')"
ck v124_serve_mamba_fp8_absent 0 "$(docker exec lsv-gate grep -c -- '--mamba-ssm-cache-dtype fp8' /root/serve_user.sh | tr -d '[:space:]')"
V124RT=$(docker exec lsv-gate /opt/venv/bin/python3 -c "import torch; from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator as C; a=C.gated_delta_net_state_dtype(torch.float16, 'auto', 'fp8_e4m3'); b=C.gated_delta_net_state_dtype(torch.float16, 'auto', 'fp8_e5m2'); print(a[1], b[1])" 2>&1)
echo "v124_runtime_fp8_resolution: $V124RT"
echo "$V124RT" | grep -q float8_e4m3fn && echo "GATE-OK   v124_fp8_e4m3_resolves" || { echo "GATE-FAIL v124_fp8_e4m3_resolves"; FAIL=1; }
echo "$V124RT" | grep -q float8_e5m2 && echo "GATE-OK   v124_fp8_e5m2_resolves" || { echo "GATE-FAIL v124_fp8_e5m2_resolves"; FAIL=1; }

'''
assert anchor in src, 'gates teardown anchor not found'
open(p, 'w').write(src.replace(anchor, add + anchor, 1))
print('GATES_V1224_BUILT')
PYEOF
chmod +x gates_v1224_lane.sh
sh -n gates_v1224_lane.sh

# --- 2. ship_v1224.sh ---
sed -e 's/v1223/v1224/g' -e 's/V1223/V1224/g' -e 's/v1\.2\.23/v1.2.24/g' \
    ship_v1223.sh > ship_v1224.sh
python3 - <<'PYEOF'
p = '/root/build/ship_v1224.sh'
src = open(p).read()
old = "sed -i 's|nohup bash repro_bootV1221.sh|nohup bash repro_bootV1224_prod.sh|' /root/build/lane_watchdog.sh"
new = "sed -i 's|repro_bootV[0-9][0-9]*_prod.sh|repro_bootV1224_prod.sh|' /root/build/lane_watchdog.sh"
assert old in src, 'watchdog repoint line not found'
open(p, 'w').write(src.replace(old, new, 1))
print('SHIP_V1224_BUILT')
PYEOF
chmod +x ship_v1224.sh
sh -n ship_v1224.sh

# --- 3. cc_battery_v1224.sh ---
sed -e 's/v1223/v1224/g' -e 's/V1223/V1224/g' cc_battery_fix_v1223.sh > cc_battery_v1224.sh
chmod +x cc_battery_v1224.sh
sh -n cc_battery_v1224.sh

echo BUILD_SHIP_V1224_FILES_DONE
