#!/bin/bash
# bake_v1214.sh — v59 §2 bake llm-scaler-exp:v1.2.14 (STALFIX + XGrammar-2 0.2.7)
# and boot it. Self-logging. Run on host.
# Precondition: current lsv-test container has xgrammar 0.2.7 (verified by
# xg2_preflight.sh) on top of v1.2.13 (STALFIX baked).
set -u
LOG=/root/build/lce1/bake_v1214.log
{
echo "=== BAKE_V1214 $(date +%H:%M:%S) ==="

echo "--- step1: confirm container xgrammar 0.2.7 before commit"
V=$(docker exec lsv-test /opt/venv/bin/python3 -c 'from importlib.metadata import version; print(version("xgrammar"))' 2>&1)
echo "container xgrammar: $V"
if [ "$V" != "0.2.7" ]; then echo "ABORT: container not on 0.2.7"; exit 9; fi

echo "--- step2: sanity lane still healthy before we take it down"
curl -s -o /dev/null -w 'pre-bake health: %{http_code}\n' -m 4 http://localhost:8000/health

echo "--- step3: bake"
docker exec lsv-test touch /root/.llm_scaler_exp_v1214_baked
docker commit lsv-test llm-scaler-exp:v1.2.14 2>&1 | tail -1
docker images llm-scaler-exp:v1.2.14 --format 'v1.2.14={{.ID}} size={{.Size}}'

echo "--- step4: repoint boot script image v1.2.13 -> v1.2.14"
cp /root/build/repro_bootV1212.sh /root/build/repro_bootV1212.sh.pre_v1214
sed -i 's/llm-scaler-exp:v1\.2\.13/llm-scaler-exp:v1.2.14/g; s/(image v1\.2\.13)/(image v1.2.14)/g; s/v1\.2\.13 baked/v1.2.14 baked/g' /root/build/repro_bootV1212.sh
grep -n 'v1.2.14' /root/build/repro_bootV1212.sh | head -4

echo "--- step5: boot v1.2.14"
bash /root/build/repro_bootV1212.sh V1214XG2
BOOT_RC=$?
echo "boot_rc=$BOOT_RC"

echo "--- step6: post-boot xgrammar in engine container"
docker exec lsv-test /opt/venv/bin/python3 -c 'from importlib.metadata import version; print("booted xgrammar:", version("xgrammar"))' 2>&1

echo "--- step7: stall patcher baked proof (ALREADY x3)"
grep -c ALREADY /root/build/lce1/stall59_apply_bootV1212.log

echo "=== BAKE_V1214 COMPLETE $(date +%H:%M:%S) ==="
} > "$LOG" 2>&1
tail -25 "$LOG"
echo BAKE_DONE
