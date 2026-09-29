#!/bin/bash
# p22a_repair_v125.sh — repair the 18:49 lane image regression discovered
# after run 5 (all three P22A legs failed on a v1.2.23 container).
#
# ROOT CAUSE (verified): the lane-watchdog SERVICE process started
# 15:06:10, BEFORE the 17:07 repoint to repro_bootV1224_prod.sh. Bash
# parses the whole `while` loop at first execution — editing
# lane_watchdog.sh on disk never updates the running process. The 18:49:21
# trigger (dbg window serve-down) fired the IN-MEMORY V1223 boot script
# -> docker rm + run of tag v1.2.23 -> container a2f00e326d8e on
# 23a07b1e5ff6 (v1.2.23). That recreate wiped ALL in-container round
# state: the P22B2 python patch (which also defines the
# _gdn_fp8_native_enabled helper), B2 serve logs, and the original
# .v125bak (v1.2.24's stock .so). v1.2.23 lacks the baked fp8
# MambaDType choices (v1.2.24 b13f56543057 HAS them: Literal["auto",
# "float32","float16","fp8_e4m3","fp8_e5m2"]) -> run-5 e4m3ns/e5m2ns
# died at argparse; ctrl-ns died in _arstage dynamo ('Unsupported
# hasattr call') = v1.2.23 artifact. All run-5 leg failures are
# image-regression artifacts, NOT P22A signals.
#
# REPAIR:
#   1. wait for run-5 P22A_V125_DONE (no boot races)
#   2. pause watchdog; systemctl restart lane-watchdog (fresh parse =
#      V1224 boot text); verify service ActiveEnterTimestamp advanced
#   3. bash repro_bootV1224_prod.sh REPAIR_V125 — full prod boot chain
#      on tag v1.2.24 (patchers idempotent, baked-marker gate exit 9,
#      serve-config gate exit 10, health loop)
#   4. posture gates: container image id MUST be b13f56543057...;
#      async-scheduling REQUIRED; spec-4 present; mamba dtype fp16
#      certified; barrier env =0
#   5. re-apply p22b2_python_patch.py (SYCL decode split + the
#      _gdn_fp8_native_enabled helper the P22A gate imports)
#   6. quick math probe through the fresh serve; re-arm watchdog
# Next (manual): re-run p22a_run_v125.sh (run 6) — on the fresh
# container every step re-applies cleanly (.v125bak = b13f stock .so,
# gate marker absent, fp8 CLI admitted).
set -u
L=/root/build/lce1/p22a_repair_v125.log
SP=/opt/venv/lib/python3.12/site-packages
WANT_IMG=sha256:b13f56543057906057fa25893d5cc17f63c81a7b2a417e84bf977eb57e95542e
touch /root/build/lane_watchdog.paused
{
echo "=== P22A REPAIR V125 start $(date +%F' '%T) ==="

echo "--- 1. wait for run-5 terminal state ---"
# Run 5 exited via ABORT_RESTORE_BOOT (v1.2.23 container + gate patch
# imports the wiped _gdn_fp8_native_enabled helper -> engine death) —
# accept DONE *or* abort marker; what matters is the runner is gone so
# no boot race with this repair.
i=0
while [ $i -lt 90 ]; do
  grep -qE 'P22A_V125_DONE|ABORT_RESTORE_BOOT' /root/build/lce1/p22a_v125.log 2>/dev/null && { echo "run-5 terminal state confirmed"; break; }
  sleep 20; i=$((i+20))
done
grep -qE 'P22A_V125_DONE|ABORT_RESTORE_BOOT' /root/build/lce1/p22a_v125.log || { echo "ABORT_RUN5_TIMEOUT"; exit 1; }
while pgrep -f 'p22a_run_v125.sh' >/dev/null; do
  sleep 10; i=$((i+10)); [ $i -gt 180 ] && { echo "ABORT_RUNNER_STILL_ALIVE"; exit 1; }
done
echo "run-5 runner exited"

echo "--- 2. restart lane-watchdog (fresh parse = V1224 boot text) ---"
systemctl restart lane-watchdog
sleep 3
NEW_TS=$(systemctl show lane-watchdog -p ActiveEnterTimestamp --value)
echo "lane-watchdog ActiveEnterTimestamp=$NEW_TS (was Mon 2026-09-28 15:06:10 UTC)"
[ -n "$NEW_TS" ] || { echo "ABORT_WATCHDOG_RESTART"; exit 1; }
tail -1 /root/build/lane_watchdog.log | sed 's/^/wd log: /'

echo "--- 3. full prod boot on v1.2.24 (repro_bootV1224_prod.sh) ---"
cd /root/build
if bash repro_bootV1224_prod.sh REPAIR_V125 > /root/build/lce1/boot_REPAIR_V125.out 2>&1; then
  echo "boot script exit 0"
else
  rc=$?
  echo "ABORT_BOOT_RC=$rc"; tail -25 /root/build/lce1/boot_REPAIR_V125.out; exit 1
fi
tail -8 /root/build/lce1/boot_REPAIR_V125.out | sed 's/^/boot: /'

echo "--- 4. posture gates ---"
IMG=$(docker inspect lsv-test --format '{{.Image}}')
echo "container image=$IMG"
echo "want           =$WANT_IMG"
[ "$IMG" = "$WANT_IMG" ] || { echo "ABORT_WRONG_IMAGE"; exit 1; }
ASY=$(docker exec lsv-test sh -c "grep -c -- '--async-scheduling' /root/serve_user.sh" || true)
SPE=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_user.sh" || true)
SSM=$(docker exec lsv-test sh -c "grep -c -- '--mamba-ssm-cache-dtype float16' /root/serve_user.sh" || true)
XG=$(docker exec lsv-test sh -c "grep -c 'VLLM_GUIDED_DECODING_BACKEND=xgrammar' /root/serve_user.sh" || true)
echo "posture: async=$ASY (>=1) spec=$SPE (>=1) ssm_fp16=$SSM (>=1) xgrammar_env_in_script=$XG"
[ "$ASY" -ge 1 ] && [ "$SPE" -ge 1 ] && [ "$SSM" -ge 1 ] || { echo "ABORT_POSTURE"; exit 1; }
docker exec lsv-test sh -c "grep -i 'mamba_ssm_cache_dtype' /root/serve_full.log | head -1" || true
SPECLOG=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log" || true)
echo "serve log spec lines=$SPECLOG (>=1)"
[ "$SPECLOG" -ge 1 ] || { echo "ABORT_SERVE_NOT_SPEC4"; exit 1; }
DT=$(docker exec lsv-test grep -n 'MambaDType = Literal' $SP/vllm/config/cache.py)
echo "baked MambaDType: $DT"
echo "$DT" | grep -q 'fp8_e4m3' || { echo "ABORT_NO_FP8_CHOICES"; exit 1; }

echo "--- 5. re-apply P22B2 python patch (SYCL split + fp8 helper) ---"
docker cp /root/build/p22b2_python_patch.py lsv-test:/root/p22b2_python_patch.py
POUT=$(docker exec lsv-test /opt/venv/bin/python /root/p22b2_python_patch.py)
echo "$POUT"
echo "$POUT" | grep -qE "P22B2_PYTHON_PATCHED|ALREADY_APPLIED" || { echo "ABORT_B2_PATCH"; exit 1; }
docker exec lsv-test /opt/venv/bin/python -c "import ast; ast.parse(open('$SP/vllm/_xpu_ops.py').read()); print('SYNTAX_OK _xpu_ops.py')" \
  || { echo "ABORT_B2_SYNTAX"; exit 1; }
docker exec lsv-test /opt/venv/bin/python -c "from vllm._xpu_ops import _gdn_fp8_native_enabled; print('helper import OK')" \
  || { echo "ABORT_HELPER_IMPORT"; exit 1; }

echo "--- 6. math probe + re-arm ---"
sleep 5
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
echo "$PR" | grep -q 'contains391=True' || echo "WARNING: math probe not 391 — inspect before legs"

rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed on V1224 lineage ($(date +%T))"
echo "=== P22A REPAIR V125 DONE $(date +%F' '%T) ==="
echo "P22A_REPAIR_V125_DONE"
} > "$L" 2>&1
tail -60 "$L"
echo P22A_REPAIR_SCRIPT_DONE
