#!/bin/bash
# p22a_run_v125.sh — v125 P22A: ESIMD fp8-native swap + op smoke + legs.
# The rebuilt custom_esimd_kernels_lgrf .so (fp8 StateT templating,
# build OK 18:14, sha 5f9b0378) is swapped into lsv-test and validated:
#   0. pause watchdog; backup old .so (.v125bak); swap; fresh import
#   1. p22a_smoke.py — op-level fp8 correctness gate (zero-state parity,
#      exact-load identity, NaN scans; both ops, both formats, prod TP2
#      shape). ABORT => restore old .so, re-arm, exit (no serve touched)
#   2. p22a_gate_patch.py — admit fp8 SSM pools into the ESIMD decode
#      path (ONLY after the matching .so is installed + smoke-passed)
#   3. legs, SPEC STRIPPED (the ESIMD surface engages only on non-spec
#      decode; spec x4 batches belong to the SYCL surface = P22B2):
#        ctrl-ns  fp16, no env  -> .so+gate neutrality (ref solo ~73-74;
#                                  must show NO ESIMD_FP8_ELIGIBLE line)
#        e4m3ns   fp8_e4m3 + VLLM_XPU_GDN_FP8_NATIVE=1 -> ESIMD fp8
#        e5m2ns   fp8_e5m2 + env -> ESIMD fp8
#      Fallback signal: 'P22B FP8_NATIVE ENGAGED' (SYCL) appearing in a
#      no-spec fp8 leg means the ESIMD gate did NOT engage — recorded.
#   4. restore certified fp16 spec-4 serve + jitwarm + watchdog re-arm
set -u
L=/root/build/lce1/p22a_v125.log
M=qwen3.8-27b-fp8
SP=/opt/venv/lib/python3.12/site-packages
NEW_SO=/root/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/custom_esimd_kernels_lgrf.cpython-312-x86_64-linux-gnu.so
OLD_SO=$SP/custom_esimd_kernels_vllm/custom_esimd_kernels_lgrf.cpython-312-x86_64-linux-gnu.so
touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused for P22A round ($(date +%T))"
{
echo "=== P22A V125 start $(date +%F' '%T) ==="

wait_health () {
  local i=0 t=${1:-480}
  while [ $i -lt $t ]; do
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && return 0
    sleep 5; i=$((i+5))
  done
  return 1
}

abort_restore () {
  echo "ABORT_RESTORE: putting back the pre-P22A .so"
  docker exec lsv-test sh -c "cp -a '$OLD_SO.v125bak' '$OLD_SO'" && echo "old .so restored"
  docker exec -d lsv-test bash /root/serve_user.sh
  wait_health 480 && echo "certified serve back up after abort"
  python3 /root/build/probe_jitwarm_v63.py http://127.0.0.1:8000 $M 731 >/dev/null 2>&1 || true
  rm -f /root/build/lane_watchdog.paused
  echo "lane_watchdog re-armed after abort ($(date +%T))"
  exit 1
}

echo "--- 0. swap .so ---"
[ -f "$NEW_SO" ] || { echo ABORT_NO_NEW_SO; exit 1; }
SOBAK_OK=$(docker exec lsv-test sh -c "[ -f '$OLD_SO.v125bak' ] && echo 1 || echo 0")
if [ "$SOBAK_OK" = "0" ]; then
  docker exec lsv-test cp -a "$OLD_SO" "$OLD_SO.v125bak" && echo "backup .v125bak created"
else
  echo "backup .v125bak already present"
fi
docker exec lsv-test sha256sum "$OLD_SO" | awk '{print "old so sha:", $1}'
docker cp "$NEW_SO" lsv-test:"$OLD_SO"
docker exec lsv-test sha256sum "$OLD_SO" | awk '{print "new so sha:", $1}'
sha256sum "$NEW_SO" | awk '{print "src so sha:", $1}'
docker exec lsv-test /opt/venv/bin/python -c "
import custom_esimd_kernels_vllm as c
from custom_esimd_kernels_vllm import esimd_gdn_conv_fused, esimd_gdn_conv_fused_seq
print('IMPORT_OK', c.__file__)
" || { echo ABORT_IMPORT; abort_restore; }

echo "--- 1. op-level fp8 smoke ---"
docker cp /root/build/p22a_smoke.py lsv-test:/root/p22a_smoke.py
docker exec lsv-test /opt/venv/bin/python /root/p22a_smoke.py
SMOKE_RC=$?
echo "smoke rc=$SMOKE_RC"
[ $SMOKE_RC -eq 0 ] || { echo ABORT_SMOKE; abort_restore; }

echo "--- 2. gate patch (fp8 SSM pools admitted, native env only) ---"
docker cp /root/build/p22a_gate_patch.py lsv-test:/root/p22a_gate_patch.py
GOUT=$(docker exec lsv-test /opt/venv/bin/python /root/p22a_gate_patch.py)
echo "$GOUT"
echo "$GOUT" | grep -qE "P22A_GATE_PATCHED|ALREADY_APPLIED" || { echo ABORT_GATE; abort_restore; }
docker exec lsv-test /opt/venv/bin/python -c "import ast; ast.parse(open('$SP/vllm/model_executor/layers/mamba/gdn_linear_attn.py').read()); print('SYNTAX_OK gdn_linear_attn.py')" \
  || { echo ABORT_SYNTAX; abort_restore; }

numerics () {
  python3 - "$M" <<'PYEOF'
import json, sys, urllib.request
m = sys.argv[1]
def ask(prompt, mt, tag):
    body = json.dumps({"model": m, "max_tokens": mt, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=180))
        txt = r["choices"][0]["message"].get("content") or ""
        print(f"{tag} http-ok len={len(txt)} contains391={'391' in txt} text={txt[:90]!r}")
    except Exception as e:
        print(f"{tag}_FAIL {type(e).__name__}: {e}")
ask("What is 17*23? Answer with the number only.", 40, "PROBE_MATH")
ask("List the numbers from 1 to 5, comma separated, nothing else.", 40, "PROBE_LIST")
ask("Write one paragraph (3 sentences) explaining why the sky is blue.", 120, "PROBE_PROSE")
PYEOF
}

# leg_runner_ns <fmt|fp16> <tag> <envflag 0|1>  (spec always stripped)
leg_runner_ns () {
  local fmt=$1 tag=$2 envf=$3
  echo "=========== LEG $tag: dtype=$fmt native_env=$envf spec=STRIPPED ==========="
  if [ "$fmt" = "fp16" ]; then
    docker exec lsv-test sh -c "sed '/--speculative-config/d; s|serve_full.log|serve_p22a_$tag.log|' /root/serve_user.sh > /root/serve_p22a_$tag.sh && chmod +x /root/serve_p22a_$tag.sh"
  else
    docker exec lsv-test sh -c "sed '/--speculative-config/d; s|mamba-ssm-cache-dtype float16|mamba-ssm-cache-dtype $fmt|; s|serve_full.log|serve_p22a_$tag.log|' /root/serve_user.sh > /root/serve_p22a_$tag.sh && chmod +x /root/serve_p22a_$tag.sh"
  fi
  if [ "$envf" = "1" ]; then
    docker exec lsv-test sed -i "1i export VLLM_XPU_GDN_FP8_NATIVE=1" /root/serve_p22a_$tag.sh
  fi
  local b e s
  b=$(docker exec lsv-test sh -c "grep -c 'mamba-ssm-cache-dtype $fmt' /root/serve_p22a_$tag.sh" || true)
  e=$(docker exec lsv-test sh -c "grep -c 'VLLM_XPU_GDN_FP8_NATIVE=1' /root/serve_p22a_$tag.sh" || true)
  s=$(docker exec lsv-test sh -c "grep -c 'speculative-config' /root/serve_p22a_$tag.sh" || true)
  echo "$tag markers: dtype=$b (>=1) env=$e (want $envf) spec_lines=$s (want 0)"
  [ "$b" -ge 1 ] 2>/dev/null || { echo "${tag}_VARIANT_BUILD_FAILED"; return 1; }
  [ "$s" -eq 0 ] 2>/dev/null || { echo "${tag}_SPEC_NOT_STRIPPED"; return 1; }
  docker exec lsv-test pkill -f 'vllm serve' || true
  sleep 8
  docker exec -d lsv-test bash /root/serve_p22a_$tag.sh
  echo "$tag launched, waiting for health..."
  if wait_health 480; then
    sleep 5
    echo "== $tag boot evidence =="
    docker exec lsv-test sh -c "grep -i 'mamba_ssm_cache_dtype' /root/serve_p22a_$tag.log | head -2" || true
    docker exec lsv-test sh -c "grep -m1 'ESIMD_FP8_ELIGIBLE' /root/serve_p22a_$tag.log" || echo "$tag: no ESIMD_FP8_ELIGIBLE line"
    docker exec lsv-test sh -c "grep -c 'P22B FP8_NATIVE ENGAGED' /root/serve_p22a_$tag.log" || true
    cd /root/build
    echo "-- $tag numerics (greedy probes) --"
    numerics
    echo "== $tag SOLO 1x1024 x3 =="; python3 bench_genspeed.py 1 1024 3
    echo "== $tag ASYNC 4x1024 ==";  python3 bench_genspeed.py 4 1024 1
    echo "-- $tag post-traffic health + errors --"
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && echo "post-traffic health OK"
    TB=$(docker exec lsv-test sh -c "grep -icE 'Traceback' /root/serve_p22a_$tag.log" || true)
    echo "$tag tracebacks=$TB"
  else
    echo "${tag}_BOOT_FAILED — evidence:"
    docker exec lsv-test tail -30 /root/serve_p22a_$tag.log
    docker exec lsv-test sh -c "grep -iE 'error|Traceback|not.?implemented|assert' /root/serve_p22a_$tag.log | tail -15" || true
  fi
}

leg_runner_ns float16 ctrl-ns 0
leg_runner_ns fp8_e4m3 e4m3ns 1
leg_runner_ns fp8_e5m2 e5m2ns 1

echo "--- restore certified fp16 spec-4 serve + jitwarm ---"
docker exec lsv-test pkill -f 'vllm serve' || true
sleep 8
docker exec -d lsv-test bash /root/serve_user.sh
wait_health 480 || { echo ABORT_RESTORE_BOOT; exit 1; }
sleep 5
REST=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log" || true)
echo "certified restored: spec_lines=$REST (expect >=1) $(date +%T)"
[ "$REST" -ge 1 ] 2>/dev/null || { echo ABORT_RESTORE_NOT_SPEC4; exit 1; }
python3 /root/build/probe_jitwarm_v63.py http://127.0.0.1:8000 $M 731 >/dev/null 2>&1 || true
echo "=== P22A V125 DONE $(date +%T) ==="
rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
echo "P22A_V125_DONE"
} > "$L" 2>&1
tail -160 "$L"
echo P22A_RUN_SCRIPT_DONE
