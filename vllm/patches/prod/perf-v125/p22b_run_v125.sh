#!/bin/bash
# p22b_run_v125.sh — v125 P22B iteration: fp8-native SYCL gdn_attention
# E2E legs on lsv-test, AFTER build_vxk_v125_inc.sh swapped the patched
# _xpu_C.abi3.so AND AFTER the v1.2.24 ship completed (the .so and the
# python patch must never be present in a container that gets committed).
#   0.  verify .so string literal + apply p22b_python_patch.py + syntax
#   0b. fp8 primitive pre-flight (unchanged from stage A)
#   1.  fp16 CONTROL leg: graphs FULL_DECODE_ONLY, no env — proves the
#       rebuilt .so is behavior-neutral on the certified path (solo ~74)
#   2.  leg e4m3 NATIVE: dtype fp8_e4m3 + VLLM_XPU_GDN_FP8_NATIVE=1 +
#       graphs FULL_DECODE_ONLY RESTORED (no torch.unique in the native
#       path -> capture-safe in principle; if capture fails, the log
#       evidence decides a NONE retry — recorded, not silent)
#   3.  leg e5m2 NATIVE: same
#   4.  restore certified fp16 serve + jitwarm
# Boot evidence gate per fp8 leg: "v125 P22B FP8_NATIVE ENGAGED" printed
# once at first fp8 forward + dtype line in serve log + graphs captured.
# Correctness BEFORE speed: greedy probes + acceptance vs stage-A refs
# (e4m3: coherent text, 46.9% acceptance via bridge; e5m2: degenerate,
# 74-79% acceptance — both expected to hold or improve natively).
# Speed refs: solo 74.19 / 4x1024 152.25 (fp16 ESIMD) — fp8 legs run on
# the SYCL path so parity is NOT expected yet (that's P22-A); the claim
# to measure is bridge-penalty removal vs stage-A's ~5x degradation.
set -u
L=/root/build/lce1/p22b_v125.log
M=qwen3.8-27b-fp8
SP=/opt/venv/lib/python3.12/site-packages
# the legs below replace the serve inside the PRODUCTION lane container while
# lane-watchdog is ARMED post-v1.2.24-ship -> pause it for the whole round
# (removed only after the certified serve is verified back up; stays paused
# on any abort so the watchdog can never race a half-booted experimental leg)
touch /root/build/lane_watchdog.paused
echo "lane_watchdog paused for P22B legs ($(date +%T))"
{
echo "=== P22B V125 start $(date +%F' '%T) ==="

wait_health () {
  local i=0 t=${1:-480}
  while [ $i -lt $t ]; do
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && return 0
    sleep 5; i=$((i+5))
  done
  return 1
}

echo "--- 0. installed .so literal + python patch ---"
SOHITS=$(docker exec lsv-test sh -c "strings $SP/vllm_xpu_kernels/_xpu_C.abi3.so | grep -c 'fp8_e4m3fn'" || true)
echo "installed .so e4m3 literal hits=$SOHITS (need >=1)"
[ "$SOHITS" -ge 1 ] 2>/dev/null || { echo ABORT_SO_NOT_SWAPPED; exit 1; }
docker cp /root/build/p22b_python_patch.py lsv-test:/root/p22b_python_patch.py
POUT=$(docker exec lsv-test /opt/venv/bin/python /root/p22b_python_patch.py)
echo "$POUT"
echo "$POUT" | grep -qE "P22B_PYTHON_PATCHED|ALREADY_APPLIED" || { echo ABORT_PATCH; exit 1; }
docker exec lsv-test /opt/venv/bin/python -c "import ast; ast.parse(open('$SP/vllm/_xpu_ops.py').read()); print('SYNTAX_OK _xpu_ops.py')" \
  || { echo ABORT_SYNTAX; exit 1; }

echo "--- 0b. fp8 XPU primitive pre-flight ---"
PF=$(docker exec -i lsv-test /opt/venv/bin/python - <<'PYEOF'
import torch
fails = 0
for dt, name in ((torch.float8_e4m3fn, 'e4m3'), (torch.float8_e5m2, 'e5m2')):
    try:
        p = torch.zeros(64, 32, 32, dtype=dt, device='xpu')
        g = p.index_select(0, torch.arange(4, device='xpu')).to(torch.float16)
        g += 1.0
        p.view(torch.uint8).index_copy_(0, torch.arange(4, device='xpu'),
                                        g.to(dt).view(torch.uint8))
        print(f"FP8_OK {name}")
    except Exception as e:
        fails += 1
        print(f"FP8_FAIL {name}: {type(e).__name__}: {e}")
print(f"PREFLT_FAILS={fails}")
PYEOF
)
echo "$PF"
echo "$PF" | grep -q "PREFLT_FAILS=0" || { echo ABORT_FP8_PRIMITIVES; exit 1; }

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
        txt = r["choices"][0]["message"].get("content", "")
        print(f"{tag} http-ok len={len(txt)} contains391={'391' in txt} text={txt[:90]!r}")
    except Exception as e:
        print(f"{tag}_FAIL {type(e).__name__}: {e}")
ask("What is 17*23? Answer with the number only.", 40, "PROBE_MATH")
ask("List the numbers from 1 to 5, comma separated, nothing else.", 40, "PROBE_LIST")
ask("Write one paragraph (3 sentences) explaining why the sky is blue.", 120, "PROBE_PROSE")
PYEOF
}

# leg_runner <fmt|fp16> <tag> <envflag 0|1>
leg_runner () {
  local fmt=$1 tag=$2 envf=$3
  echo "=========== LEG $tag: dtype=$fmt native_env=$envf graphs=FULL_DECODE_ONLY ==========="
  if [ "$fmt" = "fp16" ]; then
    docker exec lsv-test sh -c "cp /root/serve_user.sh /root/serve_p22b_$tag.sh && chmod +x /root/serve_p22b_$tag.sh"
  else
    docker exec lsv-test sh -c "sed 's|mamba-ssm-cache-dtype float16|mamba-ssm-cache-dtype $fmt|; s|serve_full.log|serve_p22b_$tag.log|' /root/serve_user.sh > /root/serve_p22b_$tag.sh && chmod +x /root/serve_p22b_$tag.sh"
  fi
  if [ "$envf" = "1" ]; then
    docker exec lsv-test sed -i "1i export VLLM_XPU_GDN_FP8_NATIVE=1" /root/serve_p22b_$tag.sh
  fi
  local b e
  b=$(docker exec lsv-test sh -c "grep -c 'mamba-ssm-cache-dtype $fmt' /root/serve_p22b_$tag.sh" || true)
  e=$(docker exec lsv-test sh -c "grep -c 'VLLM_XPU_GDN_FP8_NATIVE=1' /root/serve_p22b_$tag.sh" || true)
  echo "$tag markers: dtype=$b (>=1) env=$e (want $envf)"
  [ "$b" -ge 1 ] 2>/dev/null || { echo "${tag}_VARIANT_BUILD_FAILED"; return 1; }
  docker exec lsv-test pkill -f 'vllm serve' || true
  sleep 8
  docker exec -d lsv-test bash /root/serve_p22b_$tag.sh
  echo "$tag launched, waiting for health..."
  if wait_health 480; then
    sleep 5
    echo "== $tag boot evidence =="
    docker exec lsv-test sh -c "grep -i 'mamba_ssm_cache_dtype' /root/serve_p22b_$tag.log | head -3" || true
    docker exec lsv-test sh -c "grep -m1 'P22B FP8_NATIVE ENGAGED' /root/serve_p22b_$tag.log" || true
    docker exec lsv-test sh -c "grep -i 'capture.*cudagraph\|cudagraph.*capture\|Capturing' /root/serve_p22b_$tag.log | head -3" || true
    cd /root/build
    echo "-- $tag numerics (greedy probes) --"
    numerics
    echo "== $tag SOLO 1x1024 x3 =="; python3 bench_genspeed.py 1 1024 3
    echo "== $tag ASYNC 4x1024 ==";  python3 bench_genspeed.py 4 1024 1
    echo "-- $tag post-traffic health + acceptance + errors --"
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && echo "post-traffic health OK"
    docker exec lsv-test sh -c "grep -i 'acceptance' /root/serve_p22b_$tag.log | tail -3" || true
    TB=$(docker exec lsv-test sh -c "grep -icE 'Traceback' /root/serve_p22b_$tag.log" || true)
    echo "$tag tracebacks=$TB"
  else
    echo "${tag}_BOOT_FAILED — evidence:"
    docker exec lsv-test tail -30 /root/serve_p22b_$tag.log
    docker exec lsv-test sh -c "grep -iE 'error|Traceback|not.?implemented|assert|capture' /root/serve_p22b_$tag.log | tail -15" || true
  fi
}

# 1. fp16 control — the rebuilt .so must be neutral on the certified path
leg_runner float16 ctrlfp16 0
echo "ctrlfp16 ref check: solo median expected ~74 (certified); big drop = .so regression"

# 2+3. native fp8 legs — graphs RESTORED, native env on
leg_runner fp8_e4m3 e4m3n 1
leg_runner fp8_e5m2 e5m2n 1

echo "--- restore certified fp16 serve + jitwarm ---"
docker exec lsv-test pkill -f 'vllm serve' || true
sleep 8
docker exec -d lsv-test bash /root/serve_user.sh
wait_health 480 || { echo ABORT_RESTORE_BOOT; exit 1; }
sleep 5
REST=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log" || true)
echo "certified restored: spec_lines=$REST (expect >=1) $(date +%T)"
[ "$REST" -ge 1 ] 2>/dev/null || { echo ABORT_RESTORE_NOT_SPEC4; exit 1; }
python3 /root/build/probe_jitwarm_v63.py http://127.0.0.1:8000 $M 731 >/dev/null 2>&1 || true
echo "=== P22B V125 DONE $(date +%T) ==="
# certified serve verified back up -> re-arm the watchdog
rm -f /root/build/lane_watchdog.paused
echo "lane_watchdog re-armed ($(date +%T))"
} > "$L" 2>&1
tail -160 "$L"
echo P22B_V125_DONE
