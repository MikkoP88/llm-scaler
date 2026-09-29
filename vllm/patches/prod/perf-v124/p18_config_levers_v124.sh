#!/bin/bash
# p18_config_levers_v124.sh — P18: config lever A/Bs on the running lane.
# L1: --mamba-ssm-cache-dtype fp8_e4m3 (certified = float16) — halves mamba
#     state bytes per stream; fast accept/reject at boot, then generation
#     sanity + genspeed vs certified (solo 74.19 median / 4x1024 152.25, P16).
# L2: --max-num-batched-tokens 16384 (certified = 8192) — bigger prefill
#     chunks; measured on COLD big-prompt TTFT (probe_solo_cold seed114,
#     certified fresh-serve ref = 101.76 s, P15) + genspeed no-regression.
# Certified serve restored + verified after EACH leg. No image changes.
set -u
L=/root/build/lce1/p18_config_levers.log
M=qwen3.8-27b-fp8
{
echo "=== P18 CONFIG LEVERS start $(date +%F' '%T) ==="

wait_health () {
  local i=0 t=${1:-300}
  while [ $i -lt $t ]; do
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && return 0
    sleep 5; i=$((i+5))
  done
  return 1
}
restore_certified () {
  docker exec lsv-test pkill -f 'vllm serve' || true
  sleep 8
  docker exec -d lsv-test bash /root/serve_user.sh
  wait_health 420 || { echo ABORT_RESTORE_BOOT; return 1; }
  sleep 5
  local n; n=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log" || true)
  echo "certified restored: spec_lines=$n (expect >=1) $(date +%T)"
  python3 /root/build/probe_jitwarm_v63.py http://127.0.0.1:8000 $M 731 >/dev/null 2>&1 || true
  return 0
}

echo "=========== LEG L1: mamba-ssm-cache-dtype fp8_e4m3 ==========="
docker exec lsv-test sh -c "sed 's|mamba-ssm-cache-dtype float16|mamba-ssm-cache-dtype fp8_e4m3|; s|serve_full.log|serve_mfp8.log|' /root/serve_user.sh > /root/serve_user_mfp8.sh && chmod +x /root/serve_user_mfp8.sh"
L1B=$(docker exec lsv-test sh -c "grep -c 'mamba-ssm-cache-dtype fp8_e4m3' /root/serve_user_mfp8.sh" || true)
echo "l1_variant_marker=$L1B (expect >=1)"
if [ "$L1B" -ge 1 ] 2>/dev/null; then
  docker exec lsv-test pkill -f 'vllm serve' || true
  sleep 8
  docker exec -d lsv-test bash /root/serve_user_mfp8.sh
  echo "l1 variant launched, waiting for health..."
  if wait_health 420; then
    sleep 5
    L1D=$(docker exec lsv-test sh -c "grep -ci 'mamba_ssm_cache_dtype=fp8\|mamba_ssm_cache_dtype.:fp8\|fp8_e4m3' /root/serve_mfp8.log" || true)
    echo "l1_boot OK; fp8-confirm-lines=$L1D (expect >=1); sample dtype line:"
    docker exec lsv-test sh -c "grep -i 'mamba_ssm_cache_dtype' /root/serve_mfp8.log | head -2"
    cd /root/build
    echo "== L1 MFP8 SOLO 1x1024 x3 ==";  python3 bench_genspeed.py 1 1024 3
    echo "== L1 MFP8 ASYNC 4x1024 ==";   python3 bench_genspeed.py 4 1024 1
    echo "-- l1 generation sanity (fixed short completion) --"
    python3 - "$M" <<'PYEOF'
import json, sys, urllib.request
m = sys.argv[1]
body = json.dumps({"model": m, "max_tokens": 40, "temperature": 0,
                   "messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}]}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
try:
    r = json.load(urllib.request.urlopen(req, timeout=120))
    txt = r["choices"][0]["message"].get("content", "")
    print(f"L1_SANITY http-ok text={txt[:80]!r} contains391={'391' in txt}")
except Exception as e:
    print(f"L1_SANITY_FAIL {type(e).__name__}: {e}")
PYEOF
  else
    echo "L1_REJECTED_AT_BOOT (no health in 420s) — dtype rejection evidence:"
    docker exec lsv-test sh -c "grep -iE 'mamba|dtype|not support|invalid|error' /root/serve_mfp8.log | tail -15"
    docker exec lsv-test tail -10 /root/serve_mfp8.log
  fi
else
  echo "L1_VARIANT_BUILD_FAILED"
fi
echo "-- restoring certified (leg L1 end) --"
restore_certified || { echo ABORT_CANNOT_RESTORE; exit 1; }

echo "=========== LEG L2: max-num-batched-tokens 16384 (cold-TTFT) ==========="
docker exec lsv-test sh -c "sed 's|max-num-batched-tokens 8192|max-num-batched-tokens 16384|; s|serve_full.log|serve_mnbt16k.log|' /root/serve_user.sh > /root/serve_user_mnbt16k.sh && chmod +x /root/serve_user_mnbt16k.sh"
L2B=$(docker exec lsv-test sh -c "grep -c 'max-num-batched-tokens 16384' /root/serve_user_mnbt16k.sh" || true)
echo "l2_variant_marker=$L2B (expect >=1)"
if [ "$L2B" -ge 1 ] 2>/dev/null; then
  docker exec lsv-test pkill -f 'vllm serve' || true
  sleep 8
  docker exec -d lsv-test bash /root/serve_user_mnbt16k.sh
  echo "l2 variant launched, waiting for health..."
  if wait_health 420; then
    sleep 5
    echo "== L2 COLD big-prompt TTFT (fresh serve, seed114; certified ref 101.76s) =="
    cd /root/build
    python3 probe_solo_cold.py http://127.0.0.1:8000 $M 114
    echo "== L2 MNBT16K SOLO 1x1024 x3 (no-regression check) =="
    python3 bench_genspeed.py 1 1024 3
    echo "== L2 MNBT16K ASYNC 4x1024 =="
    python3 bench_genspeed.py 4 1024 1
  else
    echo "L2_BOOT_FAILED — evidence:"
    docker exec lsv-test tail -15 /root/serve_mnbt16k.log
  fi
else
  echo "L2_VARIANT_BUILD_FAILED"
fi
echo "-- restoring certified (leg L2 end) --"
restore_certified || { echo ABORT_CANNOT_RESTORE; exit 1; }

echo "=== P18 DONE $(date +%T) ==="
} > "$L" 2>&1
tail -80 "$L"
echo P18_DONE
