#!/bin/bash
# p195_run_v124.sh — P19.5 stage A: python plumbing + SYCL-op fp8 bridge.
# E2E validation of --mamba-ssm-cache-dtype fp8_e4m3 AND fp8_e5m2 on the
# lane (P18-L1 had proven the fork rejects both at argparse pre-patch).
#   0.  apply p195a_python_plumbing.py (idempotent; inert on fp16) + syntax
#   0b. fp8 XPU primitive pre-flight: zeros/index_select/cast/index_copy_
#       for BOTH dtypes on xpu — catches missing backend cast support
#       BEFORE spending a 2-minute boot
#   1.  leg e4m3: variant serve (dtype + cudagraph_mode NONE — the bridge
#       gather is data-dependent-shape, capture-unsafe), health, dtype
#       evidence, greedy numerics probes, spec acceptance, solo x3 + 4x1024
#   2.  leg e5m2: same
#   3.  restore certified fp16 serve (+jitwarm) — patch stays applied (inert)
# Speeds here are parity-or-worse EXPECTED (bridge casts every step); they
# are NOT a ship signal — kernel-native stage B/C is where the win must
# come from. Certified refs: solo median 74.19, 4x1024 152.25 (P16).
set -u
L=/root/build/lce1/p195_stageA.log
M=qwen3.8-27b-fp8
SP=/opt/venv/lib/python3.12/site-packages
{
echo "=== P19.5 STAGE-A start $(date +%F' '%T) ==="

wait_health () {
  local i=0 t=${1:-420}
  while [ $i -lt $t ]; do
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && return 0
    sleep 5; i=$((i+5))
  done
  return 1
}

echo "--- 0. apply python plumbing ---"
docker cp /root/build/p195a_python_plumbing.py lsv-test:/root/p195a_python_plumbing.py
POUT=$(docker exec lsv-test /opt/venv/bin/python /root/p195a_python_plumbing.py)
echo "$POUT"
echo "$POUT" | grep -q "P195A_PATCHED_OK" || { echo ABORT_PATCH; exit 1; }
for f in config/cache.py model_executor/layers/mamba/mamba_utils.py _xpu_ops.py; do
  docker exec lsv-test /opt/venv/bin/python -c "import ast; ast.parse(open('$SP/vllm/$f').read()); print('SYNTAX_OK $f')" \
    || { echo "ABORT_SYNTAX $f"; exit 1; }
done

echo "--- 0b. fp8 XPU primitive pre-flight ---"
# NB: docker exec needs -i to feed the heredoc stdin (first run produced
# empty output and a false ABORT — lesson recorded).
PF=$(docker exec -i lsv-test /opt/venv/bin/python - <<'PYEOF'
import torch
dev = 'xpu'
fails = 0
for dt, name in ((torch.float8_e4m3fn, 'e4m3'), (torch.float8_e5m2, 'e5m2')):
    try:
        p = torch.zeros(64, 32, 32, dtype=dt, device=dev)
        idx = torch.arange(4, device=dev)
        g = p.index_select(0, idx).to(torch.float16)
        g += 1.0
        # index_copy_xpu is NOT implemented for fp8 dtypes (run-3 finding);
        # the bridge scatters through a byte-identical uint8 view instead.
        p.view(torch.uint8).index_copy_(0, idx, g.to(dt).view(torch.uint8))
        _ = float(p[0, 0, 0].to(torch.float32))
        print(f"FP8_OK {name} roundtrip={float(g[0,0,0]):.3f}")
    except Exception as e:
        fails += 1
        print(f"FP8_FAIL {name}: {type(e).__name__}: {e}")
print(f"PREFLT_FAILS={fails}")
PYEOF
)
echo "$PF"
echo "$PF" | grep -q "PREFLT_FAILS=0" || { echo ABORT_FP8_PRIMITIVES_UNSUPPORTED; exit 1; }

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

run_leg () {
  local fmt=$1 tag=$2
  echo "=========== LEG $tag: mamba-ssm-cache-dtype $fmt (+graphs NONE) ==========="
  docker exec lsv-test sh -c "sed 's|mamba-ssm-cache-dtype float16|mamba-ssm-cache-dtype $fmt|; s|\"cudagraph_mode\":\"FULL_DECODE_ONLY\"|\"cudagraph_mode\":\"NONE\"|; s|serve_full.log|serve_mfp8$tag.log|' /root/serve_user.sh > /root/serve_mfp8$tag.sh && chmod +x /root/serve_mfp8$tag.sh"
  local b g
  b=$(docker exec lsv-test sh -c "grep -c 'mamba-ssm-cache-dtype $fmt' /root/serve_mfp8$tag.sh" || true)
  g=$(docker exec lsv-test sh -c "grep -c 'cudagraph_mode.:NONE\|NONE' /root/serve_mfp8$tag.sh" || true)
  echo "$tag markers: dtype=$b (>=1) graphs_none=$g (>=1)"
  [ "$b" -ge 1 ] 2>/dev/null || { echo "${tag}_VARIANT_BUILD_FAILED"; return 1; }
  docker exec lsv-test pkill -f 'vllm serve' || true
  sleep 8
  docker exec -d lsv-test bash /root/serve_mfp8$tag.sh
  echo "$tag launched, waiting for health..."
  if wait_health 480; then
    sleep 5
    echo "== $tag boot OK — dtype evidence =="
    docker exec lsv-test sh -c "grep -i 'mamba_ssm_cache_dtype' /root/serve_mfp8$tag.log | head -3; grep -ic 'cudagraph' /root/serve_mfp8$tag.log" || true
    cd /root/build
    echo "-- $tag numerics (greedy probes) --"
    numerics
    echo "== $tag SOLO 1x1024 x3 =="; python3 bench_genspeed.py 1 1024 3
    echo "== $tag ASYNC 4x1024 ==";  python3 bench_genspeed.py 4 1024 1
    echo "-- $tag post-traffic health + acceptance --"
    curl -s -o /dev/null -m 3 http://127.0.0.1:8000/health && echo "post-traffic health OK"
    docker exec lsv-test sh -c "grep -i 'acceptance' /root/serve_mfp8$tag.log | tail -3" || true
    docker exec lsv-test sh -c "grep -icE 'Traceback' /root/serve_mfp8$tag.log" || true
  else
    echo "${tag}_BOOT_FAILED — evidence:"
    docker exec lsv-test tail -25 /root/serve_mfp8$tag.log
    docker exec lsv-test sh -c "grep -iE 'error|Traceback|not.?implemented|assert' /root/serve_mfp8$tag.log | tail -12" || true
  fi
}

run_leg fp8_e4m3 e4m3
run_leg fp8_e5m2 e5m2

echo "--- restore certified fp16 serve (patch stays applied; inert on fp16) ---"
docker exec lsv-test pkill -f 'vllm serve' || true
sleep 8
docker exec -d lsv-test bash /root/serve_user.sh
wait_health 480 || { echo ABORT_RESTORE_BOOT; exit 1; }
sleep 5
REST=$(docker exec lsv-test sh -c "grep -c 'num_speculative_tokens' /root/serve_full.log" || true)
echo "certified restored: spec_lines=$REST (expect >=1) $(date +%T)"
[ "$REST" -ge 1 ] 2>/dev/null || { echo ABORT_RESTORE_NOT_SPEC4; exit 1; }
python3 /root/build/probe_jitwarm_v63.py http://127.0.0.1:8000 $M 731 >/dev/null 2>&1 || true
echo "=== P19.5 STAGE-A DONE $(date +%T) ==="
} > "$L" 2>&1
tail -120 "$L"
echo P195_STAGE_A_DONE
