#!/bin/bash
# v65 Phase-1 P1d: forced TRITON_ATTN A/B (test patch) + solo cold + revert
B=/root/build
OUT=$B/v65_p10_out.txt
V=/opt/venv/lib/python3.12/site-packages/vllm
{
echo "=== apply force-triton patch ==="
docker cp $B/patch_xpu_force_triton.py lsv-test:/root/patch_xpu_force_triton.py
docker exec lsv-test /opt/venv/bin/python3 /root/patch_xpu_force_triton.py --apply
docker exec lsv-test /opt/venv/bin/python3 /root/patch_xpu_force_triton.py --check
echo
echo "=== moe_backend resolution (fixed glob, inside container) ==="
docker exec lsv-test sh -c "grep -rn 'moe_backend' $V/config/ 2>/dev/null | grep -viE '^\s*#' | head -12"
echo
echo "=== env + relaunch ==="
docker exec lsv-test grep -q VLLM_V65_FORCE_TRITON /root/serve_user.sh || \
  docker exec lsv-test sed -i '1i export VLLM_V65_FORCE_TRITON=1' /root/serve_user.sh
docker exec lsv-test grep -n VLLM_V65_FORCE_TRITON /root/serve_user.sh | head -2
docker exec lsv-test pkill -f "vllm serve" || true
sleep 5
docker exec -d lsv-test bash /root/serve_user.sh
echo "relaunched, waiting health..."
CODE=000
for i in $(seq 1 66); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
  [ "$CODE" = "200" ] && break
  sleep 10
done
echo "HEALTH=$CODE"
if [ "$CODE" != "200" ]; then
  echo "BOOT_FAILED — evidence:"
  docker exec lsv-test sh -c "grep -iE 'V65_TEST|Using Triton|Traceback|TORCH_CHECK|error' /root/serve_full.log | tail -25"
  echo "(will revert patch+env)"
  docker exec lsv-test sed -i '/VLLM_V65_FORCE_TRITON/d' /root/serve_user.sh
  docker exec lsv-test /opt/venv/bin/python3 /root/patch_xpu_force_triton.py --revert
  docker exec lsv-test pkill -f "vllm serve" || true
  sleep 5
  docker exec -d lsv-test bash /root/serve_user.sh
  for i in $(seq 1 66); do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
    [ "$CODE" = "200" ] && break
    sleep 10
  done
  echo "RESTOCK_HEALTH=$CODE"
  echo "P10_ABORTED_BOOT_FAIL"
  exit 1
fi
echo
echo "=== backend confirmation (must say Triton) ==="
docker exec lsv-test sh -c "grep -E 'V65_TEST|Using Triton backend|Using Flash Attention backend' /root/serve_full.log | tail -5"
echo
echo "=== solo cold seed 212 (TRITON forced) ==="
date +%H:%M:%S
python3 $B/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 212 > $B/v65_p10_solo.txt 2>&1
date +%H:%M:%S
tail -4 $B/v65_p10_solo.txt
echo
echo "=== xgrammar quick sanity (2 runs) ==="
python3 $B/t2_xgrammar_probe.py http://127.0.0.1:8000 qwen3.8-27b-fp8 2 2>/dev/null | tail -4 || true
echo
echo "=== GDN cadence (step walls) ==="
docker exec lsv-test sh -c "grep 'V65_GDN' /root/serve_full.log | tail -8"
echo
echo "=== errors check ==="
docker exec lsv-test sh -c "grep -cE 'Traceback|DEVICE_LOST|FATAL' /root/serve_full.log" || true
echo
echo "=== revert patch+env + relaunch stock ==="
docker exec lsv-test sed -i '/VLLM_V65_FORCE_TRITON/d' /root/serve_user.sh
docker exec lsv-test /opt/venv/bin/python3 /root/patch_xpu_force_triton.py --revert
docker exec lsv-test /opt/venv/bin/python3 /root/patch_xpu_force_triton.py --check
docker exec lsv-test pkill -f "vllm serve" || true
sleep 5
docker exec -d lsv-test bash /root/serve_user.sh
CODE=000
for i in $(seq 1 66); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
  [ "$CODE" = "200" ] && break
  sleep 10
done
echo "RESTOCK_HEALTH=$CODE"
echo "P10_DONE"
} > $OUT 2>&1
echo "SCRIPT_EXIT=$?"
