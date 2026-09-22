#!/bin/bash
# v65 Phase-1 close-out: revert layer probe + env, relaunch stock, verify clean
B=/root/build
OUT=$B/v65_p11_out.txt
{
echo "=== revert layer probe ==="
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v65_lt.py --revert
docker exec lsv-test /opt/venv/bin/python3 /root/patch_v65_lt.py --check
echo
echo "=== remove probe env ==="
docker exec lsv-test sed -i '/VLLM_V65_LAYER_LOG/d' /root/serve_user.sh
docker exec lsv-test sh -c "grep -c VLLM_V65_LAYER_LOG /root/serve_user.sh" || true
echo
echo "=== relaunch stock ==="
docker exec lsv-test pkill -f "vllm serve" || true
sleep 5
docker exec -d lsv-test bash /root/serve_user.sh
CODE=000
for i in $(seq 1 66); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
  [ "$CODE" = "200" ] && break
  sleep 10
done
echo "HEALTH=$CODE"
echo
echo "=== clean-boot verification ==="
docker exec lsv-test sh -c "tail -300 /root/serve_full.log | grep -cE 'V65_ATTN|V65_GDN|V65_LT'" || true
echo "(above must be 0 — no probe lines in fresh boot)"
docker exec lsv-test sh -c "tail -300 /root/serve_full.log | grep -E 'Using Flash Attention backend|FlashAttention version' | tail -3"
docker exec lsv-test sh -c "tail -300 /root/serve_full.log | grep -cE 'Traceback|DEVICE_LOST|FATAL'" || true
echo
echo "=== xgrammar sanity (2 runs) ==="
python3 $B/t2_xgrammar_probe.py http://127.0.0.1:8000 qwen3.8-27b-fp8 2 2>/dev/null | tail -5 || echo "XGRAMMAR_PROBE_MISSING"
echo
echo "=== quick S1 generation sanity ==="
python3 $B/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 213 200 2>/dev/null | tail -3 || true
echo "P11_DONE"
} > $OUT 2>&1
echo "SCRIPT_EXIT=$?"
