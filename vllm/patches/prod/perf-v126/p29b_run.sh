#!/bin/bash
# p29b_run.sh — v126 P29B window: swap the v129 ESIMD .so into lsv-test
# (serve DOWN), run the op-level correctness harness (p29b_check.py) and
# the C3 speed A/B (p22c3_ops.py), then restore the pre-window lane
# (old .so + serve_user.sh) exactly as p22c3_run.sh does.
# Gate: p29b P29B_VERDICT: PASS  +  p22c3 every fp8 cell <= 1.00x fp16.
set -uo pipefail
NEW_SO=/root/build/v129_so/custom_esimd_kernels_lgrf.so
PKG=/opt/venv/lib/python3.12/site-packages/custom_esimd_kernels_vllm
SO_NAME=custom_esimd_kernels_lgrf.cpython-312-x86_64-linux-gnu.so
LOG=/root/build/lce1/p29b_window.log
{
echo "=== P29B window start $(date +%F' '%T) ==="

[ -f "$NEW_SO" ] || { echo ABORT_NO_NEW_SO; exit 1; }
touch /root/build/lane_watchdog.paused

echo "--- 1. teardown serve ---"
docker exec lsv-test pkill -f "vllm serve" 2>/dev/null || true
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://localhost:8000/health 2>/dev/null || echo 000)
  [ "$code" = "000" ] && break
  sleep 5
done
echo "health=$code (000=down)"

echo "--- 2. swap .so (backup old) ---"
docker exec lsv-test bash -c "[ -f $PKG/$SO_NAME.v129swap ] || cp -a $PKG/$SO_NAME $PKG/$SO_NAME.v129swap; ls -la $PKG/$SO_NAME*"
docker cp "$NEW_SO" lsv-test:"$PKG/$SO_NAME"
docker exec lsv-test sha256sum "$PKG/$SO_NAME"
docker exec lsv-test /opt/venv/bin/python -c \
  "from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq, esimd_gdn_conv_fused_seq_spec; print('IMPORT_OK')"

echo "--- 3. correctness (p29b) ---"
docker cp /root/build/p29b_check.py lsv-test:/root/p29b_check.py
docker exec lsv-test /opt/venv/bin/python /root/p29b_check.py 2>&1
RC=$?
echo "p29b rc=$RC"
[ $RC -eq 0 ] || { echo P29B_CHECK_FAILED; }

echo "--- 4. speed A/B (p22c3) ---"
docker cp /root/build/p22c3_ops.py lsv-test:/root/p22c3_ops.py
docker exec lsv-test /opt/venv/bin/python /root/p22c3_ops.py 2>&1 | tail -40
echo "P29B_WINDOW_DONE $(date +%T)"
} > "$LOG" 2>&1

echo "--- 5. restore pre-window lane ---"
docker exec lsv-test bash -c "cp -a $PKG/$SO_NAME.v129swap $PKG/$SO_NAME && sha256sum $PKG/$SO_NAME"
docker exec lsv-test sh -c ": > /root/serve_full.log"
docker exec -d lsv-test bash /root/serve_user.sh
for i in $(seq 1 90); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    docker exec lsv-test grep -c "'{\"method\":\"mtp\",\"num_speculative_tokens\":4}'" /root/serve_user.sh
    echo "LANE_RESTORED $(date +%H:%M:%S)"
    break
  fi
  docker ps --format '{{.Names}}' | grep -q '^lsv-test$' || { echo CONTAINER_DIED; break; }
done
tail -70 "$LOG"
echo P29B_RUN_SCRIPT_DONE
