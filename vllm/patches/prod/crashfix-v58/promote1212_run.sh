#!/bin/bash
# promote1212_run.sh — §24 U: promote llm-scaler-exp:v1.2.12 to STANDING
# production lane (user directive). Order-critical sequence:
#   1. pause watchdog (flag)
#   2. verify v1.2.12 image + marker + baked crash-free config
#   3. teardown standing Q21b lane (v1.2.10, crash-band config)
#   4. boot v1.2.12 standing (repro_bootV1212.sh, gated)
#   5. certify (promote1212_cert.sh: settle + f8ref e4m3 + q17)
#   6. re-point watchdog relaunch lineage Q21b -> V1212
#      (write+mv+systemctl restart — never edit a running bash script)
#   7. arm watchdog (rm flag) + final verification
cd /root/build
set -u
LBL=PROMOTE1212

echo "[PROMO] 1. pause watchdog $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'

echo "[PROMO] 2. verify image"
docker image inspect llm-scaler-exp:v1.2.12 --format 'image OK {{.Id}}' || { echo "IMAGE MISSING"; exit 1; }
if docker run --rm --entrypoint /bin/bash llm-scaler-exp:v1.2.12 -c \
  'test -f /root/.llm_scaler_exp_v1212_baked && grep -q -- "--gpu-memory-utilization 0.8" /root/serve_user.sh && grep -q -- "--block-size 64" /root/serve_user.sh && grep -q -- "--kv-cache-dtype fp8_e4m3" /root/serve_user.sh && grep -q -- "--enable-prefix-caching" /root/serve_user.sh' \
  && ! docker run --rm --entrypoint /bin/bash llm-scaler-exp:v1.2.12 -c \
  'grep -q -- "--async-scheduling" /root/serve_user.sh'; then
  echo "BAKED CONFIG OK (gmu0.8 bs64 e4m3 pc-ON async-OFF)"
else
  echo "BAKED CONFIG WRONG"; exit 2
fi

echo "[PROMO] 3. teardown standing lane $(date +%H:%M:%S)"
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10

echo "[PROMO] 4. boot v1.2.12 standing $(date +%H:%M:%S)"
bash repro_bootV1212.sh "$LBL" > lce1/boot_$LBL.out 2>&1
grep -m1 HEALTH_OK lce1/boot_$LBL.out || { echo "BOOT_FAIL"; tail -25 lce1/boot_$LBL.out; exit 3; }
grep -E 'container launched|baked config verified|live_resets' lce1/boot_$LBL.out
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
docker exec lsv-test sh -c 'grep -oE -- "--gpu-memory-utilization [0-9.]+|--block-size [0-9]+|--kv-cache-dtype [a-z0-9_]+|--enable-prefix-caching" /root/serve_user.sh; echo async_count=$(grep -c -- "--async-scheduling" /root/serve_user.sh)'

echo "[PROMO] 5. certify $(date +%H:%M:%S)"
bash promote1212_cert.sh 2>&1 | tee lce1/pcert_$LBL.out

echo "[PROMO] 6. re-point watchdog lineage $(date +%H:%M:%S)"
echo "old refs: $(grep -c 'repro_bootQ21b' lane_watchdog.sh)"
sed 's@repro_bootQ21b\.sh@repro_bootV1212.sh@g' lane_watchdog.sh > lane_watchdog.sh.new
echo "new refs: $(grep -c 'repro_bootV1212' lane_watchdog.sh.new)"
mv lane_watchdog.sh.new lane_watchdog.sh
systemctl restart lane-watchdog
sleep 2
echo "watchdog_active=$(systemctl is-active lane-watchdog)"

echo "[PROMO] 7. arm watchdog + final verify $(date +%H:%M:%S)"
rm -f /root/build/lane_watchdog.paused
echo "pause_flag_present=$( [ -f /root/build/lane_watchdog.paused ] && echo yes || echo no)"
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
curl -s -o /dev/null -w 'health=%{http_code}\n' -m 5 http://localhost:8000/health
systemctl is-active lane-watchdog
tail -3 lane_watchdog.log
echo "PROMOTE1212_DONE $(date +%H:%M:%S)"
