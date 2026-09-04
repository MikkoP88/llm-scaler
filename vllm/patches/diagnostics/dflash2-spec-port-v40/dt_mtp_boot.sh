#!/bin/bash
# dt_mtp_boot.sh — stock MTP lane boot: serve directly (no patches) + poll.
set -u
echo "== boot stock MTP lane =="
bash /root/build/qwen38-dflash2/dt_mtp_serve.sh || { echo SERVE_BOOT_FAIL; exit 3; }
sleep 3
state=$(docker inspect -f '{{.State.Status}}' lsv-test 2>/dev/null || echo missing)
[ "$state" = "running" ] || { echo "CONTAINER_NOT_RUNNING state=$state"; exit 4; }
ok=0
for i in $(seq 1 90); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || true)
  if [ "$code" = "200" ]; then echo "HEALTH OK after ~$((i*10))s"; ok=1; break; fi
  st=$(docker inspect -f '{{.State.Status}}' lsv-test 2>/dev/null || echo gone)
  if [ "$st" != "running" ]; then
    echo "CONTAINER_EXITED"
    docker cp lsv-test:/root/dt_mtp.log /tmp/dt_mtp_boot.log >/dev/null 2>&1
    tail -40 /tmp/dt_mtp_boot.log
    exit 7
  fi
done
if [ "$ok" != 1 ]; then
  echo HEALTH_TIMEOUT
  docker cp lsv-test:/root/dt_mtp.log /tmp/dt_mtp_boot.log >/dev/null 2>&1
  tail -30 /tmp/dt_mtp_boot.log
  exit 8
fi
echo MTP_BOOT_OK
