#!/bin/bash
# v39_apply.sh <dtype> <tag> — double-cycle v39a boot: fresh container,
# apply nibble-split patch, docker restart (bootp's non-baked flow, but the
# patch step is v39_tq_nibble.py). Leaves patched server healthy on :8000.
set -u
DT="${1:-turboquant_4bit_nc}"
TAG="${2:-dt_v39a}"
IMG=llm-scaler-prod:v1

poll() {
  for i in $(seq 1 60); do
    sleep 10
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null)
    if [ "$code" = "200" ]; then echo "HEALTH OK after ~$((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
      echo "CONTAINER DIED"; docker cp lsv-test:/root/${TAG}.log /tmp/${TAG}_boot.log >/dev/null 2>&1; tail -30 /tmp/${TAG}_boot.log; return 7
    fi
  done
  echo "HEALTH TIMEOUT"; docker cp lsv-test:/root/${TAG}.log /tmp/${TAG}_boot.log >/dev/null 2>&1; tail -25 /tmp/${TAG}_boot.log; return 8
}

docker rm -f lsv-test >/dev/null 2>&1
bash /root/build/serve_user_nospec.sh '' '' "$TAG" "--kv-cache-dtype ${DT}" 512 "$IMG"
poll || exit 9
docker cp /root/build/v39_tq_nibble.py lsv-test:/root/
docker exec lsv-test python3 /root/v39_tq_nibble.py || { echo PATCH_FAIL; exit 10; }
docker cp /root/build/v39_check.py lsv-test:/root/
docker restart lsv-test >/dev/null
poll || exit 11
echo "V39_BOOT_OK dtype=${DT} tag=${TAG}"
