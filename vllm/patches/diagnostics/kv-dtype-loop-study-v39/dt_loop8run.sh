#!/bin/bash
# dt_loop8run.sh <dtype> <tag> [run4096] — boot dtype on prod:v1, run the
# fixed dt_loop (optional, @4096) then dt_loop8 (@8192, the 4 known
# trappers) from the host against localhost:8000.
set -u
DT="$1"
TAG="$2"
RUN4096="${3:-yes}"
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

if [ "$RUN4096" = "yes" ]; then
  python3 /root/build/dt_loop.py "$TAG" > "/root/build/dt-loop-${TAG}.out" 2>&1
  echo "loop4096 done: $TAG"
fi
python3 /root/build/dt_loop8.py "$TAG" > "/root/build/dt-loop8-${TAG}.out" 2>&1
echo "LOOP8RUN_DONE ${TAG}"
