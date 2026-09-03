#!/bin/bash
# dt_bootbench.sh <dtype> <tag> [envstr] — boot dtype (prod:v1) with extra
# env (space-sep K=V), run dt_bench only. For env-knob A/B experiments.
set -u
DT="$1"
TAG="$2"
ENVV="${3:-}"
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
bash /root/build/serve_user_nospec.sh '' "$ENVV" "$TAG" "--kv-cache-dtype ${DT}" 512 "$IMG"
poll || exit 9
python3 /root/build/dt_bench.py "$TAG" > "/root/build/dt-bench-${TAG}.out" 2>&1
echo "BOOTBENCH_DONE ${TAG}"
