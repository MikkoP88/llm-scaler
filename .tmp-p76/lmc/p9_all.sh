#!/bin/bash
# p9_all.sh — P9 full perf matrix driver (user directive 2026-10-10):
# LMCache implementation vs baseline llm-scaler-exp:v1.2.32, prefill+decode,
# on the now-clean CCL lane. Legs sequential:
#   1. f16on  — lsv-test, mamba float16, LMCache ON  (ratio 2)
#   2. f16off — lsv-test, mamba float16, LMCache OFF
#   3. img32  — pristine container from llm-scaler-exp:v1.2.32, float16
# Ends by restoring the standing ON shape + engine on lsv-test.
set -u
B=/root/build
SP=/root/serve_lmc_v133.sh

cp $SP $B/serve_p9_on_bank.sh

bash $B/leg_p9.sh f16on float16 2 on   > $B/p9_leg_f16on.out  2>&1
echo "F16ON_EXIT=$?"
bash $B/leg_p9.sh f16off float16 2 off > $B/p9_leg_f16off.out 2>&1
echo "F16OFF_EXIT=$?"
bash $B/leg_p9img.sh                   > $B/p9_leg_img32.out  2>&1
echo "IMG32_EXIT=$?"

# restore standing ON shape + engine (post-matrix lane state)
cp $B/serve_p9_on_bank.sh $SP
docker cp $SP lsv-test:$SP
for p in $(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-test kill -TERM $p 2>/dev/null || true
done
sleep 14
for p in $(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-test kill -9 $p 2>/dev/null || true
done
sleep 4
docker exec -d lsv-test bash $SP
ok=0
for i in $(seq 1 110); do
  code=$(docker exec lsv-test sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; break; }
  sleep 10
done
echo "RESTORE_HEALTH_OK=$ok"
echo "P9_ALL_DONE $(date -u +%H:%M:%S)"
