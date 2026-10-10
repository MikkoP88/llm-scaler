#!/bin/bash
# freshleg2.sh N — redo of leg 8 WITH verification: fresh container + 8
# patched files, stdout to file, assert patched markers present in log,
# then probe N.
set -u
N=${1:-100}
B=/root/build
V=/opt/venv/lib/python3.12/site-packages/vllm
SP=/root/serve_lmc_v133.sh

echo "=== FRESHLEG2 START n=$N $(date -u +%H:%M:%S)"

# stop lsv-test engine
for p in $(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-test kill -TERM $p 2>/dev/null || true
done
sleep 14
for p in $(docker exec lsv-test pgrep -f "vllm serve|[V]LLM::" 2>/dev/null); do
  docker exec lsv-test kill -9 $p 2>/dev/null || true
done
sleep 4

docker rm -f lsv-fresh >/dev/null 2>&1 || true
# shellcheck disable=SC2046
docker run -dit --name lsv-fresh --privileged --network host --ipc host \
  -v /dev/dri:/dev/dri \
  -v /models/dflash2:/models/dflash2:ro \
  -v /models/drafter-fp8-v5:/models/drafter:ro \
  -v /models/swift-qwen3.8-27b:/models/target:ro \
  -v /root/build/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro \
  --env-file $B/fresh_env.txt \
  -w /llm-scaler/vllm --entrypoint bash \
  llm-scaler-exp:v1.2.32 || { echo DOCKER_RUN_FAIL; exit 1; }

for f in mamba_utils.py gpu_model_runner.py utils.py block_pool.py \
         kv_cache_manager.py scheduler.py output.py \
         single_type_kv_cache_manager.py; do
  case $f in
    mamba_utils.py|gpu_model_runner.py|utils.py) d=v1/worker ;;
    output.py|scheduler.py) d=v1/core/sched ;;
    *) d=v1/core ;;
  esac
  docker cp $B/pdiff/patched_$f lsv-fresh:$V/$d/$f || { echo CP_FAIL $f; exit 1; }
done
echo "8 patched files installed"
echo "--- md5 verify (patched vs installed):"
for f in mamba_utils.py scheduler.py; do
  case $f in mamba_utils.py) d=v1/worker ;; *) d=v1/core/sched ;; esac
  a=$(md5sum $B/pdiff/patched_$f | cut -d" " -f1)
  b=$(docker exec lsv-fresh md5sum $V/$d/$f | cut -d" " -f1)
  [ "$a" = "$b" ] && echo "$f MD5_MATCH" || echo "$f MD5_MISMATCH $a $b"
done

docker cp $SP lsv-fresh:/root/serve.sh
# stdout+stderr to FILE so we can verify patched markers post-boot
docker exec -d lsv-fresh bash -c "bash /root/serve.sh > /root/serve_out.log 2>&1"
ok=0
for i in $(seq 1 110); do
  code=$(docker exec lsv-fresh sh -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; echo "HEALTH_OK tick=$i $(date -u +%H:%M:%S)"; break; }
  sleep 10
done
[ "$ok" = "1" ] || { echo HEALTH_TIMEOUT; docker exec lsv-fresh tail -30 /root/serve_out.log; exit 1; }

echo "--- patched-marker check (llm-scaler markers in boot log):"
docker exec lsv-fresh sh -c "grep -c \"llm-scaler\" /root/serve_out.log" || true
docker exec lsv-fresh sh -c "grep -m3 -h \"llm-scaler\" /root/serve_out.log" || true
echo "--- which vllm did it import:"
docker exec -w /llm-scaler/vllm lsv-fresh /opt/venv/bin/python -c "import vllm; print(vllm.__file__)"
docker exec lsv-fresh sh -c "wc -c /root/serve_out.log; grep -m1 \"Application startup complete\" /root/serve_out.log; grep -m1 \"Started server process\" /root/serve_out.log; grep -m2 -i \"MambaForCausalLM\|qwen\" /root/serve_out.log | head -2" || true

echo "--- nonce probe:"
python3 $B/nonce_probe.py 2 | tail -n 2
python3 $B/nonce_probe.py $N > $B/nonce_fresh2.out 2>&1
tail -n 2 $B/nonce_fresh2.out
echo "degenerate=$(grep -c DEGENERATE $B/nonce_fresh2.out || true)"
echo "=== FRESHLEG2 DONE $(date -u +%H:%M:%S)"
