#!/bin/bash
# dt_dflash2_boot.sh — DFlash 2 era-3 lane boot orchestrator.
# 1. helper container: build patched site-package copies + compile/import check
#    (dflash2_edit.py v40 hooks + dflash2_winfix_edit.py v41 window fix +
#    dflash2_emitk_edit.py v42 emission cap + dflash2_schedk_edit.py v42c
#    async placeholder cap + dflash2_udql_edit.py v42d runner/lattice
#    width + dflash2_disp_edit.py v42d dispatcher width)
# 2. boot lsv-test (idles in await-patch wrapper)
# 3. docker cp new+patched files in, touch the flag
# 4. docker restart -> serve boots with patches; poll health
# v42: prefix DFLASH2_EMIT_K=<1..7> to set the emission width (default 7).
set -u
B=/root/build/qwen38-dflash2
IMG=llm-scaler-prod:v1
SP=/opt/venv/lib/python3.12/site-packages

echo "== phase 1: prepare patched copies =="
docker run --rm -v "$B":/w --entrypoint python3 "$IMG" /w/dflash2_edit.py || {
  echo EDIT_FAIL; exit 2; }
docker run --rm -v "$B":/w --entrypoint python3 "$IMG" /w/dflash2_winfix_edit.py || {
  echo WINFIX_EDIT_FAIL; exit 2; }
docker run --rm -v "$B":/w --entrypoint python3 "$IMG" /w/dflash2_emitk_edit.py || {
  echo EMITK_EDIT_FAIL; exit 2; }
docker run --rm -v "$B":/w --entrypoint python3 "$IMG" /w/dflash2_schedk_edit.py || {
  echo SCHEDK_EDIT_FAIL; exit 2; }
docker run --rm -v "$B":/w --entrypoint python3 "$IMG" /w/dflash2_udql_edit.py || {
  echo UDQL_EDIT_FAIL; exit 2; }
docker run --rm -v "$B":/w --entrypoint python3 "$IMG" /w/dflash2_disp_edit.py || {
  echo DISP_EDIT_FAIL; exit 2; }

echo "== phase 2: boot await-patch container =="
bash "$B/dt_dflash2_serve.sh" || { echo SERVE_BOOT_FAIL; exit 3; }
sleep 3
state=$(docker inspect -f '{{.State.Status}}' lsv-test 2>/dev/null || echo missing)
[ "$state" = "running" ] || { echo "CONTAINER_NOT_RUNNING state=$state"; exit 4; }

echo "== phase 3: inject files =="
docker cp "$B/new/qwen3_dflash2.py" "lsv-test:$SP/vllm/model_executor/models/qwen3_dflash2.py" || exit 5
docker cp "$B/new/dflash2_proposer.py" "lsv-test:$SP/vllm/v1/spec_decode/dflash2.py" || exit 5
for f in vllm/v1/spec_decode/utils.py vllm/v1/spec_decode/dflash.py \
         vllm/model_executor/models/qwen3_dflash.py \
         vllm/model_executor/models/registry.py \
         vllm/v1/core/sched/async_scheduler.py \
         vllm/config/vllm.py \
         vllm/v1/cudagraph_dispatcher.py \
         vllm/v1/worker/gpu_model_runner.py \
         vllm/v1/attention/backends/turboquant_attn.py \
         vllm/v1/attention/ops/triton_turboquant_decode.py; do
  docker cp "$B/patched/$f" "lsv-test:$SP/$f" || { echo "CP_FAIL $f"; exit 5; }
done
docker exec lsv-test touch /root/.dflash2_patched || exit 6
echo "injected 12 files"

echo "== phase 4: restart into serve + poll =="
docker restart lsv-test >/dev/null
ok=0
for i in $(seq 1 90); do
  sleep 10
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || true)
  if [ "$code" = "200" ]; then echo "HEALTH OK after ~$((i*10))s"; ok=1; break; fi
  st=$(docker inspect -f '{{.State.Status}}' lsv-test 2>/dev/null || echo gone)
  if [ "$st" != "running" ]; then
    echo "CONTAINER_EXITED"
    docker cp lsv-test:/root/dt_dflash2.log /tmp/dt_dflash2_boot.log >/dev/null 2>&1
    tail -40 /tmp/dt_dflash2_boot.log
    exit 7
  fi
done
if [ "$ok" != 1 ]; then
  echo HEALTH_TIMEOUT
  docker cp lsv-test:/root/dt_dflash2.log /tmp/dt_dflash2_boot.log >/dev/null 2>&1
  tail -30 /tmp/dt_dflash2_boot.log
  exit 8
fi
echo DFLASH2_BOOT_OK
