#!/bin/bash
# stage5_bake_v1215.sh — gate, bake, and boot llm-scaler-exp:v1.2.15.
# Pre-req: stage3 (drill x3) + stage4 (soak+battery) PASS.
# Gates: health 200; F8-guard==0 (runtime async OFF — one APIServer
# pre-flip config line is cosmetic and allowed); v60 marks; barrier
# True/0 (every step); full v1.2.12 pedigree; STALFIX; xgrammar 0.2.7.
# Then: marker touch -> docker commit v1.2.15 -> boot script pointed at
# v1.2.15 (+ explicit barrier env) -> fresh V1215 boot -> re-verify.
set -u
L=/root/build/lce1/v60_bake.log
B=/root/build/repro_bootV1212.sh
{
echo "=== V60 BAKE v1.2.15 $(date +%F' '%T) ==="
SP=/opt/venv/lib/python3.12/site-packages/vllm

FAIL=0
ck() { # name expected actual
  if [ "$2" = "$3" ]; then echo "GATE-OK   $1=$3"; else echo "GATE-FAIL $1 expected=$2 got=$3"; FAIL=1; fi
}

H=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)
ck health 200 "$H"

F8=$(docker exec lsv-test sh -c "grep -c 'F8 guard' /root/serve_full.log 2>/dev/null || true")
ck f8_guard_banner 0 "$F8"

BAN=$(docker exec lsv-test sh -c "grep -c 'Asynchronous scheduling is enabled' /root/serve_full.log 2>/dev/null || true")
case "$BAN" in
  0) echo "GATE-OK   async_banner=0 (best case)";;
  1) echo "GATE-OK   async_banner=1 (APIServer pre-flip cosmetic only)";;
  *) echo "GATE-FAIL async_banner=$BAN (AsyncScheduler live)"; FAIL=1;;
esac

BARR=$(docker exec lsv-test /opt/venv/bin/python3 -c 'from vllm.v1.worker import gpu_model_runner as g; print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_MIN_CTX)' 2>/dev/null | tail -1)
ck barrier "True 0" "$BARR"

ck mark_xpu_v60   1 "$(docker exec lsv-test grep -c 'llm-scaler v60' $SP/platforms/xpu.py)"
ck mark_gmr_v60b  1 "$(docker exec lsv-test grep -c 'llm-scaler v60 (wedge fix): barrier DEFAULT ON' $SP/v1/worker/gpu_model_runner.py)"
ck mark_gmr_v60g  1 "$(docker exec lsv-test grep -c 'llm-scaler v60g (WEDGEFIX-E)' $SP/v1/worker/gpu_model_runner.py)"
ck mark_gmr_v60e  0 "$(docker exec lsv-test grep -c 'llm-scaler v60e' $SP/v1/worker/gpu_model_runner.py)"
ck mark_f15b      1 "$(docker exec lsv-test sh -c "grep -c 'llm-scaler f15b' $SP/v1/worker/gpu_model_runner.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_v553      1 "$(docker exec lsv-test sh -c "grep -c 'llm-scaler v55.3' $SP/v1/worker/gpu_model_runner.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_arstage   1 "$(docker exec lsv-test sh -c "grep -c 'llm-scaler arstage' $SP/distributed/device_communicators/xpu_communicator.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_stalfix   1 "$(docker exec lsv-test sh -c "grep -c STALFIX $SP/entrypoints/openai/responses/serving.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_sched_v60c 1 "$(docker exec lsv-test sh -c "grep -c 'llm-scaler v60c' $SP/v1/core/sched/scheduler.py | awk '{print (\$1>=1)?1:0}'")"
ck mark_parser    1 "$(docker exec lsv-test sh -c "grep -c _stalfix_name_emitted $SP/tool_parsers/qwen3xml_tool_parser.py | awk '{print (\$1>=1)?1:0}'")"

XG=$(docker exec lsv-test /opt/venv/bin/python3 -c 'from importlib.metadata import version; print(version("xgrammar"))' 2>/dev/null | tail -1)
ck xgrammar 0.2.7 "$XG"

docker exec lsv-test test -f /root/.llm_scaler_exp_v1214_baked && echo "GATE-OK   v1214_marker" || { echo "GATE-FAIL v1214_marker"; FAIL=1; }

if [ "$FAIL" != "0" ]; then
  echo "BAKE ABORTED — gates failed"
  echo "=== BAKE DONE (ABORT) $(date +%T) ==="
  exit 1
fi

echo "--- gates PASS: baking v1.2.15 ---"
docker exec lsv-test touch /root/.llm_scaler_exp_v1215_baked
docker commit lsv-test llm-scaler-exp:v1.2.15
echo "commit_rc=$?"
docker images llm-scaler-exp:v1.2.15 --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'

echo "--- point boot script at v1.2.15 + explicit barrier env ---"
if grep -q 'llm-scaler-exp:v1.2.15' "$B"; then
  echo "boot script already on v1.2.15"
else
  cp "$B" "$B.pre_v1215"
  sed -i 's/llm-scaler-exp:v1.2.14/llm-scaler-exp:v1.2.15/g' "$B"
  grep -n 'llm-scaler-exp:v1.2.15' "$B" | head -3
fi
if grep -q 'VLLM_XPU_SPEC_DRAFT_BARRIER=1' "$B"; then
  echo "barrier env already present"
else
  sed -i 's|-e VLLM_V55_DECODE_TIMEOUT_S=30|-e VLLM_V55_DECODE_TIMEOUT_S=30 -e VLLM_XPU_SPEC_DRAFT_BARRIER=1 -e VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX=0|' "$B"
  grep -n 'SPEC_DRAFT_BARRIER' "$B" | head -2
fi

echo "--- fresh boot from committed image (V1215) ---"
bash "$B" V1215
echo "boot_rc=$?"

H2=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)
ck health_v1215 200 "$H2"
F8b=$(docker exec lsv-test sh -c "grep -c 'F8 guard' /root/serve_full.log 2>/dev/null || true")
ck f8_guard_v1215 0 "$F8b"
BANb=$(docker exec lsv-test sh -c "grep -c 'Asynchronous scheduling is enabled' /root/serve_full.log 2>/dev/null || true")
case "$BANb" in
  0) echo "GATE-OK   async_banner_v1215=0 (best case)";;
  1) echo "GATE-OK   async_banner_v1215=1 (APIServer pre-flip cosmetic only)";;
  *) echo "GATE-FAIL async_banner_v1215=$BANb"; FAIL=1;;
esac
BARRb=$(docker exec lsv-test /opt/venv/bin/python3 -c 'from vllm.v1.worker import gpu_model_runner as g; print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_MIN_CTX)' 2>/dev/null | tail -1)
ck barrier_v1215 "True 0" "$BARRb"
docker exec lsv-test test -f /root/.llm_scaler_exp_v1215_baked && echo "GATE-OK   v1215_marker_baked" || echo "GATE-FAIL v1215_marker_baked"
XGb=$(docker exec lsv-test /opt/venv/bin/python3 -c 'from importlib.metadata import version; print(version("xgrammar"))' 2>/dev/null | tail -1)
ck xgrammar_v1215 0.2.7 "$XGb"
docker exec lsv-test sh -c "grep -q 'llm-scaler v60c' $SP/v1/core/sched/scheduler.py" && echo "GATE-OK   mark_sched_v60c_baked" || { echo "GATE-FAIL mark_sched_v60c_baked"; FAIL=1; }

echo "--- quick functional sanity on v1.2.15 ---"
curl -s -m 60 http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer sk-dummy" \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":64,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Say READY-v1215 and nothing else."}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("SANITY", repr(d["choices"][0]["message"]["content"][:80]), d["choices"][0]["finish_reason"])' || echo SANITY_FAIL
docker exec lsv-test sh -c "grep -c 'SpecDecoding metrics' /root/serve_full.log" || true

echo "=== BAKE DONE $(date +%T) ==="
} > "$L" 2>&1
tail -60 "$L"
echo STAGE5_DONE
