#!/bin/bash
# v1.2.17 boot verify: SANITY + xgrammar + spec + C4x1 + barrier + async
R=/root/build/_v63_sanity.txt
: > $R
B=http://127.0.0.1:8000
echo "== SANITY READY-v1217 ==" >> $R
curl -s -m 60 $B/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":2000,"messages":[{"role":"user","content":"Reply with exactly this token and nothing else: READY-v1217"}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["choices"][0]; print("SANITY finish=%s content_head=%r" % (c["finish_reason"], c["message"]["content"][:60]))' >> $R 2>&1
echo "== XGRAMMAR x2 ==" >> $R
cd /root/build && python3 t2_xgrammar_probe.py >> $R 2>&1
echo "== AB C4x1 ==" >> $R
python3 v60_ab_probe.py >> $R 2>&1
echo "== SPEC METRICS ==" >> $R
curl -s -m 10 $B/metrics | grep -E 'spec_decode_(draft|accept)' | grep -v '^#' | tail -4 >> $R
echo "== BARRIER HB ==" >> $R
docker exec lsv-test sh -c 'ls -la /dev/shm/ | grep -i "llm_scaler_spec" | head -4' >> $R 2>&1
echo "== ASYNC ==" >> $R
docker exec lsv-test grep -c 'Asynchronous scheduling is enabled' /root/serve_full.log >> $R 2>&1
echo "== V63 BAKED CHECK ==" >> $R
docker exec lsv-test grep -c 'V63_TTFTFIX_ACTIVE' /root/serve_full.log >> $R 2>&1
echo "== IMAGE ==" >> $R
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | head -3 >> $R
echo "SANITY_V1217_DONE" >> $R
