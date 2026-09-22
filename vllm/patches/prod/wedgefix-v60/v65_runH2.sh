#!/bin/sh
# v65_runH2.sh — H retry: fuse_norm_quant via direct docker-exec sed (no sh -c
# re-parse). Asserts the edit landed, else aborts before boot. Output _v65_runH2.txt.
B=/root/build
OUT=$B/_v65_runH2.txt
CC_OLD='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
CC_NEW='{"cudagraph_mode":"FULL_DECODE_ONLY","pass_config":{"fuse_norm_quant":true}}'
relaunch() {
  docker exec lsv-test sh -c "pkill -f 'vllm serve' || true"
  for i in $(seq 1 30); do docker exec lsv-test sh -c "pgrep -f 'vllm serve' >/dev/null" || break; sleep 2; done
  sleep 3
  docker exec -d lsv-test bash /root/serve_user.sh
  for i in $(seq 1 60); do
    sleep 10
    c=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/health 2>/dev/null || echo 000)
    if [ "$c" = "200" ]; then echo "serve HEALTH_OK ~$((i*10))s"; return 0; fi
  done
  echo ABORT_SERVE; return 1
}
{
echo "=== V65 RUNH2 $(date +%T) ==="
docker exec lsv-test sed -i "s|$CC_OLD|$CC_NEW|" /root/serve_user.sh
echo "-- line after edit --"
docker exec lsv-test grep pass_config /root/serve_user.sh || { echo ABORT_SED_NO_MATCH; exit 1; }
if relaunch; then
  docker exec lsv-test sh -c "wc -c < /root/serve_full.log" > $B/_v65_h2_off.txt
  echo "== H2: fuse_norm_quant solo seed206 =="
  python3 $B/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 206 2>&1 | tail -3
  OFF=$(cat $B/_v65_h2_off.txt)
  echo "-- H2 steps (first 3 / last 4) --"
  docker exec lsv-test sh -c "tail -c +$OFF /root/serve_full.log | grep -A1 'V65_STEP' | grep Arguments | head -3; tail -c +$OFF /root/serve_full.log | grep -A1 'V65_STEP' | grep Arguments | tail -4"
  echo "-- H2 xgrammar quick --"
  curl -s -m 60 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.8-27b-fp8","max_tokens":200,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"response_format":{"type":"json_schema","json_schema":{"name":"t","schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}},"messages":[{"role":"user","content":"city? json"}]}' | head -c 240; echo
  echo "-- H2 fusion warn lines --"
  docker exec lsv-test sh -c "tail -c +$OFF /root/serve_full.log | grep -iE 'fusion|fuse' | head -4"
else
  echo "-- H2 BOOT FAILED; log tail --"
  docker exec lsv-test sh -c "tail -20 /root/serve_full.log | grep -iE 'error|fuse|Traceback|assert' | head -8"
fi
docker exec lsv-test sed -i "s|$CC_NEW|$CC_OLD|" /root/serve_user.sh
docker exec lsv-test grep -c pass_config /root/serve_user.sh || true
relaunch || exit 1
echo "=== V65_RUNH2 DONE $(date +%T) ==="
echo V65_RUNH2_DONE
} > $OUT 2>&1
