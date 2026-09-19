#!/bin/bash
# ltl_run3.sh — sampler forwarding + override behavior captures.
set -u
cd /root/ltl

pkill -f "ltl_capture_[s]erver.py" 2>/dev/null || true
: > captured.jsonl
nohup /usr/bin/python3 ltl_capture_server.py > capsrv.log 2>&1 < /dev/null &
sleep 1
curl -s -o /dev/null -m 3 -X POST http://127.0.0.1:8901/v1/chat/completions -d '{}'

docker rm -f litellm-test >/dev/null 2>&1 || true
IMG=$(docker inspect litellm-proxy --format '{{.Image}}')
docker run -d --name litellm-test --network host \
  -v /root/ltl/config.yaml:/app/config.yaml "$IMG" \
  --config /app/config.yaml --port 4001 --host 127.0.0.1 > /dev/null
code=000
for i in $(seq 1 40); do
  sleep 3
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 \
    -H 'Authorization: Bearer sk-dummy' http://127.0.0.1:4001/v1/models \
    2>/dev/null || echo 000)
  [ "$code" = "200" ] && break
done
echo "[T] proxy_ready code=$code"
if [ "$code" != "200" ]; then docker logs --tail 25 litellm-test; exit 1; fi

send() {
  local body="{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8"
  if [ -n "${2:-}" ]; then body="$body,${2:-}"; fi
  body="$body}"
  local out
  out=$(curl -s -m 60 -o /tmp/ltl_resp.json -w '%{http_code}' \
    -H 'Authorization: Bearer sk-dummy' -H 'Content-Type: application/json' \
    -X POST http://127.0.0.1:4001/v1/chat/completions -d "$body")
  echo "CALL $1 extra='${2:-}' -> http $out"
}

send probe-direct
send probe-direct '"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}'
send probe-default
send probe-direct '"temperature":0.9,"top_k":40'

echo "[T] === CAPTURED WIRE BODIES ==="
/usr/bin/python3 - <<'EOF'
import json
for line in open("/root/ltl/captured.jsonl"):
    d = json.loads(line)
    b = d["body"]
    keep = {k: b.get(k) for k in ("model", "temperature", "top_p", "top_k",
                                   "min_p", "presence_penalty",
                                   "repetition_penalty",
                                   "chat_template_kwargs")}
    print(json.dumps(keep))
EOF

docker rm -f litellm-test >/dev/null 2>&1 || true
pkill -f "ltl_capture_[s]erver.py" 2>/dev/null || true
echo "[T] DONE"
