#!/bin/bash
# ltl_run2.sh — run ON GPUNODE01. Boots a TEST litellm proxy (same image
# as production, port 4001, network host) + capture server (8901).
# Production litellm-proxy service is NOT touched.
set -u
cd /root/ltl

echo "[T] start capture server 8901"
pkill -f "ltl_capture_[s]erver.py" 2>/dev/null || true
: > captured.jsonl
nohup /usr/bin/python3 ltl_capture_server.py > capsrv.log 2>&1 < /dev/null &
sleep 1
curl -s -o /dev/null -m 3 -X POST http://127.0.0.1:8901/v1/chat/completions -d '{}'
echo "[T] capsrv up"

echo "[T] boot test proxy on 4001 (same image as prod)"
docker rm -f litellm-test >/dev/null 2>&1 || true
IMG=$(docker inspect litellm-proxy --format '{{.Image}}')
echo "[T] image=$IMG"
docker run -d --name litellm-test --network host \
  -v /root/ltl/config.yaml:/app/config.yaml "$IMG" \
  --config /app/config.yaml --port 4001 --host 127.0.0.1 \
  > /dev/null
code=000
for i in $(seq 1 40); do
  sleep 3
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 \
    -H 'Authorization: Bearer sk-dummy' http://127.0.0.1:4001/v1/models \
    2>/dev/null || echo 000)
  [ "$code" = "200" ] && break
done
echo "[T] proxy_ready code=$code after ~$((i*3))s"
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

echo "[T] capture calls"
send coder-string
send nonthinking-string
send coder-bool
send coder-litellm-extra-body
send coder-bool '"extra_body":{"chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"low"}}'
send coder-bool '"temperature":0.9'

echo "[T] === CAPTURED WIRE BODIES ==="
/usr/bin/python3 - <<'EOF'
import json
for line in open("/root/ltl/captured.jsonl"):
    d = json.loads(line)
    b = d["body"]
    keep = {k: b.get(k) for k in ("model", "temperature",
                                   "repetition_penalty",
                                   "chat_template_kwargs")}
    print(json.dumps(keep))
EOF

echo "[T] reachability of real vLLM"
curl -s -o /dev/null -m 5 -w 'vllm_health=%{http_code}\n' http://10.20.3.65:8000/health

echo "[T] E2E calls through test proxy -> real vLLM"
e2e() {
  local model="$1"; shift
  local body
  body=$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0}' "$model" "$1" "$2")
  local t0=$(date +%s.%N)
  local http
  http=$(curl -s -m 240 -o /tmp/e2e_$model.json -w '%{http_code}' \
    -H 'Authorization: Bearer sk-dummy' -H 'Content-Type: application/json' \
    -X POST http://127.0.0.1:4001/v1/chat/completions -d "$body")
  local t1=$(date +%s.%N)
  /usr/bin/python3 - "$model" "$http" "$t0" "$t1" <<'EOF'
import json, sys
m, http, t0, t1 = sys.argv[1:5]
try:
    d = json.load(open(f"/tmp/e2e_{m}.json"))
    ch = d["choices"][0]["message"]
    rc = ch.get("reasoning_content") or ""
    co = ch.get("content") or ""
    u = d.get("usage", {})
    print(f"E2E {m}: http={http} dt={float(t1)-float(t0):.1f}s "
          f"reasoning_len={len(rc)} content_len={len(co)} "
          f"completion_tokens={u.get('completion_tokens')} "
          f"reasoning_tokens={u.get('completion_tokens_details', {}).get('reasoning_tokens') if u.get('completion_tokens_details') else '-'}")
except Exception as e:
    print(f"E2E {m}: http={http} parse_fail={e}")
EOF
}
e2e e2e-nonthinking-bool "Reply with exactly the word: OK" 16
e2e e2e-nonthinking-leb "Reply with exactly the word: OK" 16
e2e e2e-coder-bool "What is 17*23? Think first." 96
e2e e2e-coder-leb "What is 17*23? Think first." 96

echo "[T] cleanup"
docker rm -f litellm-test >/dev/null 2>&1 || true
pkill -f "ltl_capture_[s]erver.py" 2>/dev/null || true
echo "[T] DONE (prod litellm-proxy untouched)"
