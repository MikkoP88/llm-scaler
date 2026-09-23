#!/bin/bash
# recreate_litellm.sh — recreate litellm-proxy with the full CC compat stack.
# Run this after ANY 'docker rm' of litellm-proxy. Re-establishes:
#   1. -e LITELLM_MASTER_KEY=sk-dummy (db-less master-key mode CC depends on;
#      plain recreate = 400 'No connected db.' on every route)
#   2. config mount (use_chat_completions_url_for_anthropic_messages routing
#      fix + callbacks: custom_callbacks.custom_callback)
#   3. custom_callbacks.py mounted into site-packages (tool-args un-stringify
#      repair for AskUserQuestion-class nested tool schemas)
#   4. pinned local image (2026-09-20 main-latest, 114aca7726c3) — a fresh
#      'main-latest' pull may move; re-verify thinking + tool repair after
#      any image change.
set -e
cd /root/build
docker tag 114aca7726c3 litellm-cc:pinned-20260920 2>/dev/null || true
docker rm -f litellm-proxy
docker run -d --name litellm-proxy --network host --restart unless-stopped   -e LITELLM_MASTER_KEY=sk-dummy   -v /root/litellm_config.yaml:/app/config.yaml:ro   -v /root/build/custom_callbacks.py:/app/.venv/lib/python3.13/site-packages/custom_callbacks.py:ro   litellm-cc:pinned-20260920   --config /app/config.yaml --port 4000
for i in $(seq 1 24); do
  c=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:4000/health/liveliness)
  [ "$c" = "200" ] && { echo HEALTH_OK; exit 0; }
  sleep 5
done
echo HEALTH_TIMEOUT; exit 1
