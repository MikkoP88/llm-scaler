#!/bin/bash
# recreate_litellm_4004.sh -- recreate the PRODUCTION litellm on 10.100.8.6:4004.
# (10.100.8.6 = NEVER-REBOOT host; this script only touches the proxy container.)
# Run this after ANY 'docker rm' of litellm-proxy-latest. Re-establishes:
#   1. -e LITELLM_MASTER_KEY=sk-dummy (db-less master-key mode CC depends on;
#      plain recreate = 400 'No connected db.' on every route)
#   2. config mount /root/litellm_config.yaml (read-only)
#   3. /root/claude_custom_callbacks.py -> site-packages (tool-args un-stringify
#      repair; module name imported by config callbacks:)
#   4. pinned running image 52abe19ecef0 (ghcr.io/berriai/litellm:main-latest
#      at deploy time 2026-10-08) -- do NOT blindly reuse the moving tag
#   5. host network + --port 4004 + unless-stopped
set -e
docker rm -f litellm-proxy-latest
docker run -d --name litellm-proxy-latest --network host --restart unless-stopped \
  -e LITELLM_MASTER_KEY=sk-dummy \
  -v /root/litellm_config.yaml:/app/config.yaml:ro \
  -v /root/claude_custom_callbacks.py:/app/.venv/lib/python3.13/site-packages/claude_custom_callbacks.py:ro \
  52abe19ecef09d149abd23f8a4aed6683dd7f780f90a0c20d214dfb29006f2dd \
  --config /app/config.yaml --port 4004
for i in $(seq 1 24); do
  c=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:4004/health/liveliness || echo 000)
  [ "$c" = "200" ] && { echo HEALTH_OK; exit 0; }
  sleep 5
done
echo HEALTH_TIMEOUT; exit 1
