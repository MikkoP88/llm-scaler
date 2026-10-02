#!/bin/bash
# teardown.sh — remove lsv-test, handling wedged/spinning worker procs
# that make plain `docker rm -f` hang forever.
set -u
if docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
  PIDS=$(docker exec lsv-test sh -c "ps aux | grep -E 'EngineCore|VLLM|Worker' | grep -v grep | awk '{print \$2}'" 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    docker exec lsv-test sh -c "kill -9 $PIDS" 2>/dev/null || true
    echo "killed worker pids: $PIDS"
  fi
  sleep 3
fi
timeout 60 docker rm -f lsv-test >/dev/null 2>&1 || true
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^lsv-test$'; then
  echo "WARN: container still present after rm"
else
  echo "TEARDOWN_OK $(date +%H:%M:%S)"
fi
