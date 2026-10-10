#!/bin/bash
# R1 restart diagnosis: enumerate real processes (setproctitle hides vLLM
# from comm-pgrep), inspect boot script supervision, and read the serve log
# lifecycle lines to learn whether a NEW boot happened after the SIGTERM.
echo "=== docker top (host view) ==="
docker top lsv-test -o pid,ppid,etime,args | head -n 30
echo "=== boot_v78.sh ==="
cat /root/build/p77/boot_v78.sh
echo "=== log lifecycle lines ==="
docker exec lsv-test grep -n -E "Started server process|Application shutdown complete|Shutting down|Shutting down the engine|llm-scaler v78|SHARDED-VIEW|REPLICATED|Traceback|SIGTERM|EngineCore_" /root/serve_full.log | head -n 60
echo "=== log size + mtime vs now ==="
docker exec lsv-test stat -c "%s bytes, mtime %y" /root/serve_full.log
date
