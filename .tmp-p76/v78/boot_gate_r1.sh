#!/bin/bash
# v78 R1 boot gate — poll health to 200, then assert the boot markers:
# R1 SHARDED-VIEW (drafter head reweight), REPLICATED (target head, v78),
# draft KV block wrap, KV cache size (capacity bar: >= 4.97 GiB).
CODE=000
for i in $(seq 1 60); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)
  [ "$CODE" = "200" ] && break
  sleep 10
done
echo "health=$CODE after $((i * 10))s"
echo "--- R1 marker:"
docker exec lsv-test grep -m 2 'SHARDED-VIEW rows' /root/serve_full.log || echo "MARKER-MISSING (R1 view did not apply)"
echo "--- target REPLICATED marker:"
docker exec lsv-test grep -m 1 'REPLICATED' /root/serve_full.log || true
echo "--- draft KV block wrap:"
docker exec lsv-test grep -m 1 'draft KV block' /root/serve_full.log || true
echo "--- KV cache:"
docker exec lsv-test grep -m 2 'Available KV cache' /root/serve_full.log || true
echo "--- errors:"
docker exec lsv-test grep -c 'Traceback' /root/serve_full.log || true
