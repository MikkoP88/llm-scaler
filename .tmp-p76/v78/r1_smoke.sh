#!/bin/bash
# R1 smoke + SPECTIMING harvest — one greedy chat completion, then collect
# the SPECTIMING lines it produced (propose/greedy/tforward host+device).
echo "=== req payload ==="
head -c 400 /root/build/p77/req_smoke.json; echo
echo "=== smoke POST ==="
curl -s -m 180 -w 'HTTP=%{http_code} time_total=%{time_total}s\n' -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @/root/build/p77/req_smoke.json -o /root/build/p77/r1_smoke_out.json
head -c 300 /root/build/p77/r1_smoke_out.json; echo
echo "=== SPECTIMING tail 30 ==="
docker exec lsv-test sh -c 'grep SPECTIMING /root/serve_full.log | tail -n 30'
