#!/bin/bash
# P85 live-watch tick — one-shot lane check (compact GREEN line or alert data).
# Posture: v1.2.31 + spec MTPx4 + async + tmpl, watchdog ARMED (tick header is stale).
# Baseline: crash-4 dmesg lines at uptime 2641/2672 (18:26:17/18:26:47 local).
TS=$(date +%H:%M:%S)
H=$(curl -s -m 8 -o /dev/null -w "%{http_code}" http://localhost:8000/health)
D=$(docker ps --format '{{.Names}} {{.Status}}' | grep lsv)
R1=$(docker exec lsv-test sh -c "grep -c 'POST /v1' /root/serve_full.log" 2>/dev/null)
ERR=$(docker exec lsv-test sh -c "grep -cE 'ERROR|Traceback' /root/serve_full.log" 2>/dev/null)
FIVEXX=$(docker exec lsv-test sh -c "grep 'HTTP/1.1\" 5' /root/serve_full.log | tail -n 3" 2>/dev/null | tr '\n' ' ')
RUN=$(docker exec lsv-test sh -c "curl -s -m 6 http://localhost:8000/metrics" 2>/dev/null | grep 'num_requests_running{' | grep -v '^#' | tr '\n' ' ')
SP=$(docker exec lsv-test sh -c "curl -s -m 6 http://localhost:8000/metrics" 2>/dev/null | grep -E '^vllm:spec_decode' | tail -n 6 | tr '\n' ' ')
P1=$(docker exec lsv-test sh -c "curl -s -m 6 http://localhost:8000/metrics" 2>/dev/null | grep -E '^vllm:(prompt|generation)_tokens_total' | tr '\n' ' ')
ST=$(docker exec lsv-test sh -c "grep SPECTIMING /root/serve_full.log | tail -n 2" | tr '\n' ' ')
DM=$(dmesg | grep -E 'Engine reset|devcoredump|fault' | tail -n 4 | tr '\n' ' ')
echo "TS=$TS H=$H | $D"
echo "REQ=$R1 ERR=$ERR 5XX[$FIVEXX]"
echo "RUN: $RUN"
echo "SPEC: $(echo "$SP" | cut -c1-500)"
echo "SPECTIMING: $ST"
echo "DMESG: $DM"
ls /root/build/lane_watchdog.paused >/dev/null 2>&1 && echo PAUSE_FLAG_PRESENT || echo NO_PAUSE_FLAG
# TPS: two 30s windows (standing order: show average tok/s every 30s)
echo "TOKENS-T1: $P1"
sleep 30
P2=$(docker exec lsv-test sh -c "curl -s -m 6 http://localhost:8000/metrics" 2>/dev/null | grep -E '^vllm:(prompt|generation)_tokens_total' | tr '\n' ' ')
echo "TOKENS-T2: $P2"
sleep 30
P3=$(docker exec lsv-test sh -c "curl -s -m 6 http://localhost:8000/metrics" 2>/dev/null | grep -E '^vllm:(prompt|generation)_tokens_total' | tr '\n' ' ')
echo "TOKENS-T3: $P3"
