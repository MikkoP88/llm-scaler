#!/bin/bash
# R1 boot anomaly probe: what is the NEW container actually running, and what
# does its serve log say about spec/drafter init?
echo "=== container state ==="
docker ps -a --format '{{.Names}} {{.Status}}' | grep lsv
echo "=== new serve cmdline (proc scan) ==="
docker exec lsv-test sh -c 'for d in /proc/[0-9]*/cmdline; do p=${d%/cmdline}; p=${p#/proc/}; c=$(tr "\0" " " < "$d" 2>/dev/null); case "$c" in *vllm*|*VLLM*) echo "pid $p: $(echo "$c" | cut -c 1-260)";; esac; done'
echo "=== spec/drafter markers (wide grep) ==="
docker exec lsv-test grep -n -i -E "speculat|mtp|draft|sharded-view|replicated|v78|v52|Proposer" /root/serve_full.log | head -n 40
echo "=== log tail 40 ==="
docker exec lsv-test sh -c 'tail -n 40 /root/serve_full.log'
echo "=== KV + worker lines ==="
docker exec lsv-test grep -n -E "Available KV|gpu memory|KV cache|model weights" /root/serve_full.log | head -n 15
echo "=== log head 12 (serve invocation banner) ==="
docker exec lsv-test sh -c 'head -n 12 /root/serve_full.log'
