#!/bin/bash
# check_stray.sh — who owns port 8000: the lsv-test container engine or a
# stray post-wedge process from the torn-down image lane?
echo "== listener pid 11829 =="
ps -o pid,ppid,etime,stat,cmd -p 11829 2>/dev/null || echo "pid 11829 gone"
echo "== cgroup of 11829 (container membership) =="
cat /proc/11829/cgroup 2>/dev/null || true
echo "== docker top lsv-test (host pids) =="
docker top lsv-test -o pid,etime,cmd 2>/dev/null | head -12
echo "== all vllm-ish processes on host =="
ps aux | grep -a -E '[v]llm serve|[f]rom vllm' | awk '{print $2, $9, $11, $12, $13}' | head -10
echo "== container serve log tail (bind errors?) =="
docker exec lsv-test sh -c 'tail -5 /root/serve_full.log' 2>/dev/null
echo "== container APIServer alive? =="
docker exec lsv-test sh -c 'grep -c "Uvicorn running" /root/serve_full.log' 2>/dev/null
echo "== P2E1F (pre-reboot passing screen) pcie gen =="
for f in /root/build/lce1/nv_P2E1F_clk_*.txt; do
  awk -F', ' 'NR==1{for(i=1;i<=NF;i++) if($i=="pcie.link.gen.current") c=i} NR==2{print FILENAME": gen="$c}' "$f" 2>/dev/null
done
