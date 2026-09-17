#!/bin/bash
# check_image.sh — was v1.2.10 rebuilt? do container pycs match the census?
echo "== image Created timestamps =="
for t in llm-scaler-exp:v1.2.10 llm-scaler-exp:v1.2.11 llm-scaler-exp:v1.2.9; do
  docker inspect $t --format "$t {{.Created}} {{.Id}}" 2>/dev/null
done
echo "== pyc census (base, /tmp/pyc_base.txt) =="
cat /tmp/pyc_base.txt 2>/dev/null
echo "== live container pyc md5s (same 6 modules) =="
docker exec lsv-test sh -c 'cd /opt/venv/lib/python3.12/site-packages/vllm && md5sum v1/worker/gpu_model_runner.pyc v1/worker/mamba_utils.pyc v1/attention/ops/triton_fp8_mq.pyc 2>/dev/null; ls v1/worker/__pycache__/ | head -8'
echo "== f8ref.py identity =="
md5sum /root/build/f8ref.py; stat -c '%y %s' /root/build/f8ref.py
echo "== serve_user.sh mtime =="
stat -c '%y' /root/build/serve_user.sh
echo "== dmesg xe errors since boot (non-reset) =="
dmesg | grep -a -E 'xe 0000.*(error|fail|warn|timeout)' | grep -av 'Engine reset' | tail -6
