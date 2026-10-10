#!/bin/bash
# R1 pre-deploy verification: host-side code parity markers + compile check
# in a throwaway image container (no pyc write, pure compile()).
echo "=== md5 (expect llm_base_proposer 2db4921364429f528f95c1da61105da0,"
echo "    logits_processor 17dfd2532e2b59430945a1e484aa64d1)"
md5sum /root/build/p77/code/llm_base_proposer.py /root/build/p77/code/logits_processor.py
echo "=== R1 markers in llm_base_proposer.py ==="
grep -n "_r1_shard_view_ok\|_r1_apply_shard_view\|SHARDED-VIEW rows\|mtp_tp1_mode" /root/build/p77/code/llm_base_proposer.py
echo "=== layer re-point elif ==="
grep -n 'engine' /root/build/p77/code/llm_base_proposer.py | grep -n 'in (' || true
echo "=== cache-marker lines in logits_processor.py ==="
grep -n "_xpu_pair_cache\|all_gather_into_tensor" /root/build/p77/code/logits_processor.py
echo "=== compile check (throwaway container) ==="
docker run --rm -v /root/build/p77/code:/c:ro -w /c --entrypoint /opt/venv/bin/python3 llm-scaler-exp:v1.2.29 -c "compile(open('llm_base_proposer.py').read(),'llm_base_proposer.py','exec'); compile(open('logits_processor.py').read(),'logits_processor.py','exec'); print('COMPILE-OK')"
