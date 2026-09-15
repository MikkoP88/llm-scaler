#!/bin/bash
# q22_validate.sh — Q22a (NEO 26.27 overlay) idle validation gates
# (§20 F): NEO census, config, VIA census (dynamic pids), P1 marker,
# boundary, soak, f8ref vs certified e5m2 refs, bench3, perf.
echo "--- NEO CENSUS (expect 26.27.39122.11 / igc 2.38.2 / gmmlib 22.10.0):"
docker exec lsv-test sh -c "dpkg -l | grep -E 'intel-opencl-icd|intel-igc|libigdgmm|libze-intel-gpu1' | awk '{print \$2, \$3}'"
echo "--- SERVE CONFIG (expect fp8_e5m2):"
docker exec lsv-test sh -c "grep -m1 'kv_cache_dtype' /root/serve_full.log" | cut -c1-200 || true
echo "--- VIA CENSUS per worker pid (expect True>0, False=0):"
PIDS=$(docker exec lsv-test sh -c "ps -o pid= -C python3" | tr -d '\r')
for p in $PIDS; do
  t=$(docker exec lsv-test sh -c "grep -c 'via_env=True' /tmp/fr_$p.log 2>/dev/null || echo 0")
  f=$(docker exec lsv-test sh -c "grep -c 'via_env=False' /tmp/fr_$p.log 2>/dev/null || echo 0")
  echo "pid $p: via_env=True=$t via_env=False=$f"
done
echo "--- P1 MARKER (expect 2):"
docker exec lsv-test grep -c "llm-scaler v58 P1" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
echo "--- BOUNDARY PROBE:"
cd /root/build
python3 -u p1_len5probe.py 2>&1 | tee lce1/p1_len5probe_Q22.out
echo "--- GUARD FIRES:"
docker exec lsv-test sh -c 'grep -c "v58 P1 uniform-decode prefill guard" /root/serve_full.log'
echo "--- SOAK:"
python3 -u p1_soak.py 2>&1 | tee lce1/p1_soak_Q22.out
echo "--- F8REF e5m2 (MUST equal cb8c3851b897 / 68332ec7c31b / 05c88ff03b0c):"
python3 f8ref.py q22e5m2 2>&1 | tee lce1/f8ref_q22e5m2.out
echo "--- BENCH3 spot:"
python3 dt_bench3.py 2>&1 | tee lce1/bench3_Q22.out
echo "--- PERF SPOT:"
bash q17_perf.sh 2>&1 | tee lce1/q22_perf.out
echo Q22_VALIDATE_SH_DONE
