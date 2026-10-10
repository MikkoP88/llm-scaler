#!/bin/bash
# restorelmc.sh N — post-CCL-fix restoration: lmcache package back,
# patched v139 adapter + patched connector in, mamba float16 + connector ON,
# boot, probe N, report store/lookup telemetry.
set -u
N=${1:-100}
B=/root/build
S=/opt/venv/lib/python3.12/site-packages
V=$S/vllm
SP=/root/serve_lmc_v133.sh

echo "=== RESTORELMC START n=$N $(date -u +%H:%M:%S)"

# 1. lmcache package back
docker exec lsv-test sh -c "if [ -d $S/lmcache.off ]; then mv $S/lmcache.off $S/lmcache; fi; \
  if [ -d $S/lmcache.off.dist-info ]; then mv $S/lmcache.off.dist-info $S/lmcache-0.5.5.dist-info; fi; \
  ls -d $S/lmcache $S/lmcache-0.5.5.dist-info"
docker exec lsv-test /opt/venv/bin/python -c "import lmcache; print('lmcache', lmcache.__version__ if hasattr(lmcache,'__version__') else 'ok')" \
  || { echo LMCACHE_IMPORT_FAIL; exit 1; }

# 2. patched adapter (v139) + connector
docker cp $B/adapter_v139.py lsv-test:$S/lmcache/integration/vllm/vllm_v1_adapter.py || { echo CP_ADAPTER_FAIL; exit 1; }
docker cp $B/lmcache_connector.py lsv-test:$V/distributed/device_communicators/lmcache_connector.py || { echo CP_CONN_FAIL; exit 1; }
docker exec lsv-test sh -c "grep -c _lmc_flush_pending $S/lmcache/integration/vllm/vllm_v1_adapter.py; grep -c llm-scaler $V/distributed/device_communicators/lmcache_connector.py; rm -rf $S/lmcache/integration/vllm/__pycache__ $V/distributed/device_communicators/__pycache__; true"
echo "adapter+connector installed"

# 3. serve script: float16 + connector ON ratio 2
python3 - "$SP" <<'PYEOF'
import sys
sp = sys.argv[1]
lines = open(sp).read().split("\n")
for i, l in enumerate(lines):
    if "mamba-ssm-cache-dtype" in l:
        lines[i] = "  --mamba-ssm-cache-dtype float16 \\"
        mline = i
        break
else:
    raise SystemExit("MAMBA_LINE_NOT_FOUND")
KV = ('  --kv-transfer-config \'{"kv_connector":"LMCacheConnectorV1",'
      '"kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,'
      '"lmcache.extra_config":{"mamba_pool_ratio":2}},"engine_id":"lsv65"}\' \\')
found = any("kv-transfer-config" in l for l in lines)
if not found:
    lines.insert(mline + 1, KV)
open(sp, "w").write("\n".join(lines))
print("EDIT_OK mamba_line=%d" % mline)
PYEOF
[ $? -eq 0 ] || { echo EDIT_ABORT; exit 1; }
grep -n "mamba-ssm-cache-dtype\|kv-transfer-config" $SP

# 4. clear sentinels + boot via legboot
bash $B/legboot.sh lmcon $N

# 5. store/lookup telemetry from engine log
echo "--- LMCache telemetry:"
docker exec lsv-test sh -c "grep -c \"lmc store\|LMC store\" /root/serve_full.log 2>/dev/null; grep -m5 -i \"lmcache\" /root/serve_full.log | head -5" || true
echo "=== RESTORELMC DONE $(date -u +%H:%M:%S)"
