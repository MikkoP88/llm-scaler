#!/usr/bin/env python3
"""driftcheck_v135_edit.py — rebase watchdog_v2_drift_check.sh to the v135
era (v1.2.35 bake, P10 model-runner patch). All replacements uniqueness-
asserted; raises before writing on any miss (atomic)."""
P = "/root/build/v127_stage/watchdog_v2_drift_check.sh"
src = open(P).read()


def rep(old, new):
    global src
    n = src.count(old)
    if n != 1:
        raise SystemExit("MISS n=%d for %r" % (n, old[:70]))
    src = src.replace(old, new, 1)


# 1. image default v1.2.34 -> v1.2.35
rep('EXPECT_IMAGE="${V1227_EXPECT_IMAGE:-llm-scaler-exp:v1.2.34}"',
    'EXPECT_IMAGE="${V1227_EXPECT_IMAGE:-llm-scaler-exp:v1.2.35}"')

# 2. comment refresh (content-gate wording covers both files now)
rep("# informational; the certified-CONTENT gate is the adapter md5 (dcb23fbe,\n# v141 deployed on both adapter paths).",
    "# informational; the certified-CONTENT gates are the adapter md5 (dcb23fbe,\n# v141 deployed on both adapter paths) and the gpu_model_runner md5\n# (6520f916, v135 P10 queue-kick).")

# 3. GMR config vars after the adapter path var
rep('EXPECT_ADAPTER_PATH="/opt/venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py"',
    'EXPECT_ADAPTER_PATH="/opt/venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py"\n'
    '# v135 P10: _p10_wait_counts queue-kick in the model runner (long-ctx\n'
    '# chunked-prefill event-visibility stalls fixed 2026-10-10, baked v1.2.35)\n'
    'EXPECT_GMR_MD5="${V1227_EXPECT_GMR_MD5:-6520f91686ac8db3d296595bb43c13e2}"\n'
    'EXPECT_GMR_PATH="/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"')

# 4. GMR check block after the adapter check block
rep('  echo "DRIFT: adapter md5 \'$ADAPTER_MD5\' != $EXPECT_ADAPTER_MD5 (v141 certified content)"\nfi\n',
    '  echo "DRIFT: adapter md5 \'$ADAPTER_MD5\' != $EXPECT_ADAPTER_MD5 (v141 certified content)"\nfi\n\n'
    '# content gate 2 (v135): P10 model-runner patch\n'
    'GMR_MD5=$(docker exec lsv-test md5sum $EXPECT_GMR_PATH 2>/dev/null | awk \'{print $1}\')\n'
    'GMR_OK=0\n'
    'if [ "$GMR_MD5" = "$EXPECT_GMR_MD5" ]; then\n'
    '  GMR_OK=1\n'
    'else\n'
    '  echo "DRIFT: gpu_model_runner md5 \'$GMR_MD5\' != $EXPECT_GMR_MD5 (v135 P10 certified content)"\n'
    'fi\n')

# 5. verdict chain: add GMR_OK conjunction
rep('   && [ "$ADAPTER_OK" = "1" ] \\\n',
    '   && [ "$ADAPTER_OK" = "1" ] \\\n   && [ "$GMR_OK" = "1" ] \\\n')

# 6. gate comment accuracy
rep("# Gate = content (adapter md5) + serve-cmdline floor + spec/tmpl/async +",
    "# Gate = content (adapter+GMR md5) + serve-cmdline floor + spec/tmpl/async +")

# 7. verdict string v134 -> v135
rep('posture=certified-lmc-v134-spec-async-tmpl',
    'posture=certified-lmc-v135-spec-async-tmpl')

open(P, "w").write(src)
print("EDIT_OK")
