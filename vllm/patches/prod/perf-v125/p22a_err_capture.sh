#!/bin/bash
# p22a_err_capture.sh — re-run the ESIMD build incrementally and capture
# the real compiler errors (first run's tail -40 window lost them).
set -u
docker exec esimd-inc bash -c '
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  export VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
  export MAX_JOBS=8 CC=icx CXX=icpx
  cd /src/esimd
  /opt/venv/bin/python setup_gdn_only.py build_ext --inplace > /tmp/esimd_build2.out 2>&1
  echo "BUILD_RC=$?"
  echo ---ERRORS---
  grep -E "error:|FAILED" /tmp/esimd_build2.out | head -40
  echo ---CONTEXT---
  grep -B2 -A6 "error:" /tmp/esimd_build2.out | head -100
'
