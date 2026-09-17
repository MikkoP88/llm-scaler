#!/bin/bash
# bs64_mk.sh — build --block-size 64 variants (user directive 2026-09-17:
# "rerun Bench3, 3 best candidates, --block-size 64"). Baseline bands were
# ALL measured at --block-size 512 (standing since v5x lineage); 512->64 is
# the single variable. Candidates: e5m2 (standing, 50.4), e4m3 (54.4),
# tq4nc (55.5) — the three fastest validated dtypes on the 26.14 stack.
cd /root/build
set -eu

mk_variant() {  # $1=src serve script  $2=name
  sed 's@--block-size 512@--block-size 64@' "$1" > "serve_bs64_$2.sh"
  grep -q -- '--block-size 64' "serve_bs64_$2.sh" || { echo "SED_FAIL $2"; exit 1; }
  if diff <(grep -v 'block-size' "$1") <(grep -v 'block-size' "serve_bs64_$2.sh") >/dev/null; then
    echo "VARIANT_OK $2 (single-line delta vs $1)"
  else
    echo "DELTA_TOO_BIG $2"; diff "$1" "serve_bs64_$2.sh"; exit 1
  fi
}

mk_variant serve_user.sh            e5m2
mk_variant serve_user_e4m3_262k.sh  e4m3
mk_variant serve_user_tq4nc.sh      tq4nc

# boot scripts = Q21b lineage with the serve cp source swapped
sed 's@cp /root/build/serve_user.sh lsv-test@cp /root/build/serve_bs64_e5m2.sh lsv-test@'  repro_bootQ21b.sh > repro_bootBS64_E5M2.sh
sed 's@cp /root/build/serve_user.sh lsv-test@cp /root/build/serve_bs64_e4m3.sh lsv-test@'  repro_bootQ21b.sh > repro_bootBS64_E4M3.sh
sed 's@cp /root/build/serve_user.sh lsv-test@cp /root/build/serve_bs64_tq4nc.sh lsv-test@' repro_bootQ21b.sh > repro_bootBS64_TQ4NC.sh
for b in repro_bootBS64_E5M2.sh repro_bootBS64_E4M3.sh repro_bootBS64_TQ4NC.sh; do
  grep -q 'serve_bs64_' "$b" || { echo "BOOTSED_FAIL $b"; exit 1; }
done
echo "MK_DONE"
