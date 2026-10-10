#!/bin/bash
# Parse SPECTIMING lines and print median/p10/p90 per metric.
# Usage: parse_spectiming.sh <spectiming-log>
LOG="$1"
TMP=$(mktemp -d)
sed -nE 's/.*step_wall_ms=([0-9.]+).*propose h=[0-9.]+ d=([0-9.]+).*greedy h=[0-9.]+ d=([0-9.]+).*/\1 \2 \3/p' "$LOG" > "$TMP/vals"
n=$(wc -l < "$TMP/vals")
echo "parsed_lines=$n"
med() { sort -n | awk '{a[NR]=$1} END {if (NR==0) print "NA"; else if (NR%2) printf "%.1f", a[(NR+1)/2]; else printf "%.1f", (a[NR/2]+a[NR/2+1])/2}'; }
pct() { sort -n | awk -v p="$1" '{a[NR]=$1} END {if (NR==0) print "NA"; else {i=int((p/100.0)*NR+0.5); if(i<1)i=1; if(i>NR)i=NR; printf "%.1f", a[i]}}'; }
for f in step_wall propose_d greedy_d; do
  case "$f" in
    step_wall) col=1;; propose_d) col=2;; greedy_d) col=3;;
  esac
  awk -v c=$col '{print $c}' "$TMP/vals" > "$TMP/col"
  echo "$f median=$(med < "$TMP/col") p10=$(pct 10 < "$TMP/col") p90=$(pct 90 < "$TMP/col")"
done
# Pure-decode vs mixed split (step_wall < 100 ms treated as pure-decode).
awk '$1 < 100' "$TMP/vals" > "$TMP/pure"
awk '$1 >= 100' "$TMP/vals" > "$TMP/mixed"
echo "pure_decode_lines=$(wc -l < "$TMP/pure") mixed_lines=$(wc -l < "$TMP/mixed")"
for f in propose_d greedy_d step_wall; do
  case "$f" in
    step_wall) col=1;; propose_d) col=2;; greedy_d) col=3;;
  esac
  awk -v c=$col '{print $c}' "$TMP/pure" > "$TMP/col"
  echo "pure $f median=$(med < "$TMP/col")"
done
echo "greedy_d distinct values: $(awk '{print $3}' "$TMP/vals" | sort -n | uniq -c | tr '\n' ' ')"
rm -rf "$TMP"
