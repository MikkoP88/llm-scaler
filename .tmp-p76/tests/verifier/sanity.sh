#!/bin/bash
# Acceptance sanity: 10 short sequential chat requests at temp 0.3.
# Usage: sanity.sh <output-file>
OUT="$1"
: > "$OUT"
PROMPTS=(
  "What is the capital of France?"
  "Name three primary colors."
  "What is 7 + 5?"
  "Who wrote Romeo and Juliet?"
  "Which planet is closest to the sun?"
  "Give a one-sentence definition of entropy."
  "What is the chemical symbol for gold?"
  "Name one common use of a hammer."
  "How many sides does a triangle have?"
  "Translate the word 'hello' into Spanish."
)
for i in "${!PROMPTS[@]}"; do
  n=$((i+1))
  p="${PROMPTS[$i]}"
  echo "=== request $n : $p ===" >> "$OUT"
  curl -s -o /tmp/resp_$$.json -w "HTTP=%{http_code} wall=%{time_total}s" \
    --connect-timeout 10 --max-time 180 \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen3.8-27b-fp8\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"temperature\":0.3,\"max_tokens\":64}" \
    http://10.20.3.65:8000/v1/chat/completions >> "$OUT"
  echo "" >> "$OUT"
  cat /tmp/resp_$$.json >> "$OUT"
  echo "" >> "$OUT"
done
rm -f /tmp/resp_$$.json
echo "DONE" >> "$OUT"
