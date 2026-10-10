#!/bin/bash
# R1 burst validation: 6 concurrent shorts against the sharded-view drafter.
# Gates: 6x HTTP 200, spec acceptance unchanged (fork parity ~2.7 tok/step),
# SPECTIMING stable at batch>1, zero ERROR/Traceback, health 200 after.
cd /root/build/p77

snap() { curl -s --max-time 5 http://localhost:8000/metrics | grep -E '^vllm:speculative_(num_draft|num_accepted)' ; }

echo "=== pre metrics ==="
snap > /tmp/pre.txt; cat /tmp/pre.txt

for i in 1 2 3 4 5 6; do
  cat > req_burst_$i.json <<EOF
{"model": "qwen3.8-27b-fp8", "messages": [{"role": "user", "content": "Concurrent burst probe $i: list $((i * 7)) distinct uses of speculative decoding in production systems, one line each."}], "max_tokens": 400, "temperature": 0.5, "ignore_eos": true}
EOF
done

echo "=== burst fire (6 concurrent) ==="
for i in 1 2 3 4 5 6; do
  ( curl -s -m 240 -w "req$i HTTP=%{http_code} time_total=%{time_total}s\n" \
      -X POST http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' -d @req_burst_$i.json \
      -o out_burst_$i.json ) &
done
wait
echo "=== output sanity (first 160 chars each) ==="
for i in 1 2 3 4 5 6; do
  printf 'req%s: ' "$i"; head -c 160 out_burst_$i.json | tr -d '\n'; echo
done
echo "=== post metrics ==="
snap > /tmp/post.txt; cat /tmp/post.txt
echo "=== deltas (post - pre) ==="
python3 - <<'PYEOF' 2>/dev/null || paste /tmp/pre.txt /tmp/post.txt
def load(p):
    d = {}
    for line in open(p):
        parts = line.split()
        if len(parts) >= 2:
            d[parts[0]] = float(parts[-1])
    return d
pre, post = load('/tmp/pre.txt'), load('/tmp/post.txt')
for k in sorted(post):
    if k in pre:
        print(f"{k}  {pre[k]:.0f} -> {post[k]:.0f}  (delta {post[k]-pre[k]:.0f})")
    else:
        print(f"{k}  (new) {post[k]:.0f}")
PYEOF
echo "=== SPECTIMING tail 12 ==="
docker exec lsv-test sh -c 'grep SPECTIMING /root/serve_full.log | tail -n 12'
echo "=== error scan ==="
docker exec lsv-test sh -c "grep -c -E 'ERROR|Traceback' /root/serve_full.log || true"
curl -s -o /dev/null -w 'health=%{http_code}\n' --max-time 5 http://localhost:8000/health
echo "=== dmesg tail ==="
dmesg -T | tail -n 20 | grep -E 'xe|ccs|reset|devcoredump' | tail -n 5
echo "=== BURST DONE ==="
