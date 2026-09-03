# exact-kv-splits v38 — hypothesis FALSIFIED

`v38_exact_kv_splits.py` pinned an exact FA2 KV split count while hunting the
#18 fp8-KV decode bimodality. The hypothesis died on the facts: the split
machinery is inactive at short context (workload blocks wkb=2 < 16 → the
heuristic already picks num_splits=1), so the pin changed nothing.

The actual #18 root cause was the ESIMD decode kernel itself
(eagle_ops.page_attn_decode run-to-run race) — fixed in
`prod/attn-esimd-fp8-reroute-v38/`. Committed here as tooling/evidence only;
do NOT bake.
