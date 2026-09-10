"""Exact-token capacity baseline using the previously audited HTTP helper."""
import json
import live_probe as probe

for length in (65536, 131072, 261888):
    body = dict(prompt=probe.prompt_tokens(length, 20260908 + length),
                max_tokens=128, temperature=0, ignore_eos=True)
    result = probe.request(f'lce1-mtp4-tq4-{length}', body)
    print(json.dumps(result), flush=True)
    if result.get('error'):
        raise RuntimeError(result['error'])
