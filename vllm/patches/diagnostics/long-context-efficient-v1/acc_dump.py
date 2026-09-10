#!/usr/bin/env python3
"""acc_dump.py — dump speculative acceptance telemetry per mode/len/c from suite jsonl.

Reads /root/build/lce1/suite_<mode>.jsonl records; each repeat record has
top-level 'engine_acceptance_window' (dict) when the engine emitted metric lines.
Prints per (mode,len,c): repeats with data, mean accepted/drafted per step,
mean acceptance rate, per-position acceptance if present.
"""
import json
import sys
from collections import defaultdict

ROOT = '/root/build/lce1'

def load(mode):
    recs = []
    try:
        with open(f'{ROOT}/suite_{mode}.jsonl', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return recs

def walk(rep):
    """Yield every dict in the record tree that looks like an acceptance window."""
    stack = [rep]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if 'engine_acceptance_window' in cur:
                yield cur['engine_acceptance_window']
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend(x for x in cur if isinstance(x, (dict, list)))

def main():
    for mode in sys.argv[1:]:
        recs = load(mode)
        agg = defaultdict(lambda: dict(n=0, acc=0.0, acc_steps=0, drafted=0.0,
                                       steps=0, pos=None, no_window=0))
        for r in recs:
            if r.get('skipped'):
                continue
            key = (r.get('mode', mode), r.get('len') or r.get('length'),
                   r.get('c') or r.get('clients'))
            wins = list(walk(r))
            if not wins:
                agg[key]['no_window'] += 1
                continue
            for w in wins:
                a = agg[key]
                a['n'] += 1
                for k in ('accepted', 'accepted_tokens'):
                    if isinstance(w.get(k), (int, float)):
                        a['acc'] += w[k]
                        break
                for k in ('drafted', 'drafted_tokens'):
                    if isinstance(w.get(k), (int, float)):
                        a['drafted'] += w[k]
                        break
                for k in ('steps', 'decode_steps', 'windows'):
                    if isinstance(w.get(k), (int, float)):
                        a['steps'] += w[k]
                        break
                for k in ('acceptance', 'acceptance_rate', 'mean_acceptance'):
                    if isinstance(w.get(k), (int, float)):
                        a['acc_steps'] += 1
                        a['acc_rate_sum'] = a.get('acc_rate_sum', 0) + w[k]
                        break
                pk = [k for k in w if 'position' in k or k.startswith('pos_')]
                if pk and a['pos'] is None:
                    a['pos'] = {k: w[k] for k in pk}
        print(f'=== {mode}')
        for key in sorted(agg, key=lambda k: (k[1] or 0, k[2] or 0)):
            a = agg[key]
            n = max(a['n'], 1)
            line = (f"  len={key[1]} c={key[2]}: windows={a['n']} no_window={a['no_window']} "
                    f"accepted/window={a['acc']/n:.2f} drafted/window={a['drafted']/n:.2f}")
            if a.get('acc_rate_sum') is not None:
                line += f" acc_rate_mean={a['acc_rate_sum']/max(a['acc_steps'],1):.3f}"
            if a['pos']:
                line += f" pos={json.dumps(a['pos'])[:300]}"
            print(line)

if __name__ == '__main__':
    main()
