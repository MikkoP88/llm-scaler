"""Natural-language long-context suite — PLAN.md 'Next' item 1 harness.

Non-destructive HTTP suite against the standing server (127.0.0.1:8000).
Tasks: needle retrieval at depths 8/50/92%, multi-hop sum-of-three, three-
bullet summarization. Exact prompt lengths via token-ID prompts assembled
from server-tokenized segments. temperature=0, max_tokens=192, natural EOS.

Per request: usage, TTFT, per-event stamps, verifier-step clusters (stream
events grouped by <= STEP_GAP_S gaps -> committed speculative bursts, tokens
estimated by character share), quality flags (hit / bullets / repeated
n-gram), output sha. Per repeat: engine acceptance-window delta parsed from
ENGINE_LOG ('SpecDecoding metrics' lines, host-visible /root/b_<name>.log).

Exit codes: 0 ok; 42 engine unhealthy (master skips rest of block); 43 cell
invalid (e.g. length rejected — skip length, keep block).

Usage:
  python3 nlp_suite.py OUTDIR CELL MODE LENGTH CLIENTS REPEATS BASESEED
  python3 nlp_suite.py OUTDIR cancel BASESEED          # abort probe
"""
import concurrent.futures
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

BASE = os.environ.get('LCE1_BASE', 'http://127.0.0.1:8000')
MODEL = 'qwen3.8-27b-fp8'
ENGINE_LOG = os.environ.get('ENGINE_LOG', '')  # container-internal path
CONTAINER = os.environ.get('LCE1_DOCKER', 'lsv-test')  # '' -> direct file read
STEP_GAP_S = 0.003
TASKS = ('needle', 'multihop', 'summarize')
DEPTH_BY_REPEAT = (0.08, 0.50, 0.92, 0.50, 0.08)

NAMES = ['Helsinki', 'Rotterdam', 'Osaka', 'Lyon', 'Porto', 'Krakow', 'Tampa',
         'Nairobi', 'Bogota', 'Cairo', 'Perth', 'Austin', 'Delft', 'Bergen',
         'Kyiv', 'Pune', 'Lima', 'Riga', 'Tunis', 'Hanoi', 'Aalborg', 'Turku',
         'Munster', 'Gdansk', 'Exeter', 'Uppsala']
SYSTEMS = ['ingest gateway', 'scheduler', 'cache tier', 'auth service',
           'storage broker', 'event bus', 'queue worker', 'index builder',
           'replication agent', 'metering probe', 'config server', 'audit tail']
METRICS = ['requests', 'sessions', 'records', 'writes', 'alerts', 'tasks',
           'messages', 'batches', 'checkpoints', 'tokens', 'files', 'jobs']
PERIODS = ['nightly', 'morning', 'afternoon', 'evening', 'weekend', 'holiday',
           'maintenance', 'peak', 'off-peak', 'backup']
STATUSES = ['nominal', 'degraded', 'recovering', 'elevated', 'stable',
            'throttled', 'rebalanced', 'deferred']
ANIMALS = ['ZEBRA', 'FALCON', 'OTTER', 'IBEX', 'MARLIN', 'PUMA', 'HERON',
           'LYNX', 'BADGER', 'MANTIS', 'WOMBAT', 'CORMORANT']
TEMPLATES = [
    'Record {i}: the {site} {system} processed {n} {metric} during the {period} window; status {status}.',
    'Record {i}: {site} reported its {system} handled {n} {metric} with status {status} in the {period} interval.',
    'Record {i}: during {period} operations the {system} at {site} queued {n} {metric}; overall status {status}.',
    'Record {i}: the {system} team in {site} archived {n} {metric} after the {period} run; status remained {status}.',
    'Record {i}: {period} summary for {site}: {system} throughput was {n} {metric}, status {status}.',
    'Record {i}: operators in {site} flagged the {system} as {status} after {n} {metric} were retried in the {period} cycle.',
    'Record {i}: the {system} serving {site} emitted {n} {metric} in the {period} batch; post-processing status {status}.',
    'Record {i}: capacity note: {site} {system} sustained {n} {metric} across the {period} window and closed {status}.',
    'Record {i}: audit tail for the {period} window shows {site} {system} at {status} with {n} {metric} inspected.',
    'Record {i}: replication from {site} delivered {n} {metric} to the {system}; {period} status {status}.',
]

METRIC_PAT = 'SpecDecoding metrics'
ACCEPT_RE = re.compile(
    r'Mean acceptance length: (?P<mean>[\d.]+).*?Accepted: (?P<accepted>\d+).*?'
    r'Drafted: (?P<drafted>\d+).*?Per-position acceptance rate: (?P<positions>[\d., ]+?)'
    r'\s*,\s*Avg Draft acceptance rate: (?P<avg>[\d.]+)%')


def filler_text(rng, approx_tokens):
    out = []
    i = 0
    cur = 0
    target = int(approx_tokens * 5.4)
    while cur < target:
        t = TEMPLATES[rng.randrange(len(TEMPLATES))].format(
            i=i, site=rng.choice(NAMES), system=rng.choice(SYSTEMS),
            n=rng.randrange(1, 999999), metric=rng.choice(METRICS),
            period=rng.choice(PERIODS), status=rng.choice(STATUSES))
        out.append(t)
        cur += len(t) + 1
        i += 1
    return '\n'.join(out)


def tok(text):
    body = json.dumps({'model': MODEL, 'prompt': text}).encode()
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    BASE + '/tokenize', data=body,
                    headers={'Content-Type': 'application/json'}), timeout=600) as r:
                return json.load(r)['tokens']
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def build_prompt(task, length, seed, depth):
    """Assemble an exact-`length` token-ID prompt: filler + task items + question."""
    rng = random.Random(seed)
    approx = int(length * 1.15) + 512
    while True:
        filler = tok(filler_text(rng, approx))
        if len(filler) >= length:
            break
        approx = int(len(filler) * 1.3) + 4096
    if task == 'needle':
        site = rng.choice(NAMES)
        animal = rng.choice(ANIMALS)
        code = f'{animal}-{100 + rng.randrange(900)}'
        memo_ids = tok(f'MEMO {animal}: the retention code for the {site} '
                       f'datacenter is {code}.')
        q_ids = tok(f' According to MEMO {animal}, what is the retention code '
                    f'for the {site} datacenter? State the code, then write two '
                    f'sentences describing what the retention policy covers.')
        budget = length - len(memo_ids) - len(q_ids)
        cut = int(round(depth * budget))
        ids = filler[:cut] + memo_ids + filler[cut:budget] + q_ids
        meta = dict(task=task, needle_depth=depth, expected=code,
                    memo_center_token=cut + len(memo_ids) // 2)
    elif task == 'multihop':
        sites = rng.sample(NAMES, 3)
        animals = rng.sample(ANIMALS, 3)
        while True:
            amts = [rng.randrange(1200, 9800) for _ in range(3)]
            total = sum(amts)
            if 10000 <= total <= 29000:
                break
        memo_ids = [tok(f'MEMO {animals[k]}: the {sites[k]} branch allocated '
                        f'{amts[k]} units of buffer capacity.') for k in range(3)]
        q_ids = tok(' How many units of buffer capacity were allocated in '
                    'total across the three branches named in the memos? Give '
                    'the total, then name each branch and its allocation in '
                    'one sentence.')
        budget = length - sum(len(m) for m in memo_ids) - len(q_ids)
        ids = []
        prev = 0
        for frac, m in zip((0.25, 0.50, 0.75), memo_ids):
            cut = int(round(frac * budget))
            ids += filler[prev:cut] + m
            prev = cut
        ids += filler[prev:budget] + q_ids
        meta = dict(task=task, expected=str(sum(amts)), amounts=amts, sites=sites)
    else:
        q_ids = tok(' Task: Summarize the operational records above in '
                    'exactly three bullet points, each starting with "-" and '
                    'one to two sentences long.')
        ids = filler[:length - len(q_ids)] + q_ids
        meta = dict(task=task)
    assert len(ids) == length, (task, len(ids), length)
    return ids, meta


def evaluate(task, meta, output):
    q = {}
    flat = output.replace(',', '')
    if task == 'needle':
        e = meta['expected']
        q['needle_hit'] = e in output or e in flat
        q['needle_hit_ci'] = e.lower() in output.lower()
    elif task == 'multihop':
        e = meta['expected']
        q['multihop_hit'] = e in output or e in flat
    else:
        q['summarize_bullets'] = output.count('-')
        q['summarize_ok'] = q['summarize_bullets'] >= 2
    words = output.split()
    q['output_words'] = len(words)
    grams = set()
    q['repeated_10gram'] = False
    for i in range(max(0, len(words) - 9)):
        g = ' '.join(words[i:i + 10])
        if g in grams:
            q['repeated_10gram'] = True
            break
        grams.add(g)
    return q


def cluster_stats(events, total_completion):
    """events: [(t_since_start, char_count)] -> verifier-step clusters."""
    if not events:
        return {}
    groups = []
    cur = [events[0]]
    for e in events[1:]:
        if e[0] - cur[-1][0] <= STEP_GAP_S:
            cur.append(e)
        else:
            groups.append(cur)
            cur = [e]
    groups.append(cur)
    total_chars = sum(e[1] for e in events) or 1
    steps = []
    for g in groups:
        chars = sum(e[1] for e in g)
        steps.append(dict(events=len(g),
                          est_tokens=round(total_completion * chars / total_chars),
                          span_s=round(g[-1][0] - g[0][0], 4)))
    est = [s['est_tokens'] for s in steps]
    return dict(n_steps=len(groups),
                median_est_tokens_per_step=sorted(est)[len(est) // 2],
                max_est_tokens_per_step=max(est),
                first_step=steps[0],
                largest_step=max(steps, key=lambda s: s['est_tokens']))


def _log_exists():
    if not ENGINE_LOG:
        return False
    if CONTAINER and not os.path.exists(ENGINE_LOG):
        try:
            r = subprocess.run(['docker', 'exec', CONTAINER, 'test', '-f', ENGINE_LOG],
                               capture_output=True, timeout=60)
            return r.returncode == 0
        except Exception:
            return False
    return os.path.exists(ENGINE_LOG)


def _grep_cmds(pattern, tail_n):
    """(count_cmd, tail_cmd) — docker exec when the log is container-internal."""
    if CONTAINER and not os.path.exists(ENGINE_LOG):
        return (['docker', 'exec', CONTAINER, 'grep', '-a', '-i', '-c', pattern, ENGINE_LOG],
                ['docker', 'exec', CONTAINER, 'bash', '-c',
                 f'grep -a -i "{pattern}" {shlex.quote(ENGINE_LOG)} | tail -n {int(tail_n)}'])
    return (['grep', '-a', '-i', '-c', pattern, ENGINE_LOG],
            ['bash', '-c',
             f'grep -a -i "{pattern}" {shlex.quote(ENGINE_LOG)} | tail -n {int(tail_n)}'])


def grep_engine(pattern, tail):
    """Count + last `tail` matching lines of ENGINE_LOG (case-insensitive)."""
    if not ENGINE_LOG or not _log_exists():
        if not ENGINE_LOG:
            return None
        return dict(missing=ENGINE_LOG)
    try:
        c1, c2 = _grep_cmds(pattern, tail)
        c = subprocess.run(c1, capture_output=True, text=True, timeout=180)
        count = int(c.stdout.strip() or 0)
        t = subprocess.run(c2, capture_output=True, text=True, timeout=180)
        return dict(count=count, lines=t.stdout.splitlines()[-tail:])
    except Exception as e:
        return dict(error=f'{type(e).__name__}: {e}')


def acceptance_count():
    if not ENGINE_LOG or not _log_exists():
        return None
    try:
        c1, _ = _grep_cmds(METRIC_PAT, 1)
        c = subprocess.run(c1, capture_output=True, text=True, timeout=180)
        return int(c.stdout.strip() or 0)
    except Exception:
        return None


def acceptance_window(n):
    """Parse exactly the last n SpecDecoding metric lines (the cell window).

    Accepted:/Drafted: are per-interval sums, so the window is summed, not
    differenced. Mean acceptance length is per-interval; keep the series.
    """
    if not ENGINE_LOG or not _log_exists() or n <= 0:
        return None
    try:
        _, c2 = _grep_cmds(METRIC_PAT, n)
        t = subprocess.run(c2, capture_output=True, text=True, timeout=180)
        lines = t.stdout.splitlines()
    except Exception as e:
        return dict(error=f'{type(e).__name__}: {e}')
    parsed = []
    for line in lines:
        m = ACCEPT_RE.search(line)
        if m:
            parsed.append(dict(
                mean=float(m.group('mean')),
                accepted=int(m.group('accepted')),
                drafted=int(m.group('drafted')),
                positions=[float(x) for x in m.group('positions').split(',') if x.strip()],
                avg=float(m.group('avg'))))
    out = dict(metric_lines=len(lines), parsed_lines=len(parsed))
    if parsed:
        out['accepted_sum'] = sum(p['accepted'] for p in parsed)
        out['drafted_sum'] = sum(p['drafted'] for p in parsed)
        out['mean_accept_length_series'] = [p['mean'] for p in parsed]
        out['per_position_last'] = parsed[-1]['positions']
        out['avg_draft_rate_last'] = parsed[-1]['avg']
    return out


def request(label, ids, body_extra=None, path='/v1/completions',
            barrier=None, task=None, meta=None):
    if barrier:
        barrier.wait()
    start = time.perf_counter()
    events = []
    chunks = []
    usage = None
    finish = None
    done = False
    body = dict(prompt=ids, max_tokens=192, temperature=0, stream=True,
                stream_options={'include_usage': True})
    if body_extra:
        body.update(body_extra)
    result = dict(label=label, task=task, task_meta=meta, prompt_tokens=len(ids),
                  utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        req = urllib.request.Request(
            BASE + path,
            data=json.dumps(dict(body, model=MODEL)).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data: '):
                    continue
                payload = line[6:]
                if payload == '[DONE]':
                    done = True
                    break
                data = json.loads(payload)
                if data.get('error'):
                    raise RuntimeError(str(data['error']))
                if data.get('usage'):
                    usage = data['usage']
                for ch in data.get('choices', []):
                    text = ch.get('text') or (ch.get('delta') or {}).get('content') or ''
                    if text:
                        now = time.perf_counter()
                        events.append((now - start, len(text)))
                        chunks.append(text)
                    finish = ch.get('finish_reason') or finish
        if not done or usage is None or finish is None:
            raise RuntimeError('missing DONE/usage/finish_reason')
        if usage['prompt_tokens'] != len(ids):
            raise RuntimeError(f"prompt mismatch {usage['prompt_tokens']} != {len(ids)}")
        output = ''.join(chunks)
        first = events[0][0] if events else None
        last = events[-1][0] if events else None
        dw = (last - first) if (first is not None and last is not None and last > first) else None
        result.update(
            usage=usage, finish_reason=finish, ttft_s=first, last_content_s=last,
            decode_window_s=dw,
            completion_tokens_per_decode_window_s=(usage['completion_tokens'] / dw) if dw else None,
            clusters=cluster_stats(events, usage['completion_tokens']),
            output_sha256=hashlib.sha256(output.encode()).hexdigest(),
            output=output,
            quality=evaluate(task, meta or {}, output))
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    result['wall_s'] = time.perf_counter() - start
    return result


def health():
    try:
        r = request('health', tok('Health probe.'), {'max_tokens': 4})
        return 'error' not in r
    except Exception:
        return False


def run_cell(outdir, mode, length, clients, repeats, base_seed):
    records = []
    for r in range(repeats):
        task = TASKS[r % 3]
        depth = DEPTH_BY_REPEAT[r]
        barrier = threading.Barrier(clients)
        bodies = []
        build_errors = 0
        for c in range(clients):
            seed = base_seed + 7919 * (c + 1) + 104729 * r + 13
            try:
                bodies.append(build_prompt(task, length, seed, depth))
            except Exception as e:
                build_errors += 1
                print(json.dumps(dict(cell=f'{mode}-{length}-c{clients}', repeat=r,
                                      client=c, build_error=f'{type(e).__name__}: {e}')), flush=True)
        if build_errors:
            return records, 43
        c0 = acceptance_count()
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
            futs = [pool.submit(request, f'{mode}-{length}-c{clients}-r{r}-c{c}-{task}',
                                ids, None, barrier=barrier, task=task, meta=meta)
                    for c, (ids, meta) in enumerate(bodies)]
            results = [f.result() for f in futs]
        wall = time.perf_counter() - start
        c1 = acceptance_count()
        window = acceptance_window(c1 - c0) if (c0 is not None and c1 is not None) else None
        rec = dict(mode=mode, length=length, clients=clients, repeat=r, task=task,
                   needle_depth=depth if task == 'needle' else None,
                   wall_s=round(wall, 3),
                   aggregate_output_tokens_per_s=round(sum(
                       x.get('usage', {}).get('completion_tokens', 0)
                       for x in results) / wall, 3),
                   errors=sum('error' in x for x in results),
                   engine_metric_lines=[c0, c1],
                   engine_acceptance_window=window,
                   requests=results)
        records.append(rec)
        print(json.dumps(rec), flush=True)
        if rec['errors']:
            errtxt = ' '.join(str(x.get('error')) for x in results if x.get('error'))
            if 'HTTPError' in errtxt and ('400' in errtxt or '413' in errtxt):
                return records, 43
            if not health():
                return records, 42
    return records, 0


def cancel_probe(outdir, base_seed):
    ids, _ = build_prompt('summarize', 131072, base_seed, None)
    result = dict(probe='cancel', prompt_tokens=len(ids),
                  utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    start = time.perf_counter()
    aborted_at = None
    try:
        req = urllib.request.Request(
            BASE + '/v1/completions',
            data=json.dumps(dict(model=MODEL, prompt=ids, max_tokens=4096,
                                 temperature=0, ignore_eos=True,
                                 stream=True)).encode(),
            headers={'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req, timeout=3600)
        for raw in resp:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data: '):
                continue
            payload = line[6:]
            if payload == '[DONE]':
                break
            data = json.loads(payload)
            for ch in data.get('choices', []):
                if ch.get('text'):
                    aborted_at = time.perf_counter() - start
                    break
            if aborted_at is not None:
                break
        resp.close()
    except Exception as e:
        result['abort_exception'] = f'{type(e).__name__}: {e}'
    result['first_content_s'] = aborted_at
    checks = []
    base_t = time.perf_counter()
    for delay in (5, 30, 90):
        target = base_t + delay
        while time.perf_counter() < target:
            time.sleep(min(1.0, max(0.0, target - time.perf_counter())))
        h = request('post-cancel-health', tok('Health probe after cancellation.'),
                    {'max_tokens': 8})
        checks.append(dict(after_s=delay, ok='error' not in h,
                           finish=h.get('finish_reason'), wall_s=h.get('wall_s'),
                           error=h.get('error')))
    result['post_cancel_checks'] = checks
    g = grep_engine('abort', 3)
    if isinstance(g, dict):
        result['engine_abort_count'] = g.get('count')
        result['engine_abort_lines'] = g.get('lines', [])[-3:]
    print(json.dumps(result), flush=True)
    with open(os.path.join(outdir, 'cancel_probe.jsonl'), 'a') as f:
        f.write(json.dumps(result) + '\n')
    return 0 if all(c['ok'] for c in checks) else 42


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    kind = sys.argv[2]
    if kind == 'cancel':
        sys.exit(cancel_probe(outdir, int(sys.argv[3])))
    mode = kind
    length, clients, repeats, base_seed = (int(x) for x in sys.argv[3:7])
    skip_file = os.path.join(outdir, 'SKIP_CELLS')
    if os.path.exists(skip_file):
        for line in open(skip_file):
            parts = line.split()
            if len(parts) == 3 and parts[0] == mode and parts[1] == str(length) and parts[2] == str(clients):
                rec = dict(mode=mode, length=length, clients=clients, skipped=True,
                           reason='SKIP_CELLS entry',
                           utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                print(json.dumps(rec), flush=True)
                with open(os.path.join(outdir, f'cell_{mode}_{length}_c{clients}.json'), 'w') as f:
                    json.dump(rec, f)
                sys.exit(0)
    records, rc = run_cell(outdir, mode, length, clients, repeats, base_seed)
    with open(os.path.join(outdir, f'cell_{mode}_{length}_c{clients}.json'), 'w') as f:
        json.dump(dict(mode=mode, length=length, clients=clients,
                       repeats=repeats, base_seed=base_seed, records=records), f)
    sys.exit(rc)


if __name__ == '__main__':
    main()
