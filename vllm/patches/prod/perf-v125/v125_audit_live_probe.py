"""Non-destructive HTTP audit. Run on host; JSONL stdout is the evidence.

Uses server tokenization and exact token-ID prompts, separate unique prefixes,
barrier-synchronized clients, monotonic clocks and mandatory final usage.
Capacity probes deliberately ignore EOS; chat probes do not.
"""
import concurrent.futures
import hashlib
import json
import random
import sys
import threading
import time
import urllib.request

BASE = 'http://127.0.0.1:8000'
MODEL = 'qwen3.8-27b-fp8'

def post(path, body):
    return urllib.request.urlopen(urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'}), timeout=900)

def prompt_tokens(size, seed):
    rng = random.Random(seed)
    words = ['service', 'request', 'cache', 'latency', 'document', 'result',
             'memory', 'worker', 'client', 'record', 'queue', 'response']
    text = f'Audit document {seed}. Summarize the operational records.\n'
    text += '\n'.join(f'Record {i}: {rng.choice(words)} {rng.randrange(1000000)} '
                      f'{rng.choice(words)} {rng.choice(words)}.'
                      for i in range(size // 8 + 100))
    with post('/tokenize', {'model': MODEL, 'prompt': text}) as r:
        ids = json.load(r)['tokens']
    if len(ids) < size:
        raise ValueError(f'Insufficient tokens: {len(ids)} < {size}')
    return ids[:size]

def request(label, body, path='/v1/completions', barrier=None):
    if barrier:
        barrier.wait()
    start = time.perf_counter()
    first = None
    last = None
    stamps = []
    chunks = []
    usage = None
    finish = None
    done = False
    result = {'label': label, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'requested_prompt_tokens': len(body['prompt']) if isinstance(body.get('prompt'), list) else None,
              'max_tokens': body['max_tokens'], 'ignore_eos': body.get('ignore_eos', False)}
    try:
        with post(path, dict(body, model=MODEL, stream=True,
                             stream_options={'include_usage': True})) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith('data: '):
                    continue
                if line[6:] == '[DONE]':
                    done = True
                    break
                data = json.loads(line[6:])
                if data.get('error'):
                    raise RuntimeError(data['error'])
                if data.get('usage'):
                    usage = data['usage']
                for ch in data.get('choices', []):
                    delta = ch.get('delta') or {}
                    content = ch.get('text') or delta.get('content') or delta.get('reasoning') or delta.get('reasoning_content')
                    if content:
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        stamps.append(now - start)
                        chunks.append(content)
                    finish = ch.get('finish_reason') or finish
        if not done or not usage or finish is None:
            raise RuntimeError('Missing DONE, final usage, or finish_reason')
        expected = result['requested_prompt_tokens']
        if expected is not None and usage['prompt_tokens'] != expected:
            raise RuntimeError(f'Prompt token mismatch: {usage["prompt_tokens"]} != {expected}')
        output = ''.join(chunks)
        result.update(usage=usage, finish_reason=finish, done=done,
                      ttft_s=first-start if first else None,
                      last_content_s=last-start if last else None,
                      content_event_times_s=stamps,
                      output_sha256=hashlib.sha256(output.encode()).hexdigest(),
                      output=output,
                      completion_tokens_per_post_first_content_s=(usage['completion_tokens'] / (last-first)) if first and last > first else None)
    except Exception as e:
        result['error'] = str(e)
    result['wall_s'] = time.perf_counter() - start
    return result

def emit(result):
    print(json.dumps(result), flush=True)

def main():
    seed = int(time.time())
    mode = sys.argv[1] if len(sys.argv) > 1 else 'short'
    if mode == 'chat':
        for name, text in [('arithmetic', 'What is 17 * 23? Answer with only the number.'),
                           ('json', 'Return only valid JSON with keys status and count, values ok and 3.'),
                           ('stop', 'Write the word DONE exactly once and stop.')]:
            emit(request(name, {'messages': [{'role': 'user', 'content': text}],
                               'max_tokens': 512, 'temperature': 0}, '/v1/chat/completions'))
    else:
        cases = [(2048, 1), (2048, 4)] if mode == 'short' else [(131072, 1), (261888, 1)]
        if mode == 'deepmulti':
            cases = [(131072, 2), (261888, 2)]
        for size, clients in cases:
            bodies = [{'prompt': prompt_tokens(size, seed + size + i),
                       'max_tokens': 256, 'temperature': 0, 'ignore_eos': True}
                      for i in range(clients)]
            barrier = threading.Barrier(clients)
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
                results = list(pool.map(lambda i: request(f'{mode}-{size}-c{clients}-{i}', bodies[i], barrier=barrier), range(clients)))
            wall = time.perf_counter() - start
            for result in results:
                emit(result)
            emit({'summary': f'{mode}-{size}-c{clients}', 'wall_s': wall,
                  'aggregate_output_tokens_per_s': sum(r.get('usage', {}).get('completion_tokens', 0) for r in results) / wall,
                  'errors': sum('error' in r for r in results)})

if __name__ == '__main__':
    main()
