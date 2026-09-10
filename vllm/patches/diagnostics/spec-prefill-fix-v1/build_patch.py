"""Generate a checked, reviewable prefill-phase fix from the installed fork.

Usage: python build_patch.py SOURCE_SCHED_DIRECTORY OUTPUT_DIRECTORY
Never edits the source directory. Output includes a unified patch and hashes.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import sys

source_dir, dest = map(Path, sys.argv[1:])
if source_dir.resolve() == dest.resolve():
    raise SystemExit('Source and destination must differ')
dest.mkdir(parents=True, exist_ok=True)
changes = {
    'output.py': [(
        '    new_block_ids_to_zero: list[int] | None = None\n',
        '    new_block_ids_to_zero: list[int] | None = None\n\n'
        '    # Phase of this scheduled step, not the mutable Request phase.\n'
        '    # None means unavailable (e.g. an output constructed by an older\n'
        '    # producer); absence cannot establish an invalid decode result.\n'
        '    scheduled_prefill_chunk_req_ids: frozenset[str] | None = None\n',
    )],
    'async_scheduler.py': [(
        '        super()._update_after_schedule(scheduler_output)\n',
        '        super()._update_after_schedule(scheduler_output)\n'
        '        # SPF1: capture immediately after this step advances its counters.\n'
        '        # Later queued steps may change Request.is_prefill_chunk before\n'
        '        # this step returns its legitimate empty prefill output.\n'
        '        scheduler_output.scheduled_prefill_chunk_req_ids = frozenset(\n'
        '            req_id for req_id in scheduler_output.num_scheduled_tokens\n'
        '            if self.requests[req_id].is_prefill_chunk\n'
        '        )\n',
    ), (
        '        _v52m_doomed: list[str] = []\n',
        '        _v52m_doomed: list[str] = []\n'
        '        _step_prefill = getattr(\n'
        '            scheduler_output, "scheduled_prefill_chunk_req_ids", None\n'
        '        )\n',
    ), (
        '                            or _req.is_prefill_chunk\n',
        '                            or _step_prefill is None\n'
        '                            or _rid in _step_prefill\n'
        '                            or _req.is_prefill_chunk\n',
    )],
}
manifest, diffs = {}, []
for name, replacements in changes.items():
    original = (source_dir / name).read_text(encoding='utf-8')
    candidate = original
    for old, new in replacements:
        if candidate.count(old) != 1:
            raise SystemExit(f'{name}: unexpected source layout for {old!r}')
        candidate = candidate.replace(old, new)
    ast.parse(candidate)
    (dest / name).write_text(candidate, encoding='utf-8', newline='\n')
    manifest[name] = dict(
        source_sha256=hashlib.sha256((source_dir / name).read_bytes()).hexdigest(),
        patched_sha256=hashlib.sha256((dest / name).read_bytes()).hexdigest())
    diffs.extend(difflib.unified_diff(
        original.splitlines(True), candidate.splitlines(True),
        fromfile='a/vllm/v1/core/sched/' + name,
        tofile='b/vllm/v1/core/sched/' + name))
(dest / 'spec-prefill-phase.patch').write_text(''.join(diffs), encoding='utf-8', newline='\n')
(dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(manifest))
