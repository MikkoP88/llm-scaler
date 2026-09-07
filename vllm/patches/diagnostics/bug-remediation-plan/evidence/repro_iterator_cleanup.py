"""CPU-only repro of the captured merge helper; no vLLM imports or GPU access."""
import ast
import asyncio
import contextlib
import json
from pathlib import Path

tree = ast.parse(Path(__file__).with_name('async_utils.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
          and n.name == 'merge_async_iterators')
module = ast.Module(body=[ast.ImportFrom(module='__future__',
    names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
namespace = dict(asyncio=asyncio, contextlib=contextlib,
                 FIRST_COMPLETED=asyncio.FIRST_COMPLETED)
exec(compile(ast.fix_missing_locations(module), '<captured helper>', 'exec'), namespace)

async def main():
    started = asyncio.Event()
    cleaned = asyncio.Event()
    async def slow():
        started.set()
        try:
            await asyncio.Event().wait()
            yield 'never'
        finally:
            await asyncio.sleep(0.05)
            cleaned.set()
    async def fast():
        await started.wait()
        yield 'ready'
    merge = namespace['merge_async_iterators'](fast(), slow())
    await anext(merge)
    await merge.aclose()
    immediate = cleaned.is_set()
    await asyncio.sleep(0.1)
    print(json.dumps({'cleanup_completed_when_merge_close_returns': immediate,
                      'cleanup_completed_later': cleaned.is_set()}))
    assert not immediate and cleaned.is_set(), 'Captured cleanup behavior changed'

asyncio.run(main())
