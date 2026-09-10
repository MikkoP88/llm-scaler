"""Generate a reviewable source candidate, never edit a live installation.

Usage: python build_candidate.py original.py candidate.py
"""
import ast
import hashlib
from pathlib import Path
import sys

source_path, output_path = map(Path, sys.argv[1:])
if source_path.resolve() == output_path.resolve():
    raise SystemExit('Input and output must be different files')
source = source_path.read_text()
replacements = {
    'import torch\n': 'import torch\nimport os\nfrom .lce1_fp32_cache import fp32_snapshot\n'
                      '_LCE1_CACHE = os.environ.get("VLLM_LCE1_FP32_CACHE", "0") == "1"\n',
    '            self.base_kernel[side],':
        '            (fp32_snapshot(self, "base", self.base_kernel) if _LCE1_CACHE '
        'else self.base_kernel)[side],',
    '        proj = F.linear(hidden_states.float(), w.float())':
        '        w32 = fp32_snapshot(self, "projection", w) if _LCE1_CACHE else w.float()\n'
        '        proj = F.linear(hidden_states.float(), w32)',
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise SystemExit(f'Unexpected source layout: {old!r}, count={source.count(old)}')
    source = source.replace(old, new)
ast.parse(source)
output_path.write_text(source)
print('input_sha256', hashlib.sha256(source_path.read_bytes()).hexdigest())
print('candidate_sha256', hashlib.sha256(output_path.read_bytes()).hexdigest())
