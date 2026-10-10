"""Splice attn-page hashing into LMCDbgFwd (mamba_utils.py).

The g0/g1/g2 mamba tables are hash-verified at forward time, but a3 (full
attention) pages print ids only — the async-loaded attn KV was never
content-verified. Add per-page md5 for first+middle attn layer (k,v) x first
3 logical pages, scaling the dim0 index by 1024//spec.block_size so a
64-token-block attn tensor hashes a full 1024-token logical page.
Run INSIDE container: /opt/venv/bin/python /tmp/patch_mamba_a3hash.py
"""
import hashlib
import py_compile
import shutil

P = "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py"

OLD = '''                        if type(spec).__name__ != "MambaSpec":
                            parts.append(
                                f"a{gi}:{req_state.block_ids[gi][:6]}"
                            )'''

NEW = '''                        if type(spec).__name__ != "MambaSpec":
                            aids = req_state.block_ids[gi][:6]
                            ahs = []
                            try:
                                import hashlib as _hl
                                per_log = max(1, 1024 // spec.block_size)
                                gnames = kv_cache_config.kv_cache_groups[gi].layer_names
                                for ln in (gnames[0], gnames[len(gnames) // 2]):
                                    att2 = forward_context[ln]
                                    for st2 in att2.kv_cache:
                                        for b_i in range(min(3, len(aids))):
                                            lo = b_i * per_log
                                            pg = st2[lo:lo + per_log].reshape(-1)[:262144]
                                            torch.xpu.synchronize()
                                            ahs.append(_hl.md5(pg.cpu().numpy().tobytes()).hexdigest()[:8])
                            except Exception as e:
                                ahs.append("ERR" + repr(e)[:24])
                            parts.append(
                                f"a{gi}:{aids}[{','.join(ahs)}]"
                            )'''

src = open(P).read()
n = src.count(OLD)
assert n == 1, f"anchor count {n} != 1"
shutil.copyfile(P, P + ".bak_a3hash")
src = src.replace(OLD, NEW)
open(P, "w").write(src)
py_compile.compile(P, doraise=True)
print("PATCH_OK", hashlib.md5(open(P, "rb").read()).hexdigest()[:8])
