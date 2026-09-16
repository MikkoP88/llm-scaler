#!/usr/bin/env python3
"""llm-scaler v58 P2 patcher: decode hot-path H2D copy coalescing.

CONTEXT (§24 G, crashfix-v58): NEO 26.18+ ships 76e8bd47f0 ("enable copy
via lock pointer for non-compressed resources on xe2+"): every L0 copy of
a non-compressed resource takes the CPU lock-pointer path instead of a
GPU copy submission. That commit is simultaneously the crash-avoidance
for the GSD-12919 xe/GuC race AND the -24%..-32% decode tax. The tax is
PER-OP (driver overhead per copy), not per-byte: our decode step issues
~14 ad-hoc H2D uploads (pageable torch.from_numpy().to(device) and
per-step pinned allocations), each now paying full lock-pointer overhead.

FIX (P2v1): coalesce the per-step ad-hoc uploads into a small number of
persistent pinned staging buffers, one H2D copy per group, callers get
device views. Values, dtypes and ordering are preserved:
  - group A (_calc_spec_decode_metadata): 5 pageable uploads -> 2 staged
    (int32: cu_draft/cu_sampled/bonus, int64: logits/target indices)
  - group B1 (_prepare_input_ids): sampled/prev scatter indices
    2 pinned alloc+copy -> 1 staged int64 (uploaded BEFORE the scatter
    that consumes it, preserving stream order)
  - group B2: draft/prev-draft indices 2 -> 1 staged int64
  - group B3: _dtids_local_padded per-step pinned zeros + H2D ->
    persistent pinned [n, num_spec] pair, row fill + 1 H2D
  - group C (llm_base_proposer): next_token_ids device-constructor ->
    1 staged int32; token_indices pageable + per-step qsl/seq_lens
    pinned allocs -> 2 staged (int64 indices, int32 qsl|seq_lens)
Copy-op count per decode step drops ~14 -> ~6; pageable -> pinned.
This REMOVES L0 copy ops (never adds GPU copy submissions), so the GuC
race stays starved exactly as on stock 26.18.
"""
import sys

F1 = ("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/"
      "gpu_model_runner.py")
F2 = ("/opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/"
      "llm_base_proposer.py")
MARKER = "llm-scaler v58 P2"

HELPER = '''

# llm-scaler v58 P2: per-step H2D copy coalescer (see patch header).
# Batches N small host arrays into ONE pinned->gpu copy; returns device
# views. Lazy-grow, values/dtypes preserved. Removes per-op L0 copy
# overhead on NEO>=26.18 (76e8bd47f0 lock-pointer copy tax).
class _P2CopyStaging:
    __slots__ = ("device", "dtype", "cpu", "gpu", "np")

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.cpu = None
        self.gpu = None
        self.np = None

    def _ensure(self, n):
        cap = 0 if self.cpu is None else self.cpu.numel()
        if self.cpu is None or n > cap:
            new_cap = max(n, cap * 2, 1024)
            self.cpu = torch.zeros(new_cap, dtype=self.dtype,
                                   pin_memory=True)
            self.gpu = torch.zeros(new_cap, dtype=self.dtype,
                                   device=self.device)
            self.np = self.cpu.numpy()

    def upload(self, segs):
        total = 0
        for s in segs:
            total += s.shape[0]
        self._ensure(total)
        off = 0
        views = []
        for s in segs:
            m = s.shape[0]
            if isinstance(s, torch.Tensor):
                self.cpu[off:off + m] = s
            else:
                self.np[off:off + m] = s
            views.append(self.gpu[off:off + m])
            off += m
        self.gpu[:total].copy_(self.cpu[:total], non_blocking=True)
        return views

'''

HELPER_ANCHOR = "logger = init_logger(__name__)\n"

# ---- group A: spec-decode metadata (5 pageable -> 2 staged) ----
A_OLD = '''\
        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True
        )
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )
'''
A_NEW = '''\
        # llm-scaler v58 P2: coalesce the five per-step spec-metadata H2D
        # uploads (pageable from_numpy().to(device), each paying full
        # lock-pointer copy overhead on NEO>=26.18) into two staged
        # pinned copies. Values identical; index dtypes int64 (safe for
        # every downstream consumer), cu/bonus stay int32.
        _p2_i32 = getattr(self, "_p2_spec_stg_i32", None)
        if _p2_i32 is None:
            _p2_i32 = self._p2_spec_stg_i32 = _P2CopyStaging(
                self.device, torch.int32)
        _p2_i64 = getattr(self, "_p2_spec_stg_i64", None)
        if _p2_i64 is None:
            _p2_i64 = self._p2_spec_stg_i64 = _P2CopyStaging(
                self.device, torch.int64)
        _cu_d, _cu_s, _bonus = _p2_i32.upload(
            (cu_num_draft_tokens, cu_num_sampled_tokens,
             bonus_logits_indices))
        _li, _tli = _p2_i64.upload(
            (logits_indices, target_logits_indices))
        cu_num_draft_tokens = _cu_d
        cu_num_sampled_tokens = _cu_s
        logits_indices = _li
        target_logits_indices = _tli
        bonus_logits_indices = _bonus
'''

# ---- group B1: sampled/prev scatter indices (uploaded BEFORE scatter) ----
B1_OLD = '''\
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
'''
B1_NEW = '''\
        # llm-scaler v58 P2: staged pinned H2D (2 per-step pinned allocs
        # + copies -> 1 staged int64 copy). Uploaded here, BEFORE the
        # scatter below that consumes it (stream order preserved).
        _p2_b1 = getattr(self, "_p2_b1_stg", None)
        if _p2_b1 is None:
            _p2_b1 = self._p2_b1_stg = _P2CopyStaging(
                self.device, torch.int64)
        (sampled_tokens_index_tensor,
         prev_common_req_indices_tensor) = _p2_b1.upload(
            (np.asarray(sample_flattened_indices, dtype=np.int64),
             np.asarray(prev_indices, dtype=np.int64)))
'''

# ---- group B3: _dtids_local_padded per-step pinned alloc -> persistent ----
B3_OLD = '''\
        _dtids_local_padded = None
        if isinstance(self._draft_token_ids, list):
            padded = torch.zeros(
                len(self._draft_token_ids),
                self.num_spec_tokens,
                dtype=torch.int32,
                pin_memory=self.pin_memory,
            )
            for i, toks in enumerate(self._draft_token_ids):
                if toks:
                    padded[i, : len(toks)] = torch.tensor(
                        toks, dtype=torch.int32
                    )
            _dtids_local_padded = padded.to(self.device, non_blocking=True)
'''
B3_NEW = '''\
        _dtids_local_padded = None
        if isinstance(self._draft_token_ids, list):
            # llm-scaler v58 P2: persistent pinned [n, num_spec] pair
            # (per-step zeros alloc + H2D -> row fill + 1 staged H2D).
            # Only rows [0:nrows] / cols [0:len] are consumed (v42).
            _p2_d = getattr(self, "_p2_dtids_cpu", None)
            if (_p2_d is None
                    or _p2_d.shape[0] < len(self._draft_token_ids)
                    or _p2_d.shape[1] != self.num_spec_tokens):
                _p2_d = torch.zeros(
                    max(len(self._draft_token_ids), 64),
                    self.num_spec_tokens,
                    dtype=torch.int32,
                    pin_memory=True,
                )
                self._p2_dtids_cpu = _p2_d
                self._p2_dtids_gpu = torch.zeros_like(
                    _p2_d, device=self.device)
            _nrows = len(self._draft_token_ids)
            _p2_d[:_nrows].zero_()
            for i, toks in enumerate(self._draft_token_ids):
                if toks:
                    _p2_d[i, : len(toks)] = torch.tensor(
                        toks, dtype=torch.int32
                    )
            self._p2_dtids_gpu[:_nrows].copy_(
                _p2_d[:_nrows], non_blocking=True)
            _dtids_local_padded = self._p2_dtids_gpu[:_nrows]
'''

# ---- group B2: draft/prev-draft indices ----
B2_OLD = '''\
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
'''
B2_NEW = '''\
        # llm-scaler v58 P2: staged pinned H2D (2 -> 1 staged int64 copy).
        _p2_b2 = getattr(self, "_p2_b2_stg", None)
        if _p2_b2 is None:
            _p2_b2 = self._p2_b2_stg = _P2CopyStaging(
                self.device, torch.int64)
        (draft_tokens_index_tensor,
         prev_draft_token_indices_tensor) = _p2_b2.upload(
            (np.asarray(spec_flattened_indices, dtype=np.int64),
             np.asarray(prev_draft_token_indices, dtype=np.int64)))
'''

# ---- group C1 (proposer): next_token_ids device constructor ----
C1_OLD = '''\
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.input_ids.device
        )
        return next_token_ids
'''
C1_NEW = '''\
        # llm-scaler v58 P2: staged pinned H2D instead of per-step
        # torch.tensor(..., device=) pageable constructor.
        _p2_st = getattr(self, "_p2_ntid_stg", None)
        if _p2_st is None:
            _p2_st = self._p2_ntid_stg = _P2CopyStaging(
                self.input_ids.device, torch.int32)
        (_p2_ntid,) = _p2_st.upload(
            (np.asarray(next_token_ids, dtype=np.int32),))
        return _p2_ntid
'''

# ---- group C2 (proposer): token_indices + qsl/seq_lens (3 -> 2 staged) ----
C2_OLD = '''\
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
'''
C2_NEW = '''\
        # llm-scaler v58 P2: coalesce 3 per-step H2D uploads (token_indices
        # pageable; qsl/seq_lens per-step pinned allocs) into 2 staged
        # copies. CPU refs become staging views: read within the step, and
        # next step's upper_bound read materializes before this staging is
        # refilled (read-then-overwrite in program order).
        _p2_ti = getattr(self, "_p2_ti_stg", None)
        if _p2_ti is None:
            _p2_ti = self._p2_ti_stg = _P2CopyStaging(device, torch.int64)
        (token_indices,) = _p2_ti.upload(
            (token_indices_np.astype(np.int64, copy=False),))
        _p2_qs = getattr(self, "_p2_qsl_stg", None)
        if _p2_qs is None:
            _p2_qs = self._p2_qsl_stg = _P2CopyStaging(device, torch.int32)
        _m1 = new_query_start_loc_cpu.shape[0]
        _m2 = new_seq_lens_cpu.shape[0]
        _qsl_v, _sl_v = _p2_qs.upload(
            (new_query_start_loc_np, new_seq_lens_cpu))
        new_query_start_loc_cpu = _p2_qs.cpu[:_m1]
        new_seq_lens_cpu = _p2_qs.cpu[_m1:_m1 + _m2]

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=_qsl_v,
            seq_lens=_sl_v,
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
'''


def apply(path, repls, check_only=False):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for old, new, name in repls:
        if new in src:
            print(f"  SKIP (already applied): {name}")
            continue
        n = src.count(old)
        if n != 1:
            print(f"  ERROR: anchor {name} count={n} (expected 1) — abort")
            return 1
        if check_only:
            print(f"  check-ok (not applied): {name}")
            continue
        src = src.replace(old, new)
        print(f"  applied: {name}")
    if not check_only:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
    return 0


def main() -> int:
    ok = 0
    check_only = "--check" in sys.argv
    print(f"v58 P2 patcher{' CHECK-ONLY' if check_only else ''}: {F1}")
    n = open(F1, encoding="utf-8").read().count(HELPER_ANCHOR)
    if n != 1:
        print(f"  ERROR: helper anchor count={n} — abort")
        return 1
    ok |= apply(F1, [
        (HELPER_ANCHOR, HELPER_ANCHOR + HELPER, "helper class"),
        (A_OLD, A_NEW, "A spec-decode metadata 5->2"),
        (B1_OLD, B1_NEW, "B1 sampled/prev scatter indices 2->1"),
        (B3_OLD, B3_NEW, "B3 dtids pinned pair"),
        (B2_OLD, B2_NEW, "B2 draft/prev-draft indices 2->1"),
    ], check_only)
    if ok:
        return ok
    print(f"v58 P2 patcher{' CHECK-ONLY' if check_only else ''}: {F2}")
    n = open(F2, encoding="utf-8").read().count(HELPER_ANCHOR)
    if n != 1:
        print(f"  ERROR: helper anchor count={n} — abort")
        return 1
    ok |= apply(F2, [
        (HELPER_ANCHOR, HELPER_ANCHOR + HELPER, "helper class"),
        (C1_OLD, C1_NEW, "C1 next_token_ids"),
        (C2_OLD, C2_NEW, "C2 token_indices + qsl/seq_lens 3->2"),
    ], check_only)
    if ok:
        return ok
    # verification greps
    for path, expect in ((F1, 4), (F2, 2)):
        c = open(path, encoding="utf-8").read().count(MARKER)
        print(f"  verify {path.split('/')[-1]}: marker x{c}")
    print("v58 P2 PATCH OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
