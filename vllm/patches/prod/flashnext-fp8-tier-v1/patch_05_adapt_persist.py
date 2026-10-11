#!/usr/bin/env python3
"""FN8_05 (flashnext-fp8-tier-v1 P5): adaptive tier + persistence.

Windowed swaps from LIVE routing stats + profile persistence across
restarts, all inside the FN8_02A/03 store (fused_moe_modular_method.py —
single target file, 9 idempotent edits, --restore reverses):

  1. Knobs: V1X_TIER_ADAPT(0) EVERY(200) SWAPS(4) DECAY(0.995)
     MIN_SCORE(3.0) PERSIST("") PERSIST_EVERY(4).
  2. Seed merge law: _fn8_seed_hot_lists_merged = base profile counts +
     every accumulated <PERSIST>.r<rank> file, summed.  Restart
     reproduces (and extends) the adapted hot set.
  3. Store ctor: merged loader + live_stats state (EWMA scores, raw
     touch counts, real-dispatch counter).
  4. ensure_resident touch loop: every routed-to OWNED id (hit or miss)
     bumps score = score*DECAY + 1 and a raw counter — demand signal,
     free (the ids_host loop already runs).
  5. Window boundary at end of ensure_resident (every EVERY real
     dispatches): _adapt_window().
  6. _adapt_window: hottest NON-RESIDENT ids with score >= MIN_SCORE,
     up to SWAPS, pulled in via _pf_fill (the PROVEN displacement-rights
     fill: LRU victim out, side stream, event-ordered, filling
     bookkeeping).  Selection is score-based, eviction stays LRU (the
     certified v2 law).  No helper thread — the P10 queue-kick lesson
     applied by not having one; all swap work rides the forward thread
     between windows + the existing side stream.
  7. _persist_flush (every PERSIST_EVERY windows): merge ALL stores'
     raw touches into the rank matrix, atomic write <PERSIST>.r<rank>.
     Per-rank paths by the TP2 dump-collision law.  File accumulates
     across runs; each rank writes only its owned columns (EP split),
     boot-time merge sums all rank files.
  8. Telemetry tmp per-rank (fixes the Round-12 TP2 collision: both
     ranks wrote one shared tmp path — mid-traffic frames interleaved).
  9. Telemetry rec += adapt_windows/adapt_swaps/persist_flushes.

Order: apply AFTER patch_02a + patch_03 (edits live inside their text).
Laws: py_compile on temp before every in-place write; md5 printed;
backups at <file>.fn8_05.bak.
"""
import hashlib
import importlib.util
import os
import py_compile
import shutil
import sys
import tempfile

TAG = "FN8_05"
REL = "model_executor/layers/fused_moe/fused_moe_modular_method.py"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def restore():
    path = os.path.join(PKG, REL)
    bak = path + ".fn8_05.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup for {REL}")
        sys.exit(4)
    sys.exit(0)


def edit_in_place(anchor, replacement, already, expect=1):
    """Replace `anchor` with `replacement` in PKG/REL; anchor must occur
    exactly `expect` times.  Idempotent via the unique `already` marker."""
    path = os.path.join(PKG, REL)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if already in src:
        print(f"{TAG} edit already present, skip")
        return
    n = src.count(anchor)
    if n != expect:
        print(f"{TAG} FAIL: anchor occurs {n}x (want {expect})")
        sys.exit(2)
    bak = path + ".fn8_05.bak"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
        print(f"{TAG} backup {bak} md5={md5(bak)}")
    out = src.replace(anchor, replacement)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(out)
        tmp = tf.name
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"{TAG} FAIL: compile error -> {e}")
        sys.exit(3)
    finally:
        os.unlink(tmp)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"{TAG} edited, md5={md5(path)}")


if "--restore" in sys.argv:
    restore()

# ---- 1. knobs ---------------------------------------------------------------
edit_in_place(
    '_FN8_PF_BATCH = _fn8_os.environ.get("V1X_TIER_BATCH", "1") == "1"\n',
    '_FN8_PF_BATCH = _fn8_os.environ.get("V1X_TIER_BATCH", "1") == "1"\n'
    '# FN8_05 (P5): adaptive tier + persistence knobs.\n'
    '_FN8_ADAPT = _fn8_os.environ.get("V1X_TIER_ADAPT", "0") == "1"\n'
    '_FN8_ADAPT_EVERY = int(_fn8_os.environ.get("V1X_TIER_ADAPT_EVERY", "200"))\n'
    '_FN8_ADAPT_SWAPS = int(_fn8_os.environ.get("V1X_TIER_ADAPT_SWAPS", "4"))\n'
    '_FN8_ADAPT_DECAY = float(_fn8_os.environ.get("V1X_TIER_ADAPT_DECAY", "0.995"))\n'
    '_FN8_ADAPT_MIN_SCORE = float(\n'
    '    _fn8_os.environ.get("V1X_TIER_ADAPT_MIN_SCORE", "3.0"))\n'
    '_FN8_PERSIST = _fn8_os.environ.get("V1X_TIER_PERSIST", "")\n'
    '_FN8_PERSIST_EVERY = int(_fn8_os.environ.get("V1X_TIER_PERSIST_EVERY", "4"))\n'
    '_FN8_RANK = 0\n',
    already="_FN8_ADAPT_MIN_SCORE = float(",
)

# ---- 2. merged seed loader + persist helpers -------------------------------
edit_in_place(
    '        return out\n'
    '    except Exception:\n'
    '        return {}\n',
    '        return out\n'
    '    except Exception:\n'
    '        return {}\n'
    '\n'
    '\n'
    '_FN8_PERSIST_STATE = None  # rank counts matrix, lazily sized\n'
    '\n'
    '\n'
    'def _fn8_persist_rank(owned) -> int:\n'
    '    """Which TP/EP rank this process is (for per-rank persist/dump\n'
    '    paths).  RANK/LOCAL_RANK env when the executor sets them; else the\n'
    '    ownership fallback (EP halves are contiguous, expert_map law:\n'
    '    rank 0 owns the low half, so 0 in owned identifies it)."""\n'
    '    for var in ("RANK", "LOCAL_RANK", "VLLM_RANK_ID"):\n'
    '        v = _fn8_os.environ.get(var, "")\n'
    '        if v.isdigit():\n'
    '            return int(v)\n'
    '    return 0 if 0 in owned else 1\n'
    '\n'
    '\n'
    'def _fn8_persist_counts_for_rank(rank: int) -> list[list[int]]:\n'
    '    """Previous accumulated additions from THIS rank\'s persist file."""\n'
    '    if not _FN8_PERSIST:\n'
    '        return []\n'
    '    try:\n'
    '        with open("%s.r%d" % (_FN8_PERSIST, rank)) as f:\n'
    '            return [list(r) for r in _fn8_json.load(f)["counts"]]\n'
    '    except Exception:\n'
    '        return []\n'
    '\n'
    '\n'
    'def _fn8_seed_hot_lists_merged(seed_path: str) -> dict[int, list[int]]:\n'
    '    """FN8_05 seed merge law: base profile counts + every accumulated\n'
    '    rank persistence file (<PERSIST>.r*), counts summed — a restart\n'
    '    reproduces (and extends) the adapted hot set."""\n'
    '    counts = None\n'
    '    try:\n'
    '        with open(seed_path) as f:\n'
    '            counts = [list(r) for r in _fn8_json.load(f)["counts"]]\n'
    '    except Exception:\n'
    '        counts = None\n'
    '    if _FN8_PERSIST:\n'
    '        import glob as _fn8_glob\n'
    '        for pf in sorted(_fn8_glob.glob(_FN8_PERSIST + ".r*")):\n'
    '            try:\n'
    '                with open(pf) as f:\n'
    '                    per = [list(r) for r in _fn8_json.load(f)["counts"]]\n'
    '            except Exception:\n'
    '                continue\n'
    '            if counts is None:\n'
    '                counts = per\n'
    '                continue\n'
    '            if len(per) == len(counts):\n'
    '                for i in range(len(per)):\n'
    '                    if len(per[i]) == len(counts[i]):\n'
    '                        counts[i] = [\n'
    '                            a + b for a, b in zip(counts[i], per[i])]\n'
    '    if counts is None:\n'
    '        return {}\n'
    '    out = {}\n'
    '    for li, row in enumerate(counts):\n'
    '        order = sorted(range(len(row)), key=lambda e: -row[e])\n'
    '        out[li] = [e for e in order if row[e] > 0]\n'
    '    return out\n',
    already="def _fn8_seed_hot_lists_merged",
)

# ---- 3. ctor call -> merged loader ------------------------------------------
edit_in_place(
    '        seed = _fn8_seed_hot_lists(_FN8_TIER_SEED) if _FN8_TIER_SEED else {}\n',
    '        seed = (_fn8_seed_hot_lists_merged(_FN8_TIER_SEED)\n'
    '                if _FN8_TIER_SEED else {})\n',
    already="_fn8_seed_hot_lists_merged(_FN8_TIER_SEED)",
)

# ---- 4. ctor state ----------------------------------------------------------
edit_in_place(
    '        # FN8_03: replay staging + async-fill bookkeeping (P3).\n',
    '        # FN8_05 (P5): adaptive-tier live stats + persistence state.\n'
    '        self.real_dispatches = 0\n'
    '        self.live_scores = {}    # EWMA of routed-to demand, owned ids\n'
    '        self.touch_counts = {}   # raw touches since last persist flush\n'
    '        global _FN8_RANK\n'
    '        _FN8_RANK = _fn8_persist_rank(self.owned)\n'
    '        # FN8_03: replay staging + async-fill bookkeeping (P3).\n',
    already="# FN8_05 (P5): adaptive-tier live stats",
)

# ---- 5. telemetry rec keys --------------------------------------------------
edit_in_place(
    '            "pf_drops": 0,\n',
    '            "pf_drops": 0,\n'
    '            "adapt_windows": 0, "adapt_swaps": 0,\n'
    '            "persist_flushes": 0,\n',
    already='"adapt_windows": 0, "adapt_swaps": 0,',
)

# ---- 6. touch loop (EWMA + raw count) ---------------------------------------
edit_in_place(
    '        for g in needed:\n'
    '            if g in self.slot_of:\n'
    '                self.last_use[g] = self.use_tick\n'
    '                self.rec["hits"] += 1\n',
    '        for g in needed:\n'
    '            # FN8_05: every routed-to owned id is demand signal (hit or\n'
    '            # miss alike): EWMA for the adaptive hot set + a raw count\n'
    '            # for persistence.\n'
    '            if _FN8_ADAPT:\n'
    '                self.live_scores[g] = (\n'
    '                    self.live_scores.get(g, 0.0) * _FN8_ADAPT_DECAY + 1.0)\n'
    '                self.touch_counts[g] = self.touch_counts.get(g, 0) + 1\n'
    '            if g in self.slot_of:\n'
    '                self.last_use[g] = self.use_tick\n'
    '                self.rec["hits"] += 1\n',
    already="FN8_05: every routed-to owned id is demand signal",
)

# ---- 7. window boundary hook -------------------------------------------------
edit_in_place(
    '        if changed:\n'
    '            self._sync_slot_map()\n'
    '        self.rec["ensure_ms"] += _fn8_time.time() - t0\n',
    '        if changed:\n'
    '            self._sync_slot_map()\n'
    '        self.rec["ensure_ms"] += _fn8_time.time() - t0\n'
    '        # FN8_05: window boundary — adaptive swaps + persistence, on\n'
    '        # the forward thread between windows (P10 lesson applied by\n'
    '        # construction: no helper thread to park; fills ride the\n'
    '        # proven side-stream displacement machinery).\n'
    '        if _FN8_ADAPT:\n'
    '            self.real_dispatches += 1\n'
    '            if self.real_dispatches % _FN8_ADAPT_EVERY == 0:\n'
    '                self._adapt_window()\n',
    already="FN8_05: window boundary",
)

# ---- 8. _adapt_window + _persist_flush methods ------------------------------
edit_in_place(
    '    # -- dispatch ----------------------------------------------------------\n',
    '    # -- FN8_05: adaptive swaps + persistence ------------------------------\n'
    '\n'
    '    def _adapt_window(self) -> None:\n'
    '        """Pull the hottest NON-RESIDENT owned ids into the pool via\n'
    '        _pf_fill (the displacement-rights fill: LRU victim out, side\n'
    '        stream, event-ordered).  Selection is score-based (EWMA of\n'
    '        routed-to demand); eviction stays LRU (certified v2 law)."""\n'
    '        global _FN8_ADAPT\n'
    '        try:\n'
    '            self.rec["adapt_windows"] += 1\n'
    '            cands = [\n'
    '                (score, g) for g, score in self.live_scores.items()\n'
    '                if g in self.owned and g not in self.slot_of\n'
    '                and g not in self.filling\n'
    '                and score >= _FN8_ADAPT_MIN_SCORE\n'
    '            ]\n'
    '            cands.sort(key=lambda t: (-t[0], t[1]))\n'
    '            for _score, g in cands[:_FN8_ADAPT_SWAPS]:\n'
    '                self._pf_fill(g)\n'
    '                self.rec["adapt_swaps"] += 1\n'
    '            if _FN8_PERSIST and (\n'
    '                    self.rec["adapt_windows"] % _FN8_PERSIST_EVERY == 0):\n'
    '                self._persist_flush()\n'
    '        except Exception as _e:\n'
    '            _FN8_ADAPT = False\n'
    '            logger.warning(\n'
    '                "FN8_05 adapt DISABLED (layer %s): %s",\n'
    '                self.layer_idx, _e)\n'
    '\n'
    '    def _persist_flush(self) -> None:\n'
    '        """Merge every store\'s raw touches into the rank matrix and\n'
    '        atomically write <PERSIST>.r<rank> (tmp+replace; per-rank path\n'
    '        by the TP2 dump-collision law).  Accumulates across runs;\n'
    '        each rank writes only its owned columns (EP split), and boot\n'
    '        merges base seed + all rank files (seed merge law)."""\n'
    '        global _FN8_PERSIST_STATE\n'
    '        try:\n'
    '            stores = list(_FN8_PF_REG)\n'
    '            if not stores:\n'
    '                return\n'
    '            n_layers = max(st.layer_idx for st in stores) + 1\n'
    '            n_experts = max(st.global_num for st in stores)\n'
    '            if _FN8_PERSIST_STATE is None:\n'
    '                base = _fn8_persist_counts_for_rank(_FN8_RANK)\n'
    '                state = [[0] * n_experts for _ in range(n_layers)]\n'
    '                for li, row in enumerate(base[:n_layers]):\n'
    '                    if len(row) == n_experts:\n'
    '                        state[li] = list(row)\n'
    '                _FN8_PERSIST_STATE = state\n'
    '            counts = _FN8_PERSIST_STATE\n'
    '            for st in stores:\n'
    '                li = st.layer_idx\n'
    '                if not (0 <= li < len(counts)):\n'
    '                    continue\n'
    '                for g, c in st.touch_counts.items():\n'
    '                    counts[li][g] += c\n'
    '                st.touch_counts = {}\n'
    '                st.rec["persist_flushes"] += 1\n'
    '            path = "%s.r%d" % (_FN8_PERSIST, _FN8_RANK)\n'
    '            tmp = path + ".tmp"\n'
    '            with open(tmp, "w") as f:\n'
    '                _fn8_json.dump({"counts": counts}, f)\n'
    '            _fn8_os.replace(tmp, path)\n'
    '        except Exception as _e:\n'
    '            logger.warning("FN8_05 persist flush failed: %s", _e)\n'
    '\n'
    '    # -- dispatch ----------------------------------------------------------\n',
    already="def _adapt_window(self) -> None:",
)

# ---- 9. per-rank telemetry tmp ----------------------------------------------
edit_in_place(
    '        tmp = _FN8_TIER_DUMP + ".tmp"\n',
    '        # FN8_05: per-rank tmp (Round-12 TP2 collision law: both ranks\n'
    '        # shared ONE tmp path, interleaving mid-traffic frames).\n'
    '        tmp = "%s.tmp%d" % (_FN8_TIER_DUMP, _FN8_RANK)\n',
    already='tmp = "%s.tmp%d" % (_FN8_TIER_DUMP, _FN8_RANK)',
)

print(f"{TAG} ALL_EDITS_OK")
