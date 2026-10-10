#!/usr/bin/env python3
"""ce_unpin_fix.py — llm-scaler v141 double-unpin epoch ledger for
lmcache 0.5.5 cache_engine.py (P9b residual fix).

Root cause (sealed on the sdoff leg, 188 warnings): with async
scheduling, a request re-enters lookup across steps under the same
lookup_id. Step N's lookup_unpin pops lookup_pins[id] and releases the
objects, but the id's LOADING/DONE event still exists; the duplicate
unpin then takes the elif event-cleanup branch and releases the same
objects again -> memory_management.unpin() hits pin_count -1.

Fix: an id released via the pins path is marked in _lookup_released;
the elif branch skips ids in that set; a fresh lookup() that repins the
id discards the mark. Exact-match edits, hard-fails if text drifted.
"""
import sys

PATH = (
    "/opt/venv/lib/python3.12/site-packages/lmcache/v1/cache_engine.py"
)

src = open(PATH).read()
orig = src

def must(old: str, new: str, tag: str) -> None:
    global src
    if src.count(old) != 1:
        raise SystemExit(f"ABORT {tag}: expected 1 match, got {src.count(old)}")
    src = src.replace(old, new)

# 1. init the released-set next to lookup_pins
must(
    """        # lookup_id -> {location -> [pinned keys]}
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
""",
    """        # lookup_id -> {location -> [pinned keys]}
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        # llm-scaler v141: ids already released via the lookup_unpin pins
        # path; guards the async event-cleanup branch from re-releasing
        # the same objects (double unpin -> pin_count -1 warnings).
        self._lookup_released: set = set()
""",
    "init",
)

# 2. mamba flavor repin clears the mark
must(
    """                        self.lookup_pins[lookup_id] = {_loc: [deep_key]}
""",
    """                        self.lookup_pins[lookup_id] = {_loc: [deep_key]}
                        self._lookup_released.discard(lookup_id)
""",
    "mamba-repin",
)

# 3. non-layerwise repin clears the mark
must(
    """                    self.lookup_pins[lookup_id] = block_mapping
""",
    """                    self.lookup_pins[lookup_id] = block_mapping
                    self._lookup_released.discard(lookup_id)
""",
    "layerwise-repin",
)

# 4. lookup_unpin: mark on release; skip cleanup for already-released ids
must(
    """        if lookup_id in self.lookup_pins:
            assert self.storage_manager is not None
            for location, keys in self.lookup_pins.pop(lookup_id).items():
                self.storage_manager.batched_unpin(keys, [location])

        elif (
""",
    """        if lookup_id in self.lookup_pins:
            assert self.storage_manager is not None
            for location, keys in self.lookup_pins.pop(lookup_id).items():
                self.storage_manager.batched_unpin(keys, [location])
            self._lookup_released.add(lookup_id)

        elif lookup_id in self._lookup_released:
            # llm-scaler v141: already released via the pins path and not
            # repinned since; the event cleanup below would double-release.
            pass

        elif (
""",
    "unpin-guard",
)

if src == orig:
    raise SystemExit("NO_CHANGES (already patched?)")

open(PATH, "w").write(src)
print("CE_UNPIN_FIX_APPLIED")
