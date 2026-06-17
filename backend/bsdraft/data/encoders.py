"""Stable integer encoders for model inputs (brawlers, maps, modes).

Built deterministically from the reference data so indices match between training and
serving. Index 0 is reserved as an "unknown" bucket for maps and modes (brawler ids are
fully reconciled with the reference, so brawlers use a dense 0..N-1 index with no pad).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

from bsdraft.constants import MODE_CAMEL_TO_DISPLAY, RANKED_MODES
from bsdraft.data import reference as R


# --- Brawlers: id -> 0..N-1 (from reference, sorted by id) ---
@lru_cache(maxsize=1)
def brawler_encoder() -> Dict[int, int]:
    return dict(R.brawler_index())


def num_brawlers() -> int:
    return len(R.load_brawlers())


def encode_brawler(brawler_id: int) -> int:
    return brawler_encoder().get(brawler_id, 0)


# --- Maps: id -> 1..M (0 reserved for unknown/unseen) ---
@lru_cache(maxsize=1)
def map_encoder() -> Dict[int, int]:
    return {m.id: i + 1 for i, m in enumerate(R.load_ranked_maps())}


def num_maps() -> int:
    return len(R.load_ranked_maps()) + 1  # +1 for the unknown bucket at 0


def encode_map(map_id) -> int:
    return map_encoder().get(map_id, 0)


# --- Modes: display name -> 1..K (0 reserved for unknown) ---
@lru_cache(maxsize=1)
def mode_encoder() -> Dict[str, int]:
    return {name: i + 1 for i, name in enumerate(RANKED_MODES)}


def num_modes() -> int:
    return len(RANKED_MODES) + 1


def encode_mode(raw_mode: str) -> int:
    """Accepts camelCase (battle log) or display names; returns 0 if unknown."""
    display = MODE_CAMEL_TO_DISPLAY.get(raw_mode, raw_mode)
    return mode_encoder().get(display, 0)
