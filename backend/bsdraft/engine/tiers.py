"""Brawl Stars Ranked tier ladder and helpers.

The battle-log ``trophies`` field on a ranked match is really the player's Ranked tier
(1-22): Bronze / Silver / Gold / Diamond / Mythic / Legendary / Masters each split into
I/II/III, then a single Pro step at the top. A match's *bracket* is the median of its six
players' tiers — ranked matchmaking keeps them within a step of each other — and is used to
stratify draft stats by skill bracket (see :mod:`bsdraft.engine.stats`).
"""
from __future__ import annotations

from statistics import median
from typing import List, Optional

# (low, high, name) inclusive tier-index ranges, lowest → highest
_RANGES = [
    (1, 3, "Bronze"), (4, 6, "Silver"), (7, 9, "Gold"), (10, 12, "Diamond"),
    (13, 15, "Mythic"), (16, 18, "Legendary"), (19, 21, "Masters"), (22, 22, "Pro"),
]
BRACKETS: List[str] = [name for _, _, name in _RANGES]   # low → high, for ordered display
_SUB = ["I", "II", "III"]


def bracket_of_tier(tier: int) -> Optional[str]:
    for lo, hi, name in _RANGES:
        if lo <= tier <= hi:
            return name
    return None


def tier_label(tier: int) -> str:
    """1 -> 'Bronze I', 17 -> 'Legendary II', 22 -> 'Pro'."""
    for lo, hi, name in _RANGES:
        if lo <= tier <= hi:
            return name if lo == hi else f"{name} {_SUB[tier - lo]}"
    return str(tier)


def _tiers(match: dict) -> List[int]:
    out = []
    for p in match.get("team_a", []) + match.get("team_b", []):
        t = p.get("trophies")
        if isinstance(t, int) and 1 <= t <= 22:
            out.append(t)
    return out


def match_tier(match: dict) -> Optional[int]:
    """Median Ranked tier (1-22) of the match's players, or None if unavailable."""
    ts = _tiers(match)
    return int(median(ts)) if ts else None


def match_bracket(match: dict) -> Optional[str]:
    t = match_tier(match)
    return bracket_of_tier(t) if t is not None else None
