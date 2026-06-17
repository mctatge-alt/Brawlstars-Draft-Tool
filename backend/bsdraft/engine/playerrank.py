"""Resolve a player's current Ranked tier without depending on a live API call.

The deployed backend can't reach Supercell (the API key is IP-locked to the home crawler),
so we first look the tag up in the match data we already collect — every player row carries
their Ranked tier (see :mod:`bsdraft.engine.tiers`). That covers anyone we've crawled and
needs no key. When a valid key IS available (local/home), we fall back to a live battle-log
fetch for tags we haven't seen.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from bsdraft.collect.client import normalize_tag
from bsdraft.collect.match import RANKED_TYPES
from bsdraft.data.dataset import iter_matches


def build_rank_index(matches: Optional[Iterable[dict]] = None) -> Dict[str, Tuple[int, int]]:
    """Map each crawled player tag -> (latest_ts, tier) from the collected matches."""
    idx: Dict[str, Tuple[int, int]] = {}
    for r in (matches if matches is not None else iter_matches()):
        ts = int(r.get("ts") or 0)
        for p in r.get("team_a", []) + r.get("team_b", []):
            tag, tier = p.get("tag"), p.get("trophies")
            if not tag or not isinstance(tier, int) or not (1 <= tier <= 22):
                continue
            cur = idx.get(tag)
            if cur is None or ts > cur[0]:
                idx[tag] = (ts, tier)
    return idx


def latest_ranked_tier(battles: Iterable[dict], tag: str) -> Optional[int]:
    """The player's Ranked tier from their most recent ranked battle (the battle log is
    newest-first), or None. Used for the live fallback."""
    want = normalize_tag(tag)
    for entry in battles:
        battle = entry.get("battle") or {}
        if battle.get("type") not in RANKED_TYPES:
            continue
        for team in battle.get("teams") or []:
            for p in team:
                if normalize_tag(p.get("tag", "")) == want:
                    t = (p.get("brawler") or {}).get("trophies")
                    if isinstance(t, int) and 1 <= t <= 22:
                        return t
    return None
