"""Resolve a player's current Ranked tier without depending on a live API call.

The deployed backend can't reach Supercell (the API key is IP-locked to the home crawler),
so we first look the tag up in the match data we already collect — every player row carries
their Ranked tier (see :mod:`bsdraft.engine.tiers`). That covers anyone we've crawled and
needs no key. When a valid key IS available (local/home), we fall back to a live profile
fetch for tags we haven't seen.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

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


def current_ranked_tier(player: dict) -> Optional[int]:
    """The player's *current*-season Ranked tier (1-22) from their profile, or None.

    Read this from the profile's ``rankedRank``, which is the player's tier *right now*.
    Do **not** infer it from the battle log: a ranked battle's ``trophies`` is the tier the
    player *entered that match* at, so the most recent game over-states anyone who then lost
    a promotion game — they show as the tier they'd just reached even though that loss dropped
    them back down. (``highestSeasonRankedRank`` is the season peak, which is exactly that
    over-statement, so it's the wrong field for "what am I now".)"""
    t = player.get("rankedRank")
    return t if isinstance(t, int) and 1 <= t <= 22 else None
