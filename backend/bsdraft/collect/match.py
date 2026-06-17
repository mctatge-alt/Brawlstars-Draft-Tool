"""Parse a raw Brawl Stars battle-log entry into a normalized ranked Match record."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from bsdraft.collect.client import normalize_tag

# Ranked draft queue types: solo = queued with randoms, team = premade. Both use the
# draft. The distinction is itself a useful "solo vs premade" signal for modeling.
RANKED_TYPES = {"soloRanked", "teamRanked"}


@dataclass
class Match:
    match_key: str
    battle_time: str
    ts: int                   # epoch seconds (for recency weighting)
    queue_type: str           # soloRanked | teamRanked
    mode: str                 # raw camelCase, e.g. "gemGrab"
    map_id: Optional[int]
    map_name: str
    team_a: List[dict]
    team_b: List[dict]
    a_won: Optional[bool]     # did team_a win? None for draw/unlabeled
    player_tags: List[str]


def _parse_ts(battle_time: str) -> int:
    try:
        dt = datetime.strptime(battle_time, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return 0


def _player_record(p: dict) -> dict:
    br = p.get("brawler") or {}
    return {
        "tag": normalize_tag(p.get("tag", "")),
        "brawler_id": br.get("id"),
        "brawler_name": br.get("name"),
        "power": br.get("power"),
        "trophies": br.get("trophies"),
    }


def _match_key(battle_time: str, tags: List[str]) -> str:
    raw = battle_time + "|" + "|".join(sorted(tags))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def parse_match(entry: dict, queried_tag: str) -> Optional[Match]:
    """Normalize a ranked 3v3 battle, else return None.

    ``result`` in the log is relative to the queried player, so we find that player's
    team to record an absolute winner (consistent across every player's copy of the
    match — important for dedup).
    """
    battle = entry.get("battle") or {}
    if battle.get("type") not in RANKED_TYPES:
        return None
    teams = battle.get("teams")
    if not teams or len(teams) != 2 or any(len(t) != 3 for t in teams):
        return None

    team_a = [_player_record(p) for p in teams[0]]
    team_b = [_player_record(p) for p in teams[1]]
    everyone = team_a + team_b
    all_tags = [r["tag"] for r in everyone]
    if any(not t for t in all_tags) or any(r["brawler_id"] is None for r in everyone):
        return None

    result = battle.get("result")
    qtag = normalize_tag(queried_tag)
    if qtag in {r["tag"] for r in team_a}:
        a_won = {"victory": True, "defeat": False}.get(result)
    elif qtag in {r["tag"] for r in team_b}:
        a_won = {"victory": False, "defeat": True}.get(result)
    else:
        a_won = None

    event = entry.get("event") or {}
    battle_time = entry.get("battleTime", "")
    return Match(
        match_key=_match_key(battle_time, all_tags),
        battle_time=battle_time,
        ts=_parse_ts(battle_time),
        queue_type=battle.get("type", ""),
        mode=battle.get("mode") or event.get("mode") or "",
        map_id=event.get("id"),
        map_name=event.get("map", ""),
        team_a=team_a,
        team_b=team_b,
        a_won=a_won,
        player_tags=all_tags,
    )
