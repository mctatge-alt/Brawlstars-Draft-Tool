"""Player roster & mastery.

Personalizes recommendations to brawlers the player actually owns and is invested in:
power level, personal trophies (comfort), and owned star powers / gadgets / gears /
hypercharge / **buffies**. Buffies are per-item buff enhancements
(`{"gadget": bool, "starPower": bool, "hyperCharge": bool}`); a brawler lacking buffies is
under-built and gets a lower investment score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Mastery:
    brawler_id: int
    power: int
    rank: int
    trophies: int
    highest_trophies: int
    has_starpower: bool
    has_gadget: bool
    has_gears: bool
    has_hypercharge: bool
    buffies_have: int
    buffies_total: int

    @property
    def comfort(self) -> float:  # how much the player has played/succeeded on it
        return min(1.0, self.highest_trophies / 1000.0)

    @property
    def build(self) -> float:  # how fully built the brawler is (incl. buffies)
        buffie = self.buffies_have / self.buffies_total if self.buffies_total else 0.0
        return (
            0.30 * (1.0 if self.has_starpower else 0.0)
            + 0.20 * (1.0 if self.has_gadget else 0.0)
            + 0.20 * (1.0 if self.has_gears else 0.0)
            + 0.30 * buffie
        )

    @property
    def score(self) -> float:
        # Ranked normalizes every brawler to power 11, so the *power level* doesn't affect
        # in-match strength — what the player actually controls is the loadout (you can only
        # equip the star powers / gadgets / gears / buffies you own) and how much they've
        # played the brawler. So the investment score is loadout-forward, comfort second, and
        # power is deliberately left out (a maxed but under-built brawler is still under-built).
        return max(0.0, min(1.0, 0.60 * self.build + 0.40 * self.comfort))

    def gaps(self) -> List[str]:
        # Power level is intentionally omitted: Ranked maxes it to 11, so it's never a real gap
        # there — only the owned loadout is.
        out: List[str] = []
        if not self.has_starpower:
            out.append("no star power")
        if not self.has_gadget:
            out.append("no gadget")
        if self.buffies_total and self.buffies_have < self.buffies_total:
            out.append("missing buffie")
        if not self.has_hypercharge:
            out.append("no hypercharge")
        return out


def parse_roster(player: dict) -> Dict[int, Mastery]:
    roster: Dict[int, Mastery] = {}
    for b in player.get("brawlers", []):
        buf = b.get("buffies") or {}
        have = sum(1 for v in buf.values() if v) if isinstance(buf, dict) else 0
        total = len(buf) if isinstance(buf, dict) else 0
        roster[b["id"]] = Mastery(
            brawler_id=b["id"],
            power=b.get("power", 0),
            rank=b.get("rank", 0),
            trophies=b.get("trophies", 0),
            highest_trophies=b.get("highestTrophies", 0),
            has_starpower=bool(b.get("starPowers")),
            has_gadget=bool(b.get("gadgets")),
            has_gears=bool(b.get("gears")),
            has_hypercharge=bool(b.get("hyperCharges")),
            buffies_have=have,
            buffies_total=total or 3,
        )
    return roster


async def fetch_roster(client, tag: str) -> Tuple[Dict[int, Mastery], str]:
    player = await client.get_player(tag)
    return parse_roster(player), player.get("name", "")
