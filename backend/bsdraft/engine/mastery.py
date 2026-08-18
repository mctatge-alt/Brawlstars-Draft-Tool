"""Player roster & mastery.

Personalizes recommendations to brawlers the player actually owns and is invested in:
power level, personal trophies (comfort), and owned star powers / gadgets / gears /
hypercharge / **buffies**. Buffies are per-item buff enhancements
(`{"gadget": bool, "starPower": bool, "hyperCharge": bool}`). Every brawler the roster returns
carries this object with all three keys present — owned or not — so the slot total is read from
the object's own keys rather than assumed. A brawler with unowned buffie slots is under-built
and gets a lower investment score.
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
    # The *specific* owned items (ids), retained so the UI can suggest only what the player can
    # actually equip on their own pick. Gears carry names+levels since no catalog lists them.
    # These are also the raw ingredient for the planned single-item-owner win-rate inference.
    owned_star_powers: Tuple[int, ...] = ()
    owned_gadgets: Tuple[int, ...] = ()
    owned_gears: Tuple[dict, ...] = ()  # each {"id", "name", "level"}

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
        # This ranks *investment* among brawlers the player can actually field. Power level is left
        # out on purpose: a brawler below the bracket's power floor is unselectable and gets dropped
        # upstream (see ``tiers.min_power_for_bracket`` / the API's ``_roster_for``), so it never
        # reaches this scorer; among those that do, what the player controls is the loadout (you can
        # only equip the star powers / gadgets / gears / buffies you own) and how much they've played
        # it. So the score is loadout-forward, comfort second (a maxed but under-built brawler is
        # still under-built). (The Power 9–10 window that survives the floor below Mythic isn't
        # modelled here — the dataset's win rates already fold real power in.)
        return max(0.0, min(1.0, 0.60 * self.build + 0.40 * self.comfort))

    def gaps(self) -> List[str]:
        # Power level isn't a gap here: an under-floor brawler is unfieldable and filtered out before
        # scoring, so anything that reaches this list clears the floor. Only the owned loadout remains.
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


def _ids(items) -> Tuple[int, ...]:
    return tuple(i["id"] for i in items if isinstance(i, dict) and i.get("id") is not None)


def _gears(items) -> Tuple[dict, ...]:
    out = []
    for g in items or []:
        if isinstance(g, dict) and g.get("id") is not None:
            out.append({"id": g["id"], "name": g.get("name", ""), "level": g.get("level", 0)})
    return tuple(out)


def parse_roster(player: dict) -> Dict[int, Mastery]:
    roster: Dict[int, Mastery] = {}
    for b in player.get("brawlers", []):
        # Slot total is the object's own key count, not a hardcoded 3. Every roster entry
        # observed so far carries all three keys, so this reads the same — but if the object
        # is ever absent or resized, counting what's actually there degrades to "no buffie
        # gap" instead of inventing slots the player has no way to fill.
        buf = b.get("buffies") or {}
        have = sum(1 for v in buf.values() if v) if isinstance(buf, dict) else 0
        total = len(buf) if isinstance(buf, dict) else 0
        star_powers = b.get("starPowers") or []
        gadgets = b.get("gadgets") or []
        gears = b.get("gears") or []
        roster[b["id"]] = Mastery(
            brawler_id=b["id"],
            power=b.get("power", 0),
            rank=b.get("rank", 0),
            trophies=b.get("trophies", 0),
            highest_trophies=b.get("highestTrophies", 0),
            has_starpower=bool(star_powers),
            has_gadget=bool(gadgets),
            has_gears=bool(gears),
            has_hypercharge=bool(b.get("hyperCharges")),
            buffies_have=have,
            buffies_total=total,
            owned_star_powers=_ids(star_powers),
            owned_gadgets=_ids(gadgets),
            owned_gears=_gears(gears),
        )
    return roster


async def fetch_roster(client, tag: str) -> Tuple[Dict[int, Mastery], str]:
    player = await client.get_player(tag)
    return parse_roster(player), player.get("name", "")
