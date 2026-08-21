"""Player roster & mastery.

Personalizes recommendations to brawlers the player actually owns and is invested in. ``score``
reads exactly two things: the owned **star powers / gadgets / gears** (``build``) and personal
trophies (``comfort``). Power level and hypercharge are carried on :class:`Mastery` but stay out
of the score. Power is excluded on purpose — an under-floor brawler is unfieldable and already
filtered out upstream, so the comparison here is between brawlers that all clear the floor (see
the comments on ``score`` and ``gaps``). Hypercharge is simply not part of the ``build`` term;
it surfaces only as a UI hint via ``gaps``.

Buffies are left out too. The `/players/{tag}` roster does carry a per-brawler
`buffies: {"gadget": bool, "starPower": bool, "hyperCharge": bool}` object, but its `True` flags
only tell us which buffies the player *owns* — never how many *exist* for that brawler. A brawler
with no buffie released (e.g. R-T) returns all-`False`, which is indistinguishable from one whose
buffies you simply haven't unlocked yet. With no reliable slot total, a "missing buffie" signal
misfires on every brawler that has none (verified against maxed top-100 rosters), so buffies are
left out of both the build score and the loadout gaps.
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
    def build(self) -> float:  # how fully built the brawler is, over the loadout the API can measure
        # Star power weighted 1.5× a gadget or gear — the original 3:2:2 split, with the buffie term
        # dropped (see the module docstring) and the rest renormalized to reach 1.0 when fully built.
        return (
            3 * (1.0 if self.has_starpower else 0.0)
            + 2 * (1.0 if self.has_gadget else 0.0)
            + 2 * (1.0 if self.has_gears else 0.0)
        ) / 7.0

    @property
    def score(self) -> float:
        # This ranks *investment* among brawlers the player can actually field. Power level is left
        # out on purpose: a brawler below the bracket's power floor is unselectable and gets dropped
        # upstream (see ``tiers.min_power_for_bracket`` / the API's ``_roster_for``), so it never
        # reaches this scorer; among those that do, what the player controls is the loadout (you can
        # only equip the star powers / gadgets / gears you own) and how much they've played
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
            owned_star_powers=_ids(star_powers),
            owned_gadgets=_ids(gadgets),
            owned_gears=_gears(gears),
        )
    return roster


async def fetch_roster(client, tag: str) -> Tuple[Dict[int, Mastery], str]:
    player = await client.get_player(tag)
    return parse_roster(player), player.get("name", "")
