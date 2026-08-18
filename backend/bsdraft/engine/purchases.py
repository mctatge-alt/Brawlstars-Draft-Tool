"""Personalized "what to buy/upgrade next" advisor.

Given a player's ownership snapshot (power level + owned gadgets / star powers / gears /
hypercharge per brawler, from ``/players/{tag}``) this enumerates the purchases
they *haven't* made yet and ranks them by expected competitive value. It is the inverse of
:mod:`bsdraft.engine.loadout`: the loadout advisor tells you which owned item to *equip*; this
tells you which unowned item is most worth *acquiring*.

Design (locked with the product owner):

* **Rank by win-rate value**, not value-per-coin — cost is shown as context (the API exposes
  ownership but never currency balances, so affordability is unknowable; prices are fixed
  constants from ``economy.json``).
* **Pure meta strength** — ordering ignores how much the player mains a brawler; ownership only
  decides what's *buyable*. ``value = meta_strength(brawler) × impact_prior(kind)``, nudged by the
  measured item win-rate delta where the itemstats table has a significant cell.
* **Account-wide across the ranked map pool** — ``meta_strength`` is the games-weighted average of
  the brawler's win rate over the *current* ranked maps (excludes rotated-out maps), falling back
  to the global rate on thin data.
* **New-brawler unlocks included** — a brawler absent from the roster is a Credit-unlock candidate
  scored by its meta strength.

Power levels have no intrinsic modeled value (Ranked normalizes everyone to Power 11), so a power
upgrade never stands alone: its cost is *folded into* the gated item it unlocks (e.g. the
hypercharge rec on a Power-9 brawler carries the Power 9→11 cost and a "requires Power 11" note).

Pure stdlib so it stays importable on the serve path (no torch/sklearn/pandas), like
:mod:`bsdraft.engine.loadout`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional

from bsdraft.data import reference as R
from bsdraft.engine import itemstats as IS
from bsdraft.engine.stats import DraftStats

# A measured delta only breaks ties between the two items of a type (its estimand is
# item-vs-other-owners, not own-vs-not-own), so it nudges the prior-driven score, never dominates.
_DELTA_WEIGHT = 0.5
# Below this effective sample across the ranked maps, fall back to the global rate.
_MIN_MAP_GAMES = 8.0
# A new-brawler rec is labeled "measured" only once its win rate rests on real games.
_MIN_MEASURED_GAMES = 40.0

_norm_re = re.compile(r"[^a-z0-9]")


def _norm(name: str) -> str:
    return _norm_re.sub("", (name or "").lower())


@dataclass(frozen=True)
class OwnedState:
    """One brawler's ownership, distilled from a roster entry. Ids match the catalog
    (``R.load_brawlers``); gears are keyed by normalized name (no catalog id exists)."""
    power: int = 0
    star_powers: FrozenSet[int] = frozenset()
    gadgets: FrozenSet[int] = frozenset()
    gears: FrozenSet[str] = frozenset()          # normalized owned gear names
    has_hypercharge: bool = False


@dataclass
class _Rec:
    brawler_id: int
    brawler_name: str
    kind: str                    # power_upgrade|gadget|star_power|gear|hypercharge|new_brawler
    value_score: float
    meta_winrate: float
    confidence: str              # "measured" | "heuristic" | "eligibility_only"
    cost: Dict[str, int]
    rationale: str
    item_id: Optional[int] = None
    item_name: Optional[str] = None
    target_power: Optional[int] = None
    item_delta: Optional[float] = None
    gate: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "brawler_id": self.brawler_id, "brawler_name": self.brawler_name, "kind": self.kind,
            "value_score": round(self.value_score, 4), "meta_winrate": round(self.meta_winrate, 4),
            "confidence": self.confidence, "cost": self.cost, "rationale": self.rationale,
            "item_id": self.item_id, "item_name": self.item_name, "target_power": self.target_power,
            "item_delta": None if self.item_delta is None else round(self.item_delta, 4),
            "gate": self.gate,
        }


# --- economy helpers (all fail-safe: a missing section => no cost/gate context, never an error) --

def _priors(economy: dict) -> dict:
    return economy.get("impact_priors", {}) if economy else {}


def _item_cost(economy: dict, kind: str) -> Dict[str, int]:
    return dict((economy.get("item_costs", {}) or {}).get(kind, {}) or {}) if economy else {}


def _power_cost(economy: dict, cur: int, target: int) -> Dict[str, int]:
    """Coin + power-point cost to go from ``cur`` to ``target`` power, from the cumulative table."""
    table = (economy.get("power_cost_cumulative", {}) or {}) if economy else {}
    cur = max(1, min(11, int(cur or 1)))
    target = max(1, min(11, int(target)))
    out: Dict[str, int] = {}
    if target <= cur:
        return out
    for cur_key in ("coins", "power_points"):
        arr = table.get(cur_key)
        if isinstance(arr, list) and len(arr) > target:
            delta = int(arr[target]) - int(arr[cur])
            if delta > 0:
                out[cur_key] = delta
    return out


def _merge_cost(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + int(v)
    return out


def _gate_for(economy: dict, kind: str) -> Optional[int]:
    g = (economy.get("power_gates", {}) or {}).get(kind) if economy else None
    return int(g) if g else None


def _hypercharge_eligible(economy: dict, brawler_name: str) -> bool:
    """Does this brawler have a hypercharge to buy? The API can't tell (see economy.json), so this
    reads the curated availability policy. Returns False when the section is absent (fail-safe:
    don't recommend a hypercharge we can't vouch for)."""
    cfg = economy.get("hypercharge_availability") if economy else None
    if not isinstance(cfg, dict):
        return False
    names = {_norm(n) for n in (cfg.get("list") or [])}
    mode = cfg.get("mode", "all_except")
    if mode == "only":
        return _norm(brawler_name) in names
    return _norm(brawler_name) not in names   # "all_except"


# --- scoring ---------------------------------------------------------------------------------

def _meta_strength(stats: DraftStats, brawler_id: int, map_ids: List[int]) -> float:
    """Games-weighted average win rate across the current ranked maps, falling back to the global
    rate when the per-map sample is thin. A smoothed rate (~0.5 baseline) — higher ⇒ stronger."""
    num = den = 0.0
    for mid in map_ids:
        r = stats.brawler_rate(brawler_id, mid)
        num += r.winrate * r.games
        den += r.games
    if den >= _MIN_MAP_GAMES:
        return num / den
    return stats.brawler_rate(brawler_id, None).winrate


def _measured_delta(itemstats: Optional[dict], brawler_id: int, item_id: Optional[int],
                    gear_name: Optional[str] = None) -> Optional[float]:
    """The significant measured win-rate delta for an item, or None. Positive ⇒ measured better
    than the brawler's other single-owned items of that type."""
    if not itemstats:
        return None
    if gear_name is not None:
        cell = IS.gear_cell(itemstats, brawler_id, gear_name)
    else:
        cell = IS.accessory_cell(itemstats, brawler_id, item_id)
    if cell and cell.get("significant"):
        return float(cell.get("delta", 0.0))
    return None


def _best_missing_accessory(missing, itemstats, brawler_id):
    """Pick which of the missing gadgets/star powers to recommend first: the one with the best
    significant measured delta, else the catalog-first (stable)."""
    best = missing[0]
    best_delta = _measured_delta(itemstats, brawler_id, best.id)
    for acc in missing[1:]:
        d = _measured_delta(itemstats, brawler_id, acc.id)
        if d is not None and (best_delta is None or d > best_delta):
            best, best_delta = acc, d
    return best, best_delta


def recommend_purchases(owned: Dict[int, OwnedState], stats: DraftStats,
                        itemstats: Optional[dict] = None, top: int = 20,
                        ranked_maps=None, economy: Optional[dict] = None) -> List[dict]:
    """Rank the highest-value next purchases across a player's account. ``owned`` maps brawler id →
    :class:`OwnedState` (a brawler absent from the map is unowned ⇒ a new-brawler candidate)."""
    economy = economy if economy is not None else R.load_economy()
    ranked_maps = ranked_maps if ranked_maps is not None else R.load_ranked_maps()
    map_ids = [m.id for m in ranked_maps]
    priors = _priors(economy)
    gears_guide = _gear_names(economy)
    recs: List[_Rec] = []

    for b in R.load_brawlers():
        strength = _meta_strength(stats, b.id, map_ids)
        st = owned.get(b.id)

        if st is None:
            recs.append(_new_brawler_rec(b, strength, stats, priors, economy))
            continue

        # --- gadgets / star powers: recommend the best still-missing one of each type ---
        for accessories, id_field, first_key, second_key, label in (
            (b.gadgets, "gadgets", "gadget_first", "gadget_second", "gadget"),
            (b.star_powers, "star_powers", "star_power_first", "star_power_second", "star power"),
        ):
            owned_ids = getattr(st, id_field)
            missing = [a for a in accessories if a.id not in owned_ids]
            if not missing:
                continue
            acc, delta = _best_missing_accessory(missing, itemstats, b.id)
            is_first = len(owned_ids) == 0
            prior = float(priors.get(first_key if is_first else second_key, 0.5))
            recs.append(_accessory_rec(b, st, strength, acc, label, prior, delta, economy, is_first))

        # --- gears: best still-missing universal gear ---
        rec = _gear_rec(b, st, strength, gears_guide, itemstats, priors, economy)
        if rec is not None:
            recs.append(rec)

        # --- hypercharge ---
        if not st.has_hypercharge and _hypercharge_eligible(economy, b.name):
            recs.append(_hypercharge_rec(b, st, strength, priors, economy))

        # Buffies are intentionally not advised: the roster reports which buffies you *own* but not
        # how many exist per brawler, so a "slot open" can't be told from "no buffie released" (a
        # brawler like R-T has none). See engine/mastery.py for the same reasoning.

    recs.sort(key=lambda r: r.value_score, reverse=True)
    return [r.as_dict() for r in recs[:top]]


# --- per-kind rec builders -------------------------------------------------------------------

def _gear_names(economy: dict) -> list:
    """The curated universal-gear list (reuses loadout's hand-maintained gears.json guide), each as
    ``(display_name, normalized_name, base_prior, roles)``."""
    from bsdraft.engine.loadout import _gear_guide  # reuse the fail-safe loader; both stdlib-only
    out = []
    for g in _gear_guide().get("gears", []):
        name = g.get("name", "")
        out.append((name, _norm(name), float(g.get("base", 0.4)), g.get("roles", {}) or {}))
    return out


def _fold_gate(economy: dict, st: OwnedState, kind: str, base_cost: Dict[str, int]):
    """If ``kind`` is gated above the brawler's current power, fold the power-upgrade cost into the
    item cost and return a human 'requires Power N' note; else leave both untouched."""
    gate = _gate_for(economy, kind)
    if gate and st.power and st.power < gate:
        return _merge_cost(base_cost, _power_cost(economy, st.power, gate)), f"requires Power {gate}"
    return base_cost, None


def _accessory_rec(b, st, strength, acc, label, prior, delta, economy, is_first) -> _Rec:
    value = strength * prior + (_DELTA_WEIGHT * delta if delta and delta > 0 else 0.0)
    cost, gate = _fold_gate(economy, st, acc.kind, _item_cost(economy, acc.kind))
    ordinal = "first" if is_first else "second"
    if delta is not None and delta > 0:
        conf, why = "measured", (f"Best {label} on {b.name} — measured +{delta * 100:.1f}% vs her "
                                 f"other {label}s. Her {ordinal} {label}.")
    else:
        conf, why = "heuristic", f"{b.name}'s {ordinal} {label} — a core loadout slot she's missing."
    return _Rec(brawler_id=b.id, brawler_name=b.name, kind=acc.kind, value_score=value,
                meta_winrate=strength, confidence=conf, cost=cost, rationale=why,
                item_id=acc.id, item_name=acc.name, item_delta=delta, gate=gate,
                target_power=(_gate_for(economy, acc.kind) if gate else None))


def _gear_rec(b, st, strength, gears_guide, itemstats, priors, economy) -> Optional[_Rec]:
    missing = [g for g in gears_guide if g[1] not in st.gears]
    if not missing:
        return None
    # Which gear: best significant measured gear, else the best editorial fit (base + class role).
    def gear_key(g):
        d = _measured_delta(itemstats, b.id, None, gear_name=g[0])
        return (d if d is not None else -1.0, g[2] + float(g[3].get(b.cls, 0.0)))
    best = max(missing, key=gear_key)
    delta = _measured_delta(itemstats, b.id, None, gear_name=best[0])
    prior = float(priors.get("gear", 0.5))
    value = strength * prior + (_DELTA_WEIGHT * delta if delta and delta > 0 else 0.0)
    cost, gate = _fold_gate(economy, st, "gear", _item_cost(economy, "gear"))
    if delta is not None and delta > 0:
        conf, why = "measured", f"{best[0]} gear on {b.name} — measured +{delta * 100:.1f}% win rate."
    else:
        conf, why = "heuristic", f"{best[0]} gear — a strong universal gear {b.name} is missing."
    return _Rec(brawler_id=b.id, brawler_name=b.name, kind="gear", value_score=value,
                meta_winrate=strength, confidence=conf, cost=cost, rationale=why,
                item_name=best[0], item_delta=delta, gate=gate,
                target_power=(_gate_for(economy, "gear") if gate else None))


def _hypercharge_rec(b, st, strength, priors, economy) -> _Rec:
    prior = float(priors.get("hypercharge", 0.6))
    value = strength * prior
    cost, gate = _fold_gate(economy, st, "hypercharge", _item_cost(economy, "hypercharge"))
    if gate:
        why = f"Push {b.name} to Power 11 and unlock her Hypercharge — a game-swinging upgrade."
        target = 11
    else:
        why = f"Buy {b.name}'s Hypercharge — a game-swinging upgrade she's eligible for."
        target = None
    return _Rec(brawler_id=b.id, brawler_name=b.name, kind="hypercharge", value_score=value,
                meta_winrate=strength, confidence="eligibility_only", cost=cost, rationale=why,
                gate=gate, target_power=target)


def _new_brawler_rec(b, strength, stats: DraftStats, priors, economy) -> _Rec:
    prior = float(priors.get("new_brawler", 0.6))
    value = strength * prior
    credits = (economy.get("new_brawler_credits", {}) or {}).get(b.rarity) if economy else None
    cost = {"credits": int(credits)} if credits else {}
    games = stats.brawler_rate(b.id, None).games
    conf = "measured" if games >= _MIN_MEASURED_GAMES else "heuristic"
    why = f"Unlock {b.name} ({b.rarity}) — a {strength * 100:.0f}%-win-rate meta brawler you don't own."
    return _Rec(brawler_id=b.id, brawler_name=b.name, kind="new_brawler", value_score=value,
                meta_winrate=strength, confidence=conf, cost=cost, rationale=why)
