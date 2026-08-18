"""Purchase-advisor tests: given an ownership snapshot, the advisor must recommend the *unowned*
items that matter, price them from fixed constants, fold power-upgrade cost into gated items, and
rank purely by meta strength × purchase impact.

Runs on a synthetic 3-brawler catalog + hand-checkable stats so every expected direction is known
in advance, with the economy table inlined so cost math doesn't depend on the shipped JSON.

    PYTHONPATH=backend python -m pytest backend/tests/test_purchases.py   # or run directly
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple

from bsdraft.engine import purchases as P
from bsdraft.engine.purchases import OwnedState


# --- synthetic catalog -----------------------------------------------------------------------

@dataclass(frozen=True)
class _Acc:
    id: int
    name: str
    kind: str


@dataclass(frozen=True)
class _Brawler:
    id: int
    name: str
    cls: str
    rarity: str
    gadgets: Tuple[_Acc, ...] = ()
    star_powers: Tuple[_Acc, ...] = ()


CATALOG = [
    _Brawler(1, "Strong", "Assassin", "Legendary",
             gadgets=(_Acc(101, "G1a", "gadget"), _Acc(102, "G1b", "gadget")),
             star_powers=(_Acc(201, "S1a", "star_power"), _Acc(202, "S1b", "star_power"))),
    _Brawler(2, "Mid", "Support", "Epic",
             gadgets=(_Acc(111, "G2a", "gadget"), _Acc(112, "G2b", "gadget")),
             star_powers=(_Acc(211, "S2a", "star_power"), _Acc(212, "S2b", "star_power"))),
    _Brawler(3, "Fresh", "Marksman", "Mythic",
             gadgets=(_Acc(121, "G3a", "gadget"),),
             star_powers=(_Acc(221, "S3a", "star_power"),)),
]


class _Ref:
    def load_brawlers(self):
        return CATALOG


@dataclass
class _Rate:
    winrate: float
    games: float = 200.0


class _Stats:
    """Per-brawler win rate (same for every map, so meta_strength == the brawler's rate)."""
    def __init__(self, rates):
        self.rates = rates

    def brawler_rate(self, brawler_id, map_id=None):
        return _Rate(self.rates.get(brawler_id, 0.5))


@dataclass
class _Map:
    id: int


RANKED_MAPS = [_Map(10), _Map(11), _Map(12)]

# Economy inlined — mirrors data/reference/economy.json so cost assertions are deterministic.
ECON = {
    "power_cost_cumulative": {
        "power_points": [0, 0, 20, 50, 100, 180, 310, 520, 860, 1410, 2300, 3740],
        "coins":        [0, 0, 20, 55, 130, 270, 560, 1040, 1840, 3090, 4965, 7765],
    },
    "item_costs": {
        "gadget": {"coins": 1000}, "star_power": {"coins": 2000}, "gear": {"coins": 1000},
        "hypercharge": {"coins": 5000},
    },
    "new_brawler_credits": {"Epic": 170, "Mythic": 430, "Legendary": 860},
    "power_gates": {"gadget": 7, "gear": 8, "star_power": 9, "hypercharge": 11},
    "impact_priors": {
        "gadget_first": 0.85, "gadget_second": 0.45, "star_power_first": 0.90,
        "star_power_second": 0.40, "gear": 0.55, "hypercharge": 0.80,
        "new_brawler": 0.70,
    },
    "hypercharge_availability": {"mode": "all_except", "list": []},
}

GEARS = [("Shield", "shield", 0.60, {}), ("Speed", "speed", 0.47, {})]


@contextmanager
def _synthetic(gears=GEARS):
    saved = (P.R, P._gear_names)
    P.R = _Ref()
    P._gear_names = lambda economy: list(gears)
    try:
        yield
    finally:
        P.R, P._gear_names = saved


def _run(owned, rates, itemstats=None, econ=ECON, top=50):
    with _synthetic():
        recs = P.recommend_purchases(owned, _Stats(rates), itemstats=itemstats, top=top,
                                     ranked_maps=RANKED_MAPS, economy=econ)
    return recs


def _by(recs, brawler_id=None, kind=None):
    return [r for r in recs
            if (brawler_id is None or r["brawler_id"] == brawler_id)
            and (kind is None or r["kind"] == kind)]


# --- tests -----------------------------------------------------------------------------------

def test_recommends_the_missing_item_not_the_owned_one():
    """Owning gadget 101 of {101,102} ⇒ recommend 102; owning both star powers ⇒ no star-power rec."""
    owned = {1: OwnedState(power=11, gadgets=frozenset({101}), star_powers=frozenset({201, 202}))}
    recs = _run(owned, {1: 0.55})
    gadget = _by(recs, brawler_id=1, kind="gadget")
    assert len(gadget) == 1 and gadget[0]["item_id"] == 102, gadget
    assert not _by(recs, brawler_id=1, kind="star_power"), "owns both star powers → nothing to buy"


def test_first_item_outranks_second_at_equal_strength():
    """A brawler with no gadgets (first-gadget buy) beats one with a redundant second gadget when
    the two brawlers are equally strong — the first-item prior is higher."""
    owned = {
        1: OwnedState(power=11, gadgets=frozenset({101})),          # owns one → second-gadget buy
        2: OwnedState(power=11, gadgets=frozenset()),               # owns none → first-gadget buy
    }
    recs = _run(owned, {1: 0.55, 2: 0.55})
    first = _by(recs, brawler_id=2, kind="gadget")[0]
    second = _by(recs, brawler_id=1, kind="gadget")[0]
    assert first["value_score"] > second["value_score"], (first, second)


def test_power_gate_folds_the_upgrade_cost_and_annotates():
    """A Power-7 brawler's first star power is gated at 9 — the rec must carry the 7→9 upgrade cost
    on top of the 2,000-coin star power and say so."""
    owned = {2: OwnedState(power=7, gadgets=frozenset(), star_powers=frozenset())}
    recs = _run(owned, {2: 0.55})
    sp = _by(recs, brawler_id=2, kind="star_power")[0]
    # coins = 2000 (star power) + (cum_coins[9]-cum_coins[7]) = 2000 + (3090-1040) = 4050
    assert sp["cost"]["coins"] == 4050, sp["cost"]
    assert sp["cost"]["power_points"] == 1410 - 520, sp["cost"]     # 890
    assert sp["gate"] == "requires Power 9" and sp["target_power"] == 9, sp


def test_no_buffie_recommendations():
    """Buffies are unadvised: the roster reports which buffies you own but not how many exist, so
    'slot open' can't be told from 'no buffie released' (R-T has none). The advisor never emits a
    buffie rec — the old '0/3 slots filled' misfire on buffie-less brawlers is gone."""
    owned = {
        1: OwnedState(power=11, gadgets=frozenset({101, 102}), star_powers=frozenset({201, 202})),
    }
    recs = _run(owned, {1: 0.55})
    assert not _by(recs, brawler_id=1, kind="buffie")


def test_hypercharge_eligibility_and_power11_gate():
    """A Power-11 brawler without a hypercharge gets a straight buy; a Power-7 one gets the same rec
    with the Power 11 climb folded in. Mode 'only' with an empty list suppresses both."""
    owned = {
        1: OwnedState(power=11, gadgets=frozenset({101, 102}), star_powers=frozenset({201, 202})),
        2: OwnedState(power=7, gadgets=frozenset({111, 112}), star_powers=frozenset({211, 212})),
    }
    recs = _run(owned, {1: 0.55, 2: 0.55})
    hc1 = _by(recs, brawler_id=1, kind="hypercharge")[0]
    hc2 = _by(recs, brawler_id=2, kind="hypercharge")[0]
    assert hc1["gate"] is None and hc1["cost"].get("coins") == 5000, hc1
    assert hc2["gate"] == "requires Power 11" and hc2["target_power"] == 11, hc2
    assert hc2["cost"]["coins"] == 5000 + (7765 - 1040), hc2["cost"]     # + Power 7→11 coins

    econ_only = dict(ECON, hypercharge_availability={"mode": "only", "list": []})
    recs2 = _run(owned, {1: 0.55, 2: 0.55}, econ=econ_only)
    assert not _by(recs2, kind="hypercharge"), "mode 'only' + empty list ⇒ no one is eligible"


def test_missing_economy_section_suppresses_hypercharge():
    """No availability policy at all ⇒ don't recommend a hypercharge we can't vouch for."""
    owned = {1: OwnedState(power=11, gadgets=frozenset({101, 102}), star_powers=frozenset({201, 202}))}
    recs = _run(owned, {1: 0.55}, econ={k: v for k, v in ECON.items()
                                        if k != "hypercharge_availability"})
    assert not _by(recs, kind="hypercharge")


def test_unowned_brawler_is_a_new_brawler_unlock():
    """A brawler absent from the roster becomes a Credit-unlock candidate priced by its rarity."""
    owned = {1: OwnedState(power=11, gadgets=frozenset({101, 102}), star_powers=frozenset({201, 202}))}
    recs = _run(owned, {1: 0.55, 3: 0.60})
    nb = _by(recs, brawler_id=3, kind="new_brawler")
    assert len(nb) == 1 and nb[0]["cost"].get("credits") == 430, nb   # Mythic
    assert nb[0]["confidence"] == "measured"                          # 200 games ≥ threshold


def test_measured_delta_upgrades_confidence_and_nudges_score():
    """A significant itemstats cell flips a gadget rec to 'measured', records the delta, and lifts
    its score above the same rec without measurement."""
    owned = {1: OwnedState(power=11, gadgets=frozenset({101}))}     # missing 102
    itemstats = {"cells": {"1:102": {"significant": True, "delta": 0.06}}}
    measured = _by(_run(owned, {1: 0.55}, itemstats=itemstats), brawler_id=1, kind="gadget")[0]
    plain = _by(_run(owned, {1: 0.55}), brawler_id=1, kind="gadget")[0]
    assert measured["confidence"] == "measured" and measured["item_delta"] == 0.06, measured
    assert plain["confidence"] == "heuristic" and plain["item_delta"] is None
    assert measured["value_score"] > plain["value_score"], (measured, plain)


def test_degrades_without_itemstats():
    """With no itemstats table the advisor still runs — every item rec is heuristic, none measured."""
    owned = {2: OwnedState(power=11, gadgets=frozenset(), star_powers=frozenset())}
    recs = _run(owned, {2: 0.55}, itemstats=None)
    assert recs, "should still produce recommendations from priors alone"
    assert all(r["confidence"] != "measured" for r in recs if r["kind"] in ("gadget", "star_power", "gear"))


def test_meta_strength_averages_across_ranked_maps():
    """meta_winrate is the (games-weighted) average of the brawler's rate over the ranked maps —
    with a flat per-map rate that's just the rate itself."""
    owned = {3: OwnedState(power=11, gadgets=frozenset({121}), star_powers=frozenset({221}))}
    recs = _by(_run(owned, {3: 0.573}), brawler_id=3)
    assert recs and all(abs(r["meta_winrate"] - 0.573) < 1e-9 for r in recs), recs


def test_results_are_sorted_by_value_desc():
    owned = {
        1: OwnedState(power=11, gadgets=frozenset({101})),
        2: OwnedState(power=7, gadgets=frozenset(), star_powers=frozenset()),
    }
    recs = _run(owned, {1: 0.58, 2: 0.52})
    scores = [r["value_score"] for r in recs]
    assert scores == sorted(scores, reverse=True), scores


def test_top_limit_is_respected():
    owned = {1: OwnedState(power=7, gadgets=frozenset(), star_powers=frozenset())}
    recs = _run(owned, {1: 0.55}, top=2)
    assert len(recs) == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
