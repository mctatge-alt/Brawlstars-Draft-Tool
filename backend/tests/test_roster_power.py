"""The Ranked power-level gate: an owned brawler below the bracket's power floor can't be fielded,
so it must be dropped from the personalized roster before it's ever recommended.

Ranked doesn't normalize brawlers to a fixed power — each bracket hard-blocks selecting a brawler
below a per-brawler floor: Power 9 through Diamond, Power 11 from Mythic up. Recommending one the
player literally cannot pick (e.g. an un-maxed Bibi in Legendary) is the bug this guards.

    PYTHONPATH=backend python -m pytest backend/tests/test_roster_power.py    # or run directly
"""
from __future__ import annotations

import types

import bsdraft.api.main as M
from bsdraft.api import schemas as S
from bsdraft.engine.tiers import min_power_for_bracket

# Synthetic ids well outside any real brawler / boosted-rotation id, so the boosted brawlers the
# gate folds in can never collide with the ones under assertion here.
A11, B9, C10, D0, E7 = 90000011, 90000009, 90000010, 90000000, 90000007


def _roster(bracket, entries, personalize=True):
    roster = [S.OwnedBrawler(id=i, mastery=0.5, power=p) for i, p in entries]
    req = S.RecommendRequest(map_id=1, mode="Brawl Ball", rank_bracket=bracket,
                             personalize=personalize, roster=roster)
    return M._roster_for(req)


# --- the pure floor helper -------------------------------------------------------

def test_min_power_per_bracket():
    for b in ("Mythic", "Legendary", "Masters", "Pro"):
        assert min_power_for_bracket(b) == 11, b
    for b in ("Bronze", "Silver", "Gold", "Diamond"):
        assert min_power_for_bracket(b) == 9, b
    # unknown / unset → the universal Ranked floor, never a hidden Power-11 gate
    assert min_power_for_bracket(None) == 9
    assert min_power_for_bracket("Nonsense") == 9


# --- the gate in _roster_for -----------------------------------------------------

def test_p11_bracket_drops_under_eleven():
    # Legendary: only the Power-11 brawler survives; Power 9 and Power 10 are unfieldable.
    r = _roster("Legendary", [(A11, 11), (B9, 9), (C10, 10)])
    assert A11 in r and B9 not in r and C10 not in r


def test_mythic_is_also_power_eleven():
    # The floor jumps to 11 at Mythic, not Legendary — a Power-9 brawler is blocked there too.
    r = _roster("Mythic", [(A11, 11), (B9, 9)])
    assert A11 in r and B9 not in r


def test_low_bracket_keeps_power_nine():
    # Diamond: floor is 9, so Power 9/10/11 all field; only below 9 is dropped.
    r = _roster("Diamond", [(A11, 11), (B9, 9), (C10, 10), (E7, 7)])
    assert A11 in r and B9 in r and C10 in r and E7 not in r


def test_unknown_power_is_kept():
    # Power 0 == "roster didn't report it" (older client): keep it rather than hide on missing data.
    r = _roster("Legendary", [(D0, 0)])
    assert D0 in r


def test_unset_bracket_uses_universal_floor():
    r = _roster(None, [(B9, 9), (E7, 7)])
    assert B9 in r and E7 not in r


def test_personalize_off_returns_none():
    assert _roster("Legendary", [(A11, 11)], personalize=False) is None


# --- sent-but-empty vs omitted: the cross-visitor leak guard ---------------------

class _ServerMastery:
    """Engine-side Mastery stand-in: just the fields the fallback branch reads."""
    power = 11
    score = 0.9

    def gaps(self):
        return []


def test_empty_roster_never_falls_back_to_server_roster():
    # roster == [] means "sent, and nothing fieldable" — the client's power-floor filter can empty
    # a real roster. On the keyed host _engine.roster is whatever /api/roster fetched LAST (another
    # visitor), so falling through on [] scores one player's draft against another player's roster.
    stranger = 90000042
    prev = M._engine  # None under tests — the startup event that builds it never ran
    M._engine = types.SimpleNamespace(roster={stranger: _ServerMastery()})
    try:
        r = _roster("Legendary", [])
        assert r is not None, "personalize stays on for an empty roster"
        assert stranger not in r, "another visitor's roster leaked into personalization"
        from bsdraft.data import reference as R
        assert set(r) == set(R.load_ranked_boosted())  # only the free boosted brawlers field
    finally:
        M._engine = prev


def test_omitted_roster_falls_back_to_server_roster():
    # roster=None (field omitted — a client with no roster of its own): the server's roster is the
    # intended fallback for the local/home operator.
    mine = 90000043
    prev = M._engine
    M._engine = types.SimpleNamespace(roster={mine: _ServerMastery()})
    try:
        req = S.RecommendRequest(map_id=1, mode="Brawl Ball", rank_bracket="Legendary",
                                 personalize=True, roster=None)
        assert mine in M._roster_for(req)
    finally:
        M._engine = prev


def test_boosted_brawlers_are_added_and_clear_the_floor():
    # Boosted (free) brawlers arrive at Power 11 and are folded in after the gate, so they're always
    # recommendable regardless of bracket — while an owned sub-floor brawler beside them is dropped.
    from bsdraft.data import reference as R
    boosted = R.load_ranked_boosted()
    r = _roster("Legendary", [(B9, 9)])
    assert B9 not in r
    for bid in boosted:
        assert bid in r


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
