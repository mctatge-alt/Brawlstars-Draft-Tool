"""Unit tests for the personal-win-rate layer (bsdraft.engine.personal).

Pure in-memory synthetic matches — no disk, no network — so the tricky bits are pinned:
self-perspective W/L (the player isn't always on team A), match_key de-duplication, the
per-map -> overall -> global back-off, and omitting brawlers the player never touched.

    PYTHONPATH=backend python -m pytest backend/tests/test_personal.py    # or run directly
"""
from __future__ import annotations

from bsdraft.engine.personal import PersonalStats

TOL = 1e-9


def mk(key, team_a, team_b, a_won, map_id, ts=0):
    """team_a/team_b: list of (tag, brawler_id). Builds a stored-shape Match dict."""
    a = [{"tag": t, "brawler_id": b} for t, b in team_a]
    b = [{"tag": t, "brawler_id": b} for t, b in team_b]
    return {
        "match_key": key, "ts": ts, "a_won": a_won, "map_id": map_id,
        "team_a": a, "team_b": b, "player_tags": [p["tag"] for p in a + b],
    }


# AAA: win w/ brawler 1 on map 100 (team A); loss w/ 1 on map 100 (team B, a_won still True);
# win w/ 1 on map 200; plus a duplicate copy of the first game (same match_key).
MATCHES = [
    mk("k1", [("AAA", 1), ("B", 2), ("C", 3)], [("D", 4), ("E", 5), ("F", 6)], True, 100),
    mk("k2", [("D", 4), ("E", 5), ("F", 6)], [("AAA", 1), ("B", 2), ("C", 3)], True, 100),
    mk("k3", [("AAA", 1), ("B", 2), ("C", 3)], [("D", 4), ("E", 5), ("F", 6)], True, 200),
    mk("k1", [("AAA", 1), ("B", 2), ("C", 3)], [("D", 4), ("E", 5), ("F", 6)], True, 100),  # dup
]


def _ps():
    return PersonalStats("AAA", MATCHES, fallback=None, halflife_days=0.0)


def test_dedup_by_match_key():
    assert _ps().n == 3  # four rows in, one duplicate dropped


def test_self_perspective_flip():
    # On map 100 AAA went 1-1: the team-B game (a_won=True) must count as a loss.
    ps = _ps()
    r = ps.brawler_rate(1, 100)
    assert r.games == 2.0
    assert abs(r.raw_winrate - 0.5) < TOL


def test_per_map_split():
    ps = _ps()
    assert abs(ps.brawler_rate(1, 200).raw_winrate - 1.0) < TOL  # 1-0 on map 200
    assert ps.brawler_rate(1, 200).games == 1.0


def test_overall_record_and_shrinkage():
    ps = _ps()
    r = ps.brawler_rate(1)
    assert r.games == 3.0
    assert abs(r.raw_winrate - (2.0 / 3.0)) < TOL
    # smoothed toward the 0.5 prior (fallback=None) -> below raw, above 0.5
    assert 0.5 < r.winrate < r.raw_winrate


def test_unplayed_map_backs_off_to_overall():
    # AAA never played brawler 1 on map 300 -> use the overall personal sample (3 games).
    ps = _ps()
    assert ps.brawler_rate(1, 300).games == 3.0


def test_never_played_brawler_has_no_signal():
    ps = _ps()
    assert ps.games_on(4) == 0.0           # AAA only ever used brawler 1
    never = ps.brawler_rate(4)
    assert never.games == 0.0              # callers gate on this and omit the component
    assert abs(never.winrate - 0.5) < TOL  # echoes the prior, contributes nothing


def test_absent_tag_builds_empty():
    assert PersonalStats("ZZZ", MATCHES, fallback=None, halflife_days=0.0).n == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
