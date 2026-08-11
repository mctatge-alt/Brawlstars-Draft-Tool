"""Tests for the per-item win-rate estimator (single-item-owner inference) and its serve blend.

Correctness is proven on SYNTHETIC data with known effects, because the real table can't be built
without the collected ownership profiles. These cases mirror the adversarial-review fixes: effect
recovery, Mantel-Haenszel confound control, clustered (not binomial) variance, the empty-REST guard,
small-sample shrinkage + gate, gear keying, artifact round-trip, and the loadout blend/fallback.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from bsdraft.data import itemstats_build as B
from bsdraft.engine import itemstats as IS
from bsdraft.engine import loadout as L

X = 16000000            # a brawler under test (id kept out of the boosted list via boosted_ids=[])
X2 = 16000001
G1, G2 = 23000001, 23000002        # two gadget ids
SP1, SP2 = 23000101, 23000102      # two star-power ids
SHIELD_ID, DAMAGE_ID = 62000000, 62000001   # synthetic gear ids (names matched to gears.json)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _match(ts, tag, brawler, tier, won, fi):
    a = [{"tag": tag, "brawler_id": brawler, "trophies": tier},
         {"tag": f"fa{fi}0", "brawler_id": 16009001, "trophies": tier},
         {"tag": f"fa{fi}1", "brawler_id": 16009002, "trophies": tier}]
    b = [{"tag": f"fb{fi}0", "brawler_id": 16009003, "trophies": tier},
         {"tag": f"fb{fi}1", "brawler_id": 16009004, "trophies": tier},
         {"tag": f"fb{fi}2", "brawler_id": 16009005, "trophies": tier}]
    return {"ts": ts, "a_won": bool(won), "team_a": a, "team_b": b,
            "player_tags": [pp["tag"] for pp in a + b], "match_key": f"{tag}:{fi}"}


class Scenario:
    """Accumulates (player owns item i on brawler, wins at rate over N games at tier) and emits the
    matches.jsonl + profiles.jsonl the build reads."""
    def __init__(self, ts=1_000_000):
        self.ts = ts
        self.matches = []
        self.profiles = {}
        self._fi = 0

    def add(self, tag, item_id, item_type, rate, games, tier, brawler=X, gear_name=None):
        wins = int(round(rate * games))
        for j in range(games):
            self.matches.append(_match(self.ts, tag, brawler, tier, j < wins, self._fi))
            self._fi += 1
        row = self.profiles.setdefault(tag, {}).setdefault(str(brawler), {"gd": [], "sp": [], "gr": []})
        if item_type == "gr":
            row["gr"].append([item_id, gear_name, 3])
        else:
            row[item_type].append(item_id)

    def build(self, tmp_path, **params):
        mp, pp = tmp_path / "m.jsonl", tmp_path / "p.jsonl"
        _write_jsonl(mp, self.matches)
        _write_jsonl(pp, [{"tag": t, "ts": self.ts - 1000, "b": b} for t, b in self.profiles.items()])
        return B.build_itemstats(mp, pp, params=params or None, boosted_ids=[])


# ---- unit-level estimator pieces -------------------------------------------------------------

def test_pool_stats_player_clustered_n_and_floored_variance():
    # The independent unit is the PLAYER, not the game: 40 players x 30 games each -> n_eff ~ 40,
    # not 1200. And a homogeneous pool's variance is floored at binomial-at-n_eff, never 0 (the
    # degenerate z=0 the floor fixes).
    g = np.full(40, 30.0)
    homo = B.pool_stats(g, g * 0.5)
    assert homo.n_eff == pytest.approx(40.0)                       # player-clustered, not 1200
    assert homo.var == pytest.approx(0.25 / 40.0) and homo.var > 0
    # Concentrated weight deflates the effective N below the raw player count (design effect > 1).
    gu = np.array([300.0] + [1.0] * 39)
    assert B.pool_stats(gu, gu * 0.5).n_eff < 40.0


def test_cap_shares_bounds_a_single_player_and_preserves_rate():
    # One 100x grinder among 40 ones — a 5% cap is only achievable with enough players (min share
    # here is 1/41 < 5%), and the iterated cap must drive the grinder down to the cap.
    g = np.array([100.0] + [1.0] * 40)
    wn = g * 0.7                                                    # everyone wins 70%
    cg, cwn = B.cap_shares(g, wn, 0.05)
    assert cg.max() / cg.sum() <= 0.05 + 1e-3                       # no player exceeds the cap
    assert np.allclose(cwn / cg, 0.7)                              # per-player rate preserved


def test_mh_combine_empty_rest_returns_none():
    only = B.pool_stats(np.ones(20), np.full(20, 0.6))
    empty = B.pool_stats(np.array([]), np.array([]))
    assert B.mh_combine([(only, empty)]) is None                   # no contrast -> fall back


def test_bh_fdr_flags_small_pvalues_and_respects_rate():
    pvals = [0.001, 0.2, 0.5, 0.8, 0.9]
    sig, qv = B.bh_fdr(pvals, 0.05)
    assert sig[0] is True and not any(sig[1:])
    assert qv[0] <= 0.05 and all(0 <= q <= 1 for q in qv)


# ---- build-level correctness -----------------------------------------------------------------

def test_effect_recovery_positive_delta_is_significant(tmp_path):
    # 300 distinct single-owners per item: enough PLAYER-level evidence for a ~10pt edge to clear the
    # BH gate at the (deflated) player-clustered effective N.
    s = Scenario()
    for pnum in range(300):
        j = ((pnum % 7) - 3) * 0.01                                # tiny deterministic jitter
        s.add(f"a{pnum}", G1, "gd", 0.60 + j, 15, tier=8)          # item G1 wins 60%
        s.add(f"b{pnum}", G2, "gd", 0.50 + j, 15, tier=8)          # item G2 wins 50%
    doc = s.build(tmp_path)
    c1, c2 = doc["cells"][f"{X}:{G1}"], doc["cells"][f"{X}:{G2}"]
    assert c1["delta"] > 0.03 and c1["significant"] == 1           # ~+0.10 shrunk, gated significant
    assert c2["delta"] < -0.03 and c2["significant"] == 1          # mirror sign
    assert c1["item_type"] == "gadget"


def test_mantel_haenszel_defuses_tier_confound(tmp_path):
    # G1 has NO true item effect, but its single-owners skew to the high tier (which wins more).
    # Pooled naive would show G1 ahead; MH stratified by tier must net it to ~0 and NOT flag it.
    s = Scenario()
    HI, LO = 17, 8                                                 # Legendary vs Gold
    for pnum in range(45):
        s.add(f"g1hi{pnum}", G1, "gd", 0.58, 20, tier=HI)
    for pnum in range(15):
        s.add(f"g1lo{pnum}", G1, "gd", 0.42, 20, tier=LO)
    for pnum in range(15):
        s.add(f"g2hi{pnum}", G2, "gd", 0.58, 20, tier=HI)
    for pnum in range(45):
        s.add(f"g2lo{pnum}", G2, "gd", 0.42, 20, tier=LO)
    doc = s.build(tmp_path)
    c1 = doc["cells"][f"{X}:{G1}"]
    # naive pooled diff would be ~+0.08; MH risk difference must be ~0.
    assert abs(c1["delta_raw"]) < 0.03
    assert c1["significant"] == 0
    assert c1["n_strata"] == 2


def test_thin_cell_is_shrunk_and_not_significant(tmp_path):
    # 15 owners per item: enough to clear the per-stratum floor so a cell forms, but below the
    # n_min_players=30 rank gate -> the cell exists, is shrunk toward 0, and is NOT significant.
    s = Scenario()
    for pnum in range(15):
        s.add(f"a{pnum}", G1, "gd", 0.75, 12, tier=8)
        s.add(f"b{pnum}", G2, "gd", 0.50, 12, tier=8)
    doc = s.build(tmp_path)
    c1 = doc["cells"][f"{X}:{G1}"]
    assert c1["significant"] == 0 and c1["n_players"] == 15        # gate fails on sample size
    assert abs(c1["delta"]) < abs(c1["delta_raw"])                # shrunk toward 0


def test_single_item_type_yields_no_cell(tmp_path):
    s = Scenario()
    for pnum in range(60):                                         # everyone owns only G1 -> no REST
        s.add(f"a{pnum}", G1, "gd", 0.60, 20, tier=8)
    doc = s.build(tmp_path)
    assert not any(k.startswith(f"{X}:") and k.endswith(str(G1)) for k in doc["cells"])


def test_item_winrate_is_rest_anchored_not_double_counted(tmp_path):
    # A 2-item type where G1 is the strong item at 0.62 vs G2 at 0.50. The served absolute
    # item_winrate must sit near G1's own rate, NOT overshoot toward rest+full-gap double counting.
    s = Scenario()
    for pnum in range(60):
        s.add(f"a{pnum}", G1, "gd", 0.62, 20, tier=8)
        s.add(f"b{pnum}", G2, "gd", 0.50, 20, tier=8)
    doc = s.build(tmp_path)
    iw = doc["cells"][f"{X}:{G1}"]["item_winrate"]
    assert 0.5 < iw < 0.66                                         # rest(~.50)+delta(~.11), not ~.74


def test_gear_cells_are_keyed_per_brawler_and_resolve_by_name(tmp_path):
    s = Scenario()
    for pnum in range(50):
        s.add(f"s1{pnum}", SHIELD_ID, "gr", 0.58, 20, tier=8, brawler=X, gear_name="Shield")
        s.add(f"d1{pnum}", DAMAGE_ID, "gr", 0.50, 20, tier=8, brawler=X, gear_name="Damage")
        s.add(f"s2{pnum}", SHIELD_ID, "gr", 0.52, 20, tier=8, brawler=X2, gear_name="Shield")
        s.add(f"d2{pnum}", DAMAGE_ID, "gr", 0.50, 20, tier=8, brawler=X2, gear_name="Damage")
    doc = s.build(tmp_path)
    # The same shared Shield id yields DISTINCT cells per brawler, never one merged cell.
    assert f"{X}:{SHIELD_ID}" in doc["cells"] and f"{X2}:{SHIELD_ID}" in doc["cells"]
    assert doc["cells"][f"{X}:{SHIELD_ID}"]["item_type"] == "gear"
    # meta maps gear name -> id so the serve loader can resolve a guide gear (which has no id).
    assert doc["meta"]["gear_ids_by_name"].get("shield") == SHIELD_ID
    assert IS.gear_cell(doc, X, "Shield") is doc["cells"][f"{X}:{SHIELD_ID}"]


def test_null_trophies_rows_do_not_crash_build(tmp_path):
    # Real match rows can carry a null `trophies` (tier). Those rows must be DROPPED from the
    # stratified estimator, not crash the build (bracket_of_tier would TypeError on None).
    s = Scenario()
    for pnum in range(40):
        s.add(f"a{pnum}", G1, "gd", 0.60, 12, tier=8)
        s.add(f"b{pnum}", G2, "gd", 0.50, 12, tier=8)
    for pnum in range(5):
        s.add(f"n{pnum}", G1, "gd", 0.99, 12, tier=None)      # null-tier rows (would inflate G1 if kept)
    doc = s.build(tmp_path)                                    # must not raise
    assert f"{X}:{G1}" in doc["cells"]                         # valid cells still form


def test_multiword_gear_name_resolves_through_serve_lookup(tmp_path):
    # 'Gadget Charge' (a real universal gear) exposed the build(name.lower)/serve(_norm) key mismatch.
    GC_ID = 62000005
    s = Scenario()
    for pnum in range(50):
        s.add(f"gc{pnum}", GC_ID, "gr", 0.56, 20, tier=8, brawler=X, gear_name="Gadget Charge")
        s.add(f"sh{pnum}", SHIELD_ID, "gr", 0.50, 20, tier=8, brawler=X, gear_name="Shield")
    doc = s.build(tmp_path)
    assert doc["meta"]["gear_ids_by_name"].get("gadgetcharge") == GC_ID       # normalized key
    assert IS.gear_cell(doc, X, "Gadget Charge") is doc["cells"][f"{X}:{GC_ID}"]


def test_negative_measured_item_is_not_recommended(monkeypatch):
    # A lone significant cell with a NEGATIVE delta means the item is measured WORSE than its sibling;
    # it must be shown (source winrate) but NOT recommended — the stronger heuristic sibling wins.
    from bsdraft.data import reference as R
    shelly = R.brawler_by_name("Shelly")
    g0 = shelly.gadgets[0].id
    table = {"version": 1, "meta": {"gear_ids_by_name": {}}, "brawler_baseline": {},
             "cells": {f"{shelly.id}:{g0}": {"item_type": "gadget", "delta": -0.05, "significant": 1,
                                            "item_winrate": 0.46, "n_eff": 240, "n_players": 61,
                                            "n_eff_rest": 300}}}
    monkeypatch.setattr(L.IS, "get_itemstats", lambda *a, **k: table)
    adv = L.loadout_advice(shelly.id, "Knockout")
    measured = next(g for g in adv["gadgets"] if g["id"] == g0)
    other = next(g for g in adv["gadgets"] if g["id"] != g0)
    assert measured["source"] == "winrate" and measured["fit"] == pytest.approx(0.5 - 3 * 0.05)
    assert measured["recommended"] is False                   # measured-worse item is not the pick
    assert other["recommended"] is True and other["source"] == "heuristic"


def test_artifact_roundtrip_gzip_and_plain(tmp_path):
    payload = {"version": 1, "meta": {"gear_ids_by_name": {}}, "brawler_baseline": {},
               "cells": {"16000000:23000001": {"delta": 0.123456, "significant": 1}}}
    for name in ("itemstats.json", "itemstats.json.gz"):
        path = tmp_path / name
        IS.save_itemstats(payload, path)
        got = IS.load_itemstats(path)
        assert got == payload
    # gzip path is magic-byte sniffed, not extension-driven
    assert (tmp_path / "itemstats.json.gz").read_bytes()[:2] == b"\x1f\x8b"


def test_namespace_gadget_and_starpower_ids_disjoint():
    from bsdraft.data import reference as R
    gad = {a.id for b in R.load_brawlers() for a in b.gadgets}
    sp = {a.id for b in R.load_brawlers() for a in b.star_powers}
    assert gad.isdisjoint(sp)                                      # justifies not needing kind in the key math


# ---- serve blend / fallback ------------------------------------------------------------------

def _synthetic_table(sig_gadget_delta=0.06):
    # A real Shelly (16000000) gadget id so the loadout lookup hits.
    from bsdraft.data import reference as R
    shelly = R.brawler_by_name("Shelly")
    gid = shelly.gadgets[0].id
    return {
        "version": 1, "meta": {"gear_ids_by_name": {}}, "brawler_baseline": {},
        "cells": {f"{shelly.id}:{gid}": {"item_type": "gadget", "delta": sig_gadget_delta,
                                         "item_winrate": 0.55, "significant": 1, "n_eff": 240,
                                         "q": 0.01, "z": 4.0, "n_players": 61, "n_eff_rest": 300}},
    }, shelly, gid


def test_loadout_blends_significant_cell_and_flips_source(monkeypatch):
    table, shelly, gid = _synthetic_table(0.06)
    monkeypatch.setattr(L.IS, "get_itemstats", lambda *a, **k: table)
    adv = L.loadout_advice(shelly.id, "Knockout")
    measured = next(g for g in adv["gadgets"] if g["id"] == gid)
    other = next(g for g in adv["gadgets"] if g["id"] != gid)
    assert measured["source"] == "winrate" and measured["recommended"] is True
    assert measured["fit"] == pytest.approx(0.5 + 3.0 * 0.06)      # +6% -> fit 0.68
    assert "+6.0% win rate" in measured["why"]
    assert other["source"] == "heuristic"                          # untouched item stays heuristic
    assert "Measured win rates" in adv["note"]


def test_loadout_is_pure_heuristic_when_no_table(monkeypatch):
    monkeypatch.setattr(L.IS, "get_itemstats", lambda *a, **k: None)
    adv = L.loadout_advice(16000000, "Knockout")
    assert all(g["source"] == "heuristic" for g in adv["gadgets"])
    assert all(s["source"] == "heuristic" for s in adv["star_powers"])
    assert "not a live tier read" in adv["note"]
