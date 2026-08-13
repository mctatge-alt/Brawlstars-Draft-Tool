"""Tests for the loadout advice engine and the owned-item roster plumbing."""
from __future__ import annotations

from bsdraft.data import reference as R
from bsdraft.engine import loadout as L
from bsdraft.engine import mastery


def test_classify_effect_buckets():
    assert L._classify("Shelly dashes to an aimed direction, fully reloading her ammo") == "mobility"
    assert L._classify("she instantly heals for x health") == "sustain"
    assert L._classify("Super shells slow down enemies") == "control"
    assert L._classify("deals +x extra damage at max range") == "damage"
    assert L._classify("increases her attack range by one tile") == "range"
    assert L._classify("reveals enemies hidden in a bush") == "vision"
    # No matched keyword falls back to utility rather than erroring.
    assert L._classify("some brand new mechanic with no keywords") == "utility"


def test_clean_strips_value_tokens():
    assert L._clean("slows for <!card.value1.ticks> sec") == "slows for sec"
    assert L._clean("  extra   spaces ") == "extra spaces"


def test_mode_fit_is_mode_sensitive():
    # Mobility matters more in Brawl Ball; sustain more in Knockout.
    assert L._mode_fit("mobility", "Brawl Ball") > L._mode_fit("mobility", "Heist")
    assert L._mode_fit("sustain", "Knockout") > L._mode_fit("sustain", "Heist")
    assert L._mode_fit("damage", "Heist") > L._mode_fit("damage", "Gem Grab")


def test_loadout_advice_shape_and_single_recommendation():
    b = R.brawler_by_name("Shelly")
    adv = L.loadout_advice(b.id, "Knockout")
    assert adv is not None
    assert adv["brawler_id"] == b.id and adv["brawler_name"] == "Shelly"
    assert adv["gadgets"] and adv["star_powers"] and adv["gears"]
    # Exactly one gadget and one star power flagged as the best-fit pick.
    assert sum(g["recommended"] for g in adv["gadgets"]) == 1
    assert sum(s["recommended"] for s in adv["star_powers"]) == 1
    for item in adv["gadgets"] + adv["star_powers"]:
        assert 0.0 <= item["fit"] <= 1.0
        assert item["effect"] and item["why"]
        assert item["source"] == "heuristic"


def test_loadout_recommendation_changes_with_mode():
    b = R.brawler_by_name("Shelly")
    ko = {g["name"]: g for g in L.loadout_advice(b.id, "Knockout")["star_powers"]}
    heist = {g["name"]: g for g in L.loadout_advice(b.id, "Heist")["star_powers"]}
    # Band-Aid (sustain) is the Knockout pick; it should not also be the Heist pick.
    assert ko["Band-Aid"]["recommended"] is True
    assert heist["Band-Aid"]["recommended"] is False


def test_gears_are_mode_and_role_ranked():
    # Only the six stable Super Rare gears, two flagged as the core pick.
    gears = L.loadout_advice(R.brawler_by_name("Shelly").id, "Knockout")["gears"]
    assert len(gears) == 6
    assert sum(g["recommended"] for g in gears) == 2
    # For a Damage Dealer, Shield tops a survival mode (Knockout) while the Damage gear tops a burst
    # mode (Heist) — the mode weighting flips the ranking.
    ko = {g["name"]: g["fit"] for g in L.loadout_advice(R.brawler_by_name("Shelly").id, "Knockout")["gears"]}
    heist = {g["name"]: g["fit"] for g in L.loadout_advice(R.brawler_by_name("Shelly").id, "Heist")["gears"]}
    assert ko["Shield"] > ko["Damage"]
    assert heist["Damage"] > heist["Shield"]


def test_loadout_advice_unknown_brawler_is_none():
    assert L.loadout_advice(999_999_999, "Bounty") is None


def test_parse_roster_retains_owned_item_ids():
    player = {"brawlers": [{
        "id": 16000000, "power": 11,
        "starPowers": [{"id": 1, "name": "A"}],
        "gadgets": [{"id": 2, "name": "G"}, {"id": 3, "name": "H"}],
        "gears": [{"id": 5, "name": "Speed", "level": 3}],
    }]}
    m = mastery.parse_roster(player)[16000000]
    assert m.owned_star_powers == (1,)
    assert m.owned_gadgets == (2, 3)
    assert m.owned_gears == ({"id": 5, "name": "Speed", "level": 3},)
    # Backwards-compatible booleans still derived.
    assert m.has_starpower and m.has_gadget and m.has_gears


def test_no_buffie_object_is_not_flagged_missing():
    # A brawler the game hasn't given buffies yet (e.g. Mr. P) carries no `buffies` object.
    # It must NOT be reported as "missing buffie" — there is nothing to be missing.
    player = {"brawlers": [{"id": 16000000, "power": 11}]}
    m = mastery.parse_roster(player)[16000000]
    assert m.buffies_total == 0
    assert "missing buffie" not in m.gaps()


def test_partly_owned_buffies_are_flagged_missing():
    # A brawler that *does* have buffie slots but the player hasn't filled them all is a real gap.
    player = {"brawlers": [{
        "id": 16000000, "power": 11,
        "buffies": {"gadget": True, "starPower": False, "hyperCharge": False},
    }]}
    m = mastery.parse_roster(player)[16000000]
    assert (m.buffies_have, m.buffies_total) == (1, 3)
    assert "missing buffie" in m.gaps()
