"""Rank resolution across a season reset, and the ranked-map list the board offers.

Both regressions surfaced on the 2026-08-20 season flip: the board showed a player their
*previous* season's tier as fact, and offered map/mode pairs (famously "Heist: Pit Stop") that
Ranked hasn't rotated in.
"""
from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

from bsdraft.api import main as M
from bsdraft.data import reference as R


class _FakeClient:
    """Stands in for BrawlStarsClient: an async context manager whose get_player either
    returns a profile dict or raises (the IP-locked-key / offline case)."""

    def __init__(self, player=None, exc=None):
        self._player, self._exc = player, exc

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get_player(self, tag):
        if self._exc is not None:
            raise self._exc
        return self._player


@pytest.fixture(autouse=True)
def _clear_rank_cache():
    M._rank_cache.clear()
    yield
    M._rank_cache.clear()


def _client(monkeypatch, *, token, player=None, exc=None, dataset_tier=None):
    monkeypatch.setattr(M.settings, "brawlstars_api_token", token)
    monkeypatch.setattr(M, "BrawlStarsClient", _FakeClient(player=player, exc=exc))
    monkeypatch.setattr(M, "_rank_index",
                        lambda: types.SimpleNamespace(get=lambda t: dataset_tier))
    return TestClient(M.app)


def test_live_tier_wins_and_is_not_marked_stale(monkeypatch):
    c = _client(monkeypatch, token="k", player={"rankedRank": 4}, dataset_tier=16)
    r = c.get("/api/rank", params={"tag": "#ABC123"}).json()
    assert r["found"] and r["tier"] == 4 and r["source"] == "live"
    assert r["stale"] is False          # a live read is current by construction


def test_unplaced_this_season_does_not_fall_back_to_the_pre_reset_row(monkeypatch):
    # The 2026-08-20 bug: the profile loads fine and carries no rankedRank (the season just
    # reset and the player hasn't placed), and the crawl still holds their old Legendary I row.
    # Falling through to that row reports a tier they no longer hold — the live lookup already
    # answered the question, so the dataset must not get a say.
    c = _client(monkeypatch, token="k", player={"name": "x"}, dataset_tier=16)
    r = c.get("/api/rank", params={"tag": "#ABC123"}).json()
    assert r["found"] is False
    assert r["source"] == "live"
    assert r["tier"] is None and r["bracket"] is None
    assert "season" in (r["error"] or "")


def test_unreachable_live_lookup_serves_the_dataset_but_flags_it(monkeypatch):
    # An IP-lock 403 teaches us nothing about the player, so the crawl row is still the best
    # guess — but it may predate a reset, so it goes out marked rather than as fact.
    c = _client(monkeypatch, token="k", exc=RuntimeError("HTTP 403: auth/IP error"),
                dataset_tier=16)
    r = c.get("/api/rank", params={"tag": "#ABC123"}).json()
    assert r["found"] and r["tier"] == 16 and r["source"] == "dataset"
    assert r["stale"] is True


def test_keyless_host_serves_the_dataset_unflagged(monkeypatch):
    # On the public host there is no live check to miss, so the dataset isn't "stale" in the
    # sense the flag means — nothing failed.
    c = _client(monkeypatch, token="", dataset_tier=16)
    r = c.get("/api/rank", params={"tag": "#ABC123"}).json()
    assert r["found"] and r["source"] == "dataset" and r["stale"] is False


def test_blank_tag_is_a_prompt_not_an_error(monkeypatch):
    c = _client(monkeypatch, token="k", dataset_tier=16)
    r = c.get("/api/rank", params={"tag": "  "}).json()
    assert r["found"] is False and r["error"]


# --------------------------------------------------------------------------- ranked map list

def _engine_with_map_games(map_games):
    return types.SimpleNamespace(
        stats=types.SimpleNamespace(map_games=map_games),
        bracket_stats={},
        roster=None,
    )


def test_reference_offers_only_maps_ranked_actually_rotates(monkeypatch):
    # The catalog's not-retired set spans every mode's whole map pool; Ranked rotates a handful.
    # Collected games are the only rotation signal we have, so a map with none is dropped.
    all_maps = R.load_ranked_maps()
    played = {m.id: 500 for m in all_maps[:6]}
    monkeypatch.setattr(M, "_engine", _engine_with_map_games(played))
    served = TestClient(M.app).get("/api/reference").json()["maps"]
    assert {m["id"] for m in served} == set(played)
    assert len(served) < len(all_maps)
    assert all(m["games"] > 0 for m in served)


def test_reference_falls_back_to_the_full_catalog_before_stats_load(monkeypatch):
    # A cold start has no map_games yet. Showing too many maps beats showing none.
    monkeypatch.setattr(M, "_engine", _engine_with_map_games({}))
    served = TestClient(M.app).get("/api/reference").json()["maps"]
    assert len(served) == len(R.load_ranked_maps())


def test_reference_map_ids_stay_a_subset_of_the_model_vocabulary(monkeypatch):
    # The filter lives in the endpoint on purpose: load_ranked_maps() also builds the model's
    # pinned map vocabulary, so narrowing it there would shift embedding rows under the
    # trained checkpoint. Guard that the served list never grows past that vocabulary.
    all_ids = {m.id for m in R.load_ranked_maps()}
    monkeypatch.setattr(M, "_engine",
                        _engine_with_map_games({m.id: 10 for m in R.load_ranked_maps()[:3]}))
    served = TestClient(M.app).get("/api/reference").json()["maps"]
    assert {m["id"] for m in served} <= all_ids
