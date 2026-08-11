"""The crawler must survive a network outage mid-cycle instead of crashing the daemon.

Regression guard for the connectivity-stall bug: the API client re-raises ``httpx`` transport
errors after its retry budget, and ``Crawler.run`` used to catch only ``BrawlStarsError`` — so a
DNS/connection drop propagated out, killed the process (launchd then restarted it with a full
state reload), and, because tags were marked scanned *before* the fetch, poisoned the revisit
schedule for anything popped during the outage.
"""
import asyncio

import httpx
import pytest

from bsdraft.collect import crawler as C
from bsdraft.collect.client import BrawlStarsError


@pytest.fixture
def isolated_state(monkeypatch, tmp_path):
    """Point the crawler's state files at an empty tmp dir (never the real ~1 GB dataset) and
    make backoff instant so the reconnect wait doesn't actually sleep."""
    monkeypatch.setattr(C, "MATCHES_PATH", tmp_path / "matches.jsonl")
    monkeypatch.setattr(C, "VISITED_PATH", tmp_path / "visited.txt")

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(C.asyncio, "sleep", _no_sleep)
    return tmp_path


class _DownClient:
    """Every request fails with a DNS-style transport error — the network is fully down."""

    async def get_battlelog(self, tag):
        raise httpx.ConnectError("nodename nor servname provided, or not known")

    async def get_top_players(self, country="global", limit=200):
        raise httpx.ConnectError("nodename nor servname provided, or not known")


class _FlakyClient:
    """First battlelog fetch drops (transport error); after that the network is back."""

    def __init__(self):
        self.battlelog_calls = 0

    async def get_battlelog(self, tag):
        self.battlelog_calls += 1
        if self.battlelog_calls == 1:
            raise httpx.ConnectError("nodename nor servname provided, or not known")
        return []  # recovered: a valid (empty) battle log — no matches, but a real response

    async def get_top_players(self, country="global", limit=200):
        return []  # reconnect probe succeeds -> connectivity restored


def test_transport_error_does_not_crash_or_poison(isolated_state):
    """A sustained outage ends the cycle cleanly: run() returns rather than raising, collects
    nothing, and does NOT mark the un-fetched tag scanned (so it stays eligible for retry)."""
    c = C.Crawler(_DownClient(), revisit_after=0)
    c.frontier.clear()
    c.queued.clear()
    c._enqueue("ABC")

    new = asyncio.run(c.run(target_matches=5))  # must not raise httpx.ConnectError

    assert new == 0
    assert "ABC" not in c.visited                      # not falsely marked scanned
    assert "ABC" in c.queued and "ABC" in c.frontier   # requeued for a real retry next cycle


def test_transport_error_recovers_and_resumes(isolated_state):
    """A brief blip is absorbed: after the reconnect probe succeeds the crawler retries the
    same tag, fetches it for real, and marks it scanned — all without crashing."""
    client = _FlakyClient()
    c = C.Crawler(client, revisit_after=0)
    c.frontier.clear()
    c.queued.clear()
    c._enqueue("ABC")

    new = asyncio.run(c.run(target_matches=5))

    assert new == 0                       # empty battle log -> no new matches, but no crash
    assert client.battlelog_calls >= 2    # it retried the tag after reconnecting
    assert "ABC" in c.visited             # and only marked it scanned once the fetch succeeded


def test_bad_tag_still_marked_scanned(isolated_state):
    """A per-tag BrawlStarsError (e.g. private profile / 404) is still marked scanned so we
    don't hammer it again — the transport-error handling must not regress that path."""

    class _NotFoundClient:
        async def get_battlelog(self, tag):
            raise BrawlStarsError(404, "not found")

        async def get_top_players(self, country="global", limit=200):
            return []

    c = C.Crawler(_NotFoundClient(), revisit_after=0)
    c.frontier.clear()
    c.queued.clear()
    c._enqueue("ABC")

    new = asyncio.run(c.run(target_matches=5))

    assert new == 0
    assert "ABC" in c.visited  # a real per-tag failure is recorded, as before
