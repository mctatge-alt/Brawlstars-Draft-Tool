"""Unit tests for the pick-scoring signal fusion (bsdraft.engine.scoring).

Guards the 2026-08-17 mastery/personal de-weight: a clearly stronger *meta* pick must
outrank a maxed-out *comfort* pick, so the personalized board stops burying meta brawlers the
player hasn't personally mastered. Stub stats keep the assertions off the live dataset.

    PYTHONPATH=backend python -m pytest backend/tests/test_scoring.py
"""
from __future__ import annotations

from bsdraft.engine.scoring import DEFAULT_WEIGHTS, score_candidate
from bsdraft.engine.state import DraftState

# Shelly and Colt are both Damage Dealers, so role_fit is identical for the two candidates
# and cancels out of every head-to-head below — the comparisons turn purely on map/counter/mastery.
SHELLY, COLT, BULL = 16000000, 16000001, 16000002
MODE, MAP = "Brawl Ball", 15000001

# The pre-2026-08-17 weights, kept so a test can show the ranking they used to produce.
OLD_WEIGHTS = {**DEFAULT_WEIGHTS, "mastery": 0.25, "personal": 0.20}


class _Rate:
    def __init__(self, winrate, confidence=1.0, games=100):
        self.winrate = winrate
        self.confidence = confidence
        self.games = games


class _StubStats:
    """Fixed per-brawler map win rate and counter rate; synergy neutral, no model completion."""

    def __init__(self, map_rates, counter_rates):
        self._map = map_rates
        self._counter = counter_rates

    def brawler_rate(self, bid, map_id):
        return _Rate(self._map.get(bid, 0.5))

    def synergy(self, a, b):
        return _Rate(0.5)

    def counter(self, a, b):
        return _Rate(self._counter.get(a, 0.5))

    def top_brawlers(self, map_id, n=40, min_games=3):
        return []


class _Mastery:
    def __init__(self, score):
        self.score = score

    def gaps(self):
        return []


def _score(cand, stats, roster, weights):
    # One enemy drafted so the `counter` signal is live (it's absent from an empty board).
    state = DraftState(map_id=MAP, mode=MODE, our_team=[], their_team=[BULL])
    return score_candidate(state, cand, stats, model=None, weights=weights, roster=roster).score


def test_strong_meta_beats_maxed_comfort_under_new_weights():
    # SHELLY: strong on the map and into the enemy, but the player has never built her (mastery 0).
    # COLT: weak on both, but maxed mastery. The meta pick must now win.
    stats = _StubStats(map_rates={SHELLY: 0.70, COLT: 0.45},
                       counter_rates={SHELLY: 0.70, COLT: 0.45})
    roster = {SHELLY: _Mastery(0.0), COLT: _Mastery(1.0)}
    shelly = _score(SHELLY, stats, roster, DEFAULT_WEIGHTS)
    colt = _score(COLT, stats, roster, DEFAULT_WEIGHTS)
    assert shelly > colt, f"meta pick should win now (shelly={shelly:.3f} colt={colt:.3f})"


def test_old_weights_would_have_inverted_it():
    # The exact scenario above, scored with the old mastery .25 / personal .20 weights: the comfort
    # pick wins. This is the behavior the de-weight fixes, pinned so a future re-bump is caught.
    stats = _StubStats(map_rates={SHELLY: 0.70, COLT: 0.45},
                       counter_rates={SHELLY: 0.70, COLT: 0.45})
    roster = {SHELLY: _Mastery(0.0), COLT: _Mastery(1.0)}
    assert _score(COLT, stats, roster, OLD_WEIGHTS) > _score(SHELLY, stats, roster, OLD_WEIGHTS)


def test_mastery_still_breaks_ties():
    # De-weighted, not removed: with the objective signals equal, the higher-mastery brawler
    # still edges ahead, so personalization is a nudge rather than gone.
    stats = _StubStats(map_rates={SHELLY: 0.55, COLT: 0.55},
                       counter_rates={SHELLY: 0.55, COLT: 0.55})
    roster = {SHELLY: _Mastery(0.8), COLT: _Mastery(0.2)}
    assert _score(SHELLY, stats, roster, DEFAULT_WEIGHTS) > _score(COLT, stats, roster, DEFAULT_WEIGHTS)
