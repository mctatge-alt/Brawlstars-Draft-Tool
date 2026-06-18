"""Score candidate picks by fusing empirical stats, the learned model, role-fit, synergy,
and counters — each kept as a transparent, win-rate-like component for explainability.

The fused score is a re-normalized weighted average over the *active* signals (synergy
only counts once you have allies, counters once the enemy has revealed picks), so early
and late picks are scored on what's actually known.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from statistics import mean
from typing import Dict, List, Optional

from bsdraft.data import reference as R
from bsdraft.engine.state import DraftState

# How much each mode rewards each class (heuristic, tunable). Absent classes default to 0.5.
MODE_CLASS_PREF: Dict[str, Dict[str, float]] = {
    "Heist":      {"Damage Dealer": 0.90, "Marksman": 0.70, "Assassin": 0.60, "Tank": 0.50},
    "Bounty":     {"Marksman": 0.90, "Controller": 0.80, "Artillery": 0.60},
    "Brawl Ball": {"Tank": 0.90, "Assassin": 0.80, "Support": 0.60, "Damage Dealer": 0.60},
    "Knockout":   {"Marksman": 0.90, "Controller": 0.80, "Artillery": 0.60, "Support": 0.50},
    "Gem Grab":   {"Controller": 0.80, "Support": 0.70, "Tank": 0.60, "Damage Dealer": 0.60},
    "Hot Zone":   {"Controller": 0.90, "Support": 0.70, "Artillery": 0.70, "Damage Dealer": 0.60},
}
DEFAULT_PREF = 0.5

# Fusion weights. map/counter rebalanced .40/.15 -> .32/.23 per the held-out ablation
# (scripts/ablate_context.py + docs/model-evaluation.md): counter is the strongest
# head-to-head signal and was under-weighted, map slightly over-weighted. Per-map/mode
# weighting was tested and found no better than these global weights, so they stay fixed.
DEFAULT_WEIGHTS = {"map": 0.32, "model": 0.20, "counter": 0.23, "synergy": 0.15, "role": 0.10,
                   "mastery": 0.25, "personal": 0.20}


@dataclass
class PickScore:
    brawler_id: int
    name: str
    cls: str
    score: float
    map_winrate: float
    synergy: Optional[float]
    counter: Optional[float]
    role_fit: float
    win_prob: Optional[float]
    confidence: float
    mastery: Optional[float] = None
    personal_winrate: Optional[float] = None  # this player's own win rate w/ the brawler
    personal_games: Optional[float] = None     # their effective sample (recency-weighted)
    owned: bool = True
    gaps: List[str] = field(default_factory=list)
    breakdown: Dict[str, float] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _class_map() -> Dict[int, str]:
    return {b.id: b.cls for b in R.load_brawlers()}


@lru_cache(maxsize=1)
def _name_map() -> Dict[int, str]:
    return {b.id: b.name for b in R.load_brawlers()}


def _class_of(brawler_id: int) -> str:
    return _class_map().get(brawler_id, "Unclassified")


def _complete_team(base: List[int], pool: List[int], size: int, exclude: set) -> List[int]:
    team = list(base)
    for bid in pool:
        if len(team) >= size:
            break
        if bid in team or bid in exclude:
            continue
        team.append(bid)
    if len(team) < size:  # fall back to any unused brawler
        for b in R.load_brawlers():
            if len(team) >= size:
                break
            if b.id in team or b.id in exclude:
                continue
            team.append(b.id)
    return team[:size]


def role_fit(state: DraftState, cls: str) -> float:
    pref = MODE_CLASS_PREF.get(state.mode, {}).get(cls, DEFAULT_PREF)
    redundancy = [_class_of(b) for b in state.our_team].count(cls)
    return max(0.0, min(1.0, pref - 0.15 * redundancy))


def model_marginal(state: DraftState, candidate: int, model, stats) -> Optional[float]:
    """Full-comp win-prob with `candidate` on our side, completing both teams with the
    map's top empirical picks. Same completion across candidates, so it ranks fairly."""
    if model is None or not getattr(model, "available", False):
        return None
    exclude = state.picked_or_banned()
    pool = [bid for bid, _ in stats.top_brawlers(state.map_id, n=40, min_games=3)]
    our = _complete_team(state.our_team + [candidate], pool, 3, exclude | {candidate})
    their = _complete_team(state.their_team, pool, 3, exclude | set(our))
    return model.prob(our, their, state.map_id, state.mode)


def score_candidate(state: DraftState, candidate: int, stats, model=None, weights=None,
                    roster=None, personal=None) -> PickScore:
    weights = weights or DEFAULT_WEIGHTS
    cls = _class_of(candidate)

    map_rate = stats.brawler_rate(candidate, state.map_id)
    synergy = (
        mean(stats.synergy(candidate, a).winrate for a in state.our_team)
        if state.our_team else None
    )
    counter = (
        mean(stats.counter(candidate, e).winrate for e in state.their_team)
        if state.their_team else None
    )
    rfit = role_fit(state, cls)
    win_prob = model_marginal(state, candidate, model, stats)

    mastery_val: Optional[float] = None
    owned = True
    gaps: List[str] = []
    if roster is not None:
        m = roster.get(candidate)
        owned = m is not None
        if m is not None:
            mastery_val = m.score
            gaps = m.gaps()

    # The player's own win rate with this brawler (on this map when they've played it there).
    # Only counts once they've actually played it; its weight scales with confidence, so a
    # thin personal sample nudges gently and a deep one speaks up.
    personal_wr: Optional[float] = None
    personal_games: Optional[float] = None
    personal_weight = 0.0
    if personal is not None:
        pr = personal.brawler_rate(candidate, state.map_id)
        if pr.games > 0:
            personal_wr = pr.winrate
            personal_games = pr.games
            personal_weight = weights.get("personal", 0.0) * pr.confidence

    parts: Dict[str, tuple] = {"map": (map_rate.winrate, weights["map"]), "role": (rfit, weights["role"])}
    if synergy is not None:
        parts["synergy"] = (synergy, weights["synergy"])
    if counter is not None:
        parts["counter"] = (counter, weights["counter"])
    if win_prob is not None:
        parts["model"] = (win_prob, weights["model"])
    if mastery_val is not None:
        parts["mastery"] = (mastery_val, weights["mastery"])
    if personal_wr is not None and personal_weight > 0:
        parts["personal"] = (personal_wr, personal_weight)

    wsum = sum(w for _, w in parts.values())
    score = sum(v * w for v, w in parts.values()) / wsum

    return PickScore(
        brawler_id=candidate,
        name=_name_map().get(candidate, str(candidate)),
        cls=cls,
        score=score,
        map_winrate=map_rate.winrate,
        synergy=synergy,
        counter=counter,
        role_fit=rfit,
        win_prob=win_prob,
        confidence=map_rate.confidence,
        mastery=mastery_val,
        personal_winrate=personal_wr,
        personal_games=personal_games,
        owned=owned,
        gaps=gaps,
        breakdown={k: round(v, 3) for k, (v, _) in parts.items()},
    )
