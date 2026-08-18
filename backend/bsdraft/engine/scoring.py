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

from bsdraft.constants import TEAM_SIZE
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

# Fusion weights. Rebalanced 2026-08-10 per the held-out ablation rerun on 995k matches
# (scripts/ablate_components.py + scripts/sweep_blend.py + docs/model-evaluation.md): the
# retrained net now out-discriminates the empirical blend (AUC .625 vs .608; the stacker
# gives it ~78% of the weight, a full reversal of the 40k-match era), so model .20 -> .40,
# funded by map .32 -> .25, synergy .15 -> .05 (its conditional coefficient is ~0 in every
# mode — redundant with the net), counter .23 -> .20. The .40/.25/.05/.20 blend was the
# best fixed candidate (beat the old weights in 200/200 bootstrap resamples, within .0005
# AUC of the linear refit ceiling). Per-map/mode weighting was re-tested and is still no
# better than these global weights, so they stay fixed.
#
# mastery/personal are NOT part of that ablation — it tunes only the objective signals
# (map/model/counter/synergy/role; see docs/model-evaluation.md). They personalize the
# roster seat, and only apply when a roster is loaded. Cut 2026-08-17 (mastery .25 -> .10,
# personal .20 -> .08): at the old values these two "how good is THIS player on it" signals
# were ~31% of a personalized pick's score (~37% before any enemy is drafted) — out-driving
# the model and outweighing `counter`, so the board over-recommended brawlers the player had
# already mastered and buried meta picks they could grow into. Now a nudge, not a driver
# (~15% combined). Product knobs (no accuracy claim); tune freely — a per-seat meta / my-roster
# toggle is the planned next step.
DEFAULT_WEIGHTS = {"map": 0.25, "model": 0.40, "counter": 0.20, "synergy": 0.05, "role": 0.10,
                   "mastery": 0.10, "personal": 0.08}


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


def _mean_wr(values: List[float]) -> float:
    """Bit-for-bit ``statistics.mean`` for the tiny ally/enemy winrate lists averaged per
    candidate. ``statistics.mean`` sums via exact ``Fraction``s — correct, but it was ~half of a
    mid-draft recommend call. For n<=2 the plain float form is provably identical on these inputs:
    a lone value is its own mean, and ``(a+b)/2`` is exact because halving a normal double never
    rounds and winrates live in (0, 1) so ``a+b`` can't overflow. n>=3 keeps ``statistics.mean``,
    where a sequential ``sum(...)/n`` would double-round and diverge."""
    n = len(values)
    if n == 1:
        return values[0]
    if n == 2:
        return (values[0] + values[1]) / 2.0
    return mean(values)


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


def model_marginals(state: DraftState, candidates: List[int], model, stats) -> List[Optional[float]]:
    """Vectorized :func:`model_marginal` over a candidate list — one batched model call instead
    of ~100 per-candidate ones. Every candidate shares the same enemy team and differs only by
    the ally added to our side, so a partial-draft model scores them all in a single pass.

    Bit-for-bit identical to ``[model_marginal(state, c, model, stats) for c in candidates]``:
    the batched path goes through :meth:`WinProbModel.prob_marginals`, which reproduces each
    board's per-candidate accumulation exactly. Falls back to the per-candidate loop for legacy
    (non-partial) models — whose marginal completes each candidate's team differently — and for
    any model lacking ``prob_marginals`` (e.g. a test double)."""
    if model is None or not getattr(model, "available", False):
        return [None] * len(candidates)
    if getattr(model, "supports_partial", False) and hasattr(model, "prob_marginals"):
        their = state.their_team[:TEAM_SIZE]
        teams_a = [(state.our_team + [c])[:TEAM_SIZE] for c in candidates]
        teams_b = [their] * len(candidates)
        return model.prob_marginals(teams_a, teams_b, state.map_id, state.mode)
    return [model_marginal(state, c, model, stats) for c in candidates]


def model_marginal(state: DraftState, candidate: int, model, stats) -> Optional[float]:
    """Win-prob of the draft so far with `candidate` added to our side.

    Partial-draft models (``supports_partial``) score the unfinished board directly — they
    were trained on masked comps, so unknown slots marginalize over how real drafts
    continued. Legacy artifacts can only judge full 3v3s, so both teams are completed with
    the map's top empirical picks; the same completion across candidates ranks fairly."""
    if model is None or not getattr(model, "available", False):
        return None
    if getattr(model, "supports_partial", False):
        # Cap both sides at team size: the frontend requests recommendations even when our
        # side is already full (e.g. the enemy's last pick is pending, or the draft is done),
        # where the candidate can't actually join — every candidate then scores the same
        # finished board, exactly as the legacy completion's [:size] truncation behaved.
        our = (state.our_team + [candidate])[:TEAM_SIZE]
        return model.prob(our, state.their_team[:TEAM_SIZE], state.map_id, state.mode)
    exclude = state.picked_or_banned()
    pool = [bid for bid, _ in stats.top_brawlers(state.map_id, n=40, min_games=3)]
    our = _complete_team(state.our_team + [candidate], pool, 3, exclude | {candidate})
    their = _complete_team(state.their_team, pool, 3, exclude | set(our))
    return model.prob(our, their, state.map_id, state.mode)


_UNSET = object()  # sentinel: "no precomputed win_prob supplied" (distinct from a real None)


def score_candidate(state: DraftState, candidate: int, stats, model=None, weights=None,
                    roster=None, personal=None, *, win_prob=_UNSET) -> PickScore:
    weights = weights or DEFAULT_WEIGHTS
    cls = _class_of(candidate)

    map_rate = stats.brawler_rate(candidate, state.map_id)
    synergy = (
        _mean_wr([stats.synergy(candidate, a).winrate for a in state.our_team])
        if state.our_team else None
    )
    counter = (
        _mean_wr([stats.counter(candidate, e).winrate for e in state.their_team])
        if state.their_team else None
    )
    rfit = role_fit(state, cls)
    # Confidence-scale the role term toward neutral 0.5 by *this candidate's own* per-(brawler,
    # map) map-data confidence. `role_fit` is a hand-set mode+class PRIOR (unvalidated by the
    # ablation), so it should yield to real per-map outcomes where we have them and only speak up
    # where the map cell is thin:
    #   role_eff = 0.5 + (1 - conf) * (role_fit - 0.5),  conf = map_rate.confidence  (games/(games+PRIOR))
    # On a well-sampled map cell (conf -> 1) role_eff -> 0.5 for *every* candidate: a constant that
    # drops straight out of the ranking (renormalization keeps its weight, but a flat term can't
    # re-order picks). On a zero/thin-data map (conf -> 0) role_eff -> the full archetype fit, so a
    # freshly-rotated map still leans on the prior. This fixes role punching ~2.5x above its 0.10
    # weight on the maps players actually see — its wide 0.5-0.9 spread vs map_wr's compressed
    # ~0.47-0.59 band made a 0.10-weight heuristic co-equal (~29% of ranking spread) with the
    # 0.25-weight *empirical* map signal, flat-topping mode-archetype brawlers (e.g. a Controller
    # ranking ~#4 on every Gem Grab map regardless of its true per-map win-rate). Confidence is
    # keyed per (brawler, map) — the same cell as map_wr — because role is a stand-in for *this
    # brawler's* unobserved map performance, so it should retreat exactly as that brawler's own map
    # sample fills in. Applies uniformly to blind first picks and mid-draft (map cell only; board
    # state doesn't change map_rate), so both recommend paths get the treatment.
    role_eff = 0.5 + (1.0 - map_rate.confidence) * (rfit - 0.5)
    # `win_prob` may be precomputed by the batched recommender (model_marginals); otherwise
    # fall back to the per-candidate marginal. A precomputed None (model off) is respected.
    win_prob = model_marginal(state, candidate, model, stats) if win_prob is _UNSET else win_prob

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

    # `parts["role"]` carries the confidence-scaled `role_eff` (what actually moves the score and
    # shows in `breakdown`); the raw `role_fit` field below keeps the un-scaled archetype value.
    parts: Dict[str, tuple] = {"map": (map_rate.winrate, weights["map"]), "role": (role_eff, weights["role"])}
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
