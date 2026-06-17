"""Seat-aware draft search.

Minimax over the remaining 1-2-2-1 snake: we maximize our win-probability, the opponent
minimizes it (maximizes theirs), and completed 3v3 drafts are scored by the win-prob model.
Each node is pruned to the top-K candidates by a cheap empirical heuristic, and values are
memoized by (our set, their set) to collapse the many transpositions a snake draft produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set, Tuple

from bsdraft.constants import PICK_ORDER, TEAM_SIZE
from bsdraft.data import reference as R
from bsdraft.engine.scoring import _class_of, _name_map


@dataclass
class SearchRec:
    brawler_id: int
    name: str
    cls: str
    projected_winprob: float   # our win-prob after optimal continuation
    immediate: float           # cheap heuristic value of the pick itself


def _mean(values, default: float = 0.5) -> float:
    values = list(values)
    return sum(values) / len(values) if values else default


class SeatAwareSearch:
    def __init__(self, engine, branch: int = 5):
        self.stats = engine.stats
        self.model = engine.model
        self.branch = branch
        self.memo: Dict[Tuple[FrozenSet[int], FrozenSet[int]], float] = {}
        self.our_pool: List[int] = []
        self.their_pool: List[int] = []

    def _side_score(self, c: int, side: Set[int], other: Set[int], map_id: int) -> float:
        mw = self.stats.brawler_rate(c, map_id).winrate
        syn = _mean(self.stats.synergy(c, a).winrate for a in side)
        cnt = _mean(self.stats.counter(c, e).winrate for e in other)
        return 0.5 * mw + 0.25 * syn + 0.25 * cnt

    def _topk(self, available: List[int], side: Set[int], other: Set[int], map_id: int) -> List[int]:
        scored = sorted(((self._side_score(c, side, other, map_id), c) for c in available), reverse=True)
        return [c for _, c in scored[: self.branch]]

    def _value(self, our: Set[int], their: Set[int], map_id: int, mode: str, our_index: int) -> float:
        if len(our) + len(their) >= 2 * TEAM_SIZE:
            if self.model and self.model.available:
                return self.model.prob(list(our), list(their), map_id, mode)
            return 0.5
        key = (frozenset(our), frozenset(their))
        if key in self.memo:
            return self.memo[key]

        total = len(our) + len(their)
        us_turn = PICK_ORDER[total] == our_index
        if us_turn and len(our) >= TEAM_SIZE:
            us_turn = False
        elif not us_turn and len(their) >= TEAM_SIZE:
            us_turn = True

        if us_turn:
            available = [b for b in self.our_pool if b not in our and b not in their]
            cands = self._topk(available, our, their, map_id)
            val = max(self._value(our | {c}, their, map_id, mode, our_index) for c in cands)
        else:
            available = [b for b in self.their_pool if b not in our and b not in their]
            cands = self._topk(available, their, our, map_id)
            val = min(self._value(our, their | {c}, map_id, mode, our_index) for c in cands)
        self.memo[key] = val
        return val

    def recommend(self, state, top: int = 8, roster=None) -> List[SearchRec]:
        self.memo.clear()
        our_index = 0 if state.we_pick_first else 1
        all_pool = [b.id for b in R.load_brawlers() if b.id not in state.bans]
        self.their_pool = all_pool
        self.our_pool = [b for b in all_pool if roster is None or b in roster]
        our0, their0 = set(state.our_team), set(state.their_team)
        available = [b for b in self.our_pool if b not in our0 and b not in their0]
        cands = self._topk(available, our0, their0, state.map_id)

        recs: List[SearchRec] = []
        for c in cands:
            value = self._value(our0 | {c}, their0, state.map_id, state.mode, our_index)
            recs.append(SearchRec(c, _name_map().get(c, str(c)), _class_of(c),
                                  value, self._side_score(c, our0, their0, state.map_id)))
        recs.sort(key=lambda r: r.projected_winprob, reverse=True)
        return recs[:top]


def _demo() -> None:
    import time
    from bsdraft.engine.engine import DraftEngine
    from bsdraft.engine.state import DraftState
    from bsdraft.engine.stats import DraftStats
    from bsdraft.models.serve import WinProbModel

    stats = DraftStats()
    engine = DraftEngine(stats, WinProbModel())
    by_name = {b.name.lower(): b.id for b in R.load_brawlers()}
    bb = max((m for m in R.load_ranked_maps() if m.mode == "Brawl Ball"),
             key=lambda m: stats.map_games.get(m.id, 0))

    for label, st in [
        ("first pick (empty board, heaviest)", DraftState(map_id=bb.id, mode=bb.mode, we_pick_first=True)),
        ("mid-draft (we pick last vs Edgar+Mortis, ally Gene)",
         DraftState(map_id=bb.id, mode=bb.mode, we_pick_first=False,
                    our_team=[by_name["gene"]], their_team=[by_name["edgar"], by_name["mortis"]],
                    bans=[by_name["spike"], by_name["surge"]])),
    ]:
        search = SeatAwareSearch(engine)
        t0 = time.time()
        recs = search.recommend(st, top=5)
        dt = time.time() - t0
        print(f"\n{label}  ({dt*1000:.0f} ms, {len(search.memo)} states)")
        for r in recs:
            print(f"  {r.name:<14} {r.cls:<14} projected_winprob={r.projected_winprob:.3f}  immediate={r.immediate:.3f}")


if __name__ == "__main__":
    _demo()
