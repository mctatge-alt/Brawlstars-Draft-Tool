"""Draft engine: ban recommendation, candidate pick recommendation, composition meter.

(Seat-aware snake-draft lookahead and the mastery layer build on this next.)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from bsdraft.constants import BRAWLER_CLASSES
from bsdraft.data import reference as R
from bsdraft.engine.scoring import PickScore, _class_of, _name_map, score_candidate
from bsdraft.engine.state import DraftState
from bsdraft.engine.stats import DraftStats
from bsdraft.engine import composition as composition_mod
from bsdraft.engine import gameplan as gameplan_mod
from bsdraft.engine.search import SeatAwareSearch
from bsdraft.models.serve import WinProbModel


@dataclass
class BanScore:
    brawler_id: int
    name: str
    cls: str
    threat: float
    map_winrate: float
    use_rate: float
    confidence: float


class DraftEngine:
    def __init__(self, stats: Optional[DraftStats] = None, model: Optional[WinProbModel] = None,
                 bracket_stats: Optional[Dict[str, DraftStats]] = None):
        self.stats = stats if stats is not None else DraftStats()
        self.bracket_stats: Dict[str, DraftStats] = bracket_stats or {}
        self.model = model
        self.roster = None        # dict[brawler_id, Mastery] when a player roster is loaded
        self.roster_name = ""

    def _stats_for(self, state: DraftState) -> DraftStats:
        """The requested rank-bracket table when it exists, else the global stats."""
        return self.bracket_stats.get(state.rank_bracket, self.stats)

    def candidates(self, state: DraftState, roster=None) -> List[int]:
        used = state.picked_or_banned()
        ids = [b.id for b in R.load_brawlers() if b.id not in used]
        if roster is not None:
            ids = [i for i in ids if i in roster]  # only brawlers the player owns
        return ids

    def recommend_bans(self, state: DraftState, top: int = 6) -> List[BanScore]:
        """Rank brawlers by threat on this map: strong win-rate, weighted up if contested."""
        used = state.picked_or_banned()
        stats = self._stats_for(state)
        rows = []
        for b in R.load_brawlers():
            if b.id in used:
                continue
            rate = stats.brawler_rate(b.id, state.map_id)
            use = stats.use_rate(b.id, state.map_id)
            threat = 0.85 * rate.winrate + 0.15 * min(1.0, use * 3.0)
            rows.append(BanScore(b.id, _name_map().get(b.id, str(b.id)), b.cls,
                                 threat, rate.winrate, use, rate.confidence))
        rows.sort(key=lambda r: r.threat, reverse=True)
        return rows[:top]

    def recommend_picks(self, state: DraftState, top: int = 10, weights=None, roster=None,
                        personal=None) -> List[PickScore]:
        stats = self._stats_for(state)
        scored = [score_candidate(state, c, stats, self.model, weights, roster, personal)
                  for c in self.candidates(state, roster)]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top]

    def search_recommend(self, state: DraftState, top: int = 8, branch: int = 5, roster=None):
        """Seat-aware recommendation via minimax lookahead over the remaining snake."""
        return SeatAwareSearch(self, branch=branch).recommend(state, top=top, roster=roster)

    def composition_report(self, state: DraftState) -> dict:
        return composition_mod.analyze(state)

    def game_plan(self, state: DraftState) -> dict:
        return gameplan_mod.game_plan(state)

    def composition(self, state: DraftState) -> dict:
        counts = Counter(_class_of(b) for b in state.our_team)
        return {cls: counts.get(cls, 0) for cls in BRAWLER_CLASSES if counts.get(cls, 0)}


def _demo() -> None:
    stats = DraftStats()
    model = WinProbModel()
    engine = DraftEngine(stats, model)
    by_name = {b.name.lower(): b.id for b in R.load_brawlers()}

    bb_maps = [m for m in R.load_ranked_maps() if m.mode == "Brawl Ball" and stats.map_games.get(m.id, 0) > 0]
    mp = max(bb_maps, key=lambda m: stats.map_games.get(m.id, 0))
    print(f"Map: {mp.name} ({mp.mode})   model={'ON' if model.available else 'OFF'}   "
          f"games={stats.map_games.get(mp.id, 0)}")

    print("\nTop ban suggestions (deny strongest map brawlers):")
    for b in engine.recommend_bans(DraftState(map_id=mp.id, mode=mp.mode), top=6):
        print(f"  {b.name:<14} {b.cls:<14} threat={b.threat:.3f} map_wr={b.map_winrate:.3f} "
              f"use={b.use_rate:.0%} conf={b.confidence:.2f}")

    def show(title, st):
        print(f"\n{title}")
        for s in engine.recommend_picks(st, top=6):
            extra = ""
            if s.synergy is not None:
                extra += f" syn={s.synergy:.3f}"
            if s.counter is not None:
                extra += f" cnt={s.counter:.3f}"
            if s.win_prob is not None:
                extra += f" model={s.win_prob:.3f}"
            print(f"  {s.name:<14} {s.cls:<14} score={s.score:.3f} map_wr={s.map_winrate:.3f} "
                  f"role={s.role_fit:.2f}{extra} conf={s.confidence:.2f}")

    show("First-pick suggestions (nothing on the board):",
         DraftState(map_id=mp.id, mode=mp.mode, we_pick_first=True))

    enemies = [by_name.get(n) for n in ("edgar", "mortis")]
    allies = [by_name.get("gene")]
    bans = [by_name.get(n) for n in ("spike", "surge")]
    st2 = DraftState(
        map_id=mp.id, mode=mp.mode, we_pick_first=False,
        our_team=[a for a in allies if a], their_team=[e for e in enemies if e],
        bans=[b for b in bans if b],
    )
    show("We pick (ally: Gene | enemies: Edgar, Mortis | banned: Spike, Surge):", st2)
    print("\nOur composition:", engine.composition(st2) or "(empty)")


if __name__ == "__main__":
    _demo()
