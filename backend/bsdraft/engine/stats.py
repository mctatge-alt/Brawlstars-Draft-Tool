"""Empirical draft statistics from collected matches.

Per-map and global brawler win/use rates, pair synergies, and matchup counters, each with
Bayesian (shrink-to-0.5) smoothing so thin samples aren't over-trusted. This is the
robust, interpretable signal the engine fuses with the learned model, and the source of
the "why this pick" explanations and confidence indicators.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from bsdraft.data import reference as R
from bsdraft.data.dataset import iter_matches

PRIOR = 20.0  # pseudo-games at 0.5 used for smoothing


@dataclass
class Rate:
    games: int
    winrate: float       # smoothed toward 0.5
    raw_winrate: float
    confidence: float    # games / (games + PRIOR)  in [0, 1)


def _rate(wins: int, games: int, prior: float = PRIOR) -> Rate:
    raw = wins / games if games else 0.5
    smoothed = (wins + prior / 2) / (games + prior)
    return Rate(games=games, winrate=smoothed, raw_winrate=raw, confidence=games / (games + prior))


class DraftStats:
    def __init__(self, matches: Optional[Iterable[dict]] = None):
        self.n = 0
        self.b_games: dict = defaultdict(int)
        self.b_wins: dict = defaultdict(int)
        self.bm_games: dict = defaultdict(int)   # (map_id, brawler)
        self.bm_wins: dict = defaultdict(int)
        self.map_games: dict = defaultdict(int)
        self.syn_games: dict = defaultdict(int)  # frozenset{b1, b2}
        self.syn_wins: dict = defaultdict(int)
        self.cnt_games: dict = defaultdict(int)  # (attacker, defender)
        self.cnt_wins: dict = defaultdict(int)
        self._build(matches if matches is not None else iter_matches())

    def _build(self, matches: Iterable[dict]) -> None:
        for r in matches:
            won = r.get("a_won")
            if won is None:
                continue
            a = [p["brawler_id"] for p in r["team_a"]]
            b = [p["brawler_id"] for p in r["team_b"]]
            mid = r.get("map_id")
            self.n += 1
            self.map_games[mid] += 1
            for team, win in ((a, won), (b, not won)):
                for x in team:
                    self.b_games[x] += 1
                    self.bm_games[(mid, x)] += 1
                    if win:
                        self.b_wins[x] += 1
                        self.bm_wins[(mid, x)] += 1
                for i in range(len(team)):
                    for j in range(i + 1, len(team)):
                        key = frozenset((team[i], team[j]))
                        self.syn_games[key] += 1
                        if win:
                            self.syn_wins[key] += 1
            for x in a:
                for y in b:
                    self.cnt_games[(x, y)] += 1
                    self.cnt_games[(y, x)] += 1
                    if won:
                        self.cnt_wins[(x, y)] += 1
                    else:
                        self.cnt_wins[(y, x)] += 1

    # --- accessors ---
    def brawler_rate(self, brawler_id: int, map_id: Optional[int] = None) -> Rate:
        if map_id is None:
            return _rate(self.b_wins[brawler_id], self.b_games[brawler_id])
        return _rate(self.bm_wins[(map_id, brawler_id)], self.bm_games[(map_id, brawler_id)])

    def use_rate(self, brawler_id: int, map_id: int) -> float:
        games = self.map_games.get(map_id, 0)
        return self.bm_games[(map_id, brawler_id)] / games if games else 0.0

    def synergy(self, b1: int, b2: int) -> Rate:
        key = frozenset((b1, b2))
        return _rate(self.syn_wins[key], self.syn_games[key])

    def counter(self, attacker: int, defender: int) -> Rate:
        return _rate(self.cnt_wins[(attacker, defender)], self.cnt_games[(attacker, defender)])

    def top_brawlers(self, map_id: Optional[int] = None, n: int = 15, min_games: int = 10) -> List[Tuple[int, Rate]]:
        rows = []
        for brawler_id in R.brawler_index():
            rate = self.brawler_rate(brawler_id, map_id)
            if rate.games >= min_games:
                rows.append((brawler_id, rate))
        rows.sort(key=lambda t: t[1].winrate, reverse=True)
        return rows[:n]


def _name(brawler_id: int) -> str:
    idx = {b.id: b.name for b in R.load_brawlers()}
    return idx.get(brawler_id, str(brawler_id))


if __name__ == "__main__":
    stats = DraftStats()
    names = {b.id: b.name for b in R.load_brawlers()}
    map_names = {m.id: (m.name, m.mode) for m in R.load_ranked_maps()}
    print(f"matches: {stats.n}")

    print("\nGlobal top 10 brawlers (smoothed win-rate):")
    for bid, rt in stats.top_brawlers(n=10, min_games=50):
        print(f"  {names.get(bid, bid):<16} wr={rt.winrate:.3f}  games={rt.games}")

    busiest = max(stats.map_games, key=stats.map_games.get)
    mname, mmode = map_names.get(busiest, ("?", "?"))
    print(f"\nTop 10 on most-played map: {mname} ({mmode}, {stats.map_games[busiest]} games):")
    for bid, rt in stats.top_brawlers(map_id=busiest, n=10, min_games=8):
        ur = stats.use_rate(bid, busiest)
        print(f"  {names.get(bid, bid):<16} wr={rt.winrate:.3f}  use={ur:.0%}  games={rt.games}")
