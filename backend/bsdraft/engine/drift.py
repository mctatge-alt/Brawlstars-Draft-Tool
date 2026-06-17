"""Detect meta shifts from our own match data — no balance-change API to rely on.

Supercell's API exposes no patch version or balance feed (not even base brawler stats), so
we infer "the meta moved" from the matches we already collect, two complementary ways:

  * change detection — each brawler's win/pick rate in a recent window vs the window before
    it, flagged only when the shift is too large to be sampling noise (two-proportion
    z-test). Catches buffs and nerfs, including silent server-side tweaks that never appear
    in patch notes.
  * new-content detection — brawler ids that appear in play but aren't in our reference yet,
    i.e. a freshly released brawler — a reliable "an update shipped" marker.

Empirical stats are already recency-weighted (see :mod:`bsdraft.engine.stats`), so the meta
self-corrects within ~a half-life of any change on its own. This module exists to react
*faster*: it surfaces a ``shifted`` flag the crawl loop or a scheduled check can act on —
trigger a model retrain, briefly shorten the stats half-life, or mark affected brawlers
low-confidence in the UI.

    PYTHONPATH=backend python -m bsdraft.engine.drift      # report on data/raw/matches.jsonl
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from bsdraft.data import reference as R
from bsdraft.data.dataset import iter_matches

_DAY = 86400.0


@dataclass
class BrawlerShift:
    brawler_id: int
    name: str
    kind: str            # "buff" | "nerf" — direction of the win-rate move
    wr_before: float
    wr_after: float
    use_before: float
    use_after: float
    games_before: int
    games_after: int
    z: float             # two-proportion z for the win-rate change (>0 = rose)

    @property
    def wr_delta(self) -> float:
        return self.wr_after - self.wr_before


@dataclass
class MetaReport:
    shifted: bool
    recent_days: float
    prior_days: float
    n_recent: int
    n_prior: int
    new_brawlers: List[int] = field(default_factory=list)
    shifts: List[BrawlerShift] = field(default_factory=list)
    note: str = ""

    def summary(self) -> str:
        lines = [
            f"meta shifted: {self.shifted}",
            f"windows: recent {self.recent_days:g}d ({self.n_recent} matches) "
            f"vs prior {self.prior_days:g}d ({self.n_prior} matches)",
        ]
        if self.note:
            lines.append(f"note: {self.note}")
        if self.new_brawlers:
            names = ", ".join(_name(b) for b in self.new_brawlers)
            lines.append(f"new brawler(s) — an update shipped: {names}")
        if self.shifts:
            lines.append("significant win-rate shifts:")
            for s in self.shifts:
                arrow = "^" if s.z > 0 else "v"
                lines.append(
                    f"  {arrow} {s.name:<14} {s.kind:<4} "
                    f"wr {s.wr_before:.3f} -> {s.wr_after:.3f} ({s.wr_delta:+.3f})  "
                    f"use {s.use_before:.0%} -> {s.use_after:.0%}  "
                    f"z={s.z:+.1f}  n={s.games_before}/{s.games_after}"
                )
        elif not self.new_brawlers:
            lines.append("no significant shifts — meta stable over the compared windows")
        return "\n".join(lines)


def _name(brawler_id: int) -> str:
    return {b.id: b.name for b in R.load_brawlers()}.get(brawler_id, str(brawler_id))


def _two_prop_z(w1: int, n1: int, w2: int, n2: int) -> float:
    """z for H0: p1 == p2 (pooled-variance two-proportion test). Positive when the second
    window's rate is higher. ~|z|>=3 is a <0.3% two-sided fluke — conservative on purpose,
    since we test ~80 brawlers at once."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = w1 / n1, w2 / n2
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n2))
    return (p2 - p1) / se if se > 0 else 0.0


def _window_counts(rows: List[dict], lo: float, hi: float) -> Tuple[Dict[int, int], Dict[int, int], int]:
    """Per-brawler (games, wins) and match count, over matches with ts in [lo, hi)."""
    games: Dict[int, int] = defaultdict(int)
    wins: Dict[int, int] = defaultdict(int)
    n = 0
    for r in rows:
        ts = int(r.get("ts") or 0)
        if not (lo <= ts < hi):
            continue
        won = r["a_won"]
        n += 1
        for team, win in ((r["team_a"], won), (r["team_b"], not won)):
            for p in team:
                bid = p["brawler_id"]
                games[bid] += 1
                if win:
                    wins[bid] += 1
    return games, wins, n


def detect_new_content(observed_ids: Iterable[int], known_ids: Optional[Iterable[int]] = None) -> List[int]:
    """Brawler ids seen in play but missing from the known reference — a new release. Pass a
    freshly fetched /brawlers catalog as ``observed_ids`` for the cleanest signal; match-derived
    ids work offline too. ``known_ids`` defaults to the bundled reference."""
    known = set(known_ids) if known_ids is not None else set(R.brawler_index())
    return sorted(set(observed_ids) - known)


def detect_drift(
    matches: Optional[Iterable[dict]] = None,
    *,
    recent_days: float = 7.0,
    prior_days: float = 7.0,
    min_games: int = 30,
    z_threshold: float = 3.0,
) -> MetaReport:
    """Compare a recent window against the window before it and flag win-rate shifts too large
    to be noise, plus any newly released brawlers. Windows are anchored on the newest match,
    so the result is deterministic and independent of when it runs."""
    rows = [r for r in (matches if matches is not None else iter_matches()) if r.get("a_won") is not None]
    new_brawlers = detect_new_content(
        {p["brawler_id"] for r in rows for p in r["team_a"] + r["team_b"]}
    )
    tmax = max((int(r.get("ts") or 0) for r in rows), default=0)
    if tmax <= 0:
        return MetaReport(bool(new_brawlers), recent_days, prior_days, 0, 0,
                          new_brawlers=new_brawlers, note="no usable timestamps")

    recent_lo = tmax - recent_days * _DAY
    prior_lo = recent_lo - prior_days * _DAY
    rg, rw, n_recent = _window_counts(rows, recent_lo, tmax + 1)
    pg, pw, n_prior = _window_counts(rows, prior_lo, recent_lo)

    shifts: List[BrawlerShift] = []
    tested = 0
    for bid in R.brawler_index():
        gb, ga = pg.get(bid, 0), rg.get(bid, 0)
        if gb < min_games or ga < min_games:
            continue
        tested += 1
        z = _two_prop_z(pw.get(bid, 0), gb, rw.get(bid, 0), ga)
        if abs(z) >= z_threshold:
            shifts.append(BrawlerShift(
                brawler_id=bid, name=_name(bid),
                kind="buff" if z > 0 else "nerf",
                wr_before=pw.get(bid, 0) / gb, wr_after=rw.get(bid, 0) / ga,
                use_before=gb / n_prior if n_prior else 0.0,
                use_after=ga / n_recent if n_recent else 0.0,
                games_before=gb, games_after=ga, z=z,
            ))
    shifts.sort(key=lambda s: abs(s.z), reverse=True)

    note = ""
    if tested < 10:
        note = (f"low statistical power — only {tested} brawlers had >= {min_games} games in "
                f"BOTH windows; reliable detection needs steady daily crawling")

    return MetaReport(
        shifted=bool(shifts or new_brawlers),
        recent_days=recent_days, prior_days=prior_days,
        n_recent=n_recent, n_prior=n_prior,
        new_brawlers=new_brawlers, shifts=shifts, note=note,
    )


if __name__ == "__main__":
    print(detect_drift().summary())
