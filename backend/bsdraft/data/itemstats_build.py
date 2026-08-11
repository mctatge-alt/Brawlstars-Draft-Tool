"""Build the per-item win-rate table via single-item-owner inference.

Battle logs never record the equipped gadget/star power/gear, so per-item win rates can't be
measured directly. This estimates them the way community stats sites do: among players who own
*exactly one* item of a type on a brawler, attribute that brawler's matches to their one owned
item, and contrast it against the brawler's OTHER single-item owners (an investment-matched
baseline — both cohorts own exactly one item, differing only in *which*). Skill is controlled by
stratifying on the appearance's Ranked tier (Mantel-Haenszel); grinder domination and the
correlated-rows problem are controlled by aggregating to one weighted record per player and using
a robust between-player variance; small cells are shrunk toward "no effect" and gated by a
BH-FDR-corrected significance test so the served ranking doesn't chase noise.

Home-only (needs the collected ownership profiles) and numpy-heavy — deliberately NOT on the serve
import path (the serve loader is :mod:`bsdraft.engine.itemstats`, pure stdlib). The estimator
functions are split out and pure so they can be unit-tested on synthetic data with known effects.

Design + adversarial review that shaped this (incl. the clustered-variance, no-double-count,
mirror-test-dedup, and empty-REST-guard fixes) live in docs/item-winrate.md.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from bsdraft.constants import REFERENCE_DIR
from bsdraft.data import reference as R
from bsdraft.data.dataset import iter_matches
from bsdraft.engine.itemstats import _norm
from bsdraft.engine.stats import _rate
from bsdraft.engine.tiers import bracket_of_tier

_DAY = 86400.0

# Estimator knobs (also written into the artifact meta, so retuning needs no code change).
DEFAULTS = dict(
    window_days=35.0,      # only matches within this window of the newest are used
    halflife_days=21.0,    # recency half-life inside the window (matches stats.py)
    K1=30.0,               # empirical-Bayes shrinkage strength for the effect (n_eff units)
    prior_rest=20.0,       # smoothing pseudo-count for the rest-anchored absolute rate
    cap_frac=0.05,         # cap any single player's share of a pool's weighted games
    strat_floor=10.0,      # a tier stratum needs this much n_eff in BOTH pools to enter MH
    n_min_eff=50.0,        # min Kish effective N (item pool AND rest pool) to be rankable
    n_min_players=30,      # min distinct single-owners of the item to be rankable
    fdr_q=0.05,            # Benjamini-Hochberg false-discovery rate
    max_abs_delta=0.15,    # implausibility guard: suppress |delta|>this when n_eff<200
)

_TYPES = ("gd", "sp", "gr")
_TYPE_LABEL = {"gd": "gadget", "sp": "star_power", "gr": "gear"}


@dataclass
class Pool:
    S1: float          # sum of per-player weighted games
    Wn: float          # sum of per-player weighted wins
    n_eff: float       # Kish effective sample size  S1^2 / sum(g^2)
    phat: float        # weighted win rate  Wn / S1
    var: float         # robust (clustered-by-player) variance of phat
    d: int             # distinct players


def cap_shares(g: np.ndarray, wn: np.ndarray, cap_frac: float, iters: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Down-scale any player whose weighted games exceed ``cap_frac`` of the pool total, iterating
    to a fixed point (the cap shifts as the total shrinks). Scales wins in lock-step so the
    player's win rate is preserved — this bounds one grinder's influence without dropping them."""
    g = np.asarray(g, dtype=float).copy()
    wn = np.asarray(wn, dtype=float).copy()
    if g.size == 0:
        return g, wn
    for _ in range(iters):
        total = g.sum()
        if total <= 0:
            break
        cap = cap_frac * total
        over = g > cap * (1 + 1e-9)
        if not over.any():
            break
        ratio = np.ones_like(g)
        ratio[over] = cap / g[over]
        g *= ratio
        wn *= ratio
    return g, wn


def pool_stats(g: np.ndarray, wn: np.ndarray) -> Pool:
    """Aggregate per-player weighted (games, wins) into a Pool. The independent unit is the PLAYER,
    not the game: n_eff is Kish's effective *player* count (S1^2 / sum(g^2)), so the many correlated
    games one player contributes don't masquerade as independent evidence — the deflation the
    adversarial review flagged as essential. The variance is the larger of the binomial-at-n_eff
    floor and the robust between-player (sandwich) variance: the floor is the i.i.d.-player minimum
    (design effect >= 1) and also keeps a homogeneous pool from claiming zero uncertainty, while the
    sandwich takes over when players are genuinely overdispersed."""
    g = np.asarray(g, dtype=float)
    wn = np.asarray(wn, dtype=float)
    S1 = float(g.sum())
    if S1 <= 0:
        return Pool(0.0, 0.0, 0.0, 0.5, 0.0, 0)
    S2 = float((g * g).sum())
    Wn = float(wn.sum())
    d = int(g.size)
    n_eff = S1 * S1 / S2 if S2 > 0 else 0.0
    phat = Wn / S1
    binom = phat * (1.0 - phat) / n_eff if n_eff > 0 else 0.0
    if d >= 2:
        pp = wn / g                                  # per-player win rate (g>0 guaranteed by caller)
        sandwich = float((g * g * (pp - phat) ** 2).sum()) / (S1 * S1)
        var = max(sandwich, binom)
    else:
        var = binom
    return Pool(S1, Wn, n_eff, phat, var, d)


def mh_combine(strata: List[Tuple[Pool, Pool]]) -> Optional[Tuple[float, float, float]]:
    """Mantel-Haenszel risk difference of the item pool vs the rest pool across tier strata, with a
    variance built from the pools' robust per-stratum variances. Returns ``(RD, Var(RD), n_eff_item)``
    or None when no stratum has a usable contrast (empty rest pool / all thin) — the caller then
    routes the cell to the heuristic fallback rather than dividing by zero."""
    num = den = varnum = n1 = 0.0
    for si, sr in strata:
        nk = si.n_eff + sr.n_eff
        if nk <= 0:
            continue
        u = si.n_eff * sr.n_eff / nk
        num += u * (si.phat - sr.phat)
        den += u
        varnum += u * u * (si.var + sr.var)
        n1 += si.n_eff
    if den <= 0:
        return None
    return num / den, varnum / (den * den), n1


def bh_fdr(pvals: List[float], q: float) -> Tuple[List[bool], List[float]]:
    """Benjamini-Hochberg: return (significant flags, BH-adjusted q-values) for the p-value family.
    The family must already be de-duplicated (one hypothesis per unordered item pair) so mirror
    tests don't corrupt the rank thresholding."""
    m = len(pvals)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: pvals[i])
    kmax = 0
    for rank, idx in enumerate(order, 1):
        if pvals[idx] <= rank / m * q:
            kmax = rank
    sig = [False] * m
    for rank, idx in enumerate(order, 1):
        sig[idx] = rank <= kmax
    qv = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        prev = min(prev, pvals[idx] * m / rank)
        qv[idx] = min(prev, 1.0)
    return sig, qv


def _two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0)) if z == z else 1.0  # z!=z guards NaN


# ---- ownership profiles ----------------------------------------------------------------------

def load_profiles(path: Path) -> Tuple[Dict[str, dict], Dict[int, str]]:
    """Read profiles.jsonl into ``{tag: {brawler_id: {"gd":[ids], "sp":[ids], "gr":[ids]}}}`` (the
    newest row wins per tag), plus a ``{gear_id: name}`` map learned from the gear entries. Only the
    ids are kept for gadgets/star powers; gears keep both because their id has no catalog source."""
    profiles: Dict[str, dict] = {}
    order: Dict[str, int] = {}
    gear_names: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tag = rec.get("tag")
            ts = rec.get("ts", 0)
            if not tag or ts < order.get(tag, -1):
                continue
            order[tag] = ts
            brawlers: Dict[str, dict] = {}
            for bid, owned in (rec.get("b") or {}).items():
                gr_ids = []
                for g in owned.get("gr", []):
                    gid = g[0]
                    gr_ids.append(gid)
                    if len(g) >= 2 and g[1]:
                        gear_names[gid] = g[1]
                brawlers[bid] = {"gd": list(owned.get("gd", [])),
                                 "sp": list(owned.get("sp", [])),
                                 "gr": gr_ids}
            profiles[tag] = brawlers
    return profiles, gear_names


def _universal_gear_ids(gear_names: Dict[int, str]) -> Dict[int, str]:
    """Restrict the learned gear ids to the six universal 'Super Rare' gears (the curated,
    non-phased-out set), matched by normalized name to data/reference/gears.json. Returns
    ``{gear_id: canonical_name_from_guide}``."""
    try:
        doc = json.loads((REFERENCE_DIR / "gears.json").read_text(encoding="utf-8"))
        guide = {_norm(g["name"]): g["name"] for g in doc.get("gears", [])}
    except (OSError, ValueError, KeyError):
        guide = {}
    out: Dict[int, str] = {}
    for gid, name in gear_names.items():
        canon = guide.get(_norm(name))
        if canon:
            out[gid] = canon
    return out


# ---- the build -------------------------------------------------------------------------------

def build_itemstats(matches_path: Optional[Path], profiles_path: Path,
                    params: Optional[dict] = None, boosted_ids: Optional[Iterable[int]] = None) -> dict:
    """Join the recent matches with the ownership profiles and produce the item-stats payload. See
    the module docstring for the method. ``matches_path`` defaults to the synced dataset;
    ``boosted_ids`` defaults to the current season's free/boosted brawlers (injectable for tests)."""
    p = {**DEFAULTS, **(params or {})}
    profiles, gear_names = load_profiles(profiles_path)
    gear_id_to_name = _universal_gear_ids(gear_names)  # only the 6 universal gears are estimated
    universal_gear_ids = set(gear_id_to_name)
    boosted = set(boosted_ids) if boosted_ids is not None else set(R.load_ranked_boosted())

    # First pass: newest ts (for the window + recency origin).
    ts_max = 0
    for m in iter_matches(matches_path):
        ts_max = max(ts_max, m.get("ts", 0))
    if ts_max <= 0:
        return _empty_payload(p, gear_id_to_name)
    cutoff = ts_max - p["window_days"] * _DAY
    half = p["halflife_days"] * _DAY

    # acc[(X, T)][tier][tag] = [owned_item_id, weighted_games, weighted_wins]
    acc: Dict[Tuple[int, str], Dict[str, Dict[str, list]]] = {}
    none_acc: Dict[Tuple[int, str], List[float]] = {}   # (X,T) -> [S1, Wn] for owns-0 (secondary)
    brawler_acc: Dict[int, List[float]] = {}            # X -> [S1, Wn] over ALL appearances
    seen = {"rows": 0, "no_profile": 0}

    for m in iter_matches(matches_path):
        ts = m.get("ts", 0)
        if ts < cutoff:
            continue
        a_won = m.get("a_won")
        if a_won is None:               # drop draws/unlabeled (variance-consistent for the z test)
            continue
        r = 0.5 ** ((ts_max - ts) / half) if half > 0 else 1.0
        for team, won in ((m.get("team_a", []), a_won), (m.get("team_b", []), not a_won)):
            y = 1.0 if won else 0.0
            for rec in team:
                seen["rows"] += 1
                tag = rec.get("tag")
                X = rec.get("brawler_id")
                if X is None or tag is None:
                    continue
                ba = brawler_acc.setdefault(X, [0.0, 0.0])
                ba[0] += r
                ba[1] += r * y
                if X in boosted:         # boosted/free brawlers flood 0-owned players — flag, skip cells
                    continue
                prof = profiles.get(tag)
                if prof is None:
                    seen["no_profile"] += 1
                    continue
                owned_map = prof.get(str(X))
                if owned_map is None:
                    continue
                t = rec.get("trophies")
                # A null / non-int tier can't enter a skill stratum. Guard BEFORE bracket_of_tier —
                # it does `lo <= tier <= hi`, which TypeErrors (aborting the whole build) on a None,
                # and real match rows do carry null trophies.
                tier = bracket_of_tier(t) if isinstance(t, int) else None
                if tier is None:
                    continue
                for T in _TYPES:
                    ids = owned_map.get(T, [])
                    if T == "gr":
                        ids = [g for g in ids if g in universal_gear_ids]
                    cnt = len(ids)
                    key = (X, T)
                    if cnt == 1:
                        d = acc.setdefault(key, {}).setdefault(tier, {}).setdefault(tag, [ids[0], 0.0, 0.0])
                        d[1] += r
                        d[2] += r * y
                    elif cnt == 0:
                        na = none_acc.setdefault(key, [0.0, 0.0])
                        na[0] += r
                        na[1] += r * y

    cells, hypotheses = _score_cells(acc, boosted, p)
    _apply_fdr(cells, hypotheses, p)

    baselines = {}
    for X, (S1, Wn) in brawler_acc.items():
        g_global = Wn / S1 if S1 > 0 else 0.5
        baselines[str(X)] = {"g_global": round(g_global, 6), "boosted": int(X in boosted)}
    for (X, T), (S1, Wn) in none_acc.items():
        if S1 > 0:
            baselines.setdefault(str(X), {}).setdefault("owns0_rate_by_type", {})[T] = round(Wn / S1, 6)

    coverage = 1.0 - seen["no_profile"] / seen["rows"] if seen["rows"] else 0.0
    meta = {
        # built_at is left at 0 unless explicitly stamped, so a rebuild on unchanged inputs is
        # byte-identical (the sync content-hash then skips a needless reload). The CLI stamps it.
        "built_at": int(time.time()) if (params or {}).get("stamp_time") else 0,
        "window_days": p["window_days"], "halflife_days": p["halflife_days"],
        "K1": p["K1"], "prior_rest": p["prior_rest"],
        "n_min_eff": p["n_min_eff"], "n_min_players": p["n_min_players"], "fdr_q": p["fdr_q"],
        "coverage": round(coverage, 4), "n_cells": len(cells),
        "gear_ids_by_name": {_norm(name): gid for gid, name in gear_id_to_name.items()},
    }
    return {"version": 1, "meta": meta, "brawler_baseline": baselines, "cells": cells}


def _empty_payload(p: dict, gear_id_to_name: Dict[int, str]) -> dict:
    return {"version": 1,
            "meta": {"window_days": p["window_days"], "halflife_days": p["halflife_days"],
                     "K1": p["K1"], "prior_rest": p["prior_rest"], "n_min_eff": p["n_min_eff"],
                     "n_min_players": p["n_min_players"], "fdr_q": p["fdr_q"], "n_cells": 0,
                     "gear_ids_by_name": {name.lower(): gid for gid, name in gear_id_to_name.items()}},
            "brawler_baseline": {}, "cells": {}}


def _score_cells(acc, boosted, p):
    """Turn the per-(X,T,tier,player) accumulators into scored cells (pre-FDR). Returns
    ``(cells, hypotheses)`` where hypotheses is the de-duplicated BH family: one entry per unordered
    item pair, each carrying the p-value and the cell keys that share its q."""
    cells: Dict[str, dict] = {}
    hypotheses: List[dict] = []
    for (X, T), by_tier in acc.items():
        items = sorted({rec[0] for tier in by_tier.values() for rec in tier.values()})
        if len(items) < 2:            # no within-type contrast (single-item type) -> all heuristic
            continue
        # single-owner population rate for this (X,T): prior_rate for the rest-anchored absolute
        S1_all = sum(rec[1] for tier in by_tier.values() for rec in tier.values())
        Wn_all = sum(rec[2] for tier in by_tier.values() for rec in tier.values())
        pop_rate = Wn_all / S1_all if S1_all > 0 else 0.5

        scored = {}
        for i in items:
            strata, players_i, Wn_rest, S1_rest = [], set(), 0.0, 0.0
            for tier, players in by_tier.items():
                gi, wni, gr, wnr = [], [], [], []
                for tag, (owned, g, wn) in players.items():
                    if g <= 0:
                        continue
                    if owned == i:
                        gi.append(g); wni.append(wn); players_i.add(tag)
                    else:
                        gr.append(g); wnr.append(wn)
                gi, wni = cap_shares(gi, wni, p["cap_frac"])
                gr, wnr = cap_shares(gr, wnr, p["cap_frac"])
                si, sr = pool_stats(gi, wni), pool_stats(gr, wnr)
                if si.n_eff < p["strat_floor"] or sr.n_eff < p["strat_floor"]:
                    continue
                strata.append((si, sr))
                Wn_rest += sr.Wn; S1_rest += sr.S1
            mh = mh_combine(strata)
            if mh is None:
                continue
            rd, var_rd, n_eff_i = mh
            delta = rd * n_eff_i / (n_eff_i + p["K1"])
            z = rd / math.sqrt(var_rd) if var_rd > 0 else 0.0
            rest_rate = _rate(Wn_rest, S1_rest, prior=p["prior_rest"], prior_rate=pop_rate).winrate
            n_eff_rest = sum(sr.n_eff for _, sr in strata)
            key = f"{X}:{i}"
            cells[key] = {
                "item_type": _TYPE_LABEL[T],
                "delta": round(delta, 6), "delta_raw": round(rd, 6),
                "item_winrate": round(min(1.0, max(0.0, rest_rate + delta)), 6),
                "z": round(z, 4), "q": 1.0, "significant": 0,
                "n_eff": round(n_eff_i, 2), "n_players": len(players_i),
                "n_eff_rest": round(n_eff_rest, 2), "n_strata": len(strata),
                "ci": round(1.96 * math.sqrt(var_rd), 6) if var_rd > 0 else 0.0,
                "low_conf": int(X in boosted),
            }
            scored[i] = (key, z)
        # De-duplicated BH family. For a 2-item type, item-a-vs-REST and item-b-vs-REST are the SAME
        # test with the sign flipped (|z| identical) — one hypothesis, its q shared by both cells.
        # For a >=2-item type (only gears reach k>=3), each item-vs-REST is a distinct (non-mirror)
        # contrast, so it's its own hypothesis. A 1-item type produced no cell above.
        keys = list(scored)
        # Mirror-collapse only a genuine 2-ITEM type where BOTH items produced a cell — then the two
        # item-vs-REST contrasts really are one test, sign-flipped (identical |z|), sharing a q.
        # Anything else (a >=3-item gear type, or a 2-item type where only one item survived) is one
        # hypothesis per cell — the REST pools differ, so they aren't mirrors.
        if len(items) == 2 and len(keys) == 2:
            (ka, za), (kb, zb) = scored[keys[0]], scored[keys[1]]
            z_use = za if abs(za) >= abs(zb) else zb
            hypotheses.append({"p": _two_sided_p(z_use), "cells": [ka, kb]})
        else:
            for k in keys:
                key, z = scored[k]
                hypotheses.append({"p": _two_sided_p(z), "cells": [key]})
    return cells, hypotheses


def _apply_fdr(cells, hypotheses, p):
    """Run BH-FDR over the de-duplicated family and stamp each cell's q + final significance gate
    (BH pass AND sample floors AND implausibility AND not boosted)."""
    sig, qv = bh_fdr([h["p"] for h in hypotheses], p["fdr_q"])
    for h, s, q in zip(hypotheses, sig, qv):
        for k in h["cells"]:
            c = cells.get(k)
            if c is None:
                continue
            c["q"] = round(q, 6)
            gate = (s and c["n_eff"] >= p["n_min_eff"] and c["n_eff_rest"] >= p["n_min_eff"]
                    and c["n_players"] >= p["n_min_players"] and not c["low_conf"]
                    and not (abs(c["delta"]) > p["max_abs_delta"] and c["n_eff"] < 200))
            c["significant"] = int(bool(gate))
