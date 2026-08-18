"""Regression guards for the vectorized recommend hot path (Track D perf refactor).

The pick recommender was sped up by (a) scoring the win-prob marginal for all ~100 candidates
in one batched model pass instead of one call each, and (b) skipping ``statistics.mean``'s exact-
Fraction machinery for the 1- and 2-element ally/enemy winrate lists. Both are *logic-preserving*:
these tests pin the bit-for-bit equivalence so a future edit can't silently perturb rankings.

  PYTHONPATH=backend python -m pytest backend/tests/test_scoring_perf.py   # or run directly
"""
from __future__ import annotations

import json
import statistics
import tempfile
from pathlib import Path

import numpy as np

from bsdraft.engine import scoring
from bsdraft.engine.scoring import _mean_wr, model_marginal, model_marginals
from bsdraft.engine.state import DraftState
from bsdraft.models import serve as S

# Mirrors test_partial_draft's minimal synthetic export (kept independent on purpose).
N_BRAWLER, N_MAP, N_MODE, D, R = 6, 3, 2, 8, 3
MASK, HIDDEN = N_BRAWLER, 10


def _synthetic(tmp: Path) -> Path:
    rng = np.random.default_rng(1)
    rows = N_BRAWLER + 1  # + mask row
    arrays = {
        "_config": np.array(json.dumps({"synthetic": True, "mask_row": MASK})),
        "brawler.weight": rng.normal(size=(rows, D)).astype(np.float32),
        "map_emb.weight": rng.normal(size=(N_MAP, 2)).astype(np.float32),
        "mode_emb.weight": rng.normal(size=(N_MODE, 1)).astype(np.float32),
        "strength.0.weight": rng.normal(size=(HIDDEN, D + 3)).astype(np.float32),
        "strength.0.bias": rng.normal(size=(HIDDEN,)).astype(np.float32),
        "strength.3.weight": rng.normal(size=(1, HIDDEN)).astype(np.float32),
        "strength.3.bias": rng.normal(size=(1,)).astype(np.float32),
        "counter_p.weight": rng.normal(size=(rows, R)).astype(np.float32),
        "counter_q.weight": rng.normal(size=(rows, R)).astype(np.float32),
    }
    path = tmp / "winprob.npz"
    np.savez(path, **arrays)
    return path


class _Encoders:
    def encode_brawler(self, bid): return bid
    def encode_map(self, mid): return 1
    def encode_mode(self, mode): return 1


def _load(path: Path):
    S.E, prev = _Encoders(), S.E
    try:
        return S.WinProbModel(path), prev
    except Exception:
        S.E = prev
        raise


def test_prob_marginals_bit_exact_to_per_board():
    """The batched marginal must equal calling prob() per board, to the last bit — this is what
    lets the recommender fold ~100 candidate calls into one without moving any score."""
    with tempfile.TemporaryDirectory() as td:
        m, prev = _load(_synthetic(Path(td)))
        try:
            # The recommender's exact shape: many boards sharing one enemy team, differing only by
            # the added ally; also cover full/empty our-teams and a full 3-enemy team.
            for their in ([3, 4], [3, 4, 5], [3], []):
                for base in ([], [0], [0, 1], [0, 1, 2]):
                    teams_a = [(base + [c])[:3] for c in range(N_BRAWLER)]
                    teams_b = [their[:3]] * len(teams_a)
                    batched = m.prob_marginals(teams_a, teams_b, 1, "gemGrab")
                    per = [m.prob(ta, tb, 1, "gemGrab") for ta, tb in zip(teams_a, teams_b)]
                    assert batched == per, (their, base)  # exact float equality
        finally:
            S.E = prev


def test_model_marginals_matches_per_candidate():
    """The engine-level dispatch (scoring.model_marginals) equals the per-candidate loop."""
    with tempfile.TemporaryDirectory() as td:
        m, prev = _load(_synthetic(Path(td)))
        try:
            state = DraftState(map_id=1, mode="gemGrab", our_team=[0], their_team=[3, 4])
            cands = [1, 2, 5]
            batched = model_marginals(state, cands, m, None)
            per = [model_marginal(state, c, m, None) for c in cands]
            assert batched == per
            # model off -> all None, same as the per-candidate path
            assert model_marginals(state, cands, None, None) == [None, None, None]
        finally:
            S.E = prev


def test_mean_wr_matches_statistics_mean():
    """_mean_wr is a drop-in for statistics.mean on winrate-shaped inputs: exact for n<=2 (its
    fast path) and identical for n>=3 (where it defers to statistics.mean)."""
    rng = np.random.default_rng(2)
    for _ in range(20000):
        vals = [float(x) for x in rng.uniform(0.0, 1.0, size=int(rng.integers(1, 5)))]
        assert _mean_wr(vals) == statistics.mean(vals), vals


if __name__ == "__main__":
    test_prob_marginals_bit_exact_to_per_board()
    test_model_marginals_matches_per_candidate()
    test_mean_wr_matches_statistics_mean()
    print("ok")
