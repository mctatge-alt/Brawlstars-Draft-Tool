"""Serve the win-probability model in pure NumPy — no torch at runtime.

Loads the weights exported by ``scripts/export_model.py`` (``winprob.npz``) and replicates
``WinProbNet.forward`` exactly:

    logit = [ S(A, ctx) - S(B, ctx) ] + [ PA·QB - PB·QA ]

where ctx = concat(map_emb, mode_emb); S is the strength MLP over the mean brawler
embedding + ctx (Linear -> ReLU -> [Dropout, a no-op at eval] -> Linear); and P/Q are the
low-rank counter embeddings. Training still uses PyTorch (``scripts/train.py``); only
inference is reimplemented here so the deployed API needs neither torch nor the training deps.

Degrades gracefully: if no export exists yet, ``available`` is False and ``prob`` returns
0.5, so the engine can still run on empirical stats alone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from bsdraft.constants import PROCESSED_DIR
from bsdraft.data import encoders as E

DEFAULT_PATH = PROCESSED_DIR / "winprob.npz"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class WinProbModel:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.cfg: Optional[dict] = None
        self._w: Optional[Dict[str, np.ndarray]] = None
        if self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            self.cfg = json.loads(data["_config"].item())
            self._w = {k: data[k].astype(np.float32) for k in data.files if k != "_config"}

    @property
    def available(self) -> bool:
        return self._w is not None

    def prob(self, team_a_ids: Sequence[int], team_b_ids: Sequence[int], map_id: int, mode: str) -> float:
        """P(team_a beats team_b) for full 3-brawler teams (brawler ids)."""
        if not self.available:
            return 0.5
        return self.prob_batch([list(team_a_ids)], [list(team_b_ids)], map_id, mode)[0]

    def prob_batch(
        self,
        teams_a: List[List[int]],
        teams_b: List[List[int]],
        map_id: int,
        mode: str,
    ) -> List[float]:
        if not self.available:
            return [0.5] * len(teams_a)
        w = self._w
        n = len(teams_a)
        a = np.array([[E.encode_brawler(b) for b in t] for t in teams_a])  # (N, 3)
        b = np.array([[E.encode_brawler(x) for x in t] for t in teams_b])  # (N, 3)

        # ctx = concat(map_emb, mode_emb), broadcast across the batch
        ctx = np.concatenate(
            [
                np.tile(w["map_emb.weight"][E.encode_map(map_id)], (n, 1)),
                np.tile(w["mode_emb.weight"][E.encode_mode(mode)], (n, 1)),
            ],
            axis=1,
        )

        def strength(team: np.ndarray) -> np.ndarray:
            team_vec = w["brawler.weight"][team].mean(axis=1)        # (N, d_brawler), order-invariant
            h = np.concatenate([team_vec, ctx], axis=1)
            h = h @ w["strength.0.weight"].T + w["strength.0.bias"]  # Linear
            h = np.maximum(h, 0.0)                                   # ReLU (Dropout is a no-op at eval)
            out = h @ w["strength.3.weight"].T + w["strength.3.bias"]
            return out[:, 0]

        s = strength(a) - strength(b)
        pa = w["counter_p.weight"][a].sum(axis=1)  # (N, r)
        qa = w["counter_q.weight"][a].sum(axis=1)
        pb = w["counter_p.weight"][b].sum(axis=1)
        qb = w["counter_q.weight"][b].sum(axis=1)
        counter = (pa * qb).sum(axis=1) - (pb * qa).sum(axis=1)
        return _sigmoid(s + counter).tolist()
