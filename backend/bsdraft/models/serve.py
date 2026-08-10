"""Serve the win-probability model in pure NumPy — no torch at runtime.

Loads the weights exported by ``scripts/export_model.py`` (``winprob.npz``) and replicates
``WinProbNet.forward`` exactly:

    logit = [ S(A, ctx) - S(B, ctx) ] + [ PA·QB - PB·QA ]

where ctx = concat(map_emb, mode_emb); S is the strength MLP over the mean brawler
embedding + ctx (Linear -> ReLU -> [Dropout, a no-op at eval] -> Linear); and P/Q are the
low-rank counter embeddings. Training still uses PyTorch (``scripts/train.py``); only
inference is reimplemented here so the deployed API needs neither torch nor the training deps.

Degrades gracefully in two ways:

  * if no export exists yet, ``available`` is False and ``prob`` returns 0.5, so the engine can
    still run on empirical stats alone;
  * if the reference catalog is **newer than the model** — a brawler or map the export has no
    embedding row for — that id falls back to the mean embedding instead of raising. This is not
    hypothetical: the catalog ships in git (Render redeploy) while the model ships as a GitHub
    Release asset (hot-swap), so a new brawler is live in the catalog from the moment it is
    committed until the next retrain. Without the guard, indexing past the export's vocabulary
    raises ``IndexError`` and every draft touching that brawler/map 500s.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from bsdraft.constants import MODE_CAMEL_TO_DISPLAY, PROCESSED_DIR
from bsdraft.data import encoders as E

DEFAULT_PATH = PROCESSED_DIR / "winprob.npz"
logger = logging.getLogger(__name__)

# Embedding matrices, grouped by the vocabulary that indexes them.
_BRAWLER_MATRICES = ("brawler.weight", "counter_p.weight", "counter_q.weight")
_MAP_MATRICES = ("map_emb.weight",)
_MODE_MATRICES = ("mode_emb.weight",)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class WinProbModel:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.cfg: Optional[dict] = None
        self._w: Optional[Dict[str, np.ndarray]] = None
        # Vocabulary sizes this export was trained with (before the fallback row is appended).
        self._vocab: Dict[str, int] = {}
        self._warned = False
        # id -> trained row, from the vocabulary pinned into the export. Empty for older
        # artifacts, which fall back to the live positional encoders.
        self._brawler_rows: Dict[int, int] = {}
        self._map_rows: Dict[int, int] = {}
        self._mode_rows: Dict[str, int] = {}
        if self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            self.cfg = json.loads(data["_config"].item())
            self._w = {k: data[k].astype(np.float32) for k in data.files
                       if k != "_config" and not k.startswith("_vocab_")}
            self._load_vocab(data)
            self._add_fallback_rows()

    def _load_vocab(self, data) -> None:
        """Read the pinned id->row tables written by ``scripts/export_model.py``. Without them a
        catalog refresh can silently re-map an existing map onto a neighbour's trained row (map
        indices are positional over a (mode, name)-sorted list); with them the mapping is exact
        and refreshes are harmless."""
        files = set(data.files)
        if {"_vocab_brawler_ids"} <= files:
            ids = data["_vocab_brawler_ids"].tolist()
            self._brawler_rows = {int(b): i for i, b in enumerate(ids)}
        if {"_vocab_map_ids", "_vocab_map_rows"} <= files:
            self._map_rows = {int(m): int(r) for m, r in
                              zip(data["_vocab_map_ids"].tolist(), data["_vocab_map_rows"].tolist())}
        if {"_vocab_modes", "_vocab_mode_rows"} <= files:
            self._mode_rows = {str(m): int(r) for m, r in
                               zip(data["_vocab_modes"].tolist(), data["_vocab_mode_rows"].tolist())}

    def _add_fallback_rows(self) -> None:
        """Append a mean-embedding row to every lookup matrix and remember the original vocab
        size. Ids beyond that size (catalog newer than the model) are steered to this row: an
        "average" brawler/map is a neutral prior, whereas clamping to row 0 would silently
        impersonate a specific brawler (index 0 is Shelly)."""
        w = self._w
        for key in _BRAWLER_MATRICES + _MAP_MATRICES + _MODE_MATRICES:
            m = w.get(key)
            if m is None:
                continue
            self._vocab[key] = m.shape[0]
            w[key] = np.vstack([m, m.mean(axis=0, keepdims=True)])

    def _brawler_row(self, brawler_id: int) -> int:
        """Trained row for a brawler id. An id the export predates returns a sentinel past the
        vocabulary, which :meth:`_safe` turns into the mean-embedding fallback."""
        if self._brawler_rows:
            return self._brawler_rows.get(int(brawler_id), self._vocab.get("brawler.weight", 0))
        return E.encode_brawler(brawler_id)

    def _map_row(self, map_id) -> int:
        if self._map_rows:
            return self._map_rows.get(int(map_id), self._vocab.get("map_emb.weight", 0))
        return E.encode_map(map_id)

    def _mode_row(self, mode: str) -> int:
        if self._mode_rows:
            display = MODE_CAMEL_TO_DISPLAY.get(mode, mode)
            return self._mode_rows.get(display, self._vocab.get("mode_emb.weight", 0))
        return E.encode_mode(mode)

    def _safe(self, idx, key: str):
        """Map any out-of-vocabulary index to the appended mean row. Accepts a scalar or an
        ndarray; returns the same shape."""
        n = self._vocab.get(key)
        if n is None:
            return idx
        arr = np.asarray(idx)
        if bool((arr >= n).any()):
            if not self._warned:
                self._warned = True
                logger.warning(
                    "reference catalog is newer than %s (%s has %d rows): unknown ids are using "
                    "the mean embedding — retrain + re-export to give them real rows",
                    self.path.name, key, n)
            arr = np.where(arr >= n, n, arr)
        return arr if np.ndim(idx) else int(arr)

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
        # Rows come from the export's pinned vocabulary when it has one (exact, immune to catalog
        # reordering); otherwise from the live positional encoders. _safe() then steers any id the
        # export has no row for to the appended mean embedding instead of raising IndexError.
        a = self._safe(np.array([[self._brawler_row(x) for x in t] for t in teams_a]),
                       "brawler.weight")                                   # (N, 3)
        b = self._safe(np.array([[self._brawler_row(x) for x in t] for t in teams_b]),
                       "brawler.weight")                                   # (N, 3)

        # ctx = concat(map_emb, mode_emb), broadcast across the batch
        ctx = np.concatenate(
            [
                np.tile(w["map_emb.weight"][self._safe(self._map_row(map_id), "map_emb.weight")], (n, 1)),
                np.tile(w["mode_emb.weight"][self._safe(self._mode_row(mode), "mode_emb.weight")], (n, 1)),
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
