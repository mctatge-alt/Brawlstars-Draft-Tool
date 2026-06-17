"""Load the trained win-probability model and score full 3v3 comps.

Degrades gracefully: if no trained checkpoint exists yet, ``available`` is False and
``prob`` returns 0.5, so the engine can still run on empirical stats alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch

from bsdraft.constants import PROCESSED_DIR
from bsdraft.data import encoders as E
from bsdraft.models.winprob import ModelConfig, WinProbNet

DEFAULT_PATH = PROCESSED_DIR / "winprob.pt"


class WinProbModel:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.net: Optional[WinProbNet] = None
        self.cfg: Optional[ModelConfig] = None
        if self.path.exists():
            ckpt = torch.load(self.path, map_location="cpu", weights_only=True)
            self.cfg = ModelConfig(**ckpt["config"])
            self.net = WinProbNet(self.cfg)
            self.net.load_state_dict(ckpt["state_dict"])
            self.net.eval()

    @property
    def available(self) -> bool:
        return self.net is not None

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
        ta = torch.tensor([[E.encode_brawler(b) for b in t] for t in teams_a])
        tb = torch.tensor([[E.encode_brawler(b) for b in t] for t in teams_b])
        n = len(teams_a)
        mp = torch.tensor([E.encode_map(map_id)] * n)
        mo = torch.tensor([E.encode_mode(mode)] * n)
        with torch.no_grad():
            return self.net.win_prob(ta, tb, mp, mo).tolist()
