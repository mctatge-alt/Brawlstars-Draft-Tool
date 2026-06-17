"""Antisymmetric win-probability model: P(team_a beats team_b | map, mode).

Design (a clean inductive bias for drafting):

    logit = [ S(A, ctx) - S(B, ctx) ]          # context-conditioned team strength
          + [ PA·QB - PB·QA ]                   # low-rank antisymmetric counter term

where ctx = [map_emb, mode_emb]; S is a shared MLP over the mean brawler embedding + ctx;
and P/Q are low-rank "attacker/defender" embeddings whose dot products encode directed
matchups. Swapping A and B negates the logit, so P(A wins) + P(B wins) = 1 by
construction — no team-order bias and no global offset to learn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    num_brawlers: int
    num_maps: int
    num_modes: int
    d_brawler: int = 32
    d_map: int = 16
    d_mode: int = 8
    d_hidden: int = 64
    counter_rank: int = 16
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class WinProbNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.brawler = nn.Embedding(cfg.num_brawlers, cfg.d_brawler)
        self.map_emb = nn.Embedding(cfg.num_maps, cfg.d_map)
        self.mode_emb = nn.Embedding(cfg.num_modes, cfg.d_mode)
        self.strength = nn.Sequential(
            nn.Linear(cfg.d_brawler + cfg.d_map + cfg.d_mode, cfg.d_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, 1),
        )
        # low-rank counter embeddings (attacker P, defender Q)
        self.counter_p = nn.Embedding(cfg.num_brawlers, cfg.counter_rank)
        self.counter_q = nn.Embedding(cfg.num_brawlers, cfg.counter_rank)
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.brawler, self.map_emb, self.mode_emb):
            nn.init.normal_(emb.weight, std=0.1)
        nn.init.normal_(self.counter_p.weight, std=0.05)
        nn.init.normal_(self.counter_q.weight, std=0.05)

    def _ctx(self, map_idx: torch.Tensor, mode_idx: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.map_emb(map_idx), self.mode_emb(mode_idx)], dim=-1)

    def _strength(self, team: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        team_vec = self.brawler(team).mean(dim=1)  # (B, d_brawler) — order-invariant
        return self.strength(torch.cat([team_vec, ctx], dim=-1)).squeeze(-1)  # (B,)

    def forward(
        self,
        team_a: torch.Tensor,
        team_b: torch.Tensor,
        map_idx: torch.Tensor,
        mode_idx: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self._ctx(map_idx, mode_idx)
        strength = self._strength(team_a, ctx) - self._strength(team_b, ctx)
        pa = self.counter_p(team_a).sum(dim=1)  # (B, r)
        qa = self.counter_q(team_a).sum(dim=1)
        pb = self.counter_p(team_b).sum(dim=1)
        qb = self.counter_q(team_b).sum(dim=1)
        counter = (pa * qb).sum(-1) - (pb * qa).sum(-1)  # (B,)
        return strength + counter

    @torch.no_grad()
    def win_prob(self, team_a, team_b, map_idx, mode_idx) -> torch.Tensor:
        return torch.sigmoid(self.forward(team_a, team_b, map_idx, mode_idx))
