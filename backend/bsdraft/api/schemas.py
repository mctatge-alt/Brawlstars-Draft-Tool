"""Pydantic request/response schemas for the draft API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class RecommendRequest(BaseModel):
    map_id: int
    mode: str
    our_team: List[int] = []
    their_team: List[int] = []
    bans: List[int] = []
    we_pick_first: bool = True
    solo_queue: bool = True
    rank_bracket: Optional[str] = None   # condition stats on this rank bracket, e.g. "Masters"
    phase: str = "pick"          # "pick" | "ban"
    use_search: bool = False     # seat-aware minimax lookahead
    personalize: bool = False    # weight by the player's roster / mastery
    top: int = 8


class PickRec(BaseModel):
    brawler_id: int
    name: str
    cls: str
    score: float
    map_winrate: float
    synergy: Optional[float] = None
    counter: Optional[float] = None
    role_fit: float
    win_prob: Optional[float] = None
    confidence: float
    projected_winprob: Optional[float] = None  # set when seat-aware search is used
    mastery: Optional[float] = None
    owned: bool = True
    gaps: List[str] = []
    breakdown: Dict[str, float]


class BanRec(BaseModel):
    brawler_id: int
    name: str
    cls: str
    threat: float
    map_winrate: float
    use_rate: float
    confidence: float


class Warning(BaseModel):
    text: str
    severity: str  # "info" | "warn" | "critical"


class RoleTip(BaseModel):
    name: str
    cls: str
    role: str


class ThreatTip(BaseModel):
    name: str
    cls: str
    tip: str


class GamePlan(BaseModel):
    objective: str = ""
    win_condition: str = ""
    archetype: str = ""
    playstyle: str = ""
    roles: List[RoleTip] = []
    threats: List[ThreatTip] = []
    tips: List[str] = []
    avoid: List[str] = []
    compensate: List[str] = []


class RecommendResponse(BaseModel):
    phase: str
    picks: List[PickRec] = []
    bans: List[BanRec] = []
    composition: Dict[str, int] = {}
    warnings: List[Warning] = []
    game_plan: Optional[GamePlan] = None
    next_to_act: Optional[str] = None


class BrawlerRef(BaseModel):
    id: int
    name: str
    cls: str
    rarity: str
    image_url: str


class MapRef(BaseModel):
    id: int
    name: str
    mode: str
    image_url: str
    games: int = 0


class ReferenceResponse(BaseModel):
    brawlers: List[BrawlerRef]
    maps: List[MapRef]
    modes: List[str]
    brackets: List[str] = []     # rank brackets with enough data to condition on


class OwnedBrawler(BaseModel):
    id: int
    mastery: float
    gaps: List[str] = []


class RosterResponse(BaseModel):
    loaded: bool
    tag: str
    name: str
    owned: List[OwnedBrawler] = []
    error: Optional[str] = None


class RankResponse(BaseModel):
    found: bool
    tag: str
    tier: Optional[int] = None          # Ranked tier index 1-22
    tier_label: Optional[str] = None    # e.g. "Legendary II"
    bracket: Optional[str] = None       # e.g. "Legendary"
    source: Optional[str] = None        # "dataset" | "live"
    error: Optional[str] = None


class MetaShift(BaseModel):
    brawler_id: int
    name: str
    kind: str          # "buff" | "nerf"
    wr_before: float
    wr_after: float
    use_before: float
    use_after: float
    z: float


class MetaResponse(BaseModel):
    shifted: bool
    n_recent: int
    n_prior: int
    new_brawlers: List[str] = []   # names of brawlers seen in play but not yet in the reference
    shifts: List[MetaShift] = []
    note: str = ""
