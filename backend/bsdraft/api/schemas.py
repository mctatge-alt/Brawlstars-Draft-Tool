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


class RecommendResponse(BaseModel):
    phase: str
    picks: List[PickRec] = []
    bans: List[BanRec] = []
    composition: Dict[str, int] = {}
    warnings: List[Warning] = []
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
