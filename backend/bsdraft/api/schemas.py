"""Pydantic request/response schemas for the draft API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class OwnedGear(BaseModel):
    id: int
    name: str
    level: int = 0


class OwnedBrawler(BaseModel):
    id: int
    mastery: float
    gaps: List[str] = []
    # The specific items this player owns on the brawler, so the client can restrict loadout
    # suggestions on the user's own pick to what they can actually equip. Empty on the recommend
    # request path (the client only sends id/mastery/gaps there); populated by /api/roster.
    owned_star_powers: List[int] = []
    owned_gadgets: List[int] = []
    owned_gears: List[OwnedGear] = []


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
    personal_tag: Optional[str] = None   # fold in this player's own win rates (resolved from data)
    # The player's roster (owned brawlers + mastery + loadout gaps), sent by the client so the
    # public backend can personalize despite being unable to fetch it itself (IP-locked out of
    # Supercell). Used only when ``personalize`` is set; falls back to the server's own roster.
    roster: Optional[List[OwnedBrawler]] = None
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
    personal_winrate: Optional[float] = None   # this player's own win rate with the brawler
    personal_games: Optional[float] = None      # their effective (recency-weighted) sample
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


class TopPicksRequest(BaseModel):
    """Current board for the full-loadout rail. No roster/personalize fields — the rail is
    deliberately the population meta (every brawler at a full loadout), so it never depends
    on what the player owns."""
    map_id: int
    mode: str
    our_team: List[int] = []
    their_team: List[int] = []
    bans: List[int] = []
    rank_bracket: Optional[str] = None
    top: int = 10


class TopPick(BaseModel):
    brawler_id: int
    name: str
    cls: str
    score: float
    map_winrate: float


class TopPicksResponse(BaseModel):
    """Strongest picks for the current board at a full loadout, with no roster — re-ranks as
    the draft fills in (used brawlers drop out, synergy/counters fold in)."""
    map_id: int
    mode: str
    rank_bracket: Optional[str] = None
    picks: List[TopPick] = []


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
    boosted: List[int] = []      # ids of this season's free/"boosted" Ranked brawlers


class RosterResponse(BaseModel):
    loaded: bool
    tag: str
    name: str
    owned: List[OwnedBrawler] = []
    error: Optional[str] = None


class LoadoutItem(BaseModel):
    id: Optional[int] = None       # catalog id (gadgets/star powers); None for gears (no catalog)
    name: str
    kind: str                      # "gadget" | "star_power" | "gear"
    image_url: str = ""
    effect: str = ""               # short effect label, e.g. "Mobility · reload"
    description: str = ""          # cleaned catalog text (gadgets/SP) or guide text (gears)
    fit: float = 0.0               # mode-fit score 0..1 (heuristic; swapped for a win-rate in Phase 2)
    recommended: bool = False      # the best-fit item of its kind for this mode
    why: str = ""                  # one-line reasoning tied to the mode/effect
    source: str = "heuristic"      # "heuristic" (effect-based) | "curated" (gear guide)


class LoadoutResponse(BaseModel):
    """Which gadget / star power / gear to equip on a drafted brawler, given the mode.

    Effect-based heuristic (same spirit as the game plan), not a live tier read — the match data
    can't attribute wins to a specific item yet. ``source``/``fit`` are the seam for the planned
    single-item-owner win-rate upgrade. Ownership is overlaid client-side for the user's own seat."""
    brawler_id: int
    brawler_name: str
    cls: str = ""
    mode: str = ""
    gadgets: List[LoadoutItem] = []
    star_powers: List[LoadoutItem] = []
    gears: List[LoadoutItem] = []
    note: str = ""


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
