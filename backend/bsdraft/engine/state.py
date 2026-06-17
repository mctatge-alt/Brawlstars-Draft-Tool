"""Draft state for Brawl Stars ranked (1-2-2-1 snake, 6 bans, unique picks)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from bsdraft.constants import PICK_ORDER, TEAM_SIZE


@dataclass
class DraftState:
    map_id: int
    mode: str
    our_team: List[int] = field(default_factory=list)    # brawler ids we've picked
    their_team: List[int] = field(default_factory=list)
    bans: List[int] = field(default_factory=list)
    we_pick_first: bool = True

    # refinement context
    solo_queue: bool = True                               # solo vs premade
    roster: Optional[Dict[int, dict]] = None             # brawler_id -> mastery info
    rank_bracket: Optional[str] = None

    def picked_or_banned(self) -> set:
        return set(self.our_team) | set(self.their_team) | set(self.bans)

    @property
    def our_slots_left(self) -> int:
        return TEAM_SIZE - len(self.our_team)

    @property
    def their_slots_left(self) -> int:
        return TEAM_SIZE - len(self.their_team)

    @property
    def complete(self) -> bool:
        return len(self.our_team) + len(self.their_team) >= 2 * TEAM_SIZE

    def next_to_act(self) -> Optional[str]:
        """'us', 'them', or None (picks complete), per the 1-2-2-1 snake order."""
        total = len(self.our_team) + len(self.their_team)
        if total >= 2 * TEAM_SIZE:
            return None
        our_index = 0 if self.we_pick_first else 1
        return "us" if PICK_ORDER[total] == our_index else "them"
