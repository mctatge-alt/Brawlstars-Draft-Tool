"""Composition analysis: class coverage + heuristic warnings about comp holes."""
from __future__ import annotations

from collections import Counter
from typing import List

from bsdraft.engine.scoring import _class_of
from bsdraft.engine.state import DraftState

FRONTLINE = {"Tank", "Assassin"}
RANGE = {"Marksman", "Controller", "Artillery"}


def analyze(state: DraftState) -> dict:
    classes = [_class_of(b) for b in state.our_team]
    counts = Counter(classes)
    n = len(classes)
    warnings: List[dict] = []

    def warn(text: str, severity: str = "warn") -> None:
        warnings.append({"text": text, "severity": severity})

    for cls, c in counts.items():
        if c >= 3:
            warn(f"All three are {cls}s — very one-dimensional.", "critical")
        elif c >= 2 and cls in ("Tank", "Artillery", "Marksman"):
            warn(f"Two {cls}s — watch for redundancy.", "info")

    has_front = any(c in FRONTLINE for c in classes)
    has_range = any(c in RANGE for c in classes)

    if n >= 2 and not has_front and state.mode in ("Brawl Ball", "Gem Grab", "Hot Zone"):
        warn("No frontline (Tank/Assassin) to contest space.")
    if n >= 2 and not has_range:
        warn("No long range — weak to pokes & throwers.")
    if state.mode == "Bounty" and n >= 1 and "Marksman" not in classes and counts.get("Tank", 0) == 0:
        warn("Bounty rewards long range — consider a Marksman.", "info")
    if state.mode == "Heist" and n >= 2 and not (counts.get("Damage Dealer") or counts.get("Marksman")):
        warn("Heist needs burst to crack the safe — add a Damage Dealer/Marksman.")

    enemy_tanks = sum(1 for b in state.their_team if _class_of(b) == "Tank")
    if enemy_tanks >= 2 and "Marksman" not in classes and n >= 1:
        warn("Enemy is tank-heavy — a Marksman shreds them.")

    return {
        "classes": dict(counts),
        "has_frontline": has_front,
        "has_range": has_range,
        "warnings": warnings,
    }
