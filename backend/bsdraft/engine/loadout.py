"""Post-draft loadout advice: which gadget / star power / gear to equip on a drafted brawler.

Rule-based, in the same spirit as :mod:`bsdraft.engine.gameplan`. It classifies each of a
brawler's gadgets and star powers by *effect* (parsed from the catalog description) and scores
that effect against the game mode, rather than asserting a tier-list read the match data can't
support (battle logs record the brawler, never the equipped item). Gears come from a small
curated guide (``data/reference/gears.json``) since no catalog exposes them.

The response carries a ``fit`` score (0..1) and a ``source`` tag per item so the UI can label it
and a later data-driven pass can swap the heuristic for a measured win rate without changing the
shape. That upgrade — *single-item-owner inference*: attribute a player's recent brawler matches
to their one owned gadget/star power/gear and diff against zero-owners — is the planned Phase 2,
and it reuses the owned-item ids now retained in :mod:`bsdraft.engine.mastery`.

Pure stdlib so it stays importable on the serve path (no torch/sklearn/pandas).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import List, Optional

from bsdraft.constants import REFERENCE_DIR
from bsdraft.data import reference as R

# ---- effect taxonomy -------------------------------------------------------------------------
# Each gadget/star power is bucketed into one effect by scanning its description for keywords.
# Order matters: the first matching bucket wins, so the list is tuned "headline effect first"
# (a dash-that-also-reloads reads as mobility) and DAMAGE sits late because many descriptions
# mention "damage" only incidentally ("deals x damage").
_EFFECT_KEYWORDS = [
    ("mobility", ("dash", "dashes", "jump", "jumps", "leap", "teleport", "sprint", "roll",
                  "faster", "speed", "charges forward", "moves to", "moves ", "hop")),
    ("reload",   ("reload", "reloads", "ammo", "ammunition")),
    ("sustain",  ("heal", "heals", "healing", "restore", "restores", "shield", "regenerat",
                  "regain", "invulnerab", "immune")),
    ("control",  ("slow", "slows", "stun", "stuns", "freeze", "frozen", "knock", "push",
                  "pushes", "pull", "pulls", "root", "snare", "silence")),
    ("vision",   ("vision", "reveal", "reveals", "bush", "bushes", "detect")),
    ("damage",   ("damage", "deals", "pierc", "poison", "burn", "explo", "destroy")),
    # `range` sits last (before the utility fallback): a lot of descriptions mention "range"
    # incidentally ("+damage at max range"), so a real damage/effect keyword should win first.
    ("range",    ("range", "longer", "farther", "further", "reach")),
]
_EFFECT_META = {
    "mobility": ("Mobility", "reposition, engage, or escape"),
    "reload":   ("Reload / ammo", "more attack uptime and burst"),
    "sustain":  ("Sustain", "heal or shield to survive trades"),
    "control":  ("Control / CC", "slow, push, or lock down enemies"),
    "vision":   ("Vision", "reveal enemies in bushes"),
    "range":    ("Range", "poke from a safer distance"),
    "damage":   ("Damage", "raw damage output"),
    "utility":  ("Utility", "situational value"),
}
# How valuable each effect is per mode (0..1). Absent effects fall back to _NEUTRAL. Kept in sync
# with the mode priorities encoded in engine.gameplan (survive Knockout, mobility for Brawl Ball,
# burst for Heist, range for Bounty, control/sustain for the zone modes).
_NEUTRAL = 0.35
_MODE_EFFECT = {
    "Gem Grab":   {"sustain": 0.80, "control": 0.75, "reload": 0.60, "vision": 0.60,
                   "mobility": 0.50, "damage": 0.50, "range": 0.50},
    "Brawl Ball": {"mobility": 0.85, "control": 0.70, "damage": 0.60, "sustain": 0.55,
                   "reload": 0.50, "range": 0.45, "vision": 0.45},
    "Knockout":   {"sustain": 0.80, "control": 0.70, "vision": 0.65, "range": 0.62,
                   "damage": 0.58, "reload": 0.55, "mobility": 0.55},
    "Heist":      {"damage": 0.85, "reload": 0.75, "mobility": 0.60, "range": 0.52,
                   "control": 0.50, "sustain": 0.45, "vision": 0.35},
    "Hot Zone":   {"control": 0.80, "sustain": 0.75, "damage": 0.60, "reload": 0.55,
                   "mobility": 0.50, "vision": 0.50, "range": 0.50},
    "Bounty":     {"range": 0.80, "damage": 0.70, "vision": 0.65, "mobility": 0.55,
                   "sustain": 0.55, "control": 0.55, "reload": 0.55},
}

_TOKEN_RE = re.compile(r"<!.*?>")
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Strip the catalog's unfilled ``<!card.value…>`` tokens and collapse whitespace."""
    return _WS_RE.sub(" ", _TOKEN_RE.sub("", text or "")).strip()


def _classify(description: str) -> str:
    d = (description or "").lower()
    for effect, keywords in _EFFECT_KEYWORDS:
        if any(k in d for k in keywords):
            return effect
    return "utility"


def _mode_fit(effect: str, mode: str) -> float:
    return _MODE_EFFECT.get(mode, {}).get(effect, _NEUTRAL)


@lru_cache(maxsize=1)
def _by_id() -> dict:
    return {b.id: b for b in R.load_brawlers()}


@lru_cache(maxsize=1)
def _gear_guide() -> dict:
    """Curated gear guide (hand-maintained; gears aren't in any catalog). Fail-safe to empty so a
    missing/broken file degrades to 'no gear tips' rather than erroring the whole endpoint."""
    path = REFERENCE_DIR / "gears.json"
    if not path.exists():
        return {"note": "", "gears": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"note": "", "gears": []}
    return {"note": doc.get("note", ""), "gears": doc.get("gears", []) or []}


def _advise_accessory(acc: R.Accessory, mode: str) -> dict:
    effect = _classify(acc.description)
    label, blurb = _EFFECT_META[effect]
    return {
        "id": acc.id,
        "name": acc.name,
        "kind": acc.kind,
        "image_url": acc.image_url,
        "effect": label,
        "description": _clean(acc.description),
        "fit": round(_mode_fit(effect, mode), 3),
        "recommended": False,  # set by _mark_best once the whole kind is scored
        "why": f"{label}: {blurb}.",
        "source": "heuristic",
    }


def _mark_best(items: List[dict], mode: str) -> None:
    """Flag the single best-fit item of a kind, and enrich its reasoning with the mode."""
    if not items:
        return
    best = max(items, key=lambda it: it["fit"])
    best["recommended"] = True
    best["why"] = f"Best {best['kind'].replace('_', ' ')} fit for {mode} — {best['why'][0].lower()}{best['why'][1:]}"


def _gear_items(cls: str, mode: str, top: int = 2) -> List[dict]:
    guide = _gear_guide()
    scored = []
    for g in guide["gears"]:
        base = float(g.get("base", 0.4))
        fit = base + float(g.get("modes", {}).get(mode, 0.0)) + float(g.get("roles", {}).get(cls, 0.0))
        fit = max(0.0, min(1.0, fit))
        scored.append((fit, g))
    scored.sort(key=lambda t: t[0], reverse=True)
    out: List[dict] = []
    for rank, (fit, g) in enumerate(scored):
        recommended = rank < top
        why = g.get("effect", "")
        why = (f"Core pick for {mode}" if recommended else "Situational") + (f" — {why.lower()}." if why else ".")
        out.append({
            "id": None,
            "name": g.get("name", ""),
            "kind": "gear",
            "image_url": "",
            "effect": g.get("effect", ""),
            "description": g.get("description", ""),
            "fit": round(fit, 3),
            "recommended": recommended,
            "why": why,
            "source": "curated",
        })
    return out


def loadout_advice(brawler_id: int, mode: str, map_id: Optional[int] = None) -> Optional[dict]:
    """Loadout advice for a drafted brawler, or ``None`` if the brawler id is unknown.

    ``map_id`` is accepted for a future per-map slice (Phase 2); the heuristic is mode-level today.
    """
    b = _by_id().get(brawler_id)
    if b is None:
        return None
    gadgets = [_advise_accessory(a, mode) for a in b.gadgets]
    star_powers = [_advise_accessory(a, mode) for a in b.star_powers]
    _mark_best(gadgets, mode)
    _mark_best(star_powers, mode)
    gears = _gear_items(b.cls, mode)
    note = f"Effect-based fit for {mode} — not a live tier read yet."
    return {
        "brawler_id": b.id,
        "brawler_name": b.name,
        "cls": b.cls,
        "mode": mode,
        "gadgets": gadgets,
        "star_powers": star_powers,
        "gears": gears,
        "note": note,
    }
