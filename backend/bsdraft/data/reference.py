"""Load and clean Brawl Stars reference data (brawlers, maps, modes).

Source: the keyless Brawlify/BrawlAPI JSON snapshots in ``data/reference/``. Provides
clean, typed accessors plus a stable contiguous brawler index for model embeddings.

Pure stdlib so it runs without installing the ML/runtime dependencies:

    PYTHONPATH=backend python -m bsdraft.data.reference
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

from bsdraft.constants import (
    BRAWLER_CLASSES,
    RANKED_MODES,
    REFERENCE_DIR,
    UNCLASSIFIED,
)

CLASS_OVERRIDES_PATH = Path(__file__).resolve().parent / "class_overrides.json"
RANKED_BOOSTED_PATH = REFERENCE_DIR / "ranked_boosted.json"
ECONOMY_PATH = REFERENCE_DIR / "economy.json"


@dataclass(frozen=True)
class Accessory:
    id: int
    name: str
    kind: str  # "star_power" | "gadget"
    image_url: str = ""
    description: str = ""  # raw catalog text (carries unfilled `x`/`<!card...>` value tokens)


@dataclass(frozen=True)
class Brawler:
    id: int
    name: str
    cls: str  # a member of BRAWLER_CLASSES, or UNCLASSIFIED
    rarity: str
    star_powers: tuple
    gadgets: tuple
    image_url: str


@dataclass(frozen=True)
class GameMap:
    id: int
    name: str
    mode: str
    environment: str
    image_url: str


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def class_overrides() -> dict:
    """Manual class assignments for brawlers Brawlify hasn't tagged yet."""
    if not CLASS_OVERRIDES_PATH.exists():
        return {}
    data = _load_json(CLASS_OVERRIDES_PATH)
    if isinstance(data, dict):
        return data.get("overrides", data)
    return {}


def _resolve_class(raw_brawler: dict, overrides: dict) -> str:
    cls = (raw_brawler.get("class") or {}).get("name")
    if not cls or cls == "Unknown":
        cls = overrides.get(raw_brawler["name"], UNCLASSIFIED)
    return cls if cls in BRAWLER_CLASSES else UNCLASSIFIED


@lru_cache(maxsize=1)
def load_brawlers() -> tuple:
    """All brawlers, sorted by id, with classes cleaned via overrides."""
    raw = _load_json(REFERENCE_DIR / "brawlers.json")["list"]
    overrides = class_overrides()
    brawlers = [
        Brawler(
            id=x["id"],
            name=x["name"],
            cls=_resolve_class(x, overrides),
            rarity=(x.get("rarity") or {}).get("name", ""),
            star_powers=tuple(
                Accessory(sp["id"], sp["name"], "star_power",
                          sp.get("imageUrl", ""), sp.get("description", ""))
                for sp in (x.get("starPowers") or [])
            ),
            gadgets=tuple(
                Accessory(g["id"], g["name"], "gadget",
                          g.get("imageUrl", ""), g.get("description", ""))
                for g in (x.get("gadgets") or [])
            ),
            image_url=x.get("imageUrl", ""),
        )
        for x in raw
    ]
    brawlers.sort(key=lambda b: b.id)
    return tuple(brawlers)


@lru_cache(maxsize=1)
def brawler_index() -> dict:
    """Stable brawler id -> contiguous index (0..N-1), sorted by id. For embeddings."""
    return {b.id: i for i, b in enumerate(load_brawlers())}


@lru_cache(maxsize=1)
def _by_name() -> dict:
    return {b.name.lower(): b for b in load_brawlers()}


def brawler_by_name(name: str) -> Optional[Brawler]:
    return _by_name().get(name.strip().lower())


@lru_cache(maxsize=1)
def load_ranked_maps() -> tuple:
    """Active maps belonging to the 5 ranked modes, sorted by (mode, name)."""
    raw = _load_json(REFERENCE_DIR / "maps.json")["list"]
    maps = []
    for x in raw:
        if x.get("disabled"):
            continue
        mode = (x.get("gameMode") or {}).get("name")
        if mode not in RANKED_MODES:
            continue
        maps.append(
            GameMap(
                id=x["id"],
                name=x["name"],
                mode=mode,
                environment=(x.get("environment") or {}).get("name", ""),
                image_url=x.get("imageUrl", ""),
            )
        )
    maps.sort(key=lambda m: (m.mode, m.name))
    return tuple(maps)


@lru_cache(maxsize=1)
def _ranked_boosted_doc() -> Optional[dict]:
    """Parsed ``ranked_boosted.json``, or None when absent/unreadable. Only the file parse is
    cached — the ``valid_until`` guard lives in :func:`load_ranked_boosted` and runs per call,
    because the deployed API is deliberately kept warm for days: a process-start-only check would
    keep serving an expired rotation long after the season flipped."""
    if not RANKED_BOOSTED_PATH.exists():
        return None
    try:
        return _load_json(RANKED_BOOSTED_PATH)
    except (json.JSONDecodeError, OSError):
        return None


def load_ranked_boosted() -> tuple:
    """Brawler ids of the current season's Ranked **free / "boosted" brawlers** — the maxed
    brawlers everyone may use in Ranked regardless of ownership. Read from the committed
    ``ranked_boosted.json`` (maintained by the ``boosted-watch`` scraper, see
    :mod:`bsdraft.collect.boosted`); the ``active.brawlers`` **names** are resolved to ids here so
    a catalog rename can't strand a hard-coded id.

    Fail-safe (the list must never *mislead* — telling a player they can freely pick a brawler
    they actually can't is worse than showing none): returns ``()`` when the file is absent /
    unreadable, when no name resolves, or when the optional ``valid_until`` date (the last day the
    rotation is served, inclusive) has passed — re-checked on every call, see
    :func:`_ranked_boosted_doc`."""
    doc = _ranked_boosted_doc()
    if doc is None:
        return ()
    valid_until = (doc.get("valid_until") or "").strip()
    if valid_until:
        try:
            if date.fromisoformat(valid_until) < date.today():
                return ()
        except ValueError:
            pass  # malformed date — ignore the guard rather than drop the rotation
    active = doc.get("active") or {}
    ids = []
    for name in active.get("brawlers", []) or []:
        b = brawler_by_name(name) if isinstance(name, str) else None
        if b is not None and b.id not in ids:
            ids.append(b.id)
    return tuple(ids)


@lru_cache(maxsize=1)
def load_economy() -> dict:
    """Curated progression-economy table for the purchase advisor (costs, power gates, impact
    priors, hypercharge availability). Hand-maintained — none of this is in any catalog and the
    API never exposes prices; see ``economy.json`` and ``docs/purchase-advisor.md``.

    Fail-safe to ``{}`` so a missing/broken file degrades the advisor to 'no cost/gate context'
    rather than erroring the endpoint (the consumer treats every section as optional)."""
    if not ECONOMY_PATH.exists():
        return {}
    try:
        doc = _load_json(ECONOMY_PATH)
    except (json.JSONDecodeError, OSError):
        return {}
    return doc if isinstance(doc, dict) else {}


def summary() -> str:
    brawlers = load_brawlers()
    maps = load_ranked_maps()
    cls_counts = Counter(b.cls for b in brawlers)
    unclassified = [b.name for b in brawlers if b.cls == UNCLASSIFIED]
    map_counts = Counter(m.mode for m in maps)

    lines = [
        f"Brawlers: {len(brawlers)}  (embedding index 0..{len(brawlers) - 1})",
        "  classes: " + ", ".join(f"{k}={cls_counts[k]}" for k in BRAWLER_CLASSES),
    ]
    if unclassified:
        lines.append(f"  UNCLASSIFIED ({len(unclassified)}): " + ", ".join(unclassified))
    else:
        lines.append("  UNCLASSIFIED: 0  (all brawlers classified)")
    lines.append(f"Ranked maps: {len(maps)}")
    for mode in RANKED_MODES:
        lines.append(f"  {mode}: {map_counts[mode]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
