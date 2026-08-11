"""Serve-side loader for the per-item win-rate table (``itemstats.json[.gz]``).

The home box builds the table from the matches x ownership-profiles join (see
:mod:`bsdraft.data.itemstats_build`); the deployed API just LOADS it here so
:mod:`bsdraft.engine.loadout` can serve measured gadget/star-power picks with no in-memory join.

Pure stdlib (json + gzip), like ``loadout.py`` and ``reference.py`` — it must stay importable on
the 512 MB serve box without the ML deps. A tiny mtime-keyed cache reloads the file when
``sync_itemstats`` rewrites it, so a refreshed artifact rolls out with no restart.

Artifact shape (v1)::

    {"version": 1,
     "meta": {"built_at","window_days","halflife_days","K1","prior_rest",
              "n_min_eff","n_min_players","fdr_q","gear_ids_by_name": {"<norm name>": <gear id>}},
     "brawler_baseline": {"<brawler id>": {"g_global","boosted","owns0_rate_by_type"}},
     "cells": {"<brawler id>:<item id>": {"item_type","delta","delta_raw","item_winrate",
              "z","q","significant","n_eff","n_players","n_eff_rest","n_strata","ci","low_conf"}}}

``delta`` is the shrunk win-rate difference of the item vs the brawler's OTHER single-owned items
of the same type (the primary served number, "+X.X%"); ``item_winrate`` is the rest-anchored
absolute rate (secondary). ``significant`` already folds in the sample floors + BH-FDR gate, so the
consumer only checks that flag.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Optional

from bsdraft.constants import PROCESSED_DIR

DEFAULT_PATH = PROCESSED_DIR / "itemstats.json"

_norm_re = re.compile(r"[^a-z0-9]")


def _norm(name: str) -> str:
    return _norm_re.sub("", (name or "").lower())


def load_itemstats(path: Path) -> Optional[dict]:
    """Read the table from ``path`` (gzip-framed or plain JSON — sniffed by magic bytes). Returns
    None if the file is absent/unreadable, so the loadout path degrades to the effect heuristic."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        doc = json.loads(raw)
    except Exception:  # noqa: BLE001 — a truncated gzip raises EOFError/zlib.error (not OSError/
        return None    # ValueError); any corrupt artifact must degrade to the heuristic, not 500
    return doc if isinstance(doc, dict) and "cells" in doc else None


_cache: dict = {"key": None, "data": None}


def get_itemstats(path: Path = DEFAULT_PATH) -> Optional[dict]:
    """Cached accessor. Reloads only when the file's (mtime, size) changes — so a ``sync_itemstats``
    rewrite is picked up on the next request with no restart, and an absent file is cheap."""
    try:
        st = path.stat()
    except OSError:
        _cache["key"], _cache["data"] = None, None
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if _cache["key"] != key:
        _cache["key"] = key
        _cache["data"] = load_itemstats(path)
    return _cache["data"]


def accessory_cell(data: Optional[dict], brawler_id: int, item_id: int) -> Optional[dict]:
    """The measured cell for a gadget/star power (globally-unique catalog id), or None."""
    if not data or item_id is None:
        return None
    return data.get("cells", {}).get(f"{brawler_id}:{item_id}")


def gear_cell(data: Optional[dict], brawler_id: int, gear_name: str) -> Optional[dict]:
    """The measured cell for a gear, resolved via the build-time gear-name->id map (gears carry no
    catalog id — the id is learned from live profiles and pinned in ``meta.gear_ids_by_name``)."""
    if not data:
        return None
    gid = data.get("meta", {}).get("gear_ids_by_name", {}).get(_norm(gear_name))
    if gid is None:
        return None
    return data.get("cells", {}).get(f"{brawler_id}:{gid}")


def save_itemstats(payload: dict, path: Path) -> Path:
    """Write the table as gzipped JSON when ``path`` ends in .gz, else plain JSON. Used by the
    offline build/export; stdlib-only so it can live beside the loader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if path.suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=6) as fh:
            fh.write(blob)
    else:
        path.write_bytes(blob)
    return path
