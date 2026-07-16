"""Serialize the player-rank index to a compact artifact and load it back.

Lets the deployed API **load** a precomputed ``tag -> Ranked-tier`` index instead of
**building** it in memory from the full match dataset. The built form is a Python dict with one
entry per crawled tag — at 1.3M tags that's ~200 MB resident and ~45 s to build, scanning the
whole ``matches.jsonl``; it grows with the crawl and was threatening Render's 512 MB free tier
(see ``docs`` / the OOM history). The home machine (with RAM to spare) builds the full index and
publishes ``rank_index.json.gz``; the API syncs it and loads it into a compact NumPy-backed
lookup (~20 MB), with the match data never resident for this. Mirrors the stats/model
publish-load split (:mod:`bsdraft.engine.stats_store`, ``winprob.npz``).

Only the **tier** is served (1-22); the ``build_rank_index`` timestamp is just for picking the
latest tier per tag during the build, so it's dropped from the artifact. The serve form keeps the
tags as a single sorted NumPy byte-string array (``np.searchsorted`` for O(log n) lookup) and the
tiers as a ``uint8`` array — far smaller than 1.3M Python str/int/dict objects.

Format: gzipped JSON ``{"version":1, "tags":[...ascending...], "tiers":[...]}`` — parallel arrays,
tags sorted so a lookup is a binary search. (A plain ``{tag: tier}`` dict would JSON-encode just
as small but reload into exactly the ~120 MB of Python objects we're avoiding.)
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

FORMAT_VERSION = 1


def _ascii_bytes(tags) -> np.ndarray:
    """Pack tags into a sorted-order ``S`` byte array, encoding ascii-and-ignore so a stray
    non-ASCII tag can't raise (numpy ``dtype='S'`` would otherwise UnicodeEncodeError). Mirrors
    the same encoding :meth:`RankIndex.get` uses for the query, so lookups stay consistent."""
    return np.array([t.encode("ascii", "ignore") if isinstance(t, str) else t for t in tags], dtype="S")


class RankIndex:
    """Compact, read-only ``tag -> tier`` lookup backed by parallel NumPy arrays.

    ``tags`` is ascending (lexicographic, ASCII) so :meth:`get` binary-searches it. Build it via
    :meth:`from_mapping` (from the dict :func:`bsdraft.engine.playerrank.build_rank_index` returns)
    or :func:`load_rank_index` (from the published artifact)."""

    __slots__ = ("_tags", "_tiers")

    def __init__(self, tags: np.ndarray, tiers: np.ndarray):
        # tags: sorted ascending, dtype 'S*'; tiers: uint8, parallel to tags.
        self._tags = tags
        self._tiers = tiers

    @classmethod
    def from_mapping(cls, idx: Dict[str, object]) -> "RankIndex":
        """From ``{tag: (ts, tier)}`` (what ``build_rank_index`` returns) or ``{tag: tier}``."""
        items = []
        for tag, v in idx.items():
            tier = v[1] if isinstance(v, tuple) else v
            items.append((tag, int(tier)))
        items.sort(key=lambda kv: kv[0])
        tags = _ascii_bytes(t for t, _ in items)
        tiers = np.array([t for _, t in items], dtype=np.uint8)
        return cls(tags, tiers)

    @classmethod
    def from_arrays(cls, tags_sorted, tiers) -> "RankIndex":
        """From already-sorted parallel lists/arrays (the artifact's on-disk form)."""
        return cls(_ascii_bytes(tags_sorted), np.asarray(tiers, dtype=np.uint8))

    def get(self, tag: str) -> Optional[int]:
        """The Ranked tier (1-22) for ``tag``, or None if it isn't in the index."""
        if self._tags.size == 0:
            return None
        key = tag.encode("ascii", "ignore")
        i = int(np.searchsorted(self._tags, key))
        if i < self._tags.size and self._tags[i] == key:
            return int(self._tiers[i])
        return None

    def __len__(self) -> int:
        return int(self._tags.size)


def index_payload(idx: Dict[str, Tuple[int, int]]) -> dict:
    """Serialize ``build_rank_index``'s ``{tag: (ts, tier)}`` to the compact artifact dict."""
    items = sorted((tag, int(v[1] if isinstance(v, tuple) else v)) for tag, v in idx.items())
    return {
        "version": FORMAT_VERSION,
        "tags": [t for t, _ in items],
        "tiers": [t for _, t in items],
    }


def save_rank_index(idx: Dict[str, Tuple[int, int]], path) -> Path:
    """Write the ``tag -> tier`` index to ``path`` (gzipped JSON if it ends in ``.gz``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(index_payload(idx), separators=(",", ":")).encode("utf-8")
    if path.suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=6) as fh:
            fh.write(data)
    else:
        path.write_bytes(data)
    return path


def load_rank_index(path) -> RankIndex:
    """Load a :class:`RankIndex` from a (optionally gzipped) JSON artifact."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw)
    ver = payload.get("version")
    if ver != FORMAT_VERSION:  # format drift -> raise so the caller falls back to a fresh build
        raise ValueError(f"unsupported rank index format version {ver!r} (expected {FORMAT_VERSION})")
    return RankIndex.from_arrays(payload["tags"], payload["tiers"])
