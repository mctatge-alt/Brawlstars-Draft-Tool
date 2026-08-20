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


# How many bytes of the tags array to decode at a time. The chunk is the only part of the index
# that exists as Python strings at once, so this is the knob that trades load time against peak
# memory: 4 MB of JSON text is ~300k tags, a few MB of transient objects.
_TAG_CHUNK_BYTES = 4 << 20


def _element_spans(blob: bytes, start: int, end: int, approx: int):
    """Yield ``(a, b)`` offsets carving ``blob[start:end]`` — a JSON array body of strings — into
    slices that are each themselves a valid array body.

    Offsets rather than slices on purpose: the tags body is ~30 MB, and returning it (or the
    chunks) as new bytes objects would copy exactly the megabytes this loader exists to avoid.
    Cuts land on the 3-byte ``","`` separator, which cannot occur inside a Brawl Stars tag (ASCII
    alphanumerics; JSON would escape an embedded quote anyway). If no separator is found the
    remainder is yielded whole, so a surprise degrades to one big chunk rather than corrupt data.
    """
    pos = start
    while pos < end:
        if end - pos <= approx:
            yield pos, end
            return
        i = blob.find(b'","', pos + approx, end)
        if i == -1:
            yield pos, end
            return
        yield pos, i + 1        # keep this element's closing quote
        pos = i + 2             # resume at the next element's opening quote


def _array_span(blob: bytes, key: bytes):
    """``(start, end)`` of the body of ``"<key>":[ ... ]``, or None if not laid out that way."""
    at = blob.find(b'"' + key + b'":[')
    if at < 0:
        return None
    start = at + len(key) + 4
    end = blob.find(b"]", start)     # neither tags nor tiers contain a bracket
    return None if end < 0 else (start, end)


def _load_frugally(blob: bytes) -> Optional[RankIndex]:
    """Decode the artifact without ever holding the whole index as Python objects.

    ``json.loads`` on the full document is what makes this file dangerous on a 512 MB box: at
    2.5M tags it retains ~193 MB of Python strings, and :meth:`RankIndex.from_arrays` then peaks
    another ~156 MB converting them — ~350 MB transient for a 28 MB result, which is what
    OOM-killed the Render instance on 2026-08-20. Decoding the tags array in slices and folding
    each slice straight into a NumPy array keeps the peak near the result size.

    Returns None if the document isn't in the expected flat layout, so the caller can fall back
    to the plain parse rather than guess.
    """
    tags_span = _array_span(blob, b"tags")
    tiers_span = _array_span(blob, b"tiers")
    if tags_span is None or tiers_span is None:
        return None
    blocks = []
    for a, b in _element_spans(blob, tags_span[0], tags_span[1], _TAG_CHUNK_BYTES):
        if b <= a:
            continue
        # Round-trip through json for the chunk so escapes and unicode stay correct; only this
        # slice's strings are alive at once, and the list is dropped as soon as it is packed.
        blocks.append(_ascii_bytes(json.loads(b"[" + blob[a:b] + b"]")))
    if not blocks:
        tags = np.array([], dtype="S1")
    elif len(blocks) == 1:
        tags = blocks[0]
    else:
        # Concatenating different fixed widths would truncate, so widen every block first, and
        # release each one as it lands so both copies are never fully resident.
        width = max(blk.dtype.itemsize for blk in blocks)
        total = sum(blk.shape[0] for blk in blocks)
        tags = np.empty(total, dtype=f"S{width}")
        at = 0
        for i in range(len(blocks)):
            blk = blocks[i]
            tags[at:at + blk.shape[0]] = blk
            at += blk.shape[0]
            blocks[i] = None
        del blocks
    # Tiers are 1-22, so every int the parser produces is a cached CPython singleton — the list
    # is pointers, not objects, and is cheap enough to take in one bite.
    a, b = tiers_span
    tiers = np.array(json.loads(b"[" + blob[a:b] + b"]"), dtype=np.uint8)
    if tags.size != tiers.size:
        return None
    return RankIndex(tags, tiers)


def load_rank_index(path) -> RankIndex:
    """Load a :class:`RankIndex` from a (optionally gzipped) JSON artifact."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    ver_at = raw.find(b'"version":')
    ver = None
    if ver_at >= 0:
        try:
            ver = int(raw[ver_at + 10:ver_at + 20].split(b",")[0].split(b"}")[0])
        except ValueError:
            ver = None
    if ver is None:                       # not laid out as expected — parse properly to find out
        ver = json.loads(raw).get("version")
    if ver != FORMAT_VERSION:  # format drift -> raise so the caller falls back to a fresh build
        raise ValueError(f"unsupported rank index format version {ver!r} (expected {FORMAT_VERSION})")
    frugal = _load_frugally(raw)
    if frugal is not None:
        return frugal
    payload = json.loads(raw)
    return RankIndex.from_arrays(payload["tags"], payload["tiers"])
