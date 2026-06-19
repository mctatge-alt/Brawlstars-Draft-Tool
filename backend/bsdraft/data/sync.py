"""Pull published artifacts (the matches dataset and the win-prob model) from remote URLs.

The home crawler publishes ``data/raw/matches.jsonl`` (gzipped) and, after a retrain, the
``winprob.npz`` model to a GitHub Release. The deployed API calls :func:`sync_matches` and
:func:`sync_model` periodically to refresh its local copies so it can rebuild draft stats and
hot-swap the model without a restart. Each downloads to the same path the engine reads by
default, so a plain rebuild/reload picks up the new bytes.

Robust by design: a conditional GET (ETag) skips the download when nothing changed, a content
hash avoids needless rebuilds when the bytes are identical, and any network/HTTP failure
leaves the last-good local copy in place (returns False rather than raising).
"""
from __future__ import annotations

import gzip
import hashlib
import logging
from pathlib import Path

import httpx

from bsdraft.constants import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

MATCHES_PATH = RAW_DIR / "matches.jsonl"
_ETAG_PATH = RAW_DIR / ".matches.etag"
_SHA_PATH = RAW_DIR / ".matches.sha"

MODEL_PATH = PROCESSED_DIR / "winprob.npz"
_MODEL_ETAG_PATH = PROCESSED_DIR / ".winprob.etag"
_MODEL_SHA_PATH = PROCESSED_DIR / ".winprob.sha"

STATS_PATH = PROCESSED_DIR / "stats.json"
_STATS_ETAG_PATH = PROCESSED_DIR / ".stats.etag"
_STATS_SHA_PATH = PROCESSED_DIR / ".stats.sha"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _decompress(raw: bytes) -> bytes:
    """Gunzip if gzip-framed, else pass through (so a URL may point at .gz or the raw file;
    winprob.npz is zip-framed, not gzip, so it passes through untouched)."""
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def _sync_file(url: str, dest: Path, etag_path: Path, sha_path: Path,
               timeout: float, label: str) -> bool:
    """Refresh ``dest`` from ``url`` if it changed. Returns True iff the local copy was
    rewritten. Conditional GET (ETag) skips the download when nothing changed; a content hash
    skips the rewrite when the bytes are identical; any network/HTTP failure leaves the
    last-good local copy in place (returns False, never raises)."""
    if not url:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    etag = _read(etag_path)
    if etag and dest.exists():
        headers["If-None-Match"] = etag

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url, headers=headers)
        if resp.status_code == 304:
            return False
        resp.raise_for_status()
        data = _decompress(resp.content)
    except Exception as e:  # noqa: BLE001 — never let a sync failure take down serving
        logger.warning("%s sync failed (%s); keeping last-good copy", label, e)
        return False

    new_etag = resp.headers.get("ETag", "")
    if new_etag:
        etag_path.write_text(new_etag, encoding="utf-8")

    sha = hashlib.sha256(data).hexdigest()
    if sha == _read(sha_path) and dest.exists():
        return False  # bytes identical — nothing downstream to rebuild

    tmp = dest.parent / (dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)  # atomic swap on POSIX
    sha_path.write_text(sha, encoding="utf-8")
    logger.info("%s updated (%.2f MB)", label, len(data) / 1e6)
    return True


def sync_matches(url: str, timeout: float = 60.0) -> bool:
    """Refresh the local matches dataset from ``url``. Returns True iff local data changed."""
    return _sync_file(url, MATCHES_PATH, _ETAG_PATH, _SHA_PATH, timeout, "matches")


def sync_model(url: str, timeout: float = 60.0) -> bool:
    """Refresh the local win-prob model (winprob.npz) from ``url``. Returns True iff it changed,
    so the caller can reload and hot-swap the served model."""
    return _sync_file(url, MODEL_PATH, _MODEL_ETAG_PATH, _MODEL_SHA_PATH, timeout, "model")


def sync_stats(url: str, timeout: float = 60.0) -> bool:
    """Refresh the precomputed empirical stats (stats.json) from ``url``. Returns True iff it
    changed, so the caller can reload and hot-swap the served stats — no in-memory rebuild from
    the full match dataset (which OOMs a small instance as the data grows)."""
    return _sync_file(url, STATS_PATH, _STATS_ETAG_PATH, _STATS_SHA_PATH, timeout, "stats")
