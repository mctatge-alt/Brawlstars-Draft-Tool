"""Pull the published matches dataset from a remote URL (the GitHub Release asset).

The home crawler publishes ``data/raw/matches.jsonl`` (gzipped) to a GitHub Release; the
deployed API calls :func:`sync_matches` periodically to refresh its local copy so it can
rebuild draft stats without a restart. Downloads to the same path ``iter_matches()`` reads
by default, so a plain ``DraftStats()`` picks up the new data.

Robust by design: a conditional GET (ETag) skips the download when nothing changed, a
content hash avoids needless rebuilds when the bytes are identical, and any network/HTTP
failure leaves the last-good local copy in place (returns False rather than raising).
"""
from __future__ import annotations

import gzip
import hashlib
import logging
from pathlib import Path

import httpx

from bsdraft.constants import RAW_DIR

logger = logging.getLogger(__name__)

MATCHES_PATH = RAW_DIR / "matches.jsonl"
_ETAG_PATH = RAW_DIR / ".matches.etag"
_SHA_PATH = RAW_DIR / ".matches.sha"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _decompress(raw: bytes) -> bytes:
    """Gunzip if gzip-framed, else pass through (so DATA_URL may point at .gz or raw .jsonl)."""
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def sync_matches(url: str, timeout: float = 60.0) -> bool:
    """Refresh the local matches file from ``url``. Returns True iff local data changed."""
    if not url:
        return False
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    headers = {}
    etag = _read(_ETAG_PATH)
    if etag and MATCHES_PATH.exists():
        headers["If-None-Match"] = etag

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url, headers=headers)
        if resp.status_code == 304:
            return False
        resp.raise_for_status()
        data = _decompress(resp.content)
    except Exception as e:  # noqa: BLE001 — never let a sync failure take down serving
        logger.warning("matches sync failed (%s); keeping last-good data", e)
        return False

    new_etag = resp.headers.get("ETag", "")
    if new_etag:
        _ETAG_PATH.write_text(new_etag, encoding="utf-8")

    sha = hashlib.sha256(data).hexdigest()
    if sha == _read(_SHA_PATH) and MATCHES_PATH.exists():
        return False  # bytes identical — nothing to rebuild

    tmp = RAW_DIR / "matches.jsonl.tmp"
    tmp.write_bytes(data)
    tmp.replace(MATCHES_PATH)  # atomic swap on POSIX
    _SHA_PATH.write_text(sha, encoding="utf-8")
    logger.info("matches updated (%.1f MB)", len(data) / 1e6)
    return True
