"""Publish the collected matches (and the trained model) to a GitHub Release for the API to pull.

Gzips ``data/raw/matches.jsonl`` and uploads it as the ``matches.jsonl.gz`` asset on a fixed
release tag (default ``data-latest``), replacing the previous asset; :func:`publish_model`
uploads ``winprob.npz`` alongside it. The cloud API's ``DATA_URL`` / ``MODEL_URL`` point at
those assets' stable download URLs:

    https://github.com/<owner>/<repo>/releases/download/data-latest/matches.jsonl.gz
    https://github.com/<owner>/<repo>/releases/download/data-latest/winprob.npz

Requires the GitHub CLI (`gh`), authenticated, run from inside the repo (gh infers the
owner/repo from the git remote). Keeping this on your machine is what lets the crawl keep
using the IP-locked Supercell key while the cloud stays free.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
from pathlib import Path

from bsdraft.constants import PROCESSED_DIR, RAW_DIR

MATCHES_PATH = RAW_DIR / "matches.jsonl"
GZ_PATH = RAW_DIR / "matches.jsonl.gz"
MODEL_PATH = PROCESSED_DIR / "winprob.npz"
STATS_PATH = PROCESSED_DIR / "stats.json.gz"
DEFAULT_TAG = "data-latest"


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def _ensure_release(tag: str) -> None:
    if _gh("release", "view", tag).returncode != 0:
        res = _gh(
            "release", "create", tag,
            "--title", "Latest dataset",
            "--notes", "Rolling ranked-match dataset powering the live draft API (updated by the crawler).",
        )
        if res.returncode != 0:
            raise RuntimeError(f"gh release create failed: {res.stderr.strip()}")


def gzip_matches(src: Path = MATCHES_PATH, dst: Path = GZ_PATH) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"No matches file at {src} — run the crawler first.")
    with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout)
    return dst


def publish(tag: str = DEFAULT_TAG) -> None:
    gz = gzip_matches()
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(gz), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload failed: {res.stderr.strip()}")
    print(f"published {gz.name} ({gz.stat().st_size / 1e6:.1f} MB) -> release '{tag}'")


def publish_model(tag: str = DEFAULT_TAG) -> None:
    """Upload winprob.npz to the release so an API with MODEL_URL set can hot-swap it. Run
    after export_model.py (the crawler does this automatically on a retrain-on-shift)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH} — export it first (scripts/export_model.py).")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(MODEL_PATH), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (model) failed: {res.stderr.strip()}")
    print(f"published {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.0f} KB) -> release '{tag}'")


def publish_stats(tag: str = DEFAULT_TAG) -> None:
    """Upload the precomputed stats (stats.json.gz) to the release so an API with STATS_URL set
    loads them instead of rebuilding from the full dataset. Run after scripts/export_stats.py
    (the crawler does this automatically each publish cycle)."""
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"No stats at {STATS_PATH} — build them first (scripts/export_stats.py).")
    _ensure_release(tag)
    res = _gh("release", "upload", tag, str(STATS_PATH), "--clobber")
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload (stats) failed: {res.stderr.strip()}")
    print(f"published {STATS_PATH.name} ({STATS_PATH.stat().st_size / 1e6:.1f} MB) -> release '{tag}'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish the dataset and/or model/stats to a GitHub Release.")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="release tag to upload to")
    ap.add_argument("--model", action="store_true", help="also upload winprob.npz (the model)")
    ap.add_argument("--stats", action="store_true", help="also upload stats.json.gz (precomputed stats)")
    ap.add_argument("--only-model", action="store_true", help="upload only winprob.npz, not the dataset")
    ap.add_argument("--only-stats", action="store_true", help="upload only stats.json.gz, not the dataset")
    args = ap.parse_args()
    if args.only_model:
        publish_model(args.tag)
        return
    if args.only_stats:
        publish_stats(args.tag)
        return
    publish(args.tag)
    if args.model:
        publish_model(args.tag)
    if args.stats:
        publish_stats(args.tag)


if __name__ == "__main__":
    main()
