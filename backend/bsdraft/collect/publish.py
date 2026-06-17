"""Publish the collected matches to a GitHub Release for the deployed API to pull.

Gzips ``data/raw/matches.jsonl`` and uploads it as the ``matches.jsonl.gz`` asset on a
fixed release tag (default ``data-latest``), replacing the previous asset. The cloud API's
``DATA_URL`` points at that asset's stable download URL:

    https://github.com/<owner>/<repo>/releases/download/data-latest/matches.jsonl.gz

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

from bsdraft.constants import RAW_DIR

MATCHES_PATH = RAW_DIR / "matches.jsonl"
GZ_PATH = RAW_DIR / "matches.jsonl.gz"
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish matches.jsonl.gz to a GitHub Release.")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="release tag to upload to")
    args = ap.parse_args()
    publish(args.tag)


if __name__ == "__main__":
    main()
