"""Build the empirical draft stats from ALL collected matches and write the compact artifact
the deployed API loads — instead of rebuilding them in memory from the full dataset, which
OOMs a small instance (Render's 512 MB free tier) as the data grows.

    PYTHONPATH=backend python backend/scripts/export_stats.py

Run on a machine with the data + RAM to spare (the home crawler box). Publish it with
``python -m bsdraft.collect.publish --only-stats`` (the crawler does both each cycle). The API
pulls it via ``STATS_URL`` and loads it in tens of MB. Output: data/processed/stats.json.gz.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from bsdraft.config import settings
from bsdraft.constants import PROCESSED_DIR
from bsdraft.engine.stats import build_bracketed
from bsdraft.engine.stats_store import save_stats

DEFAULT_OUT = PROCESSED_DIR / "stats.json.gz"


def export(out: Path = DEFAULT_OUT, max_matches: int = 0) -> Path:
    """Build the global + per-bracket stats (all matches by default) and save them to ``out``."""
    t = time.time()
    g, br = build_bracketed(halflife_days=settings.stats_halflife_days, max_matches=max_matches)
    save_stats(g, br, out)
    mb = out.stat().st_size / 1e6
    print(f"built stats from {g.n} matches ({len(br)} bracket(s)) -> {out} ({mb:.2f} MB) in {time.time()-t:.1f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + save precomputed stats for the API to load.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output artifact (.json or .json.gz)")
    ap.add_argument("--max-matches", type=int, default=0, help="cap matches used (0 = all)")
    args = ap.parse_args()
    export(args.out, args.max_matches)


if __name__ == "__main__":
    main()
