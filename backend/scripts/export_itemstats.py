"""Build the per-item win-rate table the API serves for data-driven loadout advice, from the
matches x ownership-profiles join (single-item-owner inference).

    PYTHONPATH=backend python backend/scripts/export_itemstats.py

Run on the home box (needs data/raw/profiles.jsonl from ``python -m bsdraft.collect.profiles``, plus
the matches dataset). Publish it with ``python -m bsdraft.collect.publish --only-itemstats``; the
API pulls it via ITEMSTATS_URL and /api/loadout flips from the effect heuristic to measured picks
wherever a cell clears the sample + BH-FDR gate. Output: data/processed/itemstats.json.gz.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from bsdraft.constants import PROCESSED_DIR, RAW_DIR
from bsdraft.data.itemstats_build import build_itemstats
from bsdraft.engine.itemstats import save_itemstats

DEFAULT_OUT = PROCESSED_DIR / "itemstats.json.gz"
DEFAULT_PROFILES = RAW_DIR / "profiles.jsonl"


def export(out: Path = DEFAULT_OUT, profiles: Path = DEFAULT_PROFILES,
           matches: Path | None = None) -> Path:
    if not profiles.exists():
        raise FileNotFoundError(
            f"No profiles at {profiles} — collect them first (python -m bsdraft.collect.profiles).")
    t = time.time()
    payload = build_itemstats(matches, profiles, params={"stamp_time": True})
    save_itemstats(payload, out)
    n = payload["meta"]["n_cells"]
    sig = sum(c["significant"] for c in payload["cells"].values())
    mb = out.stat().st_size / 1e6
    print(f"built item stats: {n} cells ({sig} significant), coverage "
          f"{payload['meta'].get('coverage')} -> {out} ({mb:.2f} MB) in {time.time()-t:.1f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + save the per-item win-rate table for the API to load.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output artifact (.json or .json.gz)")
    ap.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES, help="ownership profiles jsonl")
    ap.add_argument("--matches", type=Path, default=None, help="matches jsonl (default: synced dataset)")
    args = ap.parse_args()
    export(args.out, args.profiles, args.matches)


if __name__ == "__main__":
    main()
