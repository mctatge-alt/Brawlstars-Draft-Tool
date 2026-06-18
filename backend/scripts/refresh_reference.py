"""Refresh the bundled reference catalog (brawlers, maps) from the keyless Brawlify API.

Brawl Stars ships new brawlers and rotates ranked maps each season. The model's brawler
vocabulary and the UI's pickable lists come from ``data/reference/*.json`` (static snapshots),
so when an update lands you must refresh them — otherwise a new brawler is invisible to the
tool (absent from the embedding index and the pick list) and silently encodes to index 0.

This fetches the catalogs, validates them, rewrites the snapshots in place (atomic), and
prints what changed. Run it when the meta banner / drift report flags a new brawler. After a
brawler is added, retrain so it gets a real embedding row, then publish + commit:

    PYTHONPATH=backend python backend/scripts/refresh_reference.py             # fetch + write
    PYTHONPATH=backend python backend/scripts/refresh_reference.py --dry-run   # report only

    PYTHONPATH=backend python backend/scripts/train.py \\
      && PYTHONPATH=backend python backend/scripts/export_model.py \\
      && PYTHONPATH=backend python -m bsdraft.collect.publish --model
    git add data/reference/brawlers.json data/reference/maps.json && git commit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import httpx

from bsdraft.constants import RANKED_MODES, REFERENCE_DIR

BRAWLERS_URL = "https://api.brawlify.com/v1/brawlers"
MAPS_URL = "https://api.brawlify.com/v1/maps"


def _fetch(url: str, timeout: float = 30.0) -> dict:
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        resp = client.get(url)
    resp.raise_for_status()
    return resp.json()


def _validate(payload: dict, label: str) -> List[dict]:
    """Make sure the payload looks like a Brawlify catalog before trusting it to overwrite a
    good local snapshot. Raises ValueError on anything suspicious (wrong URL, empty list,
    malformed items) so a bad fetch never clobbers working data."""
    if not isinstance(payload, dict) or not isinstance(payload.get("list"), list):
        raise ValueError(f"{label}: response has no 'list' array — wrong URL or the API changed?")
    items = payload["list"]
    if not items:
        raise ValueError(f"{label}: 'list' is empty — refusing to overwrite local data")
    bad = [x for x in items
           if not isinstance(x, dict) or not isinstance(x.get("id"), int) or not x.get("name")]
    if bad:
        raise ValueError(f"{label}: {len(bad)} item(s) missing an integer id/name — refusing to overwrite")
    return items


def _current_list(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("list", [])
    except (json.JSONDecodeError, OSError):
        return []


def _brawler_names(items: List[dict]) -> Dict[int, str]:
    return {x["id"]: x.get("name", str(x["id"])) for x in items if isinstance(x.get("id"), int)}


def _ranked_map_names(items: List[dict]) -> Dict[int, str]:
    """Active maps in the ranked mode set, id -> 'Name (Mode)' — mirrors reference.load_ranked_maps."""
    out: Dict[int, str] = {}
    for x in items:
        if not isinstance(x.get("id"), int) or x.get("disabled"):
            continue
        mode = (x.get("gameMode") or {}).get("name")
        if mode in RANKED_MODES:
            out[x["id"]] = f'{x.get("name", "?")} ({mode})'
    return out


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    # Minified, matching the committed snapshots so diffs stay small.
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def refresh(brawlers_url: str = BRAWLERS_URL, maps_url: str = MAPS_URL, dry_run: bool = False) -> bool:
    """Fetch + validate both catalogs, report the diff vs the local snapshots, and (unless
    dry_run) rewrite them. Returns True iff a new brawler was detected."""
    b_path = REFERENCE_DIR / "brawlers.json"
    m_path = REFERENCE_DIR / "maps.json"

    before_b = _brawler_names(_current_list(b_path))
    before_m = _ranked_map_names(_current_list(m_path))

    b_payload = _fetch(brawlers_url)
    m_payload = _fetch(maps_url)
    after_b = _brawler_names(_validate(b_payload, "brawlers"))
    after_m = _ranked_map_names(_validate(m_payload, "maps"))

    new_brawlers = [f"{after_b[i]} (#{i})" for i in after_b if i not in before_b]
    gone_brawlers = [f"{before_b[i]} (#{i})" for i in before_b if i not in after_b]
    new_maps = sorted(after_m[i] for i in after_m if i not in before_m)

    print(f"brawlers: {len(before_b)} -> {len(after_b)}")
    if new_brawlers:
        print("  NEW: " + ", ".join(new_brawlers))
    if gone_brawlers:
        print("  removed: " + ", ".join(gone_brawlers))
    print(f"ranked maps: {len(before_m)} -> {len(after_m)}")
    if new_maps:
        print("  NEW: " + ", ".join(new_maps))
    if not (new_brawlers or gone_brawlers or new_maps):
        print("  (no changes vs local snapshots)")

    if dry_run:
        print("\n--dry-run: no files written.")
        return bool(new_brawlers)

    _write_atomic(b_path, b_payload)
    _write_atomic(m_path, m_payload)
    print(f"\nwrote {b_path}\n      {m_path}")
    if new_brawlers:
        print("\nNew brawler(s) — they encode to index 0 until you retrain. Roll out:")
        print("  python backend/scripts/train.py && python backend/scripts/export_model.py \\")
        print("    && python -m bsdraft.collect.publish --model")
        print("  git add data/reference/*.json && git commit")
    return bool(new_brawlers)


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh data/reference/{brawlers,maps}.json from Brawlify.")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; don't write")
    ap.add_argument("--brawlers-url", default=BRAWLERS_URL, help="override the brawlers catalog URL")
    ap.add_argument("--maps-url", default=MAPS_URL, help="override the maps catalog URL")
    args = ap.parse_args()
    try:
        refresh(args.brawlers_url, args.maps_url, dry_run=args.dry_run)
    except (httpx.HTTPError, ValueError) as e:
        raise SystemExit(f"refresh failed: {e}")


if __name__ == "__main__":
    main()
