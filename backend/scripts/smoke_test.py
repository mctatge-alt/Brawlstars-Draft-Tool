"""Smoke test: verify the API token works and inspect real response shapes.

After setting BRAWLSTARS_API_TOKEN and PLAYER_TAG in .env:

    PYTHONPATH=backend python backend/scripts/smoke_test.py
    # or pass a tag explicitly:
    PYTHONPATH=backend python backend/scripts/smoke_test.py '#2YULP2RUV'

Fetches your player profile + recent battle log, prints a structural summary, and
saves one raw battle to data/raw/sample_battle.json so the match parser can be
finalized against real data.
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter

from bsdraft.collect.client import BrawlStarsClient, normalize_tag
from bsdraft.config import settings
from bsdraft.constants import RAW_DIR


async def main(tag: str) -> None:
    async with BrawlStarsClient() as client:
        print(f"Fetching player #{tag} ...")
        player = await client.get_player(tag)
        print(
            f"  name={player.get('name')!r}  trophies={player.get('trophies')}  "
            f"3v3 victories={player.get('3vs3Victories')}  "
            f"brawlers owned={len(player.get('brawlers', []))}"
        )

        print("Fetching battle log ...")
        battles = await client.get_battlelog(tag)
        print(f"  {len(battles)} recent battles")

        modes = Counter((b.get("event") or {}).get("mode") for b in battles)
        types = Counter((b.get("battle") or {}).get("type") for b in battles)
        results = Counter((b.get("battle") or {}).get("result") for b in battles)
        print("  modes:  ", dict(modes))
        print("  types:  ", dict(types))
        print("  results:", dict(results))

        if battles:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            sample_path = RAW_DIR / "sample_battle.json"
            sample_path.write_text(json.dumps(battles[0], indent=2))
            print(f"\nSaved one raw battle -> {sample_path}")
            b0 = battles[0]
            print("  battle entry keys:", list(b0.keys()))
            print("  event:", b0.get("event"))
            battle = b0.get("battle", {})
            print("  battle keys:", list(battle.keys()))
            teams = battle.get("teams") or battle.get("players")
            kind = "teams" if battle.get("teams") else ("players" if battle.get("players") else "none")
            print(f"  participants under '{kind}':",
                  f"{len(teams)} group(s)" if teams else "none")

        print("\n✅ Token works and the API is reachable.")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else settings.player_tag
    if not tag:
        sys.exit("No player tag. Pass one as an argument or set PLAYER_TAG in .env.")
    asyncio.run(main(normalize_tag(tag)))
