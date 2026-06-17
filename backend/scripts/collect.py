"""Collect ranked matches via the snowball crawler.

    PYTHONPATH=backend python backend/scripts/collect.py --target 3000
    PYTHONPATH=backend python backend/scripts/collect.py --target 500 --countries global,US

Resumable: re-running continues from the existing matches/visited state in data/raw/.
"""
from __future__ import annotations

import argparse
import asyncio

from bsdraft.collect.client import BrawlStarsClient
from bsdraft.collect.crawler import MATCHES_PATH, Crawler
from bsdraft.config import settings


async def _run(target: int, countries: list) -> None:
    async with BrawlStarsClient() as client:
        crawler = Crawler(client)
        seeds = [settings.player_tag] if settings.player_tag else []
        n_seed = await crawler.seed(countries, seed_tags=seeds)
        print(f"Seeded {n_seed} tags from rankings ({', '.join(countries)}).")
        print(f"Resuming with {len(crawler.seen_matches)} matches, "
              f"{len(crawler.visited)} players already visited.")
        new = await crawler.run(target_matches=target)
        print(f"\nDone. +{new} new matches  ->  {MATCHES_PATH}")
        print(f"Total unique matches: {len(crawler.seen_matches)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Crawl ranked Brawl Stars matches.")
    ap.add_argument("--target", type=int, default=2000, help="number of NEW matches to collect")
    ap.add_argument("--countries", default=None,
                    help="comma-separated country codes; defaults to .env CRAWL_SEED_COUNTRIES")
    args = ap.parse_args()
    raw = args.countries.split(",") if args.countries else settings.seed_countries
    countries = [c.strip() for c in raw if c.strip()]
    asyncio.run(_run(args.target, countries))


if __name__ == "__main__":
    main()
