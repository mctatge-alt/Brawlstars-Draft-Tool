"""Collect ranked matches via the snowball crawler.

    # one-shot crawl (local only):
    PYTHONPATH=backend python backend/scripts/collect.py --target 3000
    PYTHONPATH=backend python backend/scripts/collect.py --target 500 --countries global,US

    # home daemon for the live site: crawl a batch, publish, sleep, repeat:
    PYTHONPATH=backend python backend/scripts/collect.py --loop 3600 --target 800 --publish

Resumable: re-running continues from the existing matches/visited state in data/raw/. Pass
--publish to upload matches.jsonl.gz to a GitHub Release (see collect/publish.py) so the
deployed API can pull it.
"""
from __future__ import annotations

import argparse
import asyncio

from bsdraft.collect import publish as publisher
from bsdraft.collect.client import BrawlStarsClient
from bsdraft.collect.crawler import MATCHES_PATH, Crawler
from bsdraft.config import settings


async def _run(target: int, countries: list) -> int:
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
        return new


def _try_publish() -> None:
    try:
        publisher.publish()
    except Exception as e:  # noqa: BLE001 — a publish hiccup shouldn't kill a long crawl loop
        print(f"publish failed: {e}")


async def _loop(target: int, countries: list, interval: int, do_publish: bool) -> None:
    cycle = 0
    while True:
        cycle += 1
        print(f"\n=== crawl cycle {cycle} ===")
        await _run(target, countries)
        if do_publish:
            _try_publish()
        print(f"sleeping {interval}s …  (Ctrl-C to stop)")
        await asyncio.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="Crawl ranked Brawl Stars matches.")
    ap.add_argument("--target", type=int, default=2000, help="number of NEW matches to collect per run")
    ap.add_argument("--countries", default=None,
                    help="comma-separated country codes; defaults to .env CRAWL_SEED_COUNTRIES")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                    help="run forever: crawl --target, publish, sleep SECONDS, repeat")
    ap.add_argument("--publish", action="store_true",
                    help="after each crawl, upload matches.jsonl.gz to the GitHub Release (live site)")
    args = ap.parse_args()
    raw = args.countries.split(",") if args.countries else settings.seed_countries
    countries = [c.strip() for c in raw if c.strip()]

    if args.loop > 0 and not args.publish:
        print("note: --loop without --publish — crawling locally only; the live site won't update.")

    try:
        if args.loop > 0:
            asyncio.run(_loop(args.target, countries, args.loop, args.publish))
        else:
            asyncio.run(_run(args.target, countries))
            if args.publish:
                _try_publish()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
