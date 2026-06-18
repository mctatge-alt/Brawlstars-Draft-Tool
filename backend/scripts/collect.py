"""Collect ranked matches via the snowball crawler.

    # one-shot crawl (local only):
    PYTHONPATH=backend python backend/scripts/collect.py --target 3000
    PYTHONPATH=backend python backend/scripts/collect.py --target 500 --countries global,US

    # home daemon for the live site: crawl a batch, publish, sleep, repeat:
    PYTHONPATH=backend python backend/scripts/collect.py --loop 3600 --target 800 --publish

    # …and auto-retrain the model whenever the meta shifts (balance change / new brawler):
    PYTHONPATH=backend python backend/scripts/collect.py --loop 3600 --target 800 --publish --retrain-on-shift

Resumable: re-running continues from the existing matches/visited state in data/raw/. Pass
--publish to upload matches.jsonl.gz to a GitHub Release (see collect/publish.py) so the
deployed API can pull it.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys

from bsdraft.collect import publish as publisher
from bsdraft.collect.client import BrawlStarsClient
from bsdraft.collect.crawler import MATCHES_PATH, Crawler
from bsdraft.config import settings
from bsdraft.constants import REPO_ROOT
from bsdraft.engine.drift import detect_drift


async def _run(target: int, countries: list, revisit_after: float) -> int:
    async with BrawlStarsClient() as client:
        crawler = Crawler(client, revisit_after=revisit_after)
        seeds = [settings.player_tag] if settings.player_tag else []
        backlog = len(crawler.frontier)  # players recovered from prior state, before seeding
        await crawler.seed(countries, seed_tags=seeds)
        queued = len(crawler.frontier)
        print(f"Resuming: {len(crawler.seen_matches)} matches, "
              f"{len(crawler.visited)} players scanned.")
        print(f"Queued {queued} players to scan ({backlog} recovered backlog + "
              f"{queued - backlog} new from rankings ({', '.join(countries)})).")
        new = await crawler.run(target_matches=target)
        print(f"\nDone. +{new} new matches  ->  {MATCHES_PATH}")
        print(f"Total unique matches: {len(crawler.seen_matches)}")
        return new


def _try_publish() -> None:
    try:
        publisher.publish()
    except Exception as e:  # noqa: BLE001 — a publish hiccup shouldn't kill a long crawl loop
        print(f"publish failed: {e}")


def _check_meta(retrain_on_shift: bool) -> None:
    """Run the meta-drift detector on the freshly crawled data and print the report. When the
    meta has shifted and ``--retrain-on-shift`` is set, kick a model retrain so recommendations
    catch up. Never raises — a drift hiccup must not kill a long crawl loop."""
    try:
        report = detect_drift()
    except Exception as e:  # noqa: BLE001
        print(f"meta check failed: {e}")
        return
    print("\n--- meta drift ---")
    print(report.summary())
    if report.shifted and retrain_on_shift:
        _retrain()


def _retrain() -> None:
    """Retrain and re-export the win-prob model so it reflects the shifted meta (re-publish
    afterwards to roll it out). Guarded so a failure can't take down the crawl loop."""
    print("meta shifted -> retraining win-prob model …")
    scripts = REPO_ROOT / "backend" / "scripts"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "backend")}
    try:
        subprocess.run([sys.executable, str(scripts / "train.py")], check=True, env=env)
        subprocess.run([sys.executable, str(scripts / "export_model.py")], check=True, env=env)
        print("retrain complete — re-run with --publish (or republish) to roll out the new model")
    except Exception as e:  # noqa: BLE001
        print(f"retrain failed: {e}")


async def _loop(target: int, countries: list, interval: int, do_publish: bool,
                meta_check: bool, retrain_on_shift: bool, revisit_after: float) -> None:
    cycle = 0
    while True:
        cycle += 1
        print(f"\n=== crawl cycle {cycle} ===")
        await _run(target, countries, revisit_after)
        if do_publish:
            _try_publish()
        if meta_check:
            _check_meta(retrain_on_shift)
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
    ap.add_argument("--no-meta-check", dest="meta_check", action="store_false",
                    help="skip the meta-drift report after each crawl (on by default)")
    ap.add_argument("--retrain-on-shift", action="store_true",
                    help="when the meta-drift check trips, retrain + re-export the win-prob model")
    ap.add_argument("--revisit-hours", type=float, default=None,
                    help="re-scan a known player after this many hours to catch their newer "
                         "ranked games (default: .env CRAWL_REVISIT_HOURS; 0 disables)")
    ap.set_defaults(meta_check=True)
    args = ap.parse_args()
    raw = args.countries.split(",") if args.countries else settings.seed_countries
    countries = [c.strip() for c in raw if c.strip()]
    revisit_hours = args.revisit_hours if args.revisit_hours is not None else settings.crawl_revisit_hours
    revisit_after = max(0.0, revisit_hours) * 3600

    if args.loop > 0 and not args.publish:
        print("note: --loop without --publish — crawling locally only; the live site won't update.")

    try:
        if args.loop > 0:
            asyncio.run(_loop(args.target, countries, args.loop, args.publish,
                              args.meta_check, args.retrain_on_shift, revisit_after))
        else:
            asyncio.run(_run(args.target, countries, revisit_after))
            if args.publish:
                _try_publish()
            if args.meta_check:
                _check_meta(args.retrain_on_shift)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
