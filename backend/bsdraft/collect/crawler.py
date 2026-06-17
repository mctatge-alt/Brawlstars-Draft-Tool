"""Snowball crawler: seed tags from rankings, expand via battle-log player tags, dedup.

The official API is player-centric, so we BFS the player graph: fetch a player's recent
battles, harvest the 5 other tags in each ranked match, enqueue them, repeat. Matches are
deduped by a stable key (battle time + sorted player tags), since the same match appears
in up to 6 players' logs. State (matches + visited tags) is persisted for resumable runs.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict
from typing import Iterable

from tqdm import tqdm

from bsdraft.collect.client import BrawlStarsClient, BrawlStarsError, normalize_tag
from bsdraft.collect.match import parse_match
from bsdraft.constants import RAW_DIR

MATCHES_PATH = RAW_DIR / "matches.jsonl"
VISITED_PATH = RAW_DIR / "visited_tags.txt"


class Crawler:
    def __init__(self, client: BrawlStarsClient):
        self.client = client
        self.known: set = set()        # every tag ever enqueued
        self.visited: set = set()      # tags whose battlelog we've fetched
        self.seen_matches: set = set()
        self.frontier: deque = deque()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        if MATCHES_PATH.exists():
            with open(MATCHES_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        self.seen_matches.add(json.loads(line)["match_key"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        if VISITED_PATH.exists():
            with open(VISITED_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    tag = line.strip()
                    if tag:
                        self.visited.add(tag)
                        self.known.add(tag)

    def _enqueue(self, tag: str) -> None:
        t = normalize_tag(tag)
        if t and t not in self.known:
            self.known.add(t)
            self.frontier.append(t)

    async def seed(self, countries: Iterable[str], seed_tags: Iterable[str] = ()) -> int:
        for tag in seed_tags:
            self._enqueue(tag)
        for country in countries:
            try:
                players = await self.client.get_top_players(country)
            except BrawlStarsError:
                continue
            for p in players:
                self._enqueue(p.get("tag", ""))
        return len(self.frontier)

    async def run(self, target_matches: int) -> int:
        new = 0
        out = open(MATCHES_PATH, "a", encoding="utf-8")
        vis = open(VISITED_PATH, "a", encoding="utf-8")
        pbar = tqdm(total=target_matches, desc="matches", unit="match")
        try:
            while self.frontier and new < target_matches:
                tag = self.frontier.popleft()
                if tag in self.visited:
                    continue
                self.visited.add(tag)
                vis.write(tag + "\n")
                vis.flush()
                try:
                    battles = await self.client.get_battlelog(tag)
                except BrawlStarsError:
                    continue
                for entry in battles:
                    match = parse_match(entry, queried_tag=tag)
                    if match is None:
                        continue
                    for ptag in match.player_tags:
                        self._enqueue(ptag)
                    if match.match_key in self.seen_matches:
                        continue
                    self.seen_matches.add(match.match_key)
                    out.write(json.dumps(asdict(match), ensure_ascii=False) + "\n")
                    out.flush()
                    new += 1
                    pbar.update(1)
                    if new >= target_matches:
                        break
        finally:
            pbar.close()
            out.close()
            vis.close()
        return new
