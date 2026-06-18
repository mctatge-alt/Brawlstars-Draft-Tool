"""Snowball crawler: seed tags from rankings, expand via battle-log player tags, dedup.

The official API is player-centric, so we BFS the player graph: fetch a player's recent
battles, harvest the 5 other tags in each ranked match, enqueue them, repeat. Matches are
deduped by a stable key (battle time + sorted player tags), since the same match appears in
up to 6 players' logs. State is persisted for resumable runs: matches in matches.jsonl, and
per-player last-fetch timestamps in visited_tags.txt.

The frontier itself is not persisted — it's rebuilt on load from the players seen in stored
matches, so a resumed run continues the BFS across the whole known backlog instead of
restarting from the leaderboard seeds.

Re-scanning: the API only exposes a player's last ~25 battles, so a single fetch is one
snapshot. We record *when* each player was last fetched and re-enqueue them once that is
older than ``revisit_after`` seconds, so an active player's later ranked games get picked up
over time. Set the window to 0 to disable re-scanning (visit each player at most once).
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict
from typing import Iterable

from tqdm import tqdm

from bsdraft.collect.client import BrawlStarsClient, BrawlStarsError, normalize_tag
from bsdraft.collect.match import parse_match
from bsdraft.constants import RAW_DIR

MATCHES_PATH = RAW_DIR / "matches.jsonl"
VISITED_PATH = RAW_DIR / "visited_tags.txt"

DEFAULT_REVISIT_AFTER = 12 * 3600  # re-scan a known player after this many seconds; 0 disables


class Crawler:
    def __init__(self, client: BrawlStarsClient, revisit_after: float = DEFAULT_REVISIT_AFTER):
        self.client = client
        self.revisit_after = revisit_after
        self.visited: dict = {}        # tag -> epoch seconds of last battlelog fetch
        self.queued: set = set()       # tags currently in the frontier (in-run dedup)
        self.seen_matches: set = set()
        self.frontier: deque = deque()
        self._now = time.time()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _eligible(self, tag: str) -> bool:
        """A tag may be (re)enqueued if never scanned, or last scanned longer ago than the
        revisit window. With ``revisit_after <= 0`` a scanned tag is never revisited."""
        last = self.visited.get(tag)
        if last is None:
            return True
        if self.revisit_after <= 0:
            return False
        return (self._now - last) >= self.revisit_after

    def _enqueue(self, tag: str) -> None:
        t = normalize_tag(tag)
        if t and t not in self.queued and self._eligible(t):
            self.queued.add(t)
            self.frontier.append(t)

    def _load_state(self) -> None:
        discovered: set = set()
        if MATCHES_PATH.exists():
            with open(MATCHES_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = rec.get("match_key")
                    if key:
                        self.seen_matches.add(key)
                    discovered.update(rec.get("player_tags", ()))
        if VISITED_PATH.exists():
            with open(VISITED_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    # Format is "TAG\tTS"; legacy rows are a bare "TAG" (no timestamp) — treat
                    # those as scanned long ago (ts 0) so they're eligible for an immediate
                    # re-scan. Keep the newest timestamp if a tag appears more than once.
                    parts = line.split("\t")
                    tag = parts[0].strip()
                    if not tag:
                        continue
                    ts = 0.0
                    if len(parts) > 1:
                        try:
                            ts = float(parts[1])
                        except ValueError:
                            ts = 0.0
                    self.visited[tag] = max(self.visited.get(tag, 0.0), ts)
        # Rebuild the frontier from every player seen in a stored match, plus any scanned
        # player now due for a re-scan. The frontier lives only in memory, so each cycle used
        # to start empty and re-seed only from the leaderboard — capping the BFS at the seeds'
        # immediate neighborhood (~3.2k scanned out of ~42k discovered).
        for tag in discovered:
            self._enqueue(tag)
        for tag in self.visited:
            self._enqueue(tag)

    async def seed(self, countries: Iterable[str], seed_tags: Iterable[str] = ()) -> int:
        self._now = time.time()
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
        self._now = time.time()
        out = open(MATCHES_PATH, "a", encoding="utf-8")
        vis = open(VISITED_PATH, "a", encoding="utf-8")
        pbar = tqdm(total=target_matches, desc="matches", unit="match")
        try:
            while self.frontier and new < target_matches:
                tag = self.frontier.popleft()
                self.queued.discard(tag)
                # Mark scanned before the fetch (as before): a failing tag shouldn't be retried
                # until the revisit window elapses. The timestamp is what makes that bounded.
                self.visited[tag] = self._now
                vis.write(f"{tag}\t{self._now}\n")
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
            self._compact_visited()
        return new

    def _compact_visited(self) -> None:
        """Rewrite visited_tags.txt from the in-memory map: dedupe the append-only log (a
        re-scanned player is appended again each visit) and migrate legacy rows to the
        timestamped format. Written via temp file + atomic replace so a crash can't truncate
        it; the per-visit append above is the crash-safety net between compactions."""
        tmp = VISITED_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for tag, ts in self.visited.items():
                fh.write(f"{tag}\t{ts}\n")
        tmp.replace(VISITED_PATH)
