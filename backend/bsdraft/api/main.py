"""FastAPI app exposing the draft engine.

    PYTHONPATH=backend uvicorn bsdraft.api.main:app --reload --port 8000

Loads the engine (empirical stats + trained model) at startup. When DATA_URL / MODEL_URL are
set, it also syncs the published dataset and model every REFRESH_SECONDS — rebuilding stats
and hot-swapping the model — so the live site stays current with no restart. Loads the
player's roster (mastery personalization) if PLAYER_TAG is set — a local-only feature (needs
the IP-locked key).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bsdraft.api import schemas as S
from bsdraft.collect.client import BrawlStarsClient, normalize_tag
from bsdraft.config import settings
from bsdraft.constants import RANKED_MODES
from bsdraft.data import reference as R
from bsdraft.data import sync
from bsdraft.engine import mastery
from bsdraft.engine.drift import detect_drift
from bsdraft.engine.engine import DraftEngine
from bsdraft.engine.personal import build_personal_stats, matches_from_battlelog
from bsdraft.engine.scoring import score_candidate
from bsdraft.engine.state import DraftState
from bsdraft.engine.playerrank import build_rank_index, latest_ranked_tier
from bsdraft.engine.stats import DraftStats, build_bracketed
from bsdraft.engine.stats_store import load_stats
from bsdraft.engine.tiers import BRACKETS, bracket_of_tier, tier_label
from bsdraft.models.serve import WinProbModel

logger = logging.getLogger("bsdraft.api")

_engine: Optional[DraftEngine] = None
_last_check: float = 0.0   # epoch of the last sync attempt (liveness)
_last_change: float = 0.0  # epoch of the last actual data change
_meta_cache = None         # (data_version, MetaReport); recomputed lazily when data changes
_rank_idx_cache = None     # (data_version, {tag: (ts, tier)}); recomputed lazily when data changes
_personal_cache: dict = {} # tag -> (data_version, PersonalStats|None); rebuilt when data changes
_personal_locks: dict = {} # tag -> Lock; single-flights the per-tag dataset scan (no stampede)
_personal_locks_guard = threading.Lock()
_roster_cache: dict = {}   # normalized tag -> (fetched_at, RosterResponse); short TTL spares the live API
_rank_cache: dict = {}     # normalized tag -> (fetched_at, RankResponse); short TTL on live rank lookups


def _build_stats():
    """Produce ``(global_stats, {bracket: stats})``. When STATS_URL is set the API **loads** the
    precomputed stats artifact (built off-box from *all* matches, ~tens of MB, no OOM); otherwise
    it **rebuilds** them from the synced matches, capped to STATS_MAX_MATCHES to bound peak RAM."""
    if settings.stats_url and sync.STATS_PATH.exists():
        try:
            return load_stats(sync.STATS_PATH)
        except Exception as e:  # noqa: BLE001 — a corrupt/old artifact must not break startup
            logger.warning("stats load failed (%s); rebuilding from matches", e)
    return build_bracketed(halflife_days=settings.stats_halflife_days,
                           max_matches=settings.stats_max_matches)


def _rank_index():
    """Cached tag -> (ts, tier) index from the synced matches; rebuilt when data changes."""
    global _rank_idx_cache
    if _rank_idx_cache is None or _rank_idx_cache[0] != _last_change:
        _rank_idx_cache = (_last_change, build_rank_index())
    return _rank_idx_cache[1]


def _personal_for(tag: Optional[str]):
    """Cached personal stats for ``tag``, derived from the synced dataset (key-free, so it
    works on the public host) and rebuilt when the data changes. Returns None when the tag
    is empty or the player has no labeled games in our data. A live battle-log augment can
    pre-seed a richer entry at startup (see lifespan), which this cache then serves."""
    t = normalize_tag(tag or "")
    if not t or _engine is None:
        return None
    hit = _personal_cache.get(t)
    if hit is not None and hit[0] == _last_change:
        return hit[1]
    # Building scans the dataset for this tag's games (seconds on the full cloud dataset). Single-
    # flight per tag so a burst of requests for the same uncached tag — rapid picks, the frontend
    # re-polling, multiple tabs — waits on one build instead of each launching its own redundant
    # scan (a cache stampede; the cache is only written once the scan finishes).
    with _personal_locks_guard:
        if len(_personal_locks) > 512:   # bound growth — one tiny Lock per unique tag
            _personal_locks.clear()
        lock = _personal_locks.setdefault(t, threading.Lock())
    with lock:
        hit = _personal_cache.get(t)     # another thread may have built it while we waited
        if hit is not None and hit[0] == _last_change:
            return hit[1]
        if len(_personal_cache) > 256:   # simple bound — tags are cheap to rebuild
            _personal_cache.clear()
        ps = build_personal_stats(t, fallback=_engine.stats)
        _personal_cache[t] = (_last_change, ps)
        return ps


async def _refresh_loop() -> None:
    """Periodically re-sync the dataset (and model) and hot-swap rebuilt stats / a reloaded
    model into the live engine, so a fresh crawl or retrain rolls out with no restart."""
    global _last_check, _last_change
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(settings.refresh_seconds)
        try:
            data_changed = (await loop.run_in_executor(None, sync.sync_matches, settings.data_url)
                            if settings.data_url else False)
            _last_check = time.time()
            if data_changed and _engine is not None:
                _last_change = time.time()  # invalidate the rank / meta / personal caches
            # Refresh the empirical stats from their source: the published artifact (STATS_URL,
            # loaded — no in-memory rebuild) or, failing that, a local rebuild from the matches.
            if settings.stats_url and _engine is not None:
                if await loop.run_in_executor(None, sync.sync_stats, settings.stats_url):
                    g, br = await loop.run_in_executor(None, _build_stats)
                    _engine.stats, _engine.bracket_stats = g, br
                    logger.info("draft stats reloaded: %d matches, %d bracket(s)", g.n, len(br))
            elif data_changed and _engine is not None:
                g, br = await loop.run_in_executor(None, _build_stats)
                _engine.stats, _engine.bracket_stats = g, br
                logger.info("draft stats rebuilt: %d matches, %d bracket(s)", g.n, len(br))
            if settings.model_url and _engine is not None:
                if await loop.run_in_executor(None, sync.sync_model, settings.model_url):
                    _engine.model = await loop.run_in_executor(None, WinProbModel)  # atomic swap
                    logger.info("win-prob model hot-swapped (available=%s)", _engine.model.available)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — a refresh hiccup must not kill the loop
            logger.warning("refresh loop error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _last_check, _last_change
    loop = asyncio.get_running_loop()
    if settings.data_url:
        await loop.run_in_executor(None, sync.sync_matches, settings.data_url)
        _last_check = _last_change = time.time()
    if settings.model_url:
        await loop.run_in_executor(None, sync.sync_model, settings.model_url)
    if settings.stats_url:
        await loop.run_in_executor(None, sync.sync_stats, settings.stats_url)
    g, br = _build_stats()
    _engine = DraftEngine(g, WinProbModel(), bracket_stats=br)
    if settings.player_tag:
        ptag = normalize_tag(settings.player_tag)
        try:
            async with BrawlStarsClient() as client:
                _engine.roster, _engine.roster_name = await mastery.fetch_roster(client, settings.player_tag)
                # Prime personal stats with the player's freshest games (needs the live key,
                # so local/home only); the public host falls back to dataset-derived stats.
                try:
                    extra = matches_from_battlelog(await client.get_battlelog(ptag), ptag)
                    _personal_cache[ptag] = (_last_change, build_personal_stats(
                        ptag, fallback=_engine.stats, extra_matches=extra))
                except Exception:
                    pass
        except Exception:
            _engine.roster, _engine.roster_name = None, ""
    task = None
    if (settings.data_url or settings.model_url or settings.stats_url) and settings.refresh_seconds > 0:
        task = asyncio.create_task(_refresh_loop())
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Brawl Stars Draft Tool", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": bool(_engine and _engine.model and _engine.model.available),
        "matches": _engine.stats.n if _engine else 0,
        "roster": bool(_engine and _engine.roster),
        "refresh_seconds": settings.refresh_seconds if settings.data_url else 0,
        "last_check": _last_check or None,
        "last_change": _last_change or None,
    }


@app.get("/api/meta", response_model=S.MetaResponse)
def meta():
    """Has the meta shifted (balance change / new brawler) recently? Computed from the synced
    match data and cached per data version, so the frontend can poll it cheaply to show a
    'meta shifted — recommendations updating' banner."""
    global _meta_cache
    if _meta_cache is None or _meta_cache[0] != _last_change:
        _meta_cache = (_last_change, detect_drift())
    rep = _meta_cache[1]
    names = {b.id: b.name for b in R.load_brawlers()}
    return S.MetaResponse(
        shifted=rep.shifted, n_recent=rep.n_recent, n_prior=rep.n_prior,
        new_brawlers=[names.get(b, str(b)) for b in rep.new_brawlers],
        shifts=[
            S.MetaShift(
                brawler_id=s.brawler_id, name=s.name, kind=s.kind,
                wr_before=round(s.wr_before, 4), wr_after=round(s.wr_after, 4),
                use_before=round(s.use_before, 4), use_after=round(s.use_after, 4),
                z=round(s.z, 2),
            )
            for s in rep.shifts
        ],
        note=rep.note,
    )


@app.get("/api/reference", response_model=S.ReferenceResponse)
def reference():
    brawlers = [
        S.BrawlerRef(id=b.id, name=b.name, cls=b.cls, rarity=b.rarity, image_url=b.image_url)
        for b in R.load_brawlers()
    ]
    maps = [
        S.MapRef(id=m.id, name=m.name, mode=m.mode, image_url=m.image_url,
                 games=int(_engine.stats.map_games.get(m.id, 0)) if _engine else 0)
        for m in R.load_ranked_maps()
    ]
    brackets = [b for b in BRACKETS if _engine and b in _engine.bracket_stats]
    return S.ReferenceResponse(brawlers=brawlers, maps=maps, modes=list(RANKED_MODES), brackets=brackets)


@app.get("/api/roster", response_model=S.RosterResponse)
async def roster(tag: Optional[str] = None):
    """The configured (or given) player's roster — owned brawlers, loadout completeness, and
    mastery — fetched live from Supercell (needs the IP-locked key, so local/home only). The
    frontend re-polls this so a long session stays current; a successful result is cached for
    ``roster_ttl_seconds`` so the polling doesn't hammer the live API."""
    t = (tag or settings.player_tag or "").strip()
    if not t:
        return S.RosterResponse(loaded=False, tag="", name="", error="no player tag configured")
    key = normalize_tag(t)
    hit = _roster_cache.get(key)
    if hit is not None and (time.time() - hit[0]) < settings.roster_ttl_seconds:
        return hit[1]
    try:
        async with BrawlStarsClient() as client:
            r, name = await mastery.fetch_roster(client, t)
        _engine.roster, _engine.roster_name = r, name
        owned = [S.OwnedBrawler(id=bid, mastery=round(m.score, 3), gaps=m.gaps()) for bid, m in r.items()]
        resp = S.RosterResponse(loaded=True, tag=t, name=name, owned=owned)
        if len(_roster_cache) > 512:   # bound growth — one entry per unique tag, TTL alone never frees it
            _roster_cache.clear()
        _roster_cache[key] = (time.time(), resp)  # cache only successful loads; errors retry next poll
        return resp
    except Exception as e:  # noqa: BLE001
        return S.RosterResponse(loaded=False, tag=t, name="", error=str(e))


async def _live_rank(tag_n: str) -> Optional[S.RankResponse]:
    """Current Ranked tier from a live battle-log fetch (needs the IP-locked key, so it only
    works local/home or via the keyed tunnel). Returns None when the lookup can't be served or
    the player has no recent ranked games, so the caller can fall back to the dataset. Cached
    briefly (``roster_ttl_seconds``) so the frontend re-polling the same tag spares the live API."""
    hit = _rank_cache.get(tag_n)
    if hit is not None and (time.time() - hit[0]) < settings.roster_ttl_seconds:
        return hit[1]
    try:
        async with BrawlStarsClient() as client:
            battles = await client.get_battlelog(tag_n)
        t = latest_ranked_tier(battles, tag_n)
    except Exception:  # noqa: BLE001 — keyless/offline host or API hiccup; fall back to the dataset
        return None
    if not t:
        return None
    resp = S.RankResponse(found=True, tag=tag_n, tier=t, tier_label=tier_label(t),
                          bracket=bracket_of_tier(t), source="live")
    if len(_rank_cache) > 512:   # bound growth — one entry per unique tag, TTL alone never frees it
        _rank_cache.clear()
    _rank_cache[tag_n] = (time.time(), resp)
    return resp


@app.get("/api/rank", response_model=S.RankResponse)
async def rank(tag: str):
    """Resolve a player's current Ranked tier. We try a live battle-log fetch first whenever a
    key is configured (local/home, or the keyed roster tunnel), because that's the player's tier
    *right now* — the collected match data is a crawl snapshot that goes stale across a Ranked
    season reset, where a player can drop several tiers, so a pre-reset row over-states them. The
    dataset is the fallback: it needs no key (the only source on the public host) and covers
    players with no recent ranked games."""
    tag_n = normalize_tag(tag)
    if not tag_n:
        return S.RankResponse(found=False, tag="", error="enter a player tag")
    if settings.brawlstars_api_token:
        live = await _live_rank(tag_n)
        if live is not None:
            return live
    hit = _rank_index().get(tag_n)
    if hit:
        t = hit[1]
        return S.RankResponse(found=True, tag=tag_n, tier=t, tier_label=tier_label(t),
                              bracket=bracket_of_tier(t), source="dataset")
    return S.RankResponse(
        found=False, tag=tag_n,
        error="no recent ranked games found" if settings.brawlstars_api_token
        else "not in our data, and live lookup isn't available here")


@app.post("/api/top_picks", response_model=S.TopPicksResponse)
def top_picks(req: S.TopPicksRequest):
    """The strongest picks for the *current board*, with every brawler judged at a full
    loadout (all gadgets, gears & star powers) and **no roster** — so nothing is filtered by
    ownership or mastery. It re-ranks as the draft fills in: brawlers already picked/banned
    drop out, and synergy with your team / counters to theirs fold into the score. This is
    the pure population meta ('who's strongest here right now'), the deliberate counterpart
    to /api/recommend, which personalizes to the player's roster & history."""
    state = DraftState(
        map_id=req.map_id, mode=req.mode,
        our_team=list(req.our_team), their_team=list(req.their_team), bans=list(req.bans),
        rank_bracket=req.rank_bracket,
    )
    picks = _engine.recommend_picks(state, top=req.top, roster=None)  # roster=None ⇒ full loadout
    return S.TopPicksResponse(
        map_id=req.map_id, mode=req.mode, rank_bracket=req.rank_bracket,
        picks=[
            S.TopPick(brawler_id=p.brawler_id, name=p.name, cls=p.cls,
                      score=round(p.score, 4), map_winrate=round(p.map_winrate, 4))
            for p in picks
        ],
    )


class _ReqMastery:
    """Lite stand-in for :class:`engine.mastery.Mastery` built from a client-sent roster entry —
    exposes just the ``.score`` and ``.gaps()`` that scoring reads. Lets the public backend
    personalize from a roster the client fetched (via the keyed tunnel) but the backend can't."""
    __slots__ = ("score", "_gaps")

    def __init__(self, score: float, gaps: List[str]):
        self.score = max(0.0, min(1.0, float(score)))
        self._gaps = list(gaps or [])

    def gaps(self) -> List[str]:
        return self._gaps


def _roster_for(req: S.RecommendRequest):
    """Roster dict ``{brawler_id: mastery-like}`` to personalize against, or None. Prefers the
    client-sent roster (the only source on the public host), then the server's own roster
    (local/home, where the IP-locked key can fetch it). Returns None unless ``personalize`` is set."""
    if not req.personalize:
        return None
    if req.roster:
        return {e.id: _ReqMastery(e.mastery, e.gaps) for e in req.roster}
    return _engine.roster or None


@app.post("/api/recommend", response_model=S.RecommendResponse)
def recommend(req: S.RecommendRequest):
    state = DraftState(
        map_id=req.map_id, mode=req.mode,
        our_team=list(req.our_team), their_team=list(req.their_team), bans=list(req.bans),
        we_pick_first=req.we_pick_first, solo_queue=req.solo_queue, rank_bracket=req.rank_bracket,
    )
    roster = _roster_for(req)
    composition = _engine.composition(state)
    warnings = _engine.composition_report(state)["warnings"]
    game_plan = S.GamePlan(**_engine.game_plan(state))
    next_to_act = state.next_to_act()

    if req.phase == "ban":
        bans = _engine.recommend_bans(state, top=req.top)
        return S.RecommendResponse(
            phase="ban", bans=[S.BanRec(**vars(b)) for b in bans],
            composition=composition, warnings=warnings, game_plan=game_plan, next_to_act=next_to_act,
        )

    # Personal win-rate signal — only feeds pick scoring, and the build scans the dataset, so don't
    # pay for it during the ban phase (the result would be discarded there) — that scan, fired on
    # every ban placement, was the bulk of the blind-pick "analyzing…" stall before the first pick.
    personal = _personal_for(req.personal_tag)
    can_search = req.use_search and _engine.model and _engine.model.available and state.our_slots_left > 0
    if can_search:
        picks = []
        for sr in _engine.search_recommend(state, top=req.top, roster=roster):
            scored = vars(score_candidate(state, sr.brawler_id, _engine._stats_for(state),
                                          _engine.model, roster=roster, personal=personal))
            scored["projected_winprob"] = sr.projected_winprob
            picks.append(S.PickRec(**scored))
    else:
        picks = [S.PickRec(**vars(p))
                 for p in _engine.recommend_picks(state, top=req.top, roster=roster, personal=personal)]

    return S.RecommendResponse(
        phase="pick", picks=picks,
        composition=composition, warnings=warnings, game_plan=game_plan, next_to_act=next_to_act,
    )
