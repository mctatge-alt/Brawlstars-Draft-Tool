"""FastAPI app exposing the draft engine.

    PYTHONPATH=backend uvicorn bsdraft.api.main:app --reload --port 8000

Loads the engine (empirical stats + trained model) at startup. When DATA_URL is set, it
also syncs the published matches dataset and rebuilds stats every REFRESH_SECONDS, so the
live site stays current with no restart. Loads the player's roster (mastery
personalization) if PLAYER_TAG is set — a local-only feature (needs the IP-locked key).
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

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
from bsdraft.engine.scoring import score_candidate
from bsdraft.engine.state import DraftState
from bsdraft.engine.playerrank import build_rank_index, latest_ranked_tier
from bsdraft.engine.stats import DraftStats, build_bracketed
from bsdraft.engine.tiers import BRACKETS, bracket_of_tier, tier_label
from bsdraft.models.serve import WinProbModel

logger = logging.getLogger("bsdraft.api")

_engine: Optional[DraftEngine] = None
_last_check: float = 0.0   # epoch of the last sync attempt (liveness)
_last_change: float = 0.0  # epoch of the last actual data change
_meta_cache = None         # (data_version, MetaReport); recomputed lazily when data changes
_rank_idx_cache = None     # (data_version, {tag: (ts, tier)}); recomputed lazily when data changes


def _build_stats():
    """Build global stats + per-rank-bracket tables with the configured recency half-life.
    Used at startup and on every live refresh, so both weight the meta identically.
    Returns ``(global_stats, {bracket: stats})``."""
    return build_bracketed(halflife_days=settings.stats_halflife_days)


def _rank_index():
    """Cached tag -> (ts, tier) index from the synced matches; rebuilt when data changes."""
    global _rank_idx_cache
    if _rank_idx_cache is None or _rank_idx_cache[0] != _last_change:
        _rank_idx_cache = (_last_change, build_rank_index())
    return _rank_idx_cache[1]


async def _refresh_loop() -> None:
    """Periodically re-sync the dataset and hot-swap rebuilt stats into the live engine."""
    global _last_check, _last_change
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(settings.refresh_seconds)
        try:
            changed = await loop.run_in_executor(None, sync.sync_matches, settings.data_url)
            _last_check = time.time()
            if changed and _engine is not None:
                g, br = await loop.run_in_executor(None, _build_stats)
                _engine.stats, _engine.bracket_stats = g, br  # swap in rebuilt tables
                _last_change = time.time()
                logger.info("draft stats refreshed: %d matches, %d bracket(s)", g.n, len(br))
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
    g, br = _build_stats()
    _engine = DraftEngine(g, WinProbModel(), bracket_stats=br)
    if settings.player_tag:
        try:
            async with BrawlStarsClient() as client:
                _engine.roster, _engine.roster_name = await mastery.fetch_roster(client, settings.player_tag)
        except Exception:
            _engine.roster, _engine.roster_name = None, ""
    task = None
    if settings.data_url and settings.refresh_seconds > 0:
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
    t = (tag or settings.player_tag or "").strip()
    if not t:
        return S.RosterResponse(loaded=False, tag="", name="", error="no player tag configured")
    try:
        async with BrawlStarsClient() as client:
            r, name = await mastery.fetch_roster(client, t)
        _engine.roster, _engine.roster_name = r, name
        owned = [S.OwnedBrawler(id=bid, mastery=round(m.score, 3), gaps=m.gaps()) for bid, m in r.items()]
        return S.RosterResponse(loaded=True, tag=t, name=name, owned=owned)
    except Exception as e:  # noqa: BLE001
        return S.RosterResponse(loaded=False, tag=t, name="", error=str(e))


@app.get("/api/rank", response_model=S.RankResponse)
async def rank(tag: str):
    """Resolve a player's current Ranked tier — from our collected match data first (no API
    key needed, works on the public host), then a live battle-log fetch when a valid key is
    configured (local/home)."""
    tag_n = normalize_tag(tag)
    if not tag_n:
        return S.RankResponse(found=False, tag="", error="enter a player tag")
    hit = _rank_index().get(tag_n)
    if hit:
        t = hit[1]
        return S.RankResponse(found=True, tag=tag_n, tier=t, tier_label=tier_label(t),
                              bracket=bracket_of_tier(t), source="dataset")
    try:
        async with BrawlStarsClient() as client:
            battles = await client.get_battlelog(tag_n)
        t = latest_ranked_tier(battles, tag_n)
        if t:
            return S.RankResponse(found=True, tag=tag_n, tier=t, tier_label=tier_label(t),
                                  bracket=bracket_of_tier(t), source="live")
        return S.RankResponse(found=False, tag=tag_n, error="no recent ranked games found")
    except Exception:
        return S.RankResponse(found=False, tag=tag_n,
                              error="not in our data, and live lookup isn't available here")


@app.post("/api/recommend", response_model=S.RecommendResponse)
def recommend(req: S.RecommendRequest):
    state = DraftState(
        map_id=req.map_id, mode=req.mode,
        our_team=list(req.our_team), their_team=list(req.their_team), bans=list(req.bans),
        we_pick_first=req.we_pick_first, solo_queue=req.solo_queue, rank_bracket=req.rank_bracket,
    )
    roster = _engine.roster if (req.personalize and _engine.roster) else None
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

    can_search = req.use_search and _engine.model and _engine.model.available and state.our_slots_left > 0
    if can_search:
        picks = []
        for sr in _engine.search_recommend(state, top=req.top, roster=roster):
            scored = vars(score_candidate(state, sr.brawler_id, _engine._stats_for(state), _engine.model, roster=roster))
            scored["projected_winprob"] = sr.projected_winprob
            picks.append(S.PickRec(**scored))
    else:
        picks = [S.PickRec(**vars(p)) for p in _engine.recommend_picks(state, top=req.top, roster=roster)]

    return S.RecommendResponse(
        phase="pick", picks=picks,
        composition=composition, warnings=warnings, game_plan=game_plan, next_to_act=next_to_act,
    )
