"""FastAPI app exposing the draft engine.

    PYTHONPATH=backend uvicorn bsdraft.api.main:app --reload --port 8000

Loads the engine (empirical stats + trained model) once at startup, and the player's
roster (for mastery personalization) if PLAYER_TAG is set.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bsdraft.api import schemas as S
from bsdraft.collect.client import BrawlStarsClient
from bsdraft.config import settings
from bsdraft.constants import RANKED_MODES
from bsdraft.data import reference as R
from bsdraft.engine import mastery
from bsdraft.engine.engine import DraftEngine
from bsdraft.engine.scoring import score_candidate
from bsdraft.engine.state import DraftState
from bsdraft.engine.stats import DraftStats
from bsdraft.models.serve import WinProbModel

_engine: Optional[DraftEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = DraftEngine(DraftStats(), WinProbModel())
    if settings.player_tag:
        try:
            async with BrawlStarsClient() as client:
                _engine.roster, _engine.roster_name = await mastery.fetch_roster(client, settings.player_tag)
        except Exception:
            _engine.roster, _engine.roster_name = None, ""
    yield


app = FastAPI(title="Brawl Stars Draft Tool", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": bool(_engine and _engine.model and _engine.model.available),
        "matches": _engine.stats.n if _engine else 0,
        "roster": bool(_engine and _engine.roster),
    }


@app.get("/api/reference", response_model=S.ReferenceResponse)
def reference():
    brawlers = [
        S.BrawlerRef(id=b.id, name=b.name, cls=b.cls, rarity=b.rarity, image_url=b.image_url)
        for b in R.load_brawlers()
    ]
    maps = [
        S.MapRef(id=m.id, name=m.name, mode=m.mode, image_url=m.image_url,
                 games=_engine.stats.map_games.get(m.id, 0) if _engine else 0)
        for m in R.load_ranked_maps()
    ]
    return S.ReferenceResponse(brawlers=brawlers, maps=maps, modes=list(RANKED_MODES))


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


@app.post("/api/recommend", response_model=S.RecommendResponse)
def recommend(req: S.RecommendRequest):
    state = DraftState(
        map_id=req.map_id, mode=req.mode,
        our_team=list(req.our_team), their_team=list(req.their_team), bans=list(req.bans),
        we_pick_first=req.we_pick_first, solo_queue=req.solo_queue,
    )
    roster = _engine.roster if (req.personalize and _engine.roster) else None
    composition = _engine.composition(state)
    warnings = _engine.composition_report(state)["warnings"]
    next_to_act = state.next_to_act()

    if req.phase == "ban":
        bans = _engine.recommend_bans(state, top=req.top)
        return S.RecommendResponse(
            phase="ban", bans=[S.BanRec(**vars(b)) for b in bans],
            composition=composition, warnings=warnings, next_to_act=next_to_act,
        )

    can_search = req.use_search and _engine.model and _engine.model.available and state.our_slots_left > 0
    if can_search:
        picks = []
        for sr in _engine.search_recommend(state, top=req.top, roster=roster):
            scored = vars(score_candidate(state, sr.brawler_id, _engine.stats, _engine.model, roster=roster))
            scored["projected_winprob"] = sr.projected_winprob
            picks.append(S.PickRec(**scored))
    else:
        picks = [S.PickRec(**vars(p)) for p in _engine.recommend_picks(state, top=req.top, roster=roster)]

    return S.RecommendResponse(
        phase="pick", picks=picks,
        composition=composition, warnings=warnings, next_to_act=next_to_act,
    )
