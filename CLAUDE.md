# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AI ranked-draft assistant for Brawl Stars: a Python win-probability model + draft engine (`backend/`, package `bsdraft`) behind a FastAPI API, with a Next.js draft board (`frontend/`). `README.md` and `docs/MODEL_CARD.md` cover the methodology and math; `PLAN.md` tracks the phased roadmap.

## Commands

Everything in the backend needs `PYTHONPATH=backend` — the `bsdraft` package lives under `backend/`, and scripts are run from the repo root.

```bash
# Setup (Python 3.11+)
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt          # full stack: torch, sklearn, pandas, fastapi
cp .env.example .env                              # add BRAWLSTARS_API_TOKEN + PLAYER_TAG

# Run the API (serves the model from data/processed/winprob.npz; reads data/raw/matches.jsonl locally)
PYTHONPATH=backend uvicorn bsdraft.api.main:app --reload --port 8000

# Data → model pipeline (one-time; collect needs the IP-locked key)
PYTHONPATH=backend python backend/scripts/collect.py --target 30000   # snowball crawl → data/raw/
PYTHONPATH=backend python backend/scripts/train.py                    # torch train → winprob.pt + docs/ charts
PYTHONPATH=backend python backend/scripts/export_model.py             # winprob.pt → winprob.npz (commit this)
PYTHONPATH=backend python backend/scripts/export_stats.py             # precomputed stats artifact (published next to matches.jsonl.gz)

# Tests (lightweight by design — currently only test_personal.py; each file also runs standalone via __main__)
PYTHONPATH=backend python -m pytest backend/tests/
PYTHONPATH=backend python backend/tests/test_personal.py

# Other scripts: smoke_test.py (verify API key + inspect real shapes), ablate_components.py /
# ablate_context.py (held-out ablations → docs/ablation*.json), refresh_reference.py (re-pull Brawlify).
# pytest/ruff/mypy are optional dev tools (not pinned in requirements).

# Frontend
npm --prefix frontend install
npm --prefix frontend run dev      # http://localhost:3000
npm --prefix frontend run build    # static export → frontend/out/ (output: "export")
```

**Frontend dev points at the deployed API by default.** `frontend/.env.local` sets `NEXT_PUBLIC_API_BASE` to the live Render URL, so `npm run dev` hits production, not your local uvicorn. To test local backend changes in-browser, override `NEXT_PUBLIC_API_BASE=http://localhost:8000`. The var is inlined at build time for the static export.

## Architecture

**Data → stats/model → engine → API → board.** A snowball crawler (`collect/`) works around the player-centric official API: it seeds top players, harvests all 6 tags from each ranked match, and dedupes by a stable match key (`battleTime` + sorted tags) into `data/raw/`. `data/dataset.py` builds training rows; `models/winprob.py` trains the embedding net; the engine fuses the model with empirical stats built at startup from the same matches.

Backend layers (`backend/bsdraft/`): `collect/` (async client, crawler, match parser, publish), `data/` (reference loaders, encoders, dataset builder, runtime release sync in `data/sync.py`), `models/` (train + serve), `engine/` (the draft brain), `api/` (FastAPI). `engine/state.py`'s `DraftState` is the object threaded through nearly everything. Beyond the core `scoring.py` / `search.py` / `stats.py` / `mastery.py` / `personal.py`, the engine also contains `playerrank.py` (tier resolution + rank index), `tiers.py` (Diamond/Masters bracket labels), `stats_store.py` (loads the precomputed stats artifact), `drift.py` (staleness/liveness), and `composition.py` + `gameplan.py` (team-composition reasoning surfaced through the API).

Four cross-cutting design decisions drive most of the non-obvious structure:

1. **Two model implementations that must stay in sync.** `models/winprob.py` is the PyTorch training model; `models/serve.py` reimplements its `forward()` in **pure NumPy**, loading the exported `winprob.npz`. The deployed API runs inference with no torch. **If you change the model architecture, update both** — the docstring in `serve.py` pins the exact forward formula (antisymmetric strength diff + low-rank counter term). `winprob.npz` is tiny (~50 KB) and committed; `winprob.pt` is not.

2. **Three dependency tiers.** `requirements.txt` (full: train + collect + serve), `requirements-collect.txt` (crawler only), `requirements-serve.txt` (deployed API — **no torch/sklearn/pandas**, NumPy serving only, fits Render's 512 MB free tier). Adding an import to a serve-path module can break the deploy build.

3. **Dependency-free core layers.** `constants.py` and `data/reference.py` are pure stdlib so they run without installing anything (`python -m bsdraft.data.reference`). `config.py` (needs `pydantic-settings`) is deliberately *not* imported by the reference layer. Keep third-party imports out of these two modules.

4. **Fused, renormalized scoring.** `engine/scoring.py` scores a pick as a weighted average over only the **active** signals (synergy needs allies, counters need a revealed enemy, mastery/personal need a roster), renormalized by the active weights. `DEFAULT_WEIGHTS` were tuned via the held-out ablation (see the comment there and `docs/model-evaluation.md`) — context-dependent per-map weighting was tested and found no better, so weights are global. `engine/search.py` adds the seat-aware top-K-pruned, memoized minimax over the 1-2-2-1 snake.

Two recommend endpoints are intentionally distinct: `/api/recommend` personalizes to the player's roster + history (mastery, personal win-rate), while `/api/top_picks` is the pure population meta — every brawler at a full loadout, **no roster filtering**.

**Deployment topology** (because the Supercell key is IP-locked): the crawler runs on a home machine (three launchd plists under `deploy/`: crawler, API, tunnel) and publishes `matches.jsonl.gz` + `winprob.npz` + a precomputed stats artifact to a GitHub Release. The Render API (`render.yaml`, no key) pulls via `DATA_URL` / `MODEL_URL` / `STATS_URL` every `REFRESH_SECONDS` and **hot-swaps rebuilt stats and a reloaded model with no restart** (see `data/sync.py` and the `_refresh_loop` / `lifespan` in `api/main.py`). The published stats artifact lets the cloud API skip a full dataset replay at boot — `STATS_MAX_MATCHES` only bounds the fallback rebuild if the artifact can't load. A scheduled GitHub Action (`.github/workflows/keepwarm.yml`) pings `/api/health` to keep Render's free tier out of cold-sleep.

Consequence of the IP lock: the public backend can't fetch a roster itself, so personalization is wired around it — the frontend pulls the player's roster from the home machine over a **Cloudflare Tunnel** (`roster.brawldraft.com` → the `com.bsdraft.api` agent; setup in `deploy/roster-tunnel.md` and `deploy/cloudflared.yml`) and **forwards it in the `/api/recommend` body** (`RecommendRequest.roster`), which drives the owned-filter + mastery/loadout scoring there; `/api/rank` likewise resolves from the collected data when no key is present. (Mastery is loadout-forward and power-neutral — Ranked normalizes every brawler to power 11.)
