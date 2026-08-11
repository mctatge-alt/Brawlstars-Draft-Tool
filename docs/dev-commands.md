# Dev Commands — setup, run, train, test, frontend

Every command needed to develop this repo: Python venv setup, running the FastAPI backend,
the data-collection → training → export pipeline, tests, and the Next.js frontend.

## Backend setup (Python 3.11+)

Everything in the backend needs `PYTHONPATH=backend` — the `bsdraft` package lives under
`backend/`, and scripts are run from the repo root.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt          # full stack: torch, sklearn, pandas, fastapi
cp .env.example .env                              # add BRAWLSTARS_API_TOKEN + PLAYER_TAG
```

There are three requirements files — see [backend-architecture.md](backend-architecture.md)
for why. `requirements.txt` is the full dev stack; `requirements-serve.txt` is what the
deployed API installs (no torch/sklearn/pandas).

## Run the API

Serves the model from `data/processed/winprob.npz`; reads `data/raw/matches.jsonl` locally.

```bash
PYTHONPATH=backend uvicorn bsdraft.api.main:app --reload --port 8000
```

## Data → model pipeline

One-time / retrain path. `collect.py` needs the IP-locked Supercell key (home machine only —
see [deployment-topology.md](deployment-topology.md)).

```bash
PYTHONPATH=backend python backend/scripts/collect.py --target 30000   # snowball crawl → data/raw/
PYTHONPATH=backend python backend/scripts/train.py                    # torch train → winprob.pt + docs/ charts
PYTHONPATH=backend python backend/scripts/export_model.py             # winprob.pt → winprob.npz (commit this)
PYTHONPATH=backend python backend/scripts/export_stats.py             # precomputed stats artifact (published next to matches.jsonl.gz)
PYTHONPATH=backend python backend/scripts/export_rank_index.py        # precomputed tag→tier rank index (compact artifact the cloud LOADS instead of building ~200 MB in RAM)
```

Other scripts under `backend/scripts/`:

- `smoke_test.py` — verify the API key works + inspect real response shapes.
- `ablate_components.py` / `ablate_context.py` — held-out ablations → `docs/ablation*.json`
  (methodology + results in [model-evaluation.md](model-evaluation.md)).
- `refresh_reference.py` — re-pull the Brawlify reference JSONs. **Careful:** refreshing
  `maps.json` without a retrain silently re-maps trained map embedding rows; brawlers are
  safe (id-sorted, append-only).

## Tests

Lightweight by design; each test file also runs standalone via `__main__`.

```bash
PYTHONPATH=backend python -m pytest backend/tests/
PYTHONPATH=backend python backend/tests/test_personal.py
```

`pytest` / `ruff` / `mypy` are optional dev tools (not pinned in requirements).

## Frontend (Next.js)

```bash
npm --prefix frontend install
npm --prefix frontend run dev      # http://localhost:3000
npm --prefix frontend run build    # static export → frontend/out/ (output: "export")
```

**Frontend dev points at the deployed API by default.** `frontend/.env.local` sets
`NEXT_PUBLIC_API_BASE` to the live Render URL, so `npm run dev` hits production, not your
local uvicorn. To test local backend changes in-browser, override
`NEXT_PUBLIC_API_BASE=http://localhost:8000`. The var is inlined at build time for the
static export.

Ads are shipped dark (env-gated) — read [adsense-go-live.md](adsense-go-live.md) before
touching `AdSlot.tsx`, `ads.txt`, or enabling AdSense.
