# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AI ranked-draft assistant for Brawl Stars: a Python win-probability model + draft engine (`backend/`, package `bsdraft`) behind a FastAPI API, with a Next.js draft board (`frontend/`). Deployed as a live site; the Supercell data key is IP-locked to a home machine, which shapes most of the deployment design.

## Universal rules

- Everything backend needs `PYTHONPATH=backend` (the `bsdraft` package lives under `backend/`; scripts run from the repo root).
- The deployed API installs `requirements-serve.txt` only — **no torch/sklearn/pandas** in any serve-path module, or the deploy build breaks (512 MB Render free tier).
- The model has two implementations that must stay in sync: torch training (`models/winprob.py`) and pure-NumPy serving (`models/serve.py`). Change one → change both.
- `backend/bsdraft/constants.py` and `data/reference.py` stay pure stdlib — no third-party imports.
- Frontend `npm run dev` hits the **deployed** API by default (`frontend/.env.local`); override `NEXT_PUBLIC_API_BASE=http://localhost:8000` to test local backend changes in-browser.

## Where things live

| Doc | What it covers | When to read it |
| --- | --- | --- |
| [docs/dev-commands.md](docs/dev-commands.md) | Every command: setup, run the API, collect→train→export pipeline, tests, frontend dev/build | Before running, training, or testing anything |
| [docs/backend-architecture.md](docs/backend-architecture.md) | Backend layers, data flow, the four cross-cutting design constraints, the two recommend endpoints | Before changing anything under `backend/bsdraft/` |
| [docs/deployment-topology.md](docs/deployment-topology.md) | Home crawler → GitHub Release artifacts → Render hot-swap; keepwarm/drift Actions; roster tunnel | Before touching `deploy/`, `data/sync.py`, `render.yaml`, `.github/workflows/`, artifact export scripts, or any `*_URL` env var — and when debugging the live site |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | The win-probability model: math, training data, calibration, limitations | Before changing model architecture, features, or training |
| [docs/model-evaluation.md](docs/model-evaluation.md) | Held-out ablations behind `DEFAULT_WEIGHTS`; why signal weights are global, not per-map | Before changing `engine/scoring.py` weights or adding/removing a scoring signal |
| [docs/item-winrate.md](docs/item-winrate.md) | Data-driven `/api/loadout` picks: single-item-owner inference (estimator, gate, biases) and the profiles→build→publish→serve pipeline | Before changing `engine/loadout.py`, `data/itemstats_build.py`, `engine/itemstats.py`, `collect/profiles.py`, or the `itemstats.json` artifact |
| [docs/adsense-go-live.md](docs/adsense-go-live.md) | Enabling the env-gated ad slots without halting ad serving | Before touching `AdSlot.tsx`, `ads.txt`, or enabling AdSense |
| [deploy/roster-tunnel.md](deploy/roster-tunnel.md) | Cloudflare Tunnel setup for per-visitor roster (`roster.brawldraft.com`) | When touching roster personalization or tunnel/launchd config |
| [PLAN.md](PLAN.md) | Goals, competitive landscape, phased roadmap | For scope/why questions or when planning new features |
| [README.md](README.md) | Public-facing overview, feature summary, quickstart | For the product pitch or user-visible behavior |

`Notes/` holds personal planning scratch notes — not maintained documentation.
