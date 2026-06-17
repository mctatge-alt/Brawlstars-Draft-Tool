# ⚔️ Brawl Draft — AI Ranked Draft Assistant

An AI-powered draft assistant for **Brawl Stars Ranked**. It recommends bans and picks in
real time using a win-probability model trained on **30,000+ real ranked matches**, fused
with empirical map statistics — and goes well beyond the usual "win-rate + synergy + counter"
tools with a **seat-aware minimax lookahead**, **composition warnings**, and **per-player
roster mastery**.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Status:** feature-complete. Built as a practical drafting tool *and* an end-to-end ML
> portfolio project — data pipeline → trained model → search/optimization → product.

<!-- Add a screenshot of the running app to docs/screenshot.png, then restore: ![Draft board](docs/screenshot.png) -->
> 🎮 Map select → ban/pick → real-time, explainable recommendations. Run it locally via the [Quickstart](#quickstart).

---

## Why it's different

Most draft tools reduce a pick to `map win-rate + pairwise synergy + pairwise counters`.
This project keeps that as a *baseline* and adds layers that no single competitor combines:

| Layer | What it does |
| --- | --- |
| 🎯 **Win-probability model** | Learned brawler + map embeddings predict `P(win)` for any 3v3 vs 3v3 on any map. Calibrated (ECE 0.012). |
| 🔮 **Seat-aware search** | Minimax lookahead over the 1-2-2-1 snake — reasons about the enemy's best responses, and captures first-pick vs last-pick value. |
| ⚖️ **Composition warnings** | Flags comp holes: no frontline, no long range, double-tank, "enemy is tank-heavy — bring a Marksman," mode-specific advice. |
| 👤 **Roster mastery** | Personalizes to *your* account: restricts to owned brawlers and weights by power level, comfort (personal trophies), and build completeness — including **buffies**. |
| 🔍 **Explainability** | Every suggestion shows a transparent per-signal breakdown (map / synergy / counter / role / model / mastery) and a sample-size confidence. |

## Features

- **Ban + pick phases** with a clickable draft board (6 bans, 3v3, unique picks).
- **Live recommendations** that re-rank instantly as the draft fills in.
- **Map-aware**: 100+ maps across the 6 current ranked modes, with per-map stats.
- **Deep-search toggle** for the seat-aware lookahead; **Personalize toggle** for roster mastery.
- **Transparent scoring** — no black box; every number is shown.

## Architecture

```mermaid
flowchart LR
    BS["Brawl Stars API<br/>(battle logs)"] -->|snowball crawl| CR[Crawler]
    CR -->|30k deduped matches| DS[(Dataset)]
    BF["Brawlify<br/>(brawlers / maps)"] --> REF[Reference]
    DS --> M["Win-prob model<br/>(PyTorch embeddings)"]
    REF --> M
    M --> ENG[Draft engine]
    ST["Empirical stats<br/>(win / synergy / counter)"] --> ENG
    RS["Player roster<br/>(mastery)"] --> ENG
    ENG -->|recommend · ban · search| API[FastAPI]
    API --> UI["Next.js draft board"]
```

## How it works

**1. Data.** The official API is player-centric, so a snowball crawler seeds top players from
the leaderboards, harvests the other 5 tags from every ranked match, and dedupes by a stable
match key (the same match appears in up to 6 players' logs). Result: 30k+ labeled
`(map, team A, team B) → winner` rows.

**2. Win-probability model.** A small PyTorch network with learned brawler, map, and mode
embeddings. The key design choice is an **antisymmetric** head:

```
logit = [ S(A, ctx) − S(B, ctx) ]      # context-conditioned team strength
      + [ PA·QB − PB·QA ]               # low-rank antisymmetric counter term
```

Swapping the two teams negates the logit, so `P(A wins) + P(B wins) = 1` *by construction* —
no team-order bias and no global offset to learn. The counter term captures specific matchups
(brawler X beats Y) that a pure strength model can't.

**3. Draft engine.** Given a draft state, it fuses the model with empirical map win-rates,
synergy, counters, role-fit, and (optionally) your mastery into a transparent score. The
**seat-aware search** runs a depth-limited, top-K-pruned, memoized minimax over the remaining
snake, evaluating completed drafts with the model — so it values picks by their projected
win-probability *after the enemy responds optimally*.

**4. App.** A FastAPI backend serves the engine; a Next.js board calls it and renders live,
explainable recommendations.

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the full methodology, training, and evaluation.

## Results

Held-out validation on ~30k matches (lower log-loss/ECE is better; higher AUC/accuracy is better):

| Model | Log-loss | Accuracy | AUC | ECE |
| --- | --- | --- | --- | --- |
| Always 0.5 | 0.6931 | 0.500 | – | – |
| Logistic regression (brawler presence) | 0.6879 | 0.541 | 0.557 | – |
| **Embedding net** | **0.6846** | **0.549** | **0.576** | **0.012** |

The embedding net beats both baselines and is **well-calibrated**. The absolute AUC is modest
*by nature of the problem*: at top ladder both teams draft competently and the outcome is
mostly decided by in-game skill, so the draft explains only a slice of the result. That's why
the tool ranks picks by *marginal* win-probability and fuses the model with lower-variance
empirical signals, rather than trusting any single number.

## Tech stack

- **ML / backend:** Python · PyTorch · scikit-learn · FastAPI
- **Data:** official Brawl Stars API (custom async crawler) + [Brawlify](https://brawlapi.com) for reference data & images; stored as JSONL/Parquet
- **Frontend:** Next.js · React · TypeScript · Tailwind CSS

## Project structure

```
backend/
  bsdraft/
    api/         FastAPI app (reference + recommend + roster)
    collect/     Async crawler: client, snowball, match parser, dedup
    data/        Reference loaders, encoders, dataset builder
    models/      PyTorch win-probability model + serving
    engine/      Stats, fused scoring, bans, seat-aware search, warnings, mastery
  scripts/       collect.py · train.py · export_model.py · smoke_test.py
frontend/        Next.js draft board
data/reference/  Brawlers, maps, modes, class overrides (committed)
deploy/          launchd sample for the home crawler (render.yaml + keepwarm.yml at root)
docs/            Model card, methodology, charts
```

## Quickstart

```bash
# 1. Backend (Python 3.11+)
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env          # add BRAWLSTARS_API_TOKEN + PLAYER_TAG

# 2. Collect data and train (one-time; needs an API key)
PYTHONPATH=backend python backend/scripts/collect.py --target 30000
PYTHONPATH=backend python backend/scripts/train.py
PYTHONPATH=backend python backend/scripts/export_model.py   # winprob.pt -> winprob.npz (served in NumPy)

# 3. Run the app
PYTHONPATH=backend uvicorn bsdraft.api.main:app --port 8000     # backend
npm --prefix frontend install && npm --prefix frontend run dev  # → http://localhost:3000
```

> Get a free API key at [developer.brawlstars.com](https://developer.brawlstars.com). Keys are
> IP-locked — allow the public IP of the machine running the crawler.

## Deployment — free, self-updating public site

The live site costs **$0/mo** and refreshes its data while in use. The constraint is the
IP-locked API key: the **crawler stays on a machine whose IP is on the key's allow-list** and
*publishes* data to a GitHub Release; a free cloud API *pulls* it on an interval and hot-swaps
rebuilt stats — no restart, and no key in the cloud.

```text
your machine            GitHub Release           Render (free)        Cloudflare Pages
collect.py --loop  ──▶  matches.jsonl.gz   ──▶   FastAPI: sync +   ◀──  Next.js board
(IP-locked key)         (tag: data-latest)       rebuild stats         (public URL)
```

The API serves the model in **pure NumPy** (exported `winprob.npz`, no torch), so it fits
Render's free 512 MB tier and cold-starts fast; PyTorch is used only for *training*, locally.

1. **Publish a dataset** (creates the `data-latest` release on first run):
   ```bash
   PYTHONPATH=backend python backend/scripts/export_model.py            # commit winprob.npz
   PYTHONPATH=backend python backend/scripts/collect.py --target 2000 --publish
   ```
2. **Backend → Render.** New → Blueprint → this repo (uses [`render.yaml`](render.yaml), free
   plan). It sets `DATA_URL` to the release asset and `REFRESH_SECONDS=600`. No API token needed.
3. **Frontend → Cloudflare Pages.** Create a Pages project from this repo with **root
   directory** `frontend`, **build command** `npx next build`, **output directory** `out`
   (the app is a static export). Add a build-time variable `NEXT_PUBLIC_API_BASE` = your
   Render URL (it's inlined at build, so it must be set before the build runs).
4. **Keep it warm.** Free instances sleep after ~15 min idle; set repo variable
   `RENDER_HEALTH_URL` = `<render-url>/api/health` to enable
   [`keepwarm.yml`](.github/workflows/keepwarm.yml).
5. **Run the home crawler** so data keeps flowing:
   ```bash
   PYTHONPATH=backend python backend/scripts/collect.py --loop 3600 --target 800 --publish
   ```
   Or install it as a login agent (macOS): edit the paths in
   [`deploy/com.bsdraft.crawler.plist`](deploy/com.bsdraft.crawler.plist), copy to
   `~/Library/LaunchAgents/`, then `launchctl load` it.

> **Tradeoffs.** The site stays up on the cloud, but data only advances while your crawler
> machine is on (watch `matches` / `last_change` at `/api/health`). **Roster/mastery
> personalization is local-only** — it needs a live call to the IP-locked key, which can't run
> from the cloud, so the public site runs without it.

## Roadmap

- [x] Data pipeline · win-prob model · draft engine · web app
- [x] Seat-aware search · composition warnings · roster mastery
- [x] Continuous data refresh · free, self-updating public deployment (NumPy serving)
- [ ] Best-of-3 series awareness · map-geometry features

## Credits

- Official Brawl Stars API — <https://developer.brawlstars.com>
- [Brawlify / BrawlAPI](https://brawlapi.com) for reference data & images
- Prior art / inspiration: [DraftStars](https://github.com/mcmckinley/DraftStars)

Not affiliated with, endorsed by, or sponsored by Supercell. *Brawl Stars* is a trademark of
Supercell Oy.

## License

[MIT](LICENSE) © 2026 Mitchell Tatge
