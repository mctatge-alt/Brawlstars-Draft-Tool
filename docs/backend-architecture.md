# Backend Architecture — layers, data flow, and the design decisions that constrain changes

How `backend/bsdraft` is structured (collect → data → models → engine → api), and the four
cross-cutting design decisions you must not break when editing backend code.

## Data flow

**Data → stats/model → engine → API → board.** A snowball crawler (`collect/`) works around
the player-centric official API: it seeds top players, harvests all 6 tags from each ranked
match, and dedupes by a stable match key (`battleTime` + sorted tags) into `data/raw/`.
`data/dataset.py` builds training rows; `models/winprob.py` trains the embedding net; the
engine fuses the model with empirical stats built at startup from the same matches.

## Layers (`backend/bsdraft/`)

- `collect/` — async client, crawler, match parser, publish.
- `data/` — reference loaders, encoders, dataset builder, runtime release sync in `data/sync.py`.
- `models/` — train (`winprob.py`) + serve (`serve.py`).
- `engine/` — the draft brain. `engine/state.py`'s `DraftState` is the object threaded
  through nearly everything. Core: `scoring.py` / `search.py` / `stats.py` / `mastery.py` /
  `personal.py`. Also: `playerrank.py` (tier resolution + rank index), `tiers.py`
  (Diamond/Masters bracket labels), `stats_store.py` (loads the precomputed stats artifact),
  `rank_store.py` (loads the precomputed rank-index artifact into a compact NumPy lookup),
  `drift.py` (staleness/liveness), and `composition.py` + `gameplan.py` (team-composition
  reasoning surfaced through the API).
- `api/` — FastAPI app (`api/main.py`).

## Four cross-cutting design decisions

1. **Two model implementations that must stay in sync.** `models/winprob.py` is the PyTorch
   training model; `models/serve.py` reimplements its `forward()` in **pure NumPy**, loading
   the exported `winprob.npz`. The deployed API runs inference with no torch. **If you change
   the model architecture, update both** — the docstring in `serve.py` pins the exact forward
   formula (antisymmetric strength diff + low-rank counter term). `winprob.npz` is tiny
   (~50 KB) and committed; `winprob.pt` is not.

2. **Three dependency tiers.** `requirements.txt` (full: train + collect + serve),
   `requirements-collect.txt` (crawler only), `requirements-serve.txt` (deployed API —
   **no torch/sklearn/pandas**, NumPy serving only, fits Render's 512 MB free tier).
   Adding an import to a serve-path module can break the deploy build.

3. **Dependency-free core layers.** `constants.py` and `data/reference.py` are pure stdlib
   so they run without installing anything (`python -m bsdraft.data.reference`). `config.py`
   (needs `pydantic-settings`) is deliberately *not* imported by the reference layer. Keep
   third-party imports out of these two modules.

4. **Fused, renormalized scoring.** `engine/scoring.py` scores a pick as a weighted average
   over only the **active** signals (synergy needs allies, counters need a revealed enemy,
   mastery/personal need a roster), renormalized by the active weights. `DEFAULT_WEIGHTS`
   were tuned via the held-out ablation (see the comment there and
   [model-evaluation.md](model-evaluation.md)) — context-dependent per-map weighting was
   tested and found no better, so weights are global. `engine/search.py` adds the seat-aware
   top-K-pruned, memoized minimax over the 1-2-2-1 snake.

## The two recommend endpoints are intentionally distinct

`/api/recommend` personalizes to the player's roster + history (mastery, personal win-rate),
while `/api/top_picks` is the pure population meta — every brawler at a full loadout,
**no roster filtering**. (Mastery is loadout-forward and power-neutral — Ranked normalizes
every brawler to power 11.)

## See also

- [MODEL_CARD.md](MODEL_CARD.md) — the win-probability model's math, training, calibration.
- [model-evaluation.md](model-evaluation.md) — signal-weighting ablations behind `DEFAULT_WEIGHTS`.
- [deployment-topology.md](deployment-topology.md) — how artifacts reach the cloud API and
  why the serve path is so constrained.
