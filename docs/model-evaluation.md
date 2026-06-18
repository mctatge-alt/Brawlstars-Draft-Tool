# Model Evaluation — How the draft signals are weighted, and whether that should change

The recommender fuses six signals into one pick score: a brawler's **map** win-rate, the
learned **model** (win-prob net), pairwise **synergy** with allies, **counter** vs. revealed
enemies, mode-based **role** fit, and player-specific **mastery**/**personal** history. The
first four are combined with fixed global weights ([`DEFAULT_WEIGHTS`](../backend/bsdraft/engine/scoring.py)).

A natural question: *should those weights depend on the map and the brawlers already picked?*
Intuitively yes — counters ought to matter more in some modes, synergy in others. This doc
records an ablation that **tested that intuition and largely refuted it**, plus the one change
the data did support.

> TL;DR — Draft is a small but real edge (AUC ~0.57–0.63 by mode; matchmaking equalizes
> teams). The interpretable empirical signals out-rank the neural net; the net mainly adds
> calibration. **Context-dependent (per-map/mode) weighting does not improve predictions** —
> the optimal weighting is effectively global. The one supported tweak was a global rebalance:
> **counter was under-weighted** (raised .15 → .23) and **map over-weighted** (lowered .40 → .32).

## Method (leakage-free by construction)

Two scripts, run over ~40,200 labeled Ranked matches:

- [`scripts/ablate_components.py`](../backend/scripts/ablate_components.py) — net vs. empirical signals, head-to-head.
- [`scripts/ablate_context.py`](../backend/scripts/ablate_context.py) — does the right weighting move by mode?

Design choices that keep the comparison honest:

1. **The net is re-trained on the train split inside the harness.** The shipped model was
   trained on data that has since grown, so reusing its holdout would leak. We retrain with
   the same recipe (incl. early stopping) so the net is calibrated, not overtrained.
2. **Empirical stats are built on the train rows only**, then scored on held-out rows.
3. Each draft signal is expressed as an **antisymmetric team-A advantage** (map/synergy as a
   team-vs-team difference, counter as the directed cross-matchup), so a positive score means
   "team A favored" and swapping teams negates it.
4. Calibration / stacking logistic regressions are fit with **cross-validation on the
   out-of-sample features**, so reported probabilities never see their own label.
5. For the per-mode test, features are **5-fold cross-fit over the full dataset** (each fold's
   stats come from the other folds) so per-mode estimates are stable rather than 1k-sample noise.

## Result 1 — the net does not subsume the empirical signals

Held-out validation (n = 6,031), all predictors well-calibrated (ECE ≤ 0.007):

| Predictor | log-loss | acc | AUC |
|---|---|---|---|
| always 0.5 | 0.6931 | .500 | — |
| net only | 0.6861 | .547 | **.567** |
| empirical blend only | 0.6831 | .556 | **.581** |
| net + empirical | 0.6826 | .560 | **.583** |

Standalone AUC of each raw signal: map `.568`, synergy `.564`, counter `.570`, blend `.581`, net `.567`.

- The **empirical blend out-discriminates the net** (.581 vs .567); even each individual signal
  matches it. The net's edge is calibration (ECE .001), not ranking.
- A stacker over both assigns **~69% of the weight to the empirical side, ~31% to the net** —
  so leaning harder on the net would *lose* discrimination. (Refutes "trust the model more.")

## Result 2 — context-dependent weighting does not help

Cross-fit over all 40,208 matches, per mode. `map/syn/cnt` are the standardized weights each
empirical signal earns; the AUC columns compare three blends — **fixed** (shipped weights),
**global-refit** (one logistic over all rows), **mode-refit** (a logistic refit within the mode):

| Mode | n | map | syn | cnt | AUC fixed | global-refit | mode-refit |
|---|---|---|---|---|---|---|---|
| Gem Grab | 6046 | .15 | .07 | .15 | .578 | .583 | .582 |
| Brawl Ball | 6284 | .13 | .08 | .16 | .580 | .586 | .584 |
| Knockout | 9846 | .11 | .06 | .12 | .563 | .568 | .567 |
| Hot Zone | 5989 | .18 | .14 | .12 | .593 | .596 | .595 |
| Heist | 5558 | .23 | .14 | .22 | .628 | .634 | .634 |
| Bounty | 6485 | .12 | .08 | .11 | .567 | .571 | .570 |

- **`mode-refit` ≈ `global-refit` in every mode** (within ±0.001 AUC). Re-deriving the weights
  per mode buys nothing over a single global set. → Context-dependent weighting isn't worth building.
- The relative ordering (counter ≈ map > synergy) is **stable across modes**. Hot Zone is the
  only reordering (synergy edges counter), and it yields no predictive gain.
- What *does* vary is **how much draft matters at all**: Heist is far more draft-decided
  (AUC .63) than Knockout (.56). That's a confidence signal, not a reweighting one.

> **Why no draft-phase test?** Completed matches only contain final 3v3 comps — there are no
> partial-draft labels — so "weight signals differently as picks come in" can't be measured
> from outcomes. The engine already handles phase structurally: synergy/counter only activate
> once allies/enemies exist, and the blend renormalizes over the active signals.

## Result 3 — the one change worth making: rebalance globally

Held-out AUC of candidate **fixed** weightings on the empirical trio (no fitting, so no overfit):

| Trio weighting | held-out AUC |
|---|---|
| current `map .40 / syn .15 / cnt .15` | 0.5826 |
| naive trim-synergy `.40 / .10 / .20` | 0.5825 |
| **chosen `map .32 / syn .15 / cnt .23`** | **0.5852** |
| global-refit ceiling | 0.5872 |
| *optimal trio (scaled to .70 budget)* | *map .24 / syn .14 / cnt .32* |

The optimum says **counter is the strongest head-to-head signal and is under-weighted**, map
is over-weighted, and synergy is about right — *not* the "synergy is over-weighted" guess a
glance might suggest (which the `trim-synergy` row shows gains nothing).

**Applied change** ([`scoring.py`](../backend/bsdraft/engine/scoring.py)): `map 0.40 → 0.32`,
`counter 0.15 → 0.23` (synergy, model, role, mastery, personal unchanged). This captures ~57%
of the available headroom (0.5826 → 0.5852, ceiling 0.5872).

**Why only a half-step, not the full optimum?** The ablation is *match-level* (full teams, a
team-A−B difference), whereas `score_candidate` ranks a *single* brawler by its raw map
win-rate — the most reliable "is this brawler good here" signal for an individual pick, even
where it's partly redundant once both full teams are known. So map wasn't cut all the way to .24.

## Limitations

- **Low ceiling.** Ranked matchmaking equalizes teams (base team-A win-rate 0.502), so no
  weighting scheme pulls much past ~0.58 AUC. The honest claim is "a small, real draft edge,"
  not "predicts winners."
- **Match-level ≠ candidate-level.** The reweight direction transfers; exact magnitudes don't.
- The reweight's downstream effect on *pick rankings* can't be validated against outcomes (no
  pick-level labels), so the change is deliberately conservative.

## Reproduce

```bash
PYTHONPATH=backend .venv/bin/python backend/scripts/ablate_components.py   # -> docs/ablation.json
PYTHONPATH=backend .venv/bin/python backend/scripts/ablate_context.py      # -> docs/ablation_context.json
```
