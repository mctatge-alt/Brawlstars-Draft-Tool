# Model Evaluation — How the draft signals are weighted, and whether that should change

The recommender fuses six signals into one pick score: a brawler's **map** win-rate, the
learned **model** (win-prob net), pairwise **synergy** with allies, **counter** vs. revealed
enemies, mode-based **role** fit, and player-specific **mastery**/**personal** history. The
first four are combined with fixed global weights ([`DEFAULT_WEIGHTS`](../backend/bsdraft/engine/scoring.py)).

This doc records the held-out ablation that tunes those weights. It has been run twice —
June 2026 on ~40k matches, and **August 2026 on ~995k matches** — and the headline finding
**reversed** between runs as the dataset grew 25×.

> TL;DR (2026-08-10, n = 995,135) — Draft is a small but real edge (AUC ~0.59–0.65 by mode;
> matchmaking equalizes teams). **The retrained net now out-discriminates every empirical
> signal** (AUC .625 vs .608 for the blend) — a full reversal of the June result, where the
> empirical side out-ranked a weaker net and earned ~69% of the stacker weight (it now earns
> ~22%). **Context-dependent (per-map/mode) weighting still does not help.** Applied
> rebalance: `model .20 → .40`, funded by `map .32 → .25`, `synergy .15 → .05` (its
> conditional coefficient is ~0 in every mode), `counter .23 → .20`.

## Method (leakage-free by construction)

Three scripts, run over the labeled Ranked matches:

- [`scripts/ablate_components.py`](../backend/scripts/ablate_components.py) — net vs. empirical signals, head-to-head.
- [`scripts/ablate_context.py`](../backend/scripts/ablate_context.py) — does the right weighting move by mode?
- [`scripts/sweep_blend.py`](../backend/scripts/sweep_blend.py) — concrete full-blend candidates (model + trio), scored as shippable fixed weight sets.

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
   stats come from the other folds) so per-mode estimates are stable.

## Result 1 — the net now subsumes most of the empirical signal (reversed from June)

Held-out validation (n = 149,270), all predictors well-calibrated (ECE ≤ 0.003):

| Predictor | log-loss | acc | AUC |
|---|---|---|---|
| always 0.5 | 0.6931 | .500 | — |
| net only | 0.6673 | .588 | **.625** |
| empirical blend only | 0.6740 | .576 | **.608** |
| net + empirical | 0.6667 | .589 | **.626** |

Standalone AUC of each raw signal: map `.608`, synergy `.584`, counter `.590`, blend `.608`, net `.625`.

- The **net out-discriminates the empirical blend** (.625 vs .608), and stacking the blend on
  top of it adds almost nothing (+.001 AUC). The embeddings have absorbed what the count
  tables know, plus interactions they can't represent.
- The stacker now assigns **~78% of the weight to the net, ~22% to the empirical side** — the
  exact mirror of June's 31/69. The June table (n = 6,031 val: net .567, blend .581) is kept
  below for the record; the flip tracks the net's training set growing 40k → 995k matches
  (shipped-model AUC .576 → .627 across the same period).

<details>
<summary>June 2026 run (n = 40,208) — the superseded result</summary>

| Predictor | log-loss | acc | AUC |
|---|---|---|---|
| net only | 0.6861 | .547 | .567 |
| empirical blend only | 0.6831 | .556 | .581 |
| net + empirical | 0.6826 | .560 | .583 |

Stacker: ~69% empirical / ~31% net. Standalone: map .568, synergy .564, counter .570.
</details>

## Result 2 — context-dependent weighting still does not help

Cross-fit over all 995,135 matches, per mode. `map/syn/cnt` are the standardized weights each
empirical signal earns; the AUC columns compare **fixed** (shipped weights), **global-refit**
(one logistic over all rows), and **mode-refit** (a logistic refit within the mode):

| Mode | n | map | syn | cnt | AUC fixed | global-refit | mode-refit |
|---|---|---|---|---|---|---|---|
| Gem Grab | 157,653 | +.24 | −.03 | +.19 | .600 | .602 | .602 |
| Brawl Ball | 159,981 | +.26 | −.10 | +.23 | .595 | .599 | .600 |
| Knockout | 168,114 | +.27 | −.02 | +.12 | .588 | .592 | .592 |
| Hot Zone | 158,166 | +.41 | −.10 | +.23 | .631 | .636 | .636 |
| Heist | 188,403 | +.51 | −.08 | +.16 | .641 | .648 | .647 |
| Bounty | 162,818 | +.28 | −.03 | +.11 | .587 | .592 | .592 |

- **`mode-refit` ≈ `global-refit` in every mode** (within ±0.001 AUC), on 26× the June sample.
  Context-dependent weighting remains not worth building.
- **Synergy's conditional coefficient is negative in all six modes** (−.02 to −.10): given map
  and counter, the pair-winrate tables add nothing — they are redundant, not informative.
  (Standalone, synergy still ranks at .584 — the redundancy is conditional.)
- What varies is still **how much draft matters at all**: Heist (.65) vs Bounty/Knockout
  (~.59). That's a confidence signal, not a reweighting one.

> **Why no draft-phase test?** Completed matches only contain final 3v3 comps — there are no
> partial-draft labels — so "weight signals differently as picks come in" can't be measured
> from outcomes. The engine already handles phase structurally: synergy/counter only activate
> once allies/enemies exist, and the blend renormalizes over the active signals.

## Result 3 — the applied change: shift weight from the trio to the model

Held-out AUC of **fixed** full-blend candidates (model + trio in one linear mix, prob-unit
values as `score_candidate` blends them; no fitting, so no overfit):

| Full-blend weighting (model/map/syn/cnt) | held-out AUC |
|---|---|
| June-2026 shipped `.20 / .32 / .15 / .23` | 0.6245 |
| pre-June `.20 / .40 / .15 / .15` | 0.6248 |
| **chosen `model-40` `.40 / .25 / .05 / .20`** | **0.6262** |
| `model-50` `.50 / .20 / .05 / .15` | 0.6260 |
| `model-60` `.60 / .16 / .02 / .12` | 0.6257 |
| `model-70` `.70 / .10 / .00 / .10` | 0.6252 |
| net-only `.90 / 0 / 0 / 0` | 0.6247 |
| refit ceiling (train-fit, val-scored) | 0.6267 |

- The curve **plateaus at model .40–.50 and falls off toward net-only** — the trio still earns
  its keep as a complement, just not as the majority partner. Note net-only ≈ the June-2026
  shipped weights: leaving the weights untuned wastes the whole model upgrade.
- `model-40` beat the shipped weights in **200/200 paired bootstrap resamples** and sits within
  .0005 AUC of the linear ceiling.
- The unconstrained refit's raw weights (e.g. a negative counter coefficient) are collinearity
  artifacts — the net has absorbed the counter signal — which is exactly why the decision is
  made on fixed candidates, not the refit.

**Applied change** ([`scoring.py`](../backend/bsdraft/engine/scoring.py)): `model 0.20 → 0.40`,
`map 0.32 → 0.25`, `synergy 0.15 → 0.05`, `counter 0.23 → 0.20` (role, mastery, personal
unchanged).

**Why keep synergy at .05 instead of 0?** Match-level redundancy is not candidate-level
uselessness: mid-draft, with two allies picked and no model-visible enemy comp, the pair
tables are still the only "fits with what we have" signal, and the blend renormalizes over
active signals — a small weight keeps that behavior (and its explanation in the UI) alive at
negligible cost (.0002 AUC vs the best syn-0 candidate).

## Limitations

- **Low ceiling.** Ranked matchmaking equalizes teams (base team-A win-rate 0.510), so no
  weighting scheme pulls far past ~0.63 AUC pooled. The honest claim is "a small, real draft
  edge," not "predicts winners."
- **Match-level ≠ candidate-level.** The reweight direction transfers; exact magnitudes don't.
- The reweight's downstream effect on *pick rankings* can't be validated against outcomes (no
  pick-level labels), so the change stays on the conservative side of the plateau.
- **These weights chase the model's quality.** The 69/31 → 22/78 flip shows the optimal blend
  moves with the net's training set. Re-run the suite after major dataset growth or an
  architecture change; the sweep exists so that check is one command.

## Reproduce

```bash
PYTHONPATH=backend .venv/bin/python backend/scripts/ablate_components.py   # -> docs/ablation.json
PYTHONPATH=backend .venv/bin/python backend/scripts/ablate_context.py      # -> docs/ablation_context.json
PYTHONPATH=backend .venv/bin/python backend/scripts/sweep_blend.py         # full-blend candidates
```
