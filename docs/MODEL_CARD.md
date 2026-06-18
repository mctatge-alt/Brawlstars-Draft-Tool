# Model Card — Brawl Stars Win-Probability Model

A compact neural model that predicts the probability that team A beats team B in a Brawl
Stars *Ranked* 3v3 match, given both teams' brawlers and the map. It is the core signal
behind the draft assistant's pick/ban recommendations.

## Intended use

- **Primary:** rank candidate picks/bans by their marginal effect on win-probability during a
  live ranked draft, and power the seat-aware lookahead search.
- **Not intended for:** predicting individual match outcomes with high confidence, betting, or
  any use that assumes the draft alone determines the result (it does not — see Limitations).

## Data

- **Source:** the official [Brawl Stars API](https://developer.brawlstars.com). It is
  *player-centric* — you fetch a known player's ~25 most recent battles; there is no global
  match feed, and the draft's **ban phase is not exposed** (only the final picked teams).
- **Collection:** a snowball crawler seeds top players from country/global leaderboards, then
  harvests the other five player tags from every ranked match to expand the frontier. Matches
  are deduped by a stable key (`battleTime` + sorted player tags), since one match appears in
  up to six players' logs.
- **Size:** ~30,200 unique ranked matches (~30,000 labeled after dropping draws). Each row is
  `(map, mode, team A brawlers[3], team B brawlers[3]) → winner`, plus per-brawler power level,
  trophies, and the queue type (`soloRanked`/`teamRanked`).
- **Population & bias:** seeded from top-ladder players, so the data reflects high-skill
  ranked play. Team-A win-rate is ~0.50 (no positional label bias). The active map pool is
  whatever is in the current ranked rotation (~26 maps at time of collection).

## Inputs / features

- `team_a`, `team_b`: three brawler indices each (105 brawlers, contiguous embedding index).
- `map`: index over the ranked-map pool (+ an unknown bucket).
- `mode`: one of the six current ranked modes (+ unknown).

Brawler classes (for role-fit/warnings) come from Brawlify, with a maintained override table
for the newest brawlers that Brawlify has not yet class-tagged.

## Architecture

A small PyTorch network (embedding dims 32 / 16 / 8; ~tens of thousands of parameters) with an
**antisymmetric** head. Let $E_b \in \mathbb{R}^{32}$ be a learned brawler embedding, and let the
map/mode **context** be the concatenation

$$
c = [\,e_{\text{map}},\ e_{\text{mode}}\,] \in \mathbb{R}^{16+8}.
$$

A single **strength** network scores either team in context — it is order-invariant because it reads
the *mean* brawler embedding — and a pair of low-rank embeddings $p_b, q_b \in \mathbb{R}^{16}$ act as
per-brawler "attacker" and "defender" vectors:

$$
S(T, c) = \mathrm{MLP}\!\Big(\big[\, \tfrac{1}{|T|}\sum_{b \in T} E_b,\ \ c \,\big]\Big),
\qquad
P_T = \sum_{b \in T} p_b,
\qquad
Q_T = \sum_{b \in T} q_b.
$$

The logit that team $A$ beats team $B$ adds a strength difference to a bilinear counter term, and the
win probability is its sigmoid:

$$
\ell(A, B \mid c) = \underbrace{\big(S(A, c) - S(B, c)\big)}_{\text{team strength}}
\;+\; \underbrace{\big(P_A \cdot Q_B - P_B \cdot Q_A\big)}_{\text{directed counters}},
\qquad
P(A \text{ wins} \mid c) = \sigma\!\big(\ell(A, B \mid c)\big).
$$

**Why antisymmetric.** Every term changes sign under the swap $A \leftrightarrow B$, so
$\ell(B, A \mid c) = -\,\ell(A, B \mid c)$. Since $\sigma(-z) = 1 - \sigma(z)$,

$$
P(A \text{ wins} \mid c) + P(B \text{ wins} \mid c) = \sigma(z) + \sigma(-z) = 1
$$

holds *by construction*. This bakes in a correct inductive bias (no team-order bias, no global offset
to learn) and makes team-swap data augmentation unnecessary. The bilinear pairing
$P_A \cdot Q_B - P_B \cdot Q_A$ encodes **directed** matchups (X beats Y) that an additive strength
model cannot express, while keeping the whole head sign-flipping.

## Training

The objective is **recency-weighted binary cross-entropy** on the win label $y_i \in \{0, 1\}$ ($1$
when team $A$ won). Each match is down-weighted by an exponential time-decay so the fit leans toward
the live meta across balance patches:

$$
\mathcal{L}(\theta) = -\sum_i w_i \big[\, y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i) \,\big],
\qquad
\hat p_i = \sigma\!\big(\ell(A_i, B_i \mid c_i)\big),
\qquad
w_i = 2^{-(t_{\max} - t_i)/\tau},
$$

with a configurable half-life $\tau$ (default $\approx 30$ days, so a game from a month ago carries
about half the weight of a fresh one; the weights are normalized to mean $1$).

- **Recency weighting** uses the same exponential time-decay as the empirical stats table, so the
  model and the stats both lean on recent matches; pass a non-positive half-life to disable it
  (uniform weights) for backtests.
- **Split:** random 85/15 train/val (seeded). Optimizer AdamW, weight decay 1e-4, early
  stopping on validation log-loss.
- **Baselines:** (a) constant 0.5, (b) logistic regression on signed brawler-presence features.

## Evaluation

Held-out validation (~4.5k matches):

| Model | Log-loss ↓ | Accuracy ↑ | AUC ↑ | ECE ↓ |
| --- | --- | --- | --- | --- |
| Always 0.5 | 0.6931 | 0.500 | – | – |
| Logistic regression | 0.6879 | 0.541 | 0.557 | – |
| **Embedding net** | **0.6846** | **0.549** | **0.576** | **0.012** |

- Beats both baselines on every metric.
- **Calibration is the headline:** ECE 0.012 means the predicted probabilities are
  trustworthy — when it says 60%, the team wins ~60% of the time. For an assistant that
  *consumes* probabilities, calibration matters more than raw accuracy.
- Charts: see [`docs/training.png`](training.png) (validation curve + reliability diagram).

## Limitations

- **Skill-dominated outcomes.** At top ladder both teams draft well; the *draft* explains only
  a slice of the result, capping achievable AUC (~0.57 here; comparable open projects on ~1M
  battles land near ~0.6). The tool therefore uses the model for *relative* pick ranking and
  fuses it with lower-variance empirical signals — it does not present any single absolute
  win-probability as gospel.
- **No ban data.** The API never exposes bans, so the model is trained on final picks; ban
  value is inferred separately from win-rate + contest rate.
- **Population shift.** Trained on top-ladder solo-queue play; lower brackets and premade
  coordination differ. A rank-bracket conditioning is on the roadmap.
- **Meta drift.** Brawler strength changes with patches; the model needs periodic retraining on
  fresh data (recency weighting mitigates but does not eliminate this).

## Ethical considerations

Uses only publicly available, game-provided match data via the official API. No personal data
beyond public player tags/handles is stored. Not affiliated with Supercell.

## How the engine uses it

The model is one component of a transparent fused score, combined with empirical signals from the
collected matches. Every raw win-rate is **Bayesian-shrunk** toward a prior $\pi$ (0.5 globally, or
the global rate for a per-bracket table) with a pseudo-count $\kappa = 20$, which also defines the
confidence shown in the UI:

$$
\widehat{w} = \frac{\text{wins} + \kappa\,\pi}{\text{games} + \kappa},
\qquad
\mathrm{conf} = \frac{\text{games}}{\text{games} + \kappa},
$$

where "wins" and "games" are recency-weighted counts $\sum_i w_i$ (the same exponential time-decay
used to train the model, here with a ~3-week half-life). The
map win-rate, synergy, counter, role-fit, model, and optional mastery signals are then fused into a
renormalized weighted average over whichever signals $\mathcal{A}$ are *active* at the current draft
state:

$$
\mathrm{score}(b) = \frac{\sum_{k \in \mathcal{A}} \omega_k\, v_k(b)}{\sum_{k \in \mathcal{A}} \omega_k}.
$$

That fused score drives the displayed pick ranking. The seat-aware **minimax** search instead uses the
model as its leaf evaluator and a cheaper heuristic to prune each node to its top-$K$ candidates,
$h(b) = 0.5\,\mathrm{mw}(b) + 0.25\,\overline{\mathrm{syn}}(b) + 0.25\,\overline{\mathrm{cnt}}(b)$ — see
the [README](../README.md#how-it-works) for the recursion. Every component is surfaced in the UI, so
recommendations are explainable rather than a black box.
