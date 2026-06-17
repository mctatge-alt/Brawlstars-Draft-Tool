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
**antisymmetric** head:

```
logit = [ S(A, ctx) − S(B, ctx) ]      # context-conditioned team strength
      + [ PA·QB − PB·QA ]               # low-rank antisymmetric counter term

ctx        = [ map_emb, mode_emb ]
S(team,ctx)= MLP( mean(brawler_emb(team)) ⊕ ctx )         # shared across both teams
PA, QA     = Σ counter_p(team_a), Σ counter_q(team_a)     # low-rank "attacker/defender"
```

**Why antisymmetric.** Swapping A and B negates the logit, so `P(A wins) + P(B wins) = 1`
holds *by construction*. This bakes in a correct inductive bias (no team-order bias, no global
offset), and makes data augmentation by team-swap unnecessary. The bilinear counter term
encodes directed matchups (X beats Y) that an additive strength model cannot express, while
remaining antisymmetric.

## Training

- **Loss:** binary cross-entropy on the win label.
- **Recency weighting:** samples are time-decayed (configurable half-life) so the model can
  lean toward recent meta — relevant because brawler strength shifts with balance patches.
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

The model is one component of a transparent fused score
(`map win-rate + synergy + counter + role-fit + model + mastery`), and the leaf evaluator for
the seat-aware minimax search. Every component is surfaced in the UI so recommendations are
explainable rather than a black box.
