"""Sweep FULL fusion-weight candidates — the model term blended with the empirical trio.

Third leg of the ablation suite. ablate_components.py answers "how much weight does the
net earn vs the empirical side?" (stacker share); ablate_context.py tunes the trio's
internals. This script closes the loop: it scores concrete DEFAULT_WEIGHTS-style
candidates — model + map + synergy + counter as one fixed linear blend, in the prob-unit
values engine/scoring.py mixes — on the same leakage-free seed-0 split
(train-only DraftStats, net re-trained on train, everything scored on held-out val).

Fixed candidates can't overfit, so the best row is directly shippable; the train-fit
logistic refit is reported as the linear ceiling, and a paired bootstrap says whether the
winner's edge over the shipped weights is real or noise.

    PYTHONPATH=backend .venv/bin/python backend/scripts/sweep_blend.py
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ablate_components import empirical_terms, load_aligned, rows_for, train_net

from bsdraft.engine.scoring import DEFAULT_WEIGHTS
from bsdraft.engine.stats import DraftStats

# Candidates as (model, map, synergy, counter), sharing the non-role 0.90 budget.
# role is excluded: it is constant within a matchup diff and unmeasurable from outcomes;
# mastery/personal are roster-only signals with no match-log labels.
CANDIDATES = {
    "june-2026 (mdl .20/map .32/syn .15/cnt .23)": (0.20, 0.32, 0.15, 0.23),
    "pre-june  (mdl .20/map .40/syn .15/cnt .15)": (0.20, 0.40, 0.15, 0.15),
    "model-40  (mdl .40/map .25/syn .05/cnt .20)": (0.40, 0.25, 0.05, 0.20),
    "model-50  (mdl .50/map .20/syn .05/cnt .15)": (0.50, 0.20, 0.05, 0.15),
    "model-60  (mdl .60/map .16/syn .02/cnt .12)": (0.60, 0.16, 0.02, 0.12),
    "model-70  (mdl .70/map .10/syn .00/cnt .10)": (0.70, 0.10, 0.00, 0.10),
    "net-only  (mdl .90/map .00/syn .00/cnt .00)": (0.90, 0.00, 0.00, 0.00),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--halflife-days", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()

    d = load_aligned()
    n = len(d["y"])
    np.random.seed(args.seed)
    idx = np.random.permutation(n)
    n_val = int(n * args.val_frac)
    val, tr = idx[:n_val], idx[n_val:]
    yv = d["y"][val]
    print(f"aligned ranked matches: {n}  split: train {len(tr)} / val {len(val)}")

    stats = DraftStats(rows_for(d, tr))
    print("computing empirical signals (map / synergy / counter)...")
    mp, sy, ct = empirical_terms(stats, d["a_ids"], d["b_ids"], d["map_ids"])
    print("re-training the net on train rows...")
    logit = train_net(d, tr, args.halflife_days, args.epochs, args.batch, args.lr, args.seed)
    pm = 1.0 / (1.0 + np.exp(-logit)) - 0.5   # model term in prob units, centered like the trio

    def blend(w):
        wm, wmap, wsyn, wcnt = w
        return wm * pm + wmap * mp + wsyn * sy + wcnt * ct

    shipped = (DEFAULT_WEIGHTS["model"], DEFAULT_WEIGHTS["map"],
               DEFAULT_WEIGHTS["synergy"], DEFAULT_WEIGHTS["counter"])
    shipped_label = (f"shipped   (mdl {shipped[0]:.2f}/map {shipped[1]:.2f}"
                     f"/syn {shipped[2]:.2f}/cnt {shipped[3]:.2f})")
    cands = {shipped_label: shipped, **CANDIDATES}

    print("\n=== fixed full-blend candidates — held-out AUC on val ===")
    scores = {}
    for label, w in cands.items():
        scores[label] = roc_auc_score(yv, blend(w)[val])
        print(f"  {label:<46}{scores[label]:.4f}")

    X = np.column_stack([pm, mp, sy, ct])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], d["y"][tr])
    print(f"  {'refit ceiling (train-fit, val-scored)':<46}"
          f"{roc_auc_score(yv, lr.decision_function(X[val])):.4f}")
    rc = lr.coef_[0]
    scaled = rc / rc.sum() * 0.90
    print(f"\n  refit weights (scaled to .90 — collinear, read with care): "
          f"model {scaled[0]:.2f} / map {scaled[1]:.2f} / syn {scaled[2]:.2f} / cnt {scaled[3]:.2f}")

    best = max(scores, key=scores.get)
    if best != shipped_label:
        sb, ss = blend(cands[best])[val], blend(shipped)[val]
        rng = np.random.RandomState(1)
        wins = sum(
            roc_auc_score(yv[bi], sb[bi]) > roc_auc_score(yv[bi], ss[bi])
            for bi in (rng.randint(0, len(val), len(val)) for _ in range(args.bootstrap))
        )
        print(f"\n  paired bootstrap ({args.bootstrap} resamples): "
              f"'{best.split()[0]}' beats shipped in {wins}/{args.bootstrap}")
    else:
        print("\n  shipped weights are already the best candidate.")


if __name__ == "__main__":
    main()
