"""Context test: does the *right* weighting of the draft signals change with the mode?

Follow-up to ablate_components.py. That showed the GLOBAL optimal net/empirical split is
~31/69 — close to the shipped fixed weights. The open question (the original one) is whether
the optimal *relative* weight of map / synergy / counter shifts by context, which a single
global fit would hide.

Scope: matches are final 3v3 comps, so "draft phase / already-picked" weighting is NOT
testable from outcomes (no partial-draft labels) — only map/mode is. We test per MODE.

To get stable per-mode estimates we cross-fit over the FULL dataset (K-fold): each row gets
an out-of-sample net logit (net retrained on the other folds) and out-of-sample empirical
terms (DraftStats built on the other folds). Then, per mode, we:
  * fit a logistic on standardized [map, synergy, counter] -> the relative weight each earns;
  * compare three blends on held-out (CV) AUC:
      fixed       = shipped weights (0.40/0.15/0.15)
      global-refit= one logistic on [map,syn,counter] over all rows
      mode-refit  = a logistic refit within the mode  (context-specific weighting)
    mode-refit > global-refit means context-specific weighting genuinely helps.

    PYTHONPATH=backend .venv/bin/python backend/scripts/ablate_context.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from ablate_components import empirical_terms, load_aligned, rows_for

from bsdraft.constants import REPO_ROOT
from bsdraft.data import encoders as E
from bsdraft.engine.scoring import DEFAULT_WEIGHTS
from bsdraft.engine.stats import DraftStats


def mode_names() -> dict:
    return {i: name for name, i in E.mode_encoder().items()}


def cross_fit_empirical(d, k, seed):
    """Out-of-sample map / synergy / counter terms for every row, via K-fold cross-fitting:
    each fold's terms come from a DraftStats built on the *other* folds (no in-sample leak)."""
    n = len(d["y"])
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, k, size=n)
    mp = np.zeros(n); sy = np.zeros(n); ct = np.zeros(n)
    for f in range(k):
        te = np.where(fold == f)[0]
        trn = np.where(fold != f)[0]
        print(f"  fold {f + 1}/{k}: stats on {len(trn)}, score {len(te)}")
        stats = DraftStats(rows_for(d, trn))
        a = [d["a_ids"][i] for i in te]; b = [d["b_ids"][i] for i in te]
        mids = [d["map_ids"][i] for i in te]
        m_, s_, c_ = empirical_terms(stats, a, b, mids)
        mp[te], sy[te], ct[te] = m_, s_, c_
    return mp, sy, ct


def std_coef(X, y, cv=5):
    """Standardized logistic coefficients + CV'd P(win), on the given rows."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    p = cross_val_predict(lr, Xs, y, cv=min(cv, 3) if len(y) < 1500 else cv,
                          method="predict_proba")[:, 1]
    lr.fit(Xs, y)
    return lr.coef_[0], p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load_aligned()
    n = len(d["y"])
    print(f"aligned ranked matches: {n}")
    print(f"cross-fitting empirical signals ({args.folds} folds)...")
    mp, sy, ct = cross_fit_empirical(d, args.folds, args.seed)
    w = DEFAULT_WEIGHTS
    emp = w["map"] * mp + w["synergy"] * sy + w["counter"] * ct      # shipped fixed weights
    y = d["y"]
    names = mode_names()

    # global reference: relative weight each empirical signal earns over ALL data
    gcoef, p_global = std_coef(np.column_stack([mp, sy, ct]), y)
    print("\n=== global standardized weights on [map, synergy, counter] ===")
    print(f"  map {gcoef[0]:+.3f}   synergy {gcoef[1]:+.3f}   counter {gcoef[2]:+.3f}")

    # per-mode: do the relative weights move, and does refitting per mode help?
    print("\n=== per-mode standardized weights + held-out AUC (fixed vs global-refit vs mode-refit) ===")
    hdr = f"{'mode':<13}{'n':>6}{'map':>8}{'syn':>8}{'cnt':>8}   {'fixed':>7}{'gl-refit':>9}{'md-refit':>9}"
    print(hdr)
    rows_out = {}
    for mi in sorted(set(d["mode_idx"].tolist())):
        if mi == 0:
            continue
        rows = np.where(d["mode_idx"] == mi)[0]
        if len(rows) < 300:
            print(f"{names.get(mi, mi):<13}{len(rows):>6}   (too few matches — skipped)")
            continue
        ym = y[rows]
        coef, _ = std_coef(np.column_stack([mp[rows], sy[rows], ct[rows]]), ym)
        auc_fixed = roc_auc_score(ym, emp[rows])
        auc_global = roc_auc_score(ym, p_global[rows])              # global-refit, sliced to mode
        _, p_mode = std_coef(np.column_stack([mp[rows], sy[rows], ct[rows]]), ym)
        auc_mode = roc_auc_score(ym, p_mode)
        print(f"{names.get(mi, mi):<13}{len(rows):>6}{coef[0]:>8.2f}{coef[1]:>8.2f}{coef[2]:>8.2f}"
              f"   {auc_fixed:>7.3f}{auc_global:>9.3f}{auc_mode:>9.3f}")
        rows_out[names.get(mi, str(mi))] = {
            "n": int(len(rows)),
            "coef": {"map": float(coef[0]), "synergy": float(coef[1]), "counter": float(coef[2])},
            "auc": {"fixed": float(auc_fixed), "global_refit": float(auc_global),
                    "mode_refit": float(auc_mode)},
        }

    print("\n  Read: if the map/syn/cnt columns keep the same rank order across modes and")
    print("  md-refit barely beats gl-refit, context-specific weighting isn't worth it.")

    # --- concrete reweight check: current fixed weights vs a synergy-trimmed alternative ---
    # (held-out AUC on the cross-fitted terms; both are FIXED combinations, so no overfitting.
    # global-refit is the linear-reweighting ceiling.)
    print("\n=== fixed-weight options on the empirical trio (held-out AUC over all rows) ===")
    blends = {
        "current  (map .40 / syn .15 / cnt .15)": 0.40 * mp + 0.15 * sy + 0.15 * ct,
        "trim-syn (map .40 / syn .10 / cnt .20)": 0.40 * mp + 0.10 * sy + 0.20 * ct,
        "chosen   (map .32 / syn .15 / cnt .23)": 0.32 * mp + 0.15 * sy + 0.23 * ct,
    }
    auc_blends = {}
    for label, blend in blends.items():
        a = roc_auc_score(y, blend)
        auc_blends[label] = float(a)
        print(f"  {label:<42}{a:.4f}")
    print(f"  {'global-refit ceiling':<42}{roc_auc_score(y, p_global):.4f}")

    # what does the optimal linear blend actually weight? (raw coefs, scaled to the trio's
    # current 0.70 budget so they're comparable to map .40 / syn .15 / cnt .15)
    rc = LogisticRegression(max_iter=2000, C=1.0).fit(np.column_stack([mp, sy, ct]), y).coef_[0]
    scaled = rc / rc.sum() * 0.70
    print(f"\n  optimal trio weights (scaled to .70): map {scaled[0]:.2f} / syn {scaled[1]:.2f} / cnt {scaled[2]:.2f}")

    out = {"n": n, "global_coef": {"map": float(gcoef[0]), "synergy": float(gcoef[1]),
                                   "counter": float(gcoef[2])},
           "reweight_check": auc_blends, "by_mode": rows_out}
    (REPO_ROOT / "docs" / "ablation_context.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {REPO_ROOT / 'docs' / 'ablation_context.json'}")


if __name__ == "__main__":
    main()
