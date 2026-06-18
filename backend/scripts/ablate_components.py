"""Ablation: do the empirical draft signals (map / synergy / counter) add head-to-head
discrimination *over the learned net*, and how much weight do they earn alongside it?

This is the diagnostic that decides whether it's worth investing in map-conditioned
synergy/counters or context-dependent blend weights. It is deliberately leakage-free:

  * one random split (seed, val-frac) into train / val;
  * the net is RE-TRAINED on train only (the shipped model saw data that has since grown,
    so we can't reuse its holdout) — same recipe as scripts/train.py;
  * the empirical DraftStats table is built on train rows ONLY;
  * everything is scored on the held-out val rows.

We then express each draft signal as an antisymmetric "team-A advantage" and ask:
  1. standalone discrimination (AUC) of each raw signal on val;
  2. calibrated accuracy/log-loss of net-only vs empirical-blend-only vs the two stacked,
     where calibration / stacking is fit on TRAIN and applied to VAL;
  3. the stacker's standardized coefficients — how much weight the empirical blend earns
     next to the net (the crux: if it's ~0, the components add nothing the net lacks).

    PYTHONPATH=backend .venv/bin/python backend/scripts/ablate_components.py
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict

from bsdraft.constants import PROCESSED_DIR, REPO_ROOT
from bsdraft.data import dataset as D
from bsdraft.data import encoders as E
from bsdraft.engine.scoring import DEFAULT_WEIGHTS
from bsdraft.engine.stats import DraftStats
from bsdraft.models.winprob import ModelConfig, WinProbNet


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i else (probs >= lo) & (probs <= hi)
        if mask.sum():
            total += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(total)


def load_aligned():
    """One pass mirroring dataset.build_dataset's filters, keeping BOTH the encoded arrays
    (for the net) and the parallel raw brawler-id / map-id rows (for the empirical signals)."""
    bidx = E.brawler_encoder()
    ta_i, tb_i, mp_i, mo_i = [], [], [], []
    a_ids, b_ids, map_ids, ys, tss = [], [], [], [], []
    for r in D.iter_matches():
        if r.get("a_won") is None:
            continue
        a = [p["brawler_id"] for p in r["team_a"]]
        b = [p["brawler_id"] for p in r["team_b"]]
        if any(x not in bidx for x in a + b):
            continue
        midx = E.encode_map(r.get("map_id"))
        if midx == 0:  # ranked maps only (matches build_dataset default)
            continue
        ta_i.append([bidx[x] for x in a]); tb_i.append([bidx[x] for x in b])
        mp_i.append(midx); mo_i.append(E.encode_mode(r.get("mode", "")))
        a_ids.append(a); b_ids.append(b); map_ids.append(r.get("map_id"))
        ys.append(1.0 if r["a_won"] else 0.0); tss.append(int(r.get("ts", 0)))
    return {
        "team_a": np.array(ta_i, np.int64), "team_b": np.array(tb_i, np.int64),
        "map_idx": np.array(mp_i, np.int64), "mode_idx": np.array(mo_i, np.int64),
        "a_ids": a_ids, "b_ids": b_ids, "map_ids": map_ids,
        "y": np.array(ys, np.float32), "ts": np.array(tss, np.int64),
    }


def rows_for(d, idx):
    """Reconstruct raw match dicts (as DraftStats expects) for a set of row indices."""
    return [
        {"a_won": bool(d["y"][i]), "map_id": d["map_ids"][i], "ts": int(d["ts"][i]),
         "team_a": [{"brawler_id": x} for x in d["a_ids"][i]],
         "team_b": [{"brawler_id": x} for x in d["b_ids"][i]]}
        for i in idx
    ]


def empirical_terms(stats: DraftStats, a_ids, b_ids, map_ids):
    """Antisymmetric team-A advantage from each draft signal (centered ~0, sign = favors A):
      map      = mean map win-rate(A) - mean map win-rate(B)
      synergy  = mean within-A pair win-rate - mean within-B pair win-rate
      counter  = mean counter(x in A, y in B) - 0.5   (already directed A-vs-B)
    """
    n = len(a_ids)
    mp = np.zeros(n); sy = np.zeros(n); ct = np.zeros(n)
    for i in range(n):
        A, B, mid = a_ids[i], b_ids[i], map_ids[i]
        mp[i] = (np.mean([stats.brawler_rate(x, mid).winrate for x in A])
                 - np.mean([stats.brawler_rate(x, mid).winrate for x in B]))
        sy[i] = (np.mean([stats.synergy(p, q).winrate for p, q in combinations(A, 2)])
                 - np.mean([stats.synergy(p, q).winrate for p, q in combinations(B, 2)]))
        ct[i] = np.mean([stats.counter(x, y).winrate for x in A for y in B]) - 0.5
    return mp, sy, ct


def train_net(d, tr, halflife_days, epochs, batch, lr, seed, patience=6):
    """Re-train WinProbNet on the train indices — same recipe as scripts/train.py, incl.
    early stopping with best-state restore (so the net is calibrated, not overtrained).
    Stops on an INNER val carved from train; the outer val stays untouched for evaluation."""
    torch.manual_seed(seed)
    ta, tb = torch.tensor(d["team_a"]), torch.tensor(d["team_b"])
    mp, mo = torch.tensor(d["map_idx"]), torch.tensor(d["mode_idx"])
    y = torch.tensor(d["y"])
    tmax = int(d["ts"].max())
    if tmax > 0:
        w = np.power(0.5, (tmax - d["ts"]) / (halflife_days * 86400.0)).astype(np.float32)
        w = w / w.mean()
    else:
        w = np.ones(len(d["y"]), np.float32)
    wt = torch.tensor(w)

    rng = np.random.RandomState(seed + 1)
    perm = rng.permutation(len(tr))
    n_iv = int(len(tr) * 0.15)
    fit = torch.tensor(tr[perm[n_iv:]])     # train the weights on this
    iv = torch.tensor(tr[perm[:n_iv]])      # early-stop on this
    yiv = d["y"][tr[perm[:n_iv]]]

    cfg = ModelConfig(E.num_brawlers(), E.num_maps(), E.num_modes())
    model = WinProbNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best_ll, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(fit))
        for k in range(0, len(fit), batch):
            bi = fit[order[k:k + batch]]
            opt.zero_grad()
            loss = (bce(model(ta[bi], tb[bi], mp[bi], mo[bi]), y[bi]) * wt[bi]).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            piv = torch.sigmoid(model(ta[iv], tb[iv], mp[iv], mo[iv])).numpy()
        vll = log_loss(yiv, piv, labels=[0, 1])
        if vll < best_ll - 1e-4:
            best_ll, bad = vll, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logit = model(ta, tb, mp, mo).numpy()  # net logit for ALL rows
    return logit


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (log_loss(y, p, labels=[0, 1]),
            float(((p > 0.5) == y.astype(bool)).mean()),
            roc_auc_score(y, p),
            expected_calibration_error(p, y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--halflife-days", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load_aligned()
    n = len(d["y"])
    print(f"aligned ranked matches: {n}  (team_a win-rate {d['y'].mean():.3f})")
    if n < 500:
        print("Not enough labeled data for a meaningful ablation yet.")
        return

    np.random.seed(args.seed)
    idx = np.random.permutation(n)
    n_val = int(n * args.val_frac)
    val, tr = idx[:n_val], idx[n_val:]
    yv, ytr = d["y"][val], d["y"][tr]
    print(f"split: train {len(tr)}  /  val {len(val)}\n")

    # --- empirical stats on TRAIN rows only (no leakage) ---
    stats = DraftStats(rows_for(d, tr))

    print("computing empirical signals (map / synergy / counter)...")
    mp, sy, ct = empirical_terms(stats, d["a_ids"], d["b_ids"], d["map_ids"])
    wts = DEFAULT_WEIGHTS
    emp = wts["map"] * mp + wts["synergy"] * sy + wts["counter"] * ct  # shipped empirical blend

    print("re-training the net on train rows...")
    net_logit = train_net(d, tr, args.halflife_days, args.epochs, args.batch, args.lr, args.seed)

    # --- 1) standalone discrimination of each raw signal (val) ---
    print("\n=== standalone discrimination on val (AUC; 0.5 = coin flip) ===")
    for name, sig in [("map win-rate diff", mp), ("synergy diff", sy),
                      ("counter (directed)", ct), ("empirical blend", emp),
                      ("net (logit)", net_logit)]:
        print(f"  {name:<22}{roc_auc_score(yv, sig[val]):>8.4f}")

    # --- 2) calibrated head-to-head on val, with clean (out-of-sample) features ---
    # net_logit and emp on val are already out-of-sample (net trained on train, stats built
    # on train), so we calibrate/stack ON val via 5-fold CV — no train-side in-sample leak.
    def cv_eval(feats):
        X = np.column_stack(feats)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        lr = LogisticRegression(max_iter=2000, C=1.0)
        p = cross_val_predict(lr, Xs, yv, cv=5, method="predict_proba")[:, 1]
        lr.fit(Xs, yv)  # full-fit only to read the standardized coefficients
        return p, lr.coef_[0]

    nv, empv = net_logit[val], emp[val]
    p_net, _ = cv_eval([nv])
    p_emp, _ = cv_eval([empv])
    p_full, coef_full = cv_eval([nv, empv])
    p_const = np.full_like(yv, 0.5)

    print("\n=== calibrated on val, 5-fold CV (logloss/ECE lower=better; acc/AUC higher=better) ===")
    print(f"{'predictor':<22}{'logloss':>10}{'acc':>8}{'AUC':>8}{'ECE':>8}")
    print(f"{'always 0.5':<22}{log_loss(yv, p_const, labels=[0,1]):>10.4f}{0.5:>8.3f}{'-':>8}{'-':>8}")
    for name, p in [("net only", p_net), ("empirical blend only", p_emp),
                    ("net + empirical", p_full)]:
        ll, acc, auc, ece = metrics(p, yv)
        print(f"{name:<22}{ll:>10.4f}{acc:>8.3f}{auc:>8.3f}{ece:>8.3f}")

    # --- 3) the crux: weight each source earns in the stacker (standardized coefs) ---
    share = abs(coef_full[1]) / (abs(coef_full[0]) + abs(coef_full[1]) + 1e-9)
    print("\n=== stacker weights (standardized coef) — net vs empirical ===")
    print(f"  net       {coef_full[0]:+.4f}")
    print(f"  empirical {coef_full[1]:+.4f}")
    print(f"\n  empirical share of stack weight: {share:.1%}")
    print("  (the meta-learner leans on whichever source carries independent signal)")

    out = {
        "n": n, "n_val": int(n_val),
        "standalone_auc": {
            "map": float(roc_auc_score(yv, mp[val])), "synergy": float(roc_auc_score(yv, sy[val])),
            "counter": float(roc_auc_score(yv, ct[val])), "empirical_blend": float(roc_auc_score(yv, emp[val])),
            "net": float(roc_auc_score(yv, net_logit[val])),
        },
        "calibrated": {
            "net": dict(zip(("logloss", "acc", "auc", "ece"), metrics(p_net, yv))),
            "empirical": dict(zip(("logloss", "acc", "auc", "ece"), metrics(p_emp, yv))),
            "full": dict(zip(("logloss", "acc", "auc", "ece"), metrics(p_full, yv))),
        },
        "stack_coef": {"net": float(coef_full[0]), "empirical": float(coef_full[1])},
        "empirical_share": float(share),
    }
    (REPO_ROOT / "docs" / "ablation.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {REPO_ROOT / 'docs' / 'ablation.json'}")


if __name__ == "__main__":
    main()
