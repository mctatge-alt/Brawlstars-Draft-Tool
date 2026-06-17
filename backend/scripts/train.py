"""Train and evaluate the win-probability model.

    PYTHONPATH=backend python backend/scripts/train.py --epochs 40

Compares against baselines (always-0.5 and logistic regression on brawler presence),
reports log-loss / accuracy / AUC / ECE on a held-out split, saves the model + config to
data/processed/winprob.pt, and writes calibration + training-curve charts to docs/.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bsdraft.constants import PROCESSED_DIR, REPO_ROOT  # noqa: E402
from bsdraft.data import dataset as D  # noqa: E402
from bsdraft.data import encoders as E  # noqa: E402
from bsdraft.models.winprob import ModelConfig, WinProbNet  # noqa: E402

DOCS = REPO_ROOT / "docs"


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(total)


def brawler_diff_features(team_a: np.ndarray, team_b: np.ndarray, n_brawlers: int) -> np.ndarray:
    """+1 per team_a brawler, -1 per team_b brawler — an antisymmetric linear baseline."""
    x = np.zeros((len(team_a), n_brawlers), dtype=np.float32)
    rows = np.arange(len(team_a))[:, None]
    np.add.at(x, (rows, team_a), 1.0)
    np.add.at(x, (rows, team_b), -1.0)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--halflife-days", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = D.build_dataset()
    n = len(ds)
    print(f"dataset: {D.summary(ds)}")
    if n < 200:
        print("Not enough labeled data yet — let the crawl collect more, then retrain.")
        return

    ta, tb = torch.tensor(ds.team_a), torch.tensor(ds.team_b)
    mp, mo = torch.tensor(ds.map_idx), torch.tensor(ds.mode_idx)
    y = torch.tensor(ds.y)

    # recency weights (time-decay, normalized to mean 1) — the "patch recency" lever
    tmax = int(ds.ts.max())
    if tmax > 0:
        w = np.power(0.5, (tmax - ds.ts) / (args.halflife_days * 86400.0)).astype(np.float32)
        w = w / w.mean()
    else:
        w = np.ones(n, dtype=np.float32)
    wt = torch.tensor(w)

    idx = np.random.permutation(n)
    n_val = int(n * args.val_frac)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    vai, tri = torch.tensor(val_i), torch.tensor(tr_i)
    yv = ds.y[val_i]

    # --- baselines ---
    const_ll = log_loss(yv, np.full_like(yv, 0.5), labels=[0, 1])
    x_tr = brawler_diff_features(ds.team_a[tr_i], ds.team_b[tr_i], E.num_brawlers())
    x_va = brawler_diff_features(ds.team_a[val_i], ds.team_b[val_i], E.num_brawlers())
    logreg = LogisticRegression(max_iter=2000, C=1.0)
    logreg.fit(x_tr, ds.y[tr_i])
    p_lr = logreg.predict_proba(x_va)[:, 1]
    lr_ll, lr_auc = log_loss(yv, p_lr, labels=[0, 1]), roc_auc_score(yv, p_lr)
    lr_acc = float(((p_lr > 0.5) == yv.astype(bool)).mean())

    # --- embedding model ---
    cfg = ModelConfig(E.num_brawlers(), E.num_maps(), E.num_modes())
    model = WinProbNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def batches(indices, bs):
        order = torch.randperm(len(indices))
        for k in range(0, len(indices), bs):
            yield indices[order[k:k + bs]]

    history, best_ll, best_state, bad, patience = [], float("inf"), None, 0, 6
    for _ in range(args.epochs):
        model.train()
        for bi in batches(tri, args.batch):
            opt.zero_grad()
            logit = model(ta[bi], tb[bi], mp[bi], mo[bi])
            loss = (bce(logit, y[bi]) * wt[bi]).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(ta[vai], tb[vai], mp[vai], mo[vai])).numpy()
        vll = log_loss(yv, pv, labels=[0, 1])
        history.append(vll)
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
        pv = torch.sigmoid(model(ta[vai], tb[vai], mp[vai], mo[vai])).numpy()
    m_ll, m_auc = log_loss(yv, pv, labels=[0, 1]), roc_auc_score(yv, pv)
    m_acc = float(((pv > 0.5) == yv.astype(bool)).mean())
    m_ece = expected_calibration_error(pv, yv)

    print("\n=== validation metrics (logloss/ECE: lower better; acc/AUC: higher better) ===")
    print(f"{'model':<22}{'logloss':>10}{'acc':>8}{'AUC':>8}{'ECE':>8}")
    print(f"{'always 0.5':<22}{const_ll:>10.4f}{0.5:>8.3f}{'-':>8}{'-':>8}")
    print(f"{'logreg (brawlers)':<22}{lr_ll:>10.4f}{lr_acc:>8.3f}{lr_auc:>8.3f}{'-':>8}")
    print(f"{'embedding net':<22}{m_ll:>10.4f}{m_acc:>8.3f}{m_auc:>8.3f}{m_ece:>8.3f}")

    # --- save artifacts ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, PROCESSED_DIR / "winprob.pt")
    DOCS.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_total": n, "n_val": int(n_val),
        "const": {"logloss": float(const_ll)},
        "logreg": {"logloss": float(lr_ll), "acc": lr_acc, "auc": float(lr_auc)},
        "embedding": {"logloss": float(m_ll), "acc": m_acc, "auc": float(m_auc), "ece": m_ece},
    }
    (DOCS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history, marker="o", ms=3)
    ax[0].axhline(0.6931, ls="--", c="gray", label="always 0.5")
    ax[0].axhline(lr_ll, ls=":", c="orange", label="logreg")
    ax[0].set_title("validation log-loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    edges = np.linspace(0, 1, 11)
    mids, accs = [], []
    for i in range(10):
        m = (pv > edges[i]) & (pv <= edges[i + 1]) if i else (pv >= edges[i]) & (pv <= edges[i + 1])
        if m.sum():
            mids.append(pv[m].mean()); accs.append(yv[m].mean())
    ax[1].plot([0, 1], [0, 1], ls="--", c="gray")
    ax[1].plot(mids, accs, marker="o")
    ax[1].set_title(f"calibration (ECE={m_ece:.3f})")
    ax[1].set_xlabel("predicted P(win)"); ax[1].set_ylabel("observed win-rate")
    fig.tight_layout()
    fig.savefig(DOCS / "training.png", dpi=120)

    print(f"\nsaved model  -> {PROCESSED_DIR / 'winprob.pt'}")
    print(f"saved charts -> {DOCS / 'training.png'}")
    print(f"saved metrics-> {DOCS / 'metrics.json'}")


if __name__ == "__main__":
    main()
