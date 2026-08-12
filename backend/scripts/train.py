"""Train and evaluate the win-probability model.

    PYTHONPATH=backend python backend/scripts/train.py --epochs 40

Trains on masked drafts: each epoch every match is re-masked — kept as the full 3v3 with
probability --p-full, otherwise cut down to a random partial draft state (k_a, k_b known
picks per side), unknown slots set to the trained mask row. The model therefore scores
unfinished drafts directly, marginalizing over how real drafts continued.

Compares against baselines (always-0.5 and logistic regression on brawler presence),
reports log-loss / accuracy / AUC / ECE on a held-out split — both on full comps
(comparable across retrains) and per partial draft state — saves the model + config to
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


# Draft states (known picks per side) sampled during training, besides the full (3, 3).
# (0, 0) is excluded: antisymmetry makes its logit identically 0 with zero gradient, so
# training on it is pure wasted compute — it still predicts exactly 0.5 at inference.
PARTIAL_STATES = np.array([(a, b) for a in range(4) for b in range(4)
                           if (a, b) not in ((3, 3), (0, 0))], dtype=np.int64)


def mask_to_known(team: np.ndarray, k, mask_row: int, rng: np.random.Generator) -> np.ndarray:
    """Keep a uniformly random subset of ``k`` picks per row of a (N, 3) team array and
    replace the rest with ``mask_row``. ``k`` is a scalar or an (N,) array."""
    known = np.argsort(rng.random(team.shape), axis=1) < np.asarray(k).reshape(-1, 1)
    return np.where(known, team, mask_row)


def mask_teams(team_a: np.ndarray, team_b: np.ndarray, mask_row: int, p_full: float,
               rng: np.random.Generator) -> tuple:
    """Masked copies of (N, 3) team arrays. Each row keeps the full comp with probability
    ``p_full``; otherwise a draft state (k_a, k_b) is drawn uniformly from PARTIAL_STATES
    and a uniformly random subset of each team beyond k known picks is replaced by
    ``mask_row``. Fresh masks per call = free augmentation across epochs."""
    n = team_a.shape[0]
    ka = np.full(n, 3, dtype=np.int64)
    kb = np.full(n, 3, dtype=np.int64)
    partial = rng.random(n) >= p_full
    if partial.any():
        states = PARTIAL_STATES[rng.integers(len(PARTIAL_STATES), size=int(partial.sum()))]
        ka[partial], kb[partial] = states[:, 0], states[:, 1]
    return mask_to_known(team_a, ka, mask_row, rng), mask_to_known(team_b, kb, mask_row, rng)


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
    ap.add_argument("--p-full", type=float, default=0.7,
                    help="probability a training example keeps its full 3v3 (rest are masked "
                         "to random partial draft states). 0.7 held full-comp parity with the "
                         "unmasked control while matching 0.5's partial-state quality; raise "
                         "it if the paired full-comp gate ever regresses")
    ap.add_argument("--max-full-delta", type=float, default=0.002,
                    help="hard gate: abort (exit 1, no artifacts written) if full-comp val "
                         "logloss exceeds the previous checkpoint's by more than this on the "
                         "same rows — keeps the unattended --retrain-on-shift path from "
                         "publishing a regressed model. Set <0 to disable.")
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

    # --- paired no-regression baseline ---
    # The previous checkpoint, evaluated on THIS run's val rows (full comps). Comparing
    # against docs/metrics.json alone would mix data drift into the delta: that file was
    # written on a different dataset snapshot and split.
    baseline = None
    prev_pt = PROCESSED_DIR / "winprob.pt"
    if prev_pt.exists():
        ck = torch.load(prev_pt, map_location="cpu", weights_only=True)
        pcfg = ModelConfig(**ck["config"])
        if (pcfg.num_brawlers, pcfg.num_maps, pcfg.num_modes) == (
                E.num_brawlers(), E.num_maps(), E.num_modes()):
            prev = WinProbNet(pcfg)
            prev.load_state_dict(ck["state_dict"])
            prev.eval()
            with torch.no_grad():
                pb = torch.sigmoid(prev(ta[vai], tb[vai], mp[vai], mo[vai])).numpy()
            baseline = {
                "logloss": float(log_loss(yv, pb, labels=[0, 1])),
                "acc": float(((pb > 0.5) == yv.astype(bool)).mean()),
                "auc": float(roc_auc_score(yv, pb)),
                "ece": float(expected_calibration_error(pb, yv)),
            }
        else:
            print("previous winprob.pt was trained on a different vocabulary — skipping the "
                  "paired baseline")

    # --- embedding model (masked partial-draft training) ---
    cfg = ModelConfig(E.num_brawlers(), E.num_maps(), E.num_modes(), mask_row=E.num_brawlers())
    model = WinProbNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(args.seed)

    # Fixed masked copy of the val split (same p-full mixture as training): early stopping
    # runs on this — it contains full comps, so full-comp regressions still move it — while
    # the headline metrics below stay unmasked and comparable across retrains.
    va_m, vb_m = mask_teams(ds.team_a[val_i], ds.team_b[val_i], cfg.mask_row, args.p_full,
                            np.random.default_rng(args.seed + 1))
    tam_v, tbm_v = torch.from_numpy(va_m), torch.from_numpy(vb_m)

    def batches(n_rows, bs):
        order = torch.randperm(n_rows)
        for k in range(0, n_rows, bs):
            yield order[k:k + bs]

    # Train-slice views, positionally indexed — only the 85% train rows get re-masked and
    # copied each epoch (the val rows have their own fixed masked copy above).
    ta_tr, tb_tr = ds.team_a[tr_i], ds.team_b[tr_i]
    y_tr, wt_tr, mp_tr, mo_tr = y[tri], wt[tri], mp[tri], mo[tri]

    # Early stopping tracks the mixed (masked) loss — the model's actual job — while the
    # full-comp loss is recorded alongside so a full-comp regression is visible per epoch
    # and gate failures (vs the paired baseline) have a remediation lever: raise --p-full.
    history_mix, history_full = [], []
    best_ll, best_state, bad, patience = float("inf"), None, 0, 6
    for _ in range(args.epochs):
        ta_m, tb_m = mask_teams(ta_tr, tb_tr, cfg.mask_row, args.p_full, rng)
        tam, tbm = torch.from_numpy(ta_m), torch.from_numpy(tb_m)
        model.train()
        for bi in batches(len(tr_i), args.batch):
            opt.zero_grad()
            logit = model(tam[bi], tbm[bi], mp_tr[bi], mo_tr[bi])
            loss = (bce(logit, y_tr[bi]) * wt_tr[bi]).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv_mix = torch.sigmoid(model(tam_v, tbm_v, mp[vai], mo[vai])).numpy()
            pv_full = torch.sigmoid(model(ta[vai], tb[vai], mp[vai], mo[vai])).numpy()
        vll = log_loss(yv, pv_mix, labels=[0, 1])
        history_mix.append(vll)
        history_full.append(float(log_loss(yv, pv_full, labels=[0, 1])))
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
        pv_mix = torch.sigmoid(model(tam_v, tbm_v, mp[vai], mo[vai])).numpy()
    m_ll, m_auc = log_loss(yv, pv, labels=[0, 1]), roc_auc_score(yv, pv)
    m_acc = float(((pv > 0.5) == yv.astype(bool)).mean())
    m_ece = expected_calibration_error(pv, yv)
    mix_ll = float(log_loss(yv, pv_mix, labels=[0, 1]))

    print("\n=== validation metrics, full comps (logloss/ECE: lower better; acc/AUC: higher better) ===")
    print(f"{'model':<22}{'logloss':>10}{'acc':>8}{'AUC':>8}{'ECE':>8}")
    print(f"{'always 0.5':<22}{const_ll:>10.4f}{0.5:>8.3f}{'-':>8}{'-':>8}")
    print(f"{'logreg (brawlers)':<22}{lr_ll:>10.4f}{lr_acc:>8.3f}{lr_auc:>8.3f}{'-':>8}")
    if baseline:
        print(f"{'prev checkpoint':<22}{baseline['logloss']:>10.4f}{baseline['acc']:>8.3f}"
              f"{baseline['auc']:>8.3f}{baseline['ece']:>8.3f}")
    print(f"{'embedding net':<22}{m_ll:>10.4f}{m_acc:>8.3f}{m_auc:>8.3f}{m_ece:>8.3f}")
    if baseline:
        delta = m_ll - baseline["logloss"]
        print(f"paired full-comp delta vs prev checkpoint (same val rows): "
              f"logloss {delta:+.4f}  AUC {m_auc - baseline['auc']:+.4f}"
              f"  — if logloss is up materially, raise --p-full and retrain")
        # Hard gate BEFORE any artifact is written: the unattended crawler path
        # (collect.py --retrain-on-shift) trains, exports, and publishes without a human, and
        # this run's winprob.pt becomes the next run's baseline — an advisory print alone
        # would let regressions ratchet in.
        if 0 <= args.max_full_delta < delta:
            raise SystemExit(
                f"full-comp regression gate: paired logloss delta {delta:+.4f} exceeds "
                f"--max-full-delta {args.max_full_delta} — no artifacts written. "
                f"Raise --p-full (more full-comp weight) and retrain.")

    # --- partial-draft states: how much is knowing more of the draft worth? ---
    # Each row masks the whole val split to one fixed (known_ours, known_theirs) state.
    # The 1v0 row is also compared against a shrunk brawler-map winrate marginal built on the
    # train split — the cheapest possible single-pick predictor. If the net loses to it, the
    # mask-in-mean design is washing out the single-pick signal and needs rework.
    wins = np.zeros((E.num_maps(), int(cfg.mask_row) + 1), dtype=np.float64)
    games = np.zeros_like(wins)
    for team, won in ((ds.team_a[tr_i], ds.y[tr_i]), (ds.team_b[tr_i], 1.0 - ds.y[tr_i])):
        for j in range(3):
            np.add.at(games, (ds.map_idx[tr_i], team[:, j]), 1.0)
            np.add.at(wins, (ds.map_idx[tr_i], team[:, j]), won)
    emp_wr = (wins + 5.0) / (games + 10.0)   # shrunk toward 0.5 with 10 pseudo-games

    partial_metrics = {"mixture_logloss": mix_ll, "p_full": args.p_full, "states": {}}
    print("\n=== partial draft states, val (value of knowing more of the draft) ===")
    print(f"{'state':<10}{'logloss':>10}{'AUC':>8}{'ECE':>8}{'mean|p-.5|':>12}")
    for ka, kb in ((1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)):
        srng = np.random.default_rng(args.seed + 100 + 10 * ka + kb)
        sa_np = mask_to_known(ds.team_a[val_i], ka, cfg.mask_row, srng)
        sb_np = mask_to_known(ds.team_b[val_i], kb, cfg.mask_row, srng)
        sa, sb = torch.from_numpy(sa_np), torch.from_numpy(sb_np)
        with torch.no_grad():
            ps = torch.sigmoid(model(sa, sb, mp[vai], mo[vai])).numpy()
        s_ll = float(log_loss(yv, ps, labels=[0, 1]))
        s_auc = float(roc_auc_score(yv, ps))
        s_ece = float(expected_calibration_error(ps, yv))
        spread = float(np.abs(ps - 0.5).mean())
        partial_metrics["states"][f"{ka}v{kb}"] = {
            "logloss": s_ll, "auc": s_auc, "ece": s_ece, "mean_abs_edge": spread,
        }
        print(f"{f'{ka}v{kb}':<10}{s_ll:>10.4f}{s_auc:>8.3f}{s_ece:>8.3f}{spread:>12.3f}")
        if (ka, kb) == (1, 0):
            known_ids = sa_np[sa_np != cfg.mask_row]                     # the one known pick/row
            p_emp = emp_wr[ds.map_idx[val_i], known_ids]
            e_ll = float(log_loss(yv, p_emp, labels=[0, 1]))
            partial_metrics["empirical_1v0_logloss"] = e_ll
            verdict = "net >= empirical marginal, OK" if s_ll <= e_ll else \
                "NET LOSES to the empirical marginal — single-pick signal is being washed out"
            print(f"{'  1v0 emp':<10}{e_ll:>10.4f}{'-':>8}{'-':>8}{'-':>12}   {verdict}")

    # --- save artifacts ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # The exact vocabulary this model was trained against, by embedding row. export_model.py
    # compares these ids to the live reference at export time — identity, not just counts, so
    # a same-size catalog swap between train and export fails loudly instead of silently
    # re-pinning ids onto neighbours' trained rows.
    trained_vocab = {
        "brawler_ids": [int(b) for b, _ in sorted(E.brawler_encoder().items(), key=lambda kv: kv[1])],
        "map_ids": [int(m) for m, _ in sorted(E.map_encoder().items(), key=lambda kv: kv[1])],
        "modes": [s for s, _ in sorted(E.mode_encoder().items(), key=lambda kv: kv[1])],
    }
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(), "vocab": trained_vocab},
               PROCESSED_DIR / "winprob.pt")
    DOCS.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_total": n, "n_val": int(n_val),
        "const": {"logloss": float(const_ll)},
        "logreg": {"logloss": float(lr_ll), "acc": lr_acc, "auc": float(lr_auc)},
        "embedding": {"logloss": float(m_ll), "acc": m_acc, "auc": float(m_auc), "ece": m_ece},
        "embedding_partial": partial_metrics,
        # Previous checkpoint scored on this run's val rows — the only drift-free comparison.
        "baseline_prev_checkpoint": baseline,
    }
    (DOCS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history_mix, marker="o", ms=3, label="masked mixture")
    ax[0].plot(history_full, marker="o", ms=3, label="full comps")
    ax[0].axhline(0.6931, ls="--", c="gray", label="always 0.5")
    ax[0].axhline(lr_ll, ls=":", c="orange", label="logreg (full comps)")
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
