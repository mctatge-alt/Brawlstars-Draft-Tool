"""Decide what to do about the full-comp regression gate, once real data is flowing again.

Background: `collect.py --retrain-on-shift` calls `train.py` with no arguments, so the gate runs
at `--max-full-delta 0.002`. Through 2026-08-18..20 it refused **38 consecutive** retrains with a
paired full-comp logloss delta of +0.0022..+0.0031, freezing the served model at 2026-08-12. The
confound: for most of that window the crawler was IP-blocked and `matches.jsonl` never grew, so
every retrain was refitting essentially the same data and the deltas may be pure run-to-run
variance rather than a real regression.

This script separates the three explanations, which call for three different fixes:

  1. **It was the frozen dataset.** With fresh matches the delta goes negative on its own.
     Fix: nothing — the gate is working.
  2. **The threshold sits below the training procedure's own noise floor.** Deltas swing by more
     than 0.002 across seeds alone, so no fixed threshold can pass reliably.
     Fix: best-of-N candidates, or widen the threshold to the measured floor.
  3. **A real ratchet.** The incumbent was a lucky draw and every honest retrain is a hair worse,
     so the gate has locked the model in permanently.
     Fix: a staleness escape hatch — accept a marginally worse full-comp model once the incumbent
     is N days old, since a stale model beats a slightly-noisier fresh one.

Usage:

    # wait for the crawl to recover, then run the default grid (this is the intended entry point)
    PYTHONPATH=backend python backend/scripts/gate_experiment.py --wait-for-data

    # data is already flowing — go now
    PYTHONPATH=backend python backend/scripts/gate_experiment.py

    # quick smoke pass (NOT comparable to production: fewer epochs changes the fit)
    PYTHONPATH=backend python backend/scripts/gate_experiment.py --epochs 8 --seeds 0,1

Every trial runs `train.py` with the gate **disabled** (`--max-full-delta -1`) so the run always
completes and always writes `docs/metrics.json` — we want the delta for the failures too, and a
threshold we can sweep offline. Because that also means every trial writes artifacts, the script
snapshots `winprob.pt` / `winprob.npz` / `docs/metrics.json` / `docs/training.png` up front and
restores them **before every trial and again on exit** (including Ctrl-C). Restoring between
trials is not tidiness — it is the experiment's control: `train.py` scores the incumbent from
`winprob.pt`, so without a restore trial 2 would be measured against trial 1's output and the
comparison would drift.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bsdraft.constants import PROCESSED_DIR, RAW_DIR, REPO_ROOT

DOCS = REPO_ROOT / "docs"
MATCHES_PATH = RAW_DIR / "matches.jsonl"
METRICS_PATH = DOCS / "metrics.json"
REPORT_PATH = PROCESSED_DIR / "gate_experiment.json"

# Everything train.py writes. All of it is restored between trials.
GUARDED = (
    PROCESSED_DIR / "winprob.pt",
    PROCESSED_DIR / "winprob.npz",
    METRICS_PATH,
    DOCS / "training.png",
)

PRODUCTION_P_FULL = 0.7        # what collect.py's argument-free train.py call actually uses
PRODUCTION_GATE = 0.002        # …and the threshold it is judged against


# --------------------------------------------------------------------------- data precondition

def count_matches(path: Path = MATCHES_PATH) -> int:
    """Rows in matches.jsonl. Cheaper than json-parsing and the only number we need — during an
    IP-lock stall this is flat while the crawler still logs '+5000 new matches' every cycle."""
    if not path.exists():
        return 0
    n = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(8 << 20):
            n += chunk.count(b"\n")
    return n


def wait_for_data(min_new: int, poll_seconds: int, max_wait_hours: float) -> dict:
    """Block until the crawl has added ``min_new`` rows since we started watching.

    The stall this script exists to work around is total — zero rows written — so *any* sustained
    growth means the key is authorized again. We still wait for a batch rather than the first row,
    because the point is to retrain on data the incumbent has not already seen.
    """
    start_count = count_matches()
    deadline = time.monotonic() + max_wait_hours * 3600
    target = start_count + min_new
    print(f"waiting for data: {start_count:,} rows now, need {target:,} "
          f"(+{min_new:,}); polling every {poll_seconds}s, giving up after {max_wait_hours}h")
    while True:
        now = count_matches()
        if now >= target:
            print(f"data is flowing again: {now:,} rows (+{now - start_count:,}) — starting trials")
            return {"start_rows": start_count, "rows_at_start_of_trials": now}
        if time.monotonic() > deadline:
            raise SystemExit(
                f"gave up after {max_wait_hours}h: {now:,} rows (+{now - start_count:,} of "
                f"{min_new:,} needed). The crawl is probably still IP-blocked — check "
                f"`curl -s https://api.ipify.org` against the key's allow-list, and look for "
                f"'0 new from rankings' in data/raw/crawl.out.log.")
        time.sleep(poll_seconds)


# --------------------------------------------------------------------------- artifact guarding

def snapshot(into: Path) -> list:
    saved = []
    for p in GUARDED:
        if p.exists():
            shutil.copy2(p, into / p.name)
            saved.append(p.name)
    return saved


def restore(frm: Path) -> None:
    """Put the incumbent back exactly as it was. Also removes a file a trial *created* that
    wasn't there before, so a trial can't leave a fake baseline behind."""
    for p in GUARDED:
        src = frm / p.name
        if src.exists():
            shutil.copy2(src, p)
        elif p.exists():
            p.unlink()


# --------------------------------------------------------------------------- one trial

def run_trial(p_full: float, seed: int, epochs: int | None, timeout: int) -> dict:
    """One train.py run with the gate off. Returns the parsed metrics, or an `error` entry."""
    cmd = [sys.executable, str(REPO_ROOT / "backend" / "scripts" / "train.py"),
           "--p-full", str(p_full), "--seed", str(seed),
           "--max-full-delta", "-1"]          # gate off: measure the delta, don't enforce it
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "backend")}
    t0 = time.monotonic()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - t0
    if res.returncode != 0:
        return {"p_full": p_full, "seed": seed, "error": (res.stderr or "").strip()[-800:],
                "seconds": round(elapsed, 1)}
    try:
        m = json.loads(METRICS_PATH.read_text())
    except Exception as e:  # noqa: BLE001
        return {"p_full": p_full, "seed": seed, "error": f"metrics.json unreadable: {e}",
                "seconds": round(elapsed, 1)}
    base = m.get("baseline_prev_checkpoint")
    if not base:
        # train.py skips the paired baseline when the checkpoint's vocabulary no longer matches
        # the reference. Without it there is no delta to measure and the whole run is moot.
        return {"p_full": p_full, "seed": seed, "seconds": round(elapsed, 1),
                "error": "no paired baseline — winprob.pt vocabulary differs from the reference; "
                         "refresh_reference + a clean retrain is needed before this experiment "
                         "means anything"}
    emb, partial = m["embedding"], m.get("embedding_partial", {})
    return {
        "p_full": p_full, "seed": seed, "seconds": round(elapsed, 1),
        "n_total": m.get("n_total"), "n_val": m.get("n_val"),
        "logloss": emb["logloss"], "baseline_logloss": base["logloss"],
        "delta_logloss": emb["logloss"] - base["logloss"],
        "delta_auc": emb["auc"] - base["auc"],
        # Raising --p-full trades partial-draft quality for full-comp parity, which is the exact
        # thing the masked-training design bought. Carry it so a "fix" that guts partial drafts
        # is visible instead of looking like a win.
        "mixture_logloss": partial.get("mixture_logloss"),
        "partial_1v0_logloss": (partial.get("states", {}).get("1v0") or {}).get("logloss"),
    }


# --------------------------------------------------------------------------- analysis

def analyze(trials: list, gate: float = PRODUCTION_GATE, comparable: bool = True) -> dict:
    ok = [t for t in trials if "error" not in t]
    if not ok:
        return {"verdict": "inconclusive", "reason": "every trial failed", "detail": {}}

    at_prod = [t for t in ok if abs(t["p_full"] - PRODUCTION_P_FULL) < 1e-9]
    by_p = {}
    for t in ok:
        by_p.setdefault(t["p_full"], []).append(t["delta_logloss"])
    per_p = {p: {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v),
                 "spread": max(v) - min(v)}
             for p, v in sorted(by_p.items())}

    prod_deltas = [t["delta_logloss"] for t in at_prod]
    prod_mean = sum(prod_deltas) / len(prod_deltas) if prod_deltas else None
    noise_floor = (max(prod_deltas) - min(prod_deltas)) if len(prod_deltas) > 1 else None
    passing = [t for t in ok if t["delta_logloss"] <= gate]

    # Order matters, and the noise floor comes first on purpose: production retrains on a single
    # seed, so a mean that sits inside the gate while the spread swamps it still fails about half
    # the time. Variance that large is the finding, whatever the mean says. Everything below then
    # judges on the *worst* trial rather than the mean, for the same reason — one unlucky draw is
    # what the unattended crawler will actually get.
    if noise_floor is not None and noise_floor >= gate:
        verdict, reason = "below-noise-floor", (
            f"Seeds alone move the delta by {noise_floor:.4f} at fixed --p-full, which is "
            f"{'more than' if noise_floor > gate else 'as much as'} the {gate} gate. A fixed "
            f"threshold this tight cannot pass reliably no matter how good the data is — and "
            f"production retrains on one seed, so it is a coin flip every cycle. Train N "
            f"candidates and keep the best, or widen --max-full-delta to the measured floor.")
    elif prod_deltas and max(prod_deltas) <= gate:
        verdict, reason = "frozen-dataset", (
            f"At the production --p-full {PRODUCTION_P_FULL}, every trial landed inside the {gate} "
            f"gate (worst {max(prod_deltas):+.4f}, mean {prod_mean:+.4f}). The 38 failures were an "
            f"artifact of retraining on a dataset the IP-lock had frozen. No gate change needed — "
            f"just keep the crawl alive.")
    else:
        better = [p for p, s in per_p.items()
                  if p > PRODUCTION_P_FULL and s["max"] <= gate]
        if better:
            p = min(better)
            verdict, reason = "raise-p-full", (
                f"--p-full {p} brings every trial inside the gate (worst {per_p[p]['max']:+.4f}, "
                f"mean {per_p[p]['mean']:+.4f}) — the remediation train.py's own message suggests. "
                f"Check mixture_logloss at that setting before adopting it: more full-comp weight "
                f"costs partial-draft quality, and partial drafts are what the board actually "
                f"scores.")
        else:
            verdict, reason = "ratchet", (
                f"Fresh data and every --p-full tried still leave the delta above the gate "
                f"(best worst-case {min(s['max'] for s in per_p.values()):+.4f}). The incumbent looks "
                f"like a lucky draw that nothing reproduces, so the gate has locked the model in. "
                f"Add a staleness escape hatch: past N days, accept a small regression rather than "
                f"serve a frozen model.")

    if not comparable:
        # A shortened run is a different fit, so its delta says nothing about the production
        # gate. Keep the numbers — they still prove the harness works — but refuse to let the
        # verdict read as an answer, because it looks exactly like one.
        reason = ("TRIALS ARE NOT COMPARABLE TO PRODUCTION (--epochs was overridden), so this "
                  "verdict is not an answer about the gate — rerun without --epochs before "
                  "acting on it. For reference, the same logic on these numbers would say: "
                  + reason)
        verdict = f"not-comparable ({verdict})"

    return {
        "verdict": verdict, "reason": reason, "comparable": comparable,
        "detail": {
            "gate": gate,
            "trials_ok": len(ok), "trials_failed": len(trials) - len(ok),
            "production_p_full": PRODUCTION_P_FULL,
            "mean_delta_at_production_p_full": prod_mean,
            "seed_noise_floor_at_production_p_full": noise_floor,
            "per_p_full": per_p,
            "n_passing_gate": len(passing),
        },
    }


def print_report(trials: list, analysis: dict) -> None:
    print("\n=== gate experiment: paired full-comp delta vs the incumbent checkpoint ===")
    print(f"{'p_full':>7}{'seed':>6}{'delta LL':>11}{'delta AUC':>11}"
          f"{'mixture LL':>12}{'1v0 LL':>10}{'secs':>8}")
    for t in trials:
        if "error" in t:
            print(f"{t['p_full']:>7}{t['seed']:>6}{'FAILED':>11}   {t['error'][:60]}")
            continue
        mix = f"{t['mixture_logloss']:.4f}" if t.get("mixture_logloss") else "-"
        p1 = f"{t['partial_1v0_logloss']:.4f}" if t.get("partial_1v0_logloss") else "-"
        print(f"{t['p_full']:>7}{t['seed']:>6}{t['delta_logloss']:>+11.4f}"
              f"{t['delta_auc']:>+11.4f}{mix:>12}{p1:>10}{t['seconds']:>8.0f}")
    if not analysis.get("comparable", True):
        print("\n" + "!" * 78)
        print("!! --epochs was overridden: these trials do not reproduce the production fit, ")
        print("!! so the delta below is not the delta the gate would see. Smoke test only.")
        print("!" * 78)
    d = analysis["detail"]
    if d.get("seed_noise_floor_at_production_p_full") is not None:
        print(f"\nseed-to-seed spread at --p-full {PRODUCTION_P_FULL}: "
              f"{d['seed_noise_floor_at_production_p_full']:.4f}  (gate is {d['gate']})")
    print(f"\nVERDICT: {analysis['verdict']}\n{analysis['reason']}")


# --------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait-for-data", action="store_true",
                    help="block until the crawl adds --min-new-matches rows before running")
    ap.add_argument("--min-new-matches", type=int, default=25_000,
                    help="rows the crawl must add before trials start (~5 cycles at --target 5000)")
    ap.add_argument("--poll-seconds", type=int, default=600)
    ap.add_argument("--max-wait-hours", type=float, default=24.0)
    ap.add_argument("--p-full", default=f"{PRODUCTION_P_FULL},0.8,0.9",
                    help="comma-separated --p-full values to sweep; must include the production "
                         "value for the noise-floor and frozen-dataset reads to work")
    ap.add_argument("--seeds", default="0,1,2",
                    help="comma-separated seeds. Production always uses 0, so >1 seed is what "
                         "measures the run-to-run noise the gate is being compared against")
    ap.add_argument("--epochs", type=int, default=None,
                    help="passed through to train.py; omit to use its default (40). A shorter "
                         "run is NOT comparable to the production fit — smoke tests only")
    ap.add_argument("--trial-timeout", type=int, default=7200)
    args = ap.parse_args()

    p_fulls = [float(x) for x in args.p_full.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not p_fulls or not seeds:
        raise SystemExit("need at least one --p-full and one --seed")

    data_note = wait_for_data(args.min_new_matches, args.poll_seconds,
                              args.max_wait_hours) if args.wait_for_data else \
        {"rows_at_start_of_trials": count_matches()}

    print(f"\n{len(p_fulls) * len(seeds)} trials: p_full={p_fulls} x seeds={seeds}")
    print("the incumbent winprob.pt is snapshotted and restored before every trial — "
          "nothing here can replace the served model")

    trials = []
    with tempfile.TemporaryDirectory(prefix="bsdraft-gate-") as tmp:
        safe = Path(tmp)
        saved = snapshot(safe)
        print(f"snapshotted: {', '.join(saved) or '(nothing to guard yet)'}\n")
        try:
            for p_full in p_fulls:
                for seed in seeds:
                    restore(safe)   # every trial is scored against the SAME incumbent
                    print(f"--- trial p_full={p_full} seed={seed} …")
                    t = run_trial(p_full, seed, args.epochs, args.trial_timeout)
                    trials.append(t)
                    if "error" in t:
                        print(f"    FAILED: {t['error'][:200]}")
                    else:
                        print(f"    delta logloss {t['delta_logloss']:+.4f} "
                              f"({t['seconds']:.0f}s, n={t['n_total']:,})")
        finally:
            restore(safe)
            print("\nincumbent artifacts restored")

    analysis = analyze(trials, comparable=args.epochs is None)
    print_report(trials, analysis)
    REPORT_PATH.write_text(json.dumps(
        {"data": data_note, "trials": trials, "analysis": analysis}, indent=2))
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
