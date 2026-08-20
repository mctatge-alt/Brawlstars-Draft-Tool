"""The gate experiment's control logic: artifact guarding and the verdict it reaches.

The training runs themselves are far too slow to test; what's worth pinning down is (a) that a
trial can never leave the served checkpoint replaced — restoring between trials is the
experiment's control, not housekeeping — and (b) that each of the three explanations produces its
own verdict, since each implies a different fix.
"""
from __future__ import annotations

import importlib.util

import pytest

from bsdraft.constants import REPO_ROOT


@pytest.fixture
def gx(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "gate_experiment_under_test", REPO_ROOT / "backend" / "scripts" / "gate_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a, b = tmp_path / "winprob.pt", tmp_path / "metrics.json"
    monkeypatch.setattr(mod, "GUARDED", (a, b))
    return mod


def _trial(p_full, seed, delta, mixture=0.66):
    return {"p_full": p_full, "seed": seed, "delta_logloss": delta, "delta_auc": -0.001,
            "seconds": 1.0, "n_total": 1_000_000, "n_val": 150_000,
            "logloss": 0.66, "baseline_logloss": 0.66 - delta,
            "mixture_logloss": mixture, "partial_1v0_logloss": 0.68}


# --------------------------------------------------------------------------- artifact guarding

def test_restore_puts_the_incumbent_back_byte_for_byte(gx, tmp_path):
    incumbent, metrics = gx.GUARDED
    incumbent.write_bytes(b"the served model")
    metrics.write_text('{"a": 1}')
    safe = tmp_path / "safe"; safe.mkdir()
    gx.snapshot(safe)

    incumbent.write_bytes(b"a trial's output")   # what a gate-disabled trial does
    metrics.write_text('{"a": 2}')
    gx.restore(safe)

    assert incumbent.read_bytes() == b"the served model"
    assert metrics.read_text() == '{"a": 1}'


def test_restore_deletes_a_file_the_trial_invented(gx, tmp_path):
    incumbent, metrics = gx.GUARDED
    incumbent.write_bytes(b"served")
    safe = tmp_path / "safe"; safe.mkdir()
    gx.snapshot(safe)              # metrics.json did not exist at snapshot time
    metrics.write_text("{}")       # …a trial creates it
    gx.restore(safe)
    # Leaving it would hand the next trial a baseline that was never the incumbent's.
    assert not metrics.exists() and incumbent.read_bytes() == b"served"


def test_snapshot_tolerates_a_cold_repo(gx, tmp_path):
    safe = tmp_path / "safe"; safe.mkdir()
    assert gx.snapshot(safe) == []
    gx.restore(safe)               # must not raise


def test_count_matches_counts_rows(gx, tmp_path):
    f = tmp_path / "matches.jsonl"
    f.write_text('{"a":1}\n{"a":2}\n{"a":3}\n')
    assert gx.count_matches(f) == 3
    assert gx.count_matches(tmp_path / "absent.jsonl") == 0


# --------------------------------------------------------------------------- verdicts

def test_fresh_data_clearing_the_gate_blames_the_frozen_dataset(gx):
    trials = [_trial(0.7, s, d) for s, d in zip((0, 1, 2), (-0.0011, -0.0004, -0.0009))]
    a = gx.analyze(trials)
    assert a["verdict"] == "frozen-dataset"
    assert a["detail"]["mean_delta_at_production_p_full"] < 0


def test_wide_seed_spread_outranks_a_mean_that_looks_fine(gx):
    # These deltas average +0.0009 — comfortably inside the gate — while swinging 0.0055 across
    # seeds. Production retrains on one seed, so "the mean passes" would be a lie: it fails about
    # half the cycles. The noise-floor read has to outrank the frozen-dataset read for that reason.
    trials = [_trial(0.7, s, d) for s, d in zip((0, 1, 2), (0.0030, -0.0025, 0.0021))]
    a = gx.analyze(trials)
    assert a["detail"]["mean_delta_at_production_p_full"] <= gx.PRODUCTION_GATE   # looks fine …
    assert a["detail"]["seed_noise_floor_at_production_p_full"] >= gx.PRODUCTION_GATE
    assert a["verdict"] == "below-noise-floor"                                    # … but isn't


def test_a_higher_p_full_that_clears_the_gate_is_recommended(gx):
    trials = ([_trial(0.7, s, d) for s, d in zip((0, 1, 2), (0.0026, 0.0028, 0.0025))]
              + [_trial(0.9, s, d) for s, d in zip((0, 1, 2), (0.0005, 0.0002, 0.0008))])
    a = gx.analyze(trials)
    assert a["verdict"] == "raise-p-full"
    assert "0.9" in a["reason"]
    # The recommendation must warn about what more full-comp weight costs.
    assert "partial" in a["reason"]


def test_persistent_regression_everywhere_is_called_a_ratchet(gx):
    trials = ([_trial(0.7, s, d) for s, d in zip((0, 1, 2), (0.0026, 0.0028, 0.0025))]
              + [_trial(0.9, s, d) for s, d in zip((0, 1, 2), (0.0031, 0.0029, 0.0033))])
    a = gx.analyze(trials)
    assert a["verdict"] == "ratchet"
    assert "escape hatch" in a["reason"]


def test_all_trials_failing_is_inconclusive_not_a_verdict(gx):
    a = gx.analyze([{"p_full": 0.7, "seed": 0, "error": "boom"}])
    assert a["verdict"] == "inconclusive"


def test_a_single_seed_cannot_claim_a_noise_floor(gx):
    # One run per setting measures nothing about variance; the read must not pretend otherwise.
    a = gx.analyze([_trial(0.7, 0, 0.0026)])
    assert a["detail"]["seed_noise_floor_at_production_p_full"] is None
    assert a["verdict"] != "below-noise-floor"


def test_a_shortened_run_refuses_to_pass_itself_off_as_an_answer(gx):
    # --epochs is a smoke-test lever; its delta is a different fit's delta. The output otherwise
    # looks identical to a real run, so the verdict itself has to carry the caveat.
    trials = [_trial(0.7, s, d) for s, d in zip((0, 1, 2), (0.0026, 0.0028, 0.0025))]
    a = gx.analyze(trials, comparable=False)
    assert a["comparable"] is False
    assert a["verdict"].startswith("not-comparable")
    assert "NOT COMPARABLE" in a["reason"]
    assert gx.analyze(trials)["comparable"] is True
