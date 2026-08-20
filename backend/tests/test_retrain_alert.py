"""The crawler's consecutive-retrain-failure alarm.

Added after a run of 38 silent `--max-full-delta` gate refusals left the served model 8 days
stale with no signal anywhere but an unread log line.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess

import pytest

from bsdraft.constants import REPO_ROOT


@pytest.fixture
def collect(tmp_path, monkeypatch):
    """scripts/collect.py isn't importable as a package module — load it by path, with its
    streak state redirected into tmp so tests never touch data/processed."""
    spec = importlib.util.spec_from_file_location(
        "collect_under_test", REPO_ROOT / "backend" / "scripts" / "collect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_RETRAIN_STATE_PATH", tmp_path / "retrain_state.json")
    return mod


def test_streak_counts_up_and_a_success_clears_it(collect):
    assert collect._record_retrain(False, "gate") == 1
    assert collect._record_retrain(False, "gate") == 2
    assert collect._record_retrain(True) == 0
    assert collect._record_retrain(False, "gate") == 1


def test_streak_survives_a_process_restart(collect):
    collect._record_retrain(False, "gate")
    collect._record_retrain(False, "gate")
    # The daemon gets kickstarted a lot; a streak that resets on restart would never alert.
    assert json.loads(collect._RETRAIN_STATE_PATH.read_text())["consecutive_failures"] == 2
    assert collect._record_retrain(False, "gate") == 3


def test_unreadable_state_starts_a_fresh_streak_rather_than_raising(collect):
    collect._RETRAIN_STATE_PATH.write_text("{not json")
    assert collect._record_retrain(False, "gate") == 1


def _calls(collect, monkeypatch):
    seen = []

    def fake_gh(*args):
        seen.append(args)
        if args[:2] == ("issue", "list"):
            return subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(collect.publisher, "_gh", fake_gh)
    return seen


def test_no_alert_below_the_threshold(collect, monkeypatch):
    seen = _calls(collect, monkeypatch)
    for streak in range(1, collect._RETRAIN_ALERT_AFTER):
        collect._alert_retrain_stalled(streak, "gate")
    assert seen == []          # a one-off failure is noise, not an incident


def test_alerts_once_on_crossing_then_daily(collect, monkeypatch):
    seen = _calls(collect, monkeypatch)
    collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER, "gate detail")
    assert any(a[:2] == ("issue", "create") for a in seen)
    # The next few cycles stay quiet …
    seen.clear()
    for extra in range(1, collect._RETRAIN_REALERT_EVERY):
        collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER + extra, "gate detail")
    assert not any(a[:2] == ("issue", "create") for a in seen)
    # … then it nags again a day later.
    seen.clear()
    collect._alert_retrain_stalled(
        collect._RETRAIN_ALERT_AFTER + collect._RETRAIN_REALERT_EVERY, "gate detail")
    assert any(a[:2] == ("issue", "create") for a in seen)


def test_an_open_issue_suppresses_a_duplicate(collect, monkeypatch):
    seen = []

    def fake_gh(*args):
        seen.append(args)
        if args[:2] == ("issue", "list"):
            return subprocess.CompletedProcess(args, 0, stdout='[{"number": 7}]', stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(collect.publisher, "_gh", fake_gh)
    collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER, "gate")
    assert not any(a[:2] == ("issue", "create") for a in seen)


def test_alert_failure_never_propagates(collect, monkeypatch):
    monkeypatch.setattr(collect.publisher, "_gh",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("gh missing")))
    collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER, "gate")  # must not raise


def test_alert_body_carries_the_gate_explanation(collect, monkeypatch):
    seen = _calls(collect, monkeypatch)
    collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER,
                                   "full-comp regression gate: paired logloss delta +0.0025")
    create = [a for a in seen if a[:2] == ("issue", "create")][0]
    assert "paired logloss delta +0.0025" in " ".join(create)


def test_missing_label_falls_back_to_an_unlabelled_issue(collect, monkeypatch):
    # `gh` errors on a label the repo doesn't have; losing the alert over a taxonomy detail
    # would defeat the point.
    seen = []

    def fake_gh(*args):
        seen.append(args)
        if args[:2] == ("issue", "list"):
            return subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")
        if "--label" in args:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="unknown label")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(collect.publisher, "_gh", fake_gh)
    collect._alert_retrain_stalled(collect._RETRAIN_ALERT_AFTER, "gate")
    creates = [a for a in seen if a[:2] == ("issue", "create")]
    assert len(creates) == 2 and "--label" not in creates[1]
