"""Partial-draft model tests: mask-row serving, legacy-artifact guard, torch↔NumPy parity.

The model scores unfinished drafts by padding short teams with a trained "unknown slot"
row (``mask_row`` in the export's config). These tests pin down:

  * serve-side padding + the ``supports_partial`` gate (legacy artifacts reject partials);
  * the antisymmetry corollaries (empty board == 0.5 exactly, P(A,B) + P(B,A) == 1);
  * exact parity between the torch net (training) and the NumPy reimplementation (serving)
    across every (known_a, known_b) draft state — the repo's two-implementation sync rule,
    previously enforced by convention only;
  * scoring.model_marginal dispatch: direct partial call on new artifacts, top-meta team
    completion on legacy ones.

The torch parity test skips cleanly where torch isn't installed (e.g. the serve deploy).

    PYTHONPATH=backend python -m pytest backend/tests/test_partial_draft.py   # or run directly
"""
from __future__ import annotations

import itertools
import json
import tempfile
from pathlib import Path

import numpy as np

from bsdraft.models import serve as S

N_BRAWLER, N_MAP, N_MODE, D, R = 6, 3, 2, 8, 3
MASK = N_BRAWLER
HIDDEN = 10


def _synthetic(tmp: Path, mask_row: bool) -> Path:
    """A minimal winprob.npz; with ``mask_row`` the brawler-indexed matrices carry the
    extra trained "unknown slot" row and the config advertises it."""
    rng = np.random.default_rng(0)
    rows = N_BRAWLER + (1 if mask_row else 0)
    cfg = {"synthetic": True}
    if mask_row:
        cfg["mask_row"] = MASK
    arrays = {
        "_config": np.array(json.dumps(cfg)),
        "brawler.weight": rng.normal(size=(rows, D)).astype(np.float32),
        "map_emb.weight": rng.normal(size=(N_MAP, 2)).astype(np.float32),
        "mode_emb.weight": rng.normal(size=(N_MODE, 1)).astype(np.float32),
        "strength.0.weight": rng.normal(size=(HIDDEN, D + 3)).astype(np.float32),
        "strength.0.bias": rng.normal(size=(HIDDEN,)).astype(np.float32),
        "strength.3.weight": rng.normal(size=(1, HIDDEN)).astype(np.float32),
        "strength.3.bias": rng.normal(size=(1,)).astype(np.float32),
        "counter_p.weight": rng.normal(size=(rows, R)).astype(np.float32),
        "counter_q.weight": rng.normal(size=(rows, R)).astype(np.float32),
    }
    path = tmp / "winprob.npz"
    np.savez(path, **arrays)
    return path


class _Encoders:
    """Identity stand-in for bsdraft.data.encoders: brawler id == embedding row."""
    def encode_brawler(self, bid):
        return bid

    def encode_map(self, mid):
        return 1

    def encode_mode(self, mode):
        return 1


def _load(path: Path):
    """Model with the encoder layer swapped for the identity stand-in; caller restores."""
    S.E, prev = _Encoders(), S.E
    try:
        return S.WinProbModel(path), prev
    except Exception:
        S.E = prev
        raise


def test_supports_partial_detection():
    with tempfile.TemporaryDirectory() as td:
        legacy = S.WinProbModel(_synthetic(Path(td), mask_row=False))
        assert not legacy.supports_partial
    with tempfile.TemporaryDirectory() as td:
        masked = S.WinProbModel(_synthetic(Path(td), mask_row=True))
        assert masked.supports_partial
    # Total on the degraded no-artifact model too (cfg is None there).
    assert not S.WinProbModel(Path("/nonexistent/winprob.npz")).supports_partial


def test_fallback_mean_excludes_mask_row():
    """The OOV fallback row is an "average brawler" — the unknown-slot row must not dilute it."""
    with tempfile.TemporaryDirectory() as td:
        path = _synthetic(Path(td), mask_row=True)
        raw = np.load(path, allow_pickle=False)
        m = S.WinProbModel(path)
        for key in ("brawler.weight", "counter_p.weight", "counter_q.weight"):
            assert np.allclose(m._w[key][-1], raw[key][:MASK].mean(axis=0), atol=1e-6), key


def test_padding_resolves_to_mask_row_with_pinned_vocab():
    """Padding happens in row space, after the id->row lookup: with a pinned vocabulary the
    mask row (which has no brawler id) must still land on exactly cfg["mask_row"], not on
    the OOV mean row."""
    with tempfile.TemporaryDirectory() as td:
        path = _synthetic(Path(td), mask_row=True)
        raw = dict(np.load(path, allow_pickle=False))
        raw["_vocab_brawler_ids"] = np.array([100, 101, 102, 103, 104, 105], dtype=np.int64)
        np.savez(path, **raw)
        m = S.WinProbModel(path)
        assert m._team_rows([100, 101]) == [0, 1, MASK]
        assert m._team_rows([]) == [MASK, MASK, MASK]


def test_mixed_length_batch():
    with tempfile.TemporaryDirectory() as td:
        m, prev = _load(_synthetic(Path(td), mask_row=True))
        try:
            probs = m.prob_batch([[0], [0, 1, 2]], [[3, 4, 5], [3, 4]], 1, "gemGrab")
            assert len(probs) == 2 and all(0.0 <= x <= 1.0 for x in probs)
        finally:
            S.E = prev


def test_partial_teams_scored_and_antisymmetric():
    with tempfile.TemporaryDirectory() as td:
        m, prev = _load(_synthetic(Path(td), mask_row=True))
        try:
            for ka, kb in itertools.product(range(4), range(4)):
                a, b = [0, 1, 2][:ka], [3, 4, 5][:kb]
                p = m.prob(a, b, map_id=1, mode="gemGrab")
                assert 0.0 < p < 1.0, (ka, kb)
                # Swapping sides negates the logit for any partial state.
                assert abs(p + m.prob(b, a, map_id=1, mode="gemGrab") - 1.0) < 1e-6, (ka, kb)
            # The empty board is exactly a coin flip by antisymmetry.
            assert m.prob([], [], map_id=1, mode="gemGrab") == 0.5
            # Explicit padding equivalence: a short team scores as if padded with the mask row.
            direct = m.prob_batch([[0, 1, MASK]], [[3, 4, 5]], 1, "gemGrab")[0]
            assert abs(m.prob([0, 1], [3, 4, 5], map_id=1, mode="gemGrab") - direct) < 1e-9
        finally:
            S.E = prev


def test_legacy_artifact_rejects_partial_teams():
    with tempfile.TemporaryDirectory() as td:
        m, prev = _load(_synthetic(Path(td), mask_row=False))
        try:
            p = m.prob([0, 1, 2], [3, 4, 5], map_id=1, mode="gemGrab")   # full 3v3 still fine
            assert 0.0 <= p <= 1.0   # synthetic weights can saturate the sigmoid
            for bad in ([0, 1], []):
                try:
                    m.prob(bad, [3, 4, 5], map_id=1, mode="gemGrab")
                    raise AssertionError("legacy artifact accepted a partial team")
                except ValueError:
                    pass
        finally:
            S.E = prev


def test_oversized_team_rejected():
    with tempfile.TemporaryDirectory() as td:
        m, prev = _load(_synthetic(Path(td), mask_row=True))
        try:
            try:
                m.prob([0, 1, 2, 3], [3, 4, 5], map_id=1, mode="gemGrab")
                raise AssertionError("accepted a 4-brawler team")
            except ValueError:
                pass
        finally:
            S.E = prev


def test_torch_numpy_parity_across_draft_states():
    """The two model implementations must agree bit-for-bit-ish on every draft state."""
    import pytest
    torch = pytest.importorskip("torch")
    from bsdraft.models.winprob import ModelConfig, WinProbNet

    cfg = ModelConfig(num_brawlers=N_BRAWLER, num_maps=N_MAP, num_modes=N_MODE,
                      d_brawler=D, d_map=2, d_mode=1, d_hidden=HIDDEN, counter_rank=R,
                      mask_row=MASK)
    torch.manual_seed(0)
    net = WinProbNet(cfg)
    net.eval()

    with tempfile.TemporaryDirectory() as td:
        # Mirror scripts/export_model.py: state_dict tensors + config JSON (vocab pinning is
        # exercised by test_serve_vocab; parity targets the math).
        weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        path = Path(td) / "winprob.npz"
        np.savez(path, _config=np.array(json.dumps(cfg.to_dict())), **weights)
        m, prev = _load(path)
        try:
            assert m.supports_partial
            mp, mo = torch.tensor([1]), torch.tensor([1])
            for ka, kb in itertools.product(range(4), range(4)):
                a, b = [0, 1, 2][:ka], [3, 4, 5][:kb]
                ta = torch.tensor([a + [MASK] * (3 - ka)])
                tb = torch.tensor([b + [MASK] * (3 - kb)])
                with torch.no_grad():
                    pt = float(net.win_prob(ta, tb, mp, mo)[0])
                pn = m.prob(a, b, map_id=1, mode="gemGrab")
                assert abs(pt - pn) < 5e-5, (ka, kb, pt, pn)
        finally:
            S.E = prev


def test_model_marginal_dispatch():
    from bsdraft.engine.scoring import model_marginal
    from bsdraft.engine.state import DraftState

    state = DraftState(map_id=1, mode="Brawl Ball", our_team=[10], their_team=[20, 21])

    class _Partial:
        available = True
        supports_partial = True
        calls = []

        def prob(self, a, b, map_id, mode):
            self.calls.append((list(a), list(b)))
            return 0.61

    class _Legacy:
        available = True
        supports_partial = False
        calls = []

        def prob(self, a, b, map_id, mode):
            self.calls.append((list(a), list(b)))
            return 0.57

    class _Stats:
        def top_brawlers(self, map_id, n, min_games):
            class _Rate:  # noqa: N801 - throwaway
                pass
            return [(i, _Rate()) for i in range(30, 30 + n)]

    partial = _Partial()
    assert model_marginal(state, 11, partial, _Stats()) == 0.61
    # The unfinished board goes to the model as-is: our picks + candidate vs their picks.
    assert partial.calls == [([10, 11], [20, 21])]

    legacy = _Legacy()
    assert model_marginal(state, 11, legacy, _Stats()) == 0.57
    (a, b), = legacy.calls
    # Legacy artifacts only judge full 3v3s, so both teams were completed to 3.
    assert len(a) == 3 and len(b) == 3
    assert a[:2] == [10, 11] and b[:2] == [20, 21]

    assert model_marginal(state, 11, None, _Stats()) is None


def test_model_marginal_caps_full_team():
    """The frontend requests recommendations even when our side is already full (enemy's
    last pick pending, or draft done). The candidate can't join a full team — the call must
    cap at TEAM_SIZE instead of sending a 4-brawler team the model rejects."""
    from bsdraft.engine.scoring import model_marginal
    from bsdraft.engine.state import DraftState

    state = DraftState(map_id=1, mode="Brawl Ball",
                       our_team=[10, 11, 12], their_team=[20, 21, 22])

    class _Partial:
        available = True
        supports_partial = True
        calls = []

        def prob(self, a, b, map_id, mode):
            self.calls.append((list(a), list(b)))
            return 0.5

    m = _Partial()
    assert model_marginal(state, 99, m, None) == 0.5
    (a, b), = m.calls
    assert a == [10, 11, 12] and b == [20, 21, 22]   # capped: candidate 99 can't join


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
