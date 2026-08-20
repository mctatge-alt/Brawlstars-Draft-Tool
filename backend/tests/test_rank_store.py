"""The rank-index artifact's load path, which has to fit a 2.5M-entry index into a 512 MB box.

`json.loads` on the whole document retained ~193 MB of Python strings and `from_arrays` peaked
another ~156 MB converting them — ~350 MB transient for a 28 MB result. That OOM-killed the
Render instance on 2026-08-20 every time anyone looked up a rank, taking the whole API with it.
The loader now decodes the tags array in slices; these tests pin that the frugal path is exactly
equivalent to the plain one, including at the chunk seams where it could silently truncate.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from bsdraft.engine import rank_store as RS


def _index(n: int, width=lambda i: 8) -> dict:
    """`{tag: (ts, tier)}` as build_rank_index returns it, with controllable tag widths."""
    return {f"{i:0{width(i)}X}"[-width(i):]: (1000 + i, (i % 22) + 1) for i in range(n)}


def _roundtrip(tmp_path, idx, name="rank_index.json.gz"):
    path = RS.save_rank_index(idx, tmp_path / name)
    return RS.load_rank_index(path)


def _plain_load(path) -> RS.RankIndex:
    """The whole-document parse the frugal path replaced — the reference implementation."""
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw)
    return RS.RankIndex.from_arrays(payload["tags"], payload["tiers"])


def _same(a: RS.RankIndex, b: RS.RankIndex) -> bool:
    width = max(a._tags.dtype.itemsize, b._tags.dtype.itemsize)
    return (np.array_equal(a._tags.astype(f"S{width}"), b._tags.astype(f"S{width}"))
            and np.array_equal(a._tiers, b._tiers))


def test_roundtrip_preserves_every_lookup(tmp_path):
    idx = _index(500)
    loaded = _roundtrip(tmp_path, idx)
    assert len(loaded) == 500
    for tag, (_, tier) in idx.items():
        assert loaded.get(tag) == tier
    assert loaded.get("NOTATAG") is None


def test_frugal_and_plain_paths_agree(tmp_path, monkeypatch):
    # A chunk size well below the payload forces many seams; the plain parse is the oracle.
    monkeypatch.setattr(RS, "_TAG_CHUNK_BYTES", 64)
    path = RS.save_rank_index(_index(2000), tmp_path / "r.json.gz")
    assert _same(RS.load_rank_index(path), _plain_load(path))


def test_seams_do_not_drop_or_duplicate_entries(tmp_path, monkeypatch):
    # The cut lands on the b'","' separator; an off-by-one there would eat a tag's quote and
    # silently shift every tier after it against the wrong tag.
    monkeypatch.setattr(RS, "_TAG_CHUNK_BYTES", 16)
    idx = _index(777)
    loaded = _roundtrip(tmp_path, idx)
    assert len(loaded) == len(idx)
    for tag, (_, tier) in idx.items():
        assert loaded.get(tag) == tier, f"{tag} came back wrong across a chunk seam"


def test_varying_tag_widths_are_not_truncated(tmp_path, monkeypatch):
    # Blocks get different fixed 'S' widths; concatenating without widening first would clip the
    # long tags down to the first block's itemsize.
    monkeypatch.setattr(RS, "_TAG_CHUNK_BYTES", 32)
    idx = {}
    for i in range(300):
        tag = ("T" * (2 + i % 12)) + f"{i:04d}"      # 6..17 chars
        idx[tag] = (1, (i % 22) + 1)
    loaded = _roundtrip(tmp_path, idx)
    for tag, (_, tier) in idx.items():
        assert loaded.get(tag) == tier


def test_single_chunk_path(tmp_path, monkeypatch):
    monkeypatch.setattr(RS, "_TAG_CHUNK_BYTES", 1 << 20)   # everything in one bite
    idx = _index(50)
    loaded = _roundtrip(tmp_path, idx)
    assert len(loaded) == 50 and loaded.get(next(iter(idx))) is not None


def test_empty_index(tmp_path):
    loaded = _roundtrip(tmp_path, {})
    assert len(loaded) == 0 and loaded.get("ANY") is None


def test_uncompressed_artifact_still_loads(tmp_path):
    idx = _index(40)
    loaded = _roundtrip(tmp_path, idx, name="rank_index.json")
    assert len(loaded) == 40


def test_version_mismatch_raises_so_the_caller_rebuilds(tmp_path):
    path = tmp_path / "r.json.gz"
    payload = {"version": RS.FORMAT_VERSION + 1, "tags": ["AA"], "tiers": [3]}
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="unsupported rank index format version"):
        RS.load_rank_index(path)


def test_unexpected_layout_falls_back_instead_of_guessing(tmp_path):
    # Keys reordered and spaced — the offset scan won't find its markers. The loader must fall
    # back to the plain parse rather than return a half-decoded index.
    path = tmp_path / "r.json.gz"
    payload = {"tiers": [5, 9], "tags": ["AAA", "BBB"], "version": RS.FORMAT_VERSION}
    path.write_bytes(gzip.compress(json.dumps(payload, indent=2).encode()))
    loaded = RS.load_rank_index(path)
    assert loaded.get("AAA") == 5 and loaded.get("BBB") == 9


def test_mismatched_array_lengths_fall_back(tmp_path, monkeypatch):
    # A truncated tiers array must not produce an index whose tags and tiers are misaligned.
    path = tmp_path / "r.json.gz"
    payload = {"version": RS.FORMAT_VERSION, "tags": ["AAA", "BBB"], "tiers": [5]}
    path.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode()))
    assert RS._load_frugally(gzip.decompress(path.read_bytes())) is None


def test_lookup_is_exact_not_prefix(tmp_path):
    # Binary search over fixed-width byte strings could match a prefix if compared loosely.
    idx = {"ABC": (1, 4), "ABCDEF": (1, 9)}
    loaded = _roundtrip(tmp_path, idx)
    assert loaded.get("ABC") == 4 and loaded.get("ABCDEF") == 9
    assert loaded.get("ABCD") is None and loaded.get("AB") is None
