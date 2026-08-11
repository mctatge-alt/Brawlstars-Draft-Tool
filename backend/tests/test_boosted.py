"""Unit tests for the Ranked boosted-brawler scraper (bsdraft.collect.boosted) and the serving
loader (bsdraft.data.reference.load_ranked_boosted).

Offline — parses a committed HTML fixture trimmed from the *real* release-notes "Ranked"
subsection, plus synthetic ``__NEXT_DATA__`` pages that exercise the layout variations the parser
must survive (nested-list rotation, flat/concatenated rotation, season ordering, layout drift).

    PYTHONPATH=backend python -m pytest backend/tests/test_boosted.py    # or run directly
"""
from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from bsdraft.collect import boosted as B
from bsdraft.data import reference as R

FIX = Path(__file__).resolve().parent / "fixtures" / "patchnotes"
JUNE_URL = "https://supercell.com/en/games/brawlstars/blog/release-notes/release-notes-june-2026/"


def _fixture(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


# --- minimal __NEXT_DATA__ builders ---------------------------------------------

def _text(value, *marks):
    return {"nodeType": "text", "value": value, "marks": [{"type": m} for m in marks], "data": {}}

def _p(value, *marks):
    return {"nodeType": "paragraph", "data": {}, "content": [_text(value, *marks)]}

def _h3(value):
    return {"nodeType": "heading-3", "data": {}, "content": [_text(value, "bold")]}

def _li(*content):
    return {"nodeType": "list-item", "data": {}, "content": list(content)}

def _ul(*items):
    return {"nodeType": "unordered-list", "data": {}, "content": list(items)}

def _leaf_ul(*names):
    """A rotation's nested list — one list-item per brawler name."""
    return _ul(*[_li(_p(n)) for n in names])

def _rt(*content):
    return {"json": {"nodeType": "document", "data": {}, "content": list(content)}}

def _mk_html(blocks, title="Release Notes X", publish_date="2026-08-04T00:00:00.000+03:00"):
    nd = {"props": {"pageProps": {"title": title, "publishDate": publish_date,
          "bodyCollection": [{"__typename": "TextBlock", "title": t, "text": rt}
                             for t, rt in blocks]}}}
    return ('<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(nd, ensure_ascii=False) + "</script>")


def _season(label, featured, names, nested=True):
    """The nodes for one season group: a bold "Season N" paragraph then its rotation list."""
    if nested:
        rot = _li(_p("Free Brawler Rotation:"), _leaf_ul(*names))
    else:  # flat layout: names concatenated onto the label line, no nested list
        rot = _li(_p("Free Brawler Rotation:" + "".join(names)))
    feat = _li(_p(f"Featured game mode: {featured}"))
    return [_p(label, "bold"), _ul(feat, rot)]


def _ranked_block(*season_groups, title="Maps, Game Modes, Environments & Rotation Changes"):
    content = [_h3("Ranked")]
    for g in season_groups:
        content.extend(g)
    return (title, _rt(*content))


# --- real fixture ----------------------------------------------------------------

def _real() -> B.BoostedReport:
    return B.parse_boosted(_fixture("release_notes_ranked_boosted.html"), JUNE_URL)


def test_real_fixture_parses_both_rotations():
    r = _real()
    assert [(x.season, x.brawlers) for x in r.rotations] == [
        ("Season 1", ["Berry", "Tara", "Meg"]),
        ("Season 2", ["Trunk", "Willow", "Kaze"]),
    ]
    assert not r.layout_warning and not r.unresolved


def test_real_fixture_active_is_current_season():
    r = _real()
    assert r.active.season == "Season 1"
    assert r.active.brawlers == ["Berry", "Tara", "Meg"]
    assert r.active.featured_mode == "Gem Grab"
    assert [u.season for u in r.upcoming] == ["Season 2"]


def test_real_fixture_all_names_resolve():
    for rot in _real().rotations:
        assert None not in rot.brawler_ids, rot.season


# --- synthetic layout variations -------------------------------------------------

def test_nested_list_layout():
    html = _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]))])
    r = B.parse_boosted(html, JUNE_URL)
    assert r.active.brawlers == ["Berry", "Tara", "Meg"]


def test_flat_concatenated_layout_uses_segmentation():
    # The looser layout where names run together on the label line ("...Rotation:BerryTaraMeg").
    html = _mk_html([_ranked_block(
        _season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"], nested=False))])
    r = B.parse_boosted(html, JUNE_URL)
    assert r.active.brawlers == ["Berry", "Tara", "Meg"]
    assert not r.layout_warning


def test_segment_names_splits_delimiter_free_run():
    assert B._segment_names("BerryTaraMeg") == ["Berry", "Tara", "Meg"]
    # longest-first: "Larry & Lawrie" wins over "Larry", and "Colette" isn't shadowed by "Cole"
    assert B._segment_names("Larry & LawrieColette") == ["Larry & Lawrie", "Colette"]


def test_active_is_lowest_numbered_even_if_listed_out_of_order():
    html = _mk_html([_ranked_block(
        _season("Season 2", "Brawl Ball", ["Trunk", "Willow", "Kaze"]),
        _season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]))])
    r = B.parse_boosted(html, JUNE_URL)
    assert r.active.season == "Season 1"


def test_ranked_subsection_found_inside_any_block():
    # "Ranked" is a heading-3 nested in a larger section, not a top-level bodyCollection block.
    html = _mk_html([
        ("Bug Fixes", _rt(_p("fixed things"))),
        _ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"])),
    ])
    r = B.parse_boosted(html, JUNE_URL)
    assert r.active.brawlers == ["Berry", "Tara", "Meg"]


def test_ranked_subsection_ends_at_next_heading():
    # A season group after a *different* heading must not be read as a ranked rotation.
    block = ("Maps", _rt(
        _h3("Ranked"), *_season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]),
        _h3("Maps"), *_season("Season 9", "Heist", ["Shelly", "Colt", "Nita"]),
    ))
    r = B.parse_boosted(_mk_html([block]), JUNE_URL)
    assert [x.season for x in r.rotations] == ["Season 1"]


# --- layout warning --------------------------------------------------------------

def test_layout_warning_when_ranked_present_but_no_rotation():
    html = _mk_html([("Maps", _rt(_h3("Ranked"), _p("General ranked prose, no rotation")))])
    r = B.parse_boosted(html, JUNE_URL)
    assert r.layout_warning and not r.rotations
    assert "layout may have changed" in r.note


def test_no_layout_warning_when_ranked_section_absent():
    html = _mk_html([("Bug Fixes", _rt(_p("fixed things")))])
    r = B.parse_boosted(html, JUNE_URL)
    assert not r.layout_warning and not r.rotations


def test_unresolved_name_is_reported():
    html = _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Zzzznotreal"]))])
    r = B.parse_boosted(html, JUNE_URL)
    assert "Zzzznotreal" in r.unresolved


# --- fingerprint / change detection ---------------------------------------------

def test_fingerprint_stable_and_content_sensitive():
    a = B.parse_boosted(_mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry"]))]), JUNE_URL)
    b = B.parse_boosted(_mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry"]))]), JUNE_URL)
    c = B.parse_boosted(_mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Tara"]))]), JUNE_URL)
    assert a.fingerprint == b.fingerprint and len(a.fingerprint) == 16
    assert a.fingerprint != c.fingerprint


@contextmanager
def _committed_at(doc):
    """Point both the scraper and the reference loader at a temp ranked_boosted.json."""
    old_b, old_r = B.BOOSTED_PATH, R.RANKED_BOOSTED_PATH
    R.load_ranked_boosted.cache_clear()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ranked_boosted.json"
        if doc is not None:
            p.write_text(json.dumps(doc), encoding="utf-8")
        B.BOOSTED_PATH = R.RANKED_BOOSTED_PATH = p
        try:
            yield p
        finally:
            B.BOOSTED_PATH, R.RANKED_BOOSTED_PATH = old_b, old_r
            R.load_ranked_boosted.cache_clear()


def test_has_changed_detects_new_rotation():
    report = B.parse_boosted(
        _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]))]), JUNE_URL)
    with _committed_at({"active": {"season": "Season 1", "brawlers": ["Berry", "Tara", "Meg"]},
                        "upcoming": []}):
        assert not B.has_changed(report)
    with _committed_at({"active": {"season": "Season 1", "brawlers": ["Shelly", "Colt", "Nita"]},
                        "upcoming": []}):
        assert B.has_changed(report)
    with _committed_at(None):
        assert B.has_changed(report)  # no committed file yet


def test_write_document_roundtrips_and_refuses_unresolved():
    good = B.parse_boosted(
        _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Tara", "Meg"]))]), JUNE_URL)
    with _committed_at(None) as p:
        B.write_document(good)
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert doc["active"]["brawlers"] == ["Berry", "Tara", "Meg"]
        assert doc["valid_until"] is None
    bad = B.parse_boosted(
        _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Zzzznotreal"]))]), JUNE_URL)
    with _committed_at(None):
        try:
            B.write_document(bad)
            raise AssertionError("expected refusal to write an unmatched name")
        except ValueError:
            pass


# --- PR rendering (inherits patchnotes' _cell/_safe_url hardening) ---------------

def test_render_pr_has_table_marker_and_confirm_warning():
    r = _real()
    title, body = B.render_pr(r)
    assert "Season 1" in title and "Berry" in title
    assert "| Free brawlers |" in body                       # rotation table
    assert "Confirm which season is live" in body            # human-review nudge
    assert f"<!-- {B.FINGERPRINT_MARKER}:{r.fingerprint} -->" in body


def test_render_pr_flags_unresolved_names():
    r = B.parse_boosted(
        _mk_html([_ranked_block(_season("Season 1", "Gem Grab", ["Berry", "Zzzznotreal"]))]), JUNE_URL)
    _title, body = B.render_pr(r)
    assert "Unmatched name" in body and "Zzzznotreal" in body


# --- serving loader (reference.load_ranked_boosted) -----------------------------

def test_reference_loads_active_ids():
    with _committed_at({"active": {"season": "S", "brawlers": ["Berry", "Tara", "Meg"]}}):
        ids = R.load_ranked_boosted()
    assert R.brawler_by_name("Berry").id in ids and len(ids) == 3


def test_reference_missing_file_returns_empty():
    with _committed_at(None):
        assert R.load_ranked_boosted() == ()


def test_reference_expired_valid_until_returns_empty():
    with _committed_at({"valid_until": "2000-01-01",
                        "active": {"brawlers": ["Berry", "Tara", "Meg"]}}):
        assert R.load_ranked_boosted() == ()


def test_reference_future_valid_until_is_active():
    with _committed_at({"valid_until": "2999-12-31", "active": {"brawlers": ["Berry"]}}):
        assert len(R.load_ranked_boosted()) == 1


def test_reference_skips_unknown_names():
    with _committed_at({"active": {"brawlers": ["Berry", "Zzzznotreal"]}}):
        assert len(R.load_ranked_boosted()) == 1


def test_committed_file_is_valid():
    # The checked-in data/reference/ranked_boosted.json must load without error to a tuple of ids.
    ids = R.load_ranked_boosted()
    assert isinstance(ids, tuple) and all(isinstance(i, int) for i in ids)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
