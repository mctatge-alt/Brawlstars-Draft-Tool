"""Unit tests for the release-notes scraper (bsdraft.collect.patchnotes).

Offline — parses committed HTML fixtures trimmed from two *real* release-notes pages that
between them exercise the cross-month structural variation the parser has to survive:

  * June 2026  — "Buffs"/"Nerfs"/"Changes" heading dividers, Title-case names, 2 new brawlers.
  * February 2026 — no dividers; direction is a per-name suffix ("MORTIS - Minor Nerf"),
    ALL-CAPS names ("JAE YONG", "LARRY & LAWRIE"), emoji + trailing-space section titles.

Plus discovery over a trimmed blog-index fixture and the off-site-link guard.

    PYTHONPATH=backend python -m pytest backend/tests/test_patchnotes.py    # or run directly
"""
from __future__ import annotations

import json
from pathlib import Path

from bsdraft.collect import patchnotes as P

FIX = Path(__file__).resolve().parent / "fixtures" / "patchnotes"
JUNE_URL = "https://supercell.com/en/games/brawlstars/blog/release-notes/release-notes-june-2026/"
FEB_URL = "https://supercell.com/en/games/brawlstars/blog/release-notes/release-notes-february-2026/"


def _fixture(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


# --- minimal __NEXT_DATA__ builders for synthetic edge-case pages ----------------

def _text(value, *marks):
    return {"nodeType": "text", "value": value, "marks": [{"type": m} for m in marks], "data": {}}

def _p(value, *marks):
    return {"nodeType": "paragraph", "data": {}, "content": [_text(value, *marks)]}

def _h3(value):
    return {"nodeType": "heading-3", "data": {}, "content": [_text(value, "bold")]}

def _ul(*items):
    return {"nodeType": "unordered-list", "data": {}, "content": [
        {"nodeType": "list-item", "data": {}, "content": [_p(it)]} for it in items]}

def _rt(*content):
    return {"json": {"nodeType": "document", "data": {}, "content": list(content)}}

def _mk_html(title, publish_date, blocks):
    """blocks: list of (section_title, rich_text_doc)."""
    nd = {"props": {"pageProps": {"title": title, "publishDate": publish_date,
          "bodyCollection": [{"__typename": "TextBlock", "title": t, "text": rt}
                             for t, rt in blocks]}}}
    return ('<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(nd, ensure_ascii=False) + "</script>")


def _june() -> P.PatchReport:
    return P.parse_release_notes(_fixture("release_notes_june_2026.html"), JUNE_URL)


def _feb() -> P.PatchReport:
    return P.parse_release_notes(_fixture("release_notes_february_2026.html"), FEB_URL)


# --- unit helpers ---------------------------------------------------------------

def test_split_name_suffix():
    assert P._split_name_suffix("MORTIS - Minor Nerf") == ("MORTIS", "nerf")
    assert P._split_name_suffix("ZIGGY - Major Buff") == ("ZIGGY", "buff")
    assert P._split_name_suffix("KIT - Change") == ("KIT", "change")
    assert P._split_name_suffix("Piper") == ("Piper", None)          # no suffix
    assert P._split_name_suffix("Jae-Yong")[0] == "Jae-Yong"          # hyphen w/o spaces kept
    assert P._split_name_suffix("LARRY & LAWRIE - Minor Buff") == ("LARRY & LAWRIE", "buff")


def test_norm_name_matches_catalog():
    # The punctuation/casing that varies month to month must still resolve to one brawler.
    idx = P._name_index()
    for printed in ("JAE YONG", "Jae-Yong", "LARRY & LAWRIE", "8-BIT", "Mr. P", "R-T"):
        assert P._resolve_id(printed, idx) is not None, printed


# --- June: heading-divider scheme ----------------------------------------------

def test_june_new_brawlers():
    assert _june().new_brawlers == ["Nori", "Wendy"]


def test_june_buffs_and_nerfs():
    r = _june()
    buffs = {c.name for c in r.buffs}
    nerfs = {c.name for c in r.nerfs}
    assert "Piper" in buffs
    assert {"Mortis", "Crow"} <= nerfs
    # every buff/nerf carries its bullet details
    assert all(c.details for c in r.buffs), "buffs should have change bullets"


def test_june_all_names_resolve_to_ids():
    assert [c.name for c in _june().all_changes if c.brawler_id is None] == []


def test_no_balance_entries_are_dropped():
    # Regression guard: Contentful often splits a paragraph so the FIRST text run is a bare "\n"
    # with no marks and the styled name follows. Reading only that first run silently dropped 31
    # of 52 entries on this real page (Mandy, Gale, Larry & Lawrie, Starr Nova, …).
    r = _june()
    assert len(r.all_changes) == 52
    names = {c.name for c in r.all_changes}
    for missed in ("Mandy", "Gale", "Larry & Lawrie", "Starr Nova", "Meeple", "Ruffs"):
        assert missed in names, missed
    assert len(_feb().all_changes) == 32


def test_marks_of_skips_blank_leading_run():
    node = {"nodeType": "paragraph", "content": [
        {"nodeType": "text", "value": "\n", "marks": []},
        {"nodeType": "text", "value": "Mandy", "marks": [{"type": "bold"}, {"type": "underline"}]},
    ]}
    assert {"bold", "underline"} <= P._marks_of(node)


def test_grey_alias_resolves_to_gray():
    # The notes print "Grey"; the catalog spells it "Gray".
    assert P._resolve_id("Grey", P._name_index()) == P._resolve_id("Gray", P._name_index())
    assert P._resolve_id("Grey", P._name_index()) is not None


def test_retitled_sections_still_route():
    html = _mk_html("Release Notes X", "2026-08-04", [
        ("Balance Changes & Adjustments", _rt(_h3("Nerfs"), _p("Mortis", "bold", "underline"),
                                              _ul("Damage reduced"))),
        ("NEW Brawlers and Skins", _rt(_h3("Kaze - Legendary - Assassin"))),
    ])
    r = P.parse_release_notes(html, JUNE_URL)
    assert [c.name for c in r.nerfs] == ["Mortis"]
    assert r.new_brawlers == ["Kaze"]


def test_june_maintenance_sections_captured():
    assert _june().maintenance_sections == ["Maintenance - August 4", "Maintenance - July 8"]


# --- February: per-name-suffix scheme, ALL CAPS, messy titles -------------------

def test_feb_suffix_direction_and_allcaps_resolution():
    r = _feb()
    by_name = {c.name: c for c in r.all_changes}
    assert by_name["MORTIS"].direction == "nerf"
    assert by_name["ZIGGY"].direction == "buff"
    assert by_name["KIT"].direction == "change"
    # ALL-CAPS + ampersand names still resolve against the Title-case catalog
    assert by_name["JAE YONG"].brawler_id is not None
    assert by_name["LARRY & LAWRIE"].brawler_id is not None


def test_feb_section_titles_stripped():
    # "Maintenance - March 19 " on the page (trailing space) must be normalized.
    assert "Maintenance - March 19" in _feb().maintenance_sections
    assert all(s == s.strip() for s in _feb().maintenance_sections)


# --- fingerprint / dedup --------------------------------------------------------

def test_fingerprint_is_stable_and_content_sensitive():
    a, b = _june().fingerprint, _june().fingerprint
    assert a == b and len(a) == 16          # deterministic, fixed width
    assert _june().fingerprint != _feb().fingerprint  # different patches -> different hash


def test_fingerprint_changes_when_a_section_changes():
    # Simulate an in-place update: same slug, an extra balance line -> a new fingerprint,
    # so the watcher re-alerts on a living-doc edit rather than staying silent.
    base = P._fingerprint("release-notes-june-2026", ["Balance Changes\nPiper buffed"])
    added = P._fingerprint("release-notes-june-2026", ["Balance Changes\nPiper buffed",
                                                       "Maintenance - Sept 1\nCrow nerfed"])
    assert base != added


# --- issue rendering ------------------------------------------------------------

def test_render_new_brawler_issue_has_rollout_and_marker():
    title, body = P.render_issue(_june())
    assert title.startswith("New brawler(s) + balance patch: Nori, Wendy")
    assert "refresh_reference.py" in body            # rollout steps for the new brawlers
    assert "| Brawler | Changes |" in body           # buff/nerf tables
    assert f"<!-- {P.FINGERPRINT_MARKER}:{_june().fingerprint} -->" in body  # dedup marker


def test_render_cell_sanitizes_table_breakers_and_marker_spoof():
    # A pipe would break the table; a comment delimiter could spoof the fingerprint marker.
    evil = P.BalanceChange(name="Evil|Name", brawler_id=None, direction="buff",
                           details=["hp 100|200\nsplit", "<!-- bs-patch:deadbeef -->"])
    r = P.PatchReport(slug="s", url="https://supercell.com/x", title="T", publish_date="2026-08-04",
                      fingerprint="abc123", buffs=[evil])
    _title, body = P.render_issue(r)
    table_lines = [ln for ln in body.splitlines() if ln.startswith("| Evil")]
    assert len(table_lines) == 1                      # no newline leaked a second row
    assert "Evil\\|Name" in table_lines[0]            # pipe escaped
    # Comment delimiters stripped from the cell, so no *valid* marker is forged; only the real
    # trailing marker uses "<!--", which is what the dedup grep keys on.
    assert "<!-- bs-patch:deadbeef -->" not in body
    assert body.count("<!--") == 1
    assert f"<!-- {P.FINGERPRINT_MARKER}:abc123 -->" in body


def test_render_title_variants():
    plain = P.PatchReport(slug="s", url="u", title="Release Notes X", publish_date="2026-08-04",
                          fingerprint="f", other_changes=[P.BalanceChange("Bo", 1, "change", ["x"])])
    assert P.render_issue(plain)[0].startswith("Balance patch:")
    empty = P.PatchReport(slug="s", url="u", title="Release Notes X", publish_date="2026-08-04",
                          fingerprint="f")
    assert P.render_issue(empty)[0].startswith("Release notes updated:")


# --- robustness fixes: emoji titles, layout warning, url guard ------------------

def test_canon_normalizes_emoji_and_punctuation():
    assert P._canon("⚖️ Balance Changes") == "balance changes"
    assert P._canon("🆕 NEW Brawlers") == "new brawlers"
    assert P._canon("Maintenance - March 19 ").startswith("maintenance")


def test_emoji_decorated_sections_still_route():
    # A future month could decorate the two content-bearing section titles with emoji; strict
    # equality would silently drop the whole patch (Supercell does emoji-decorate titles).
    html = _mk_html("Release Notes X", "2026-08-04T00:00:00.000+03:00", [
        ("⚖️ Balance Changes", _rt(_h3("Buffs"), _p("Piper", "bold", "underline"),
                                   _ul("Health increased from 2500 to 2800"))),
        ("🆕 NEW Brawlers", _rt(_h3("Kaze - Legendary - Assassin"))),
    ])
    r = P.parse_release_notes(html, JUNE_URL)
    assert [c.name for c in r.buffs] == ["Piper"]
    assert r.new_brawlers == ["Kaze"]
    assert not r.layout_warning


def test_layout_warning_when_recognized_section_present_but_empty():
    # "Balance Changes" exists but the inner rich text has no brawler entries -> likely a layout
    # change. This must be flagged (and the CI job escalates), not silently treated as balance-free.
    html = _mk_html("Release Notes X", "2026-08-04", [
        ("Balance Changes", _rt(_p("General prose, no brawler entries in a shape we recognize")))])
    r = P.parse_release_notes(html, JUNE_URL)
    assert r.layout_warning
    assert not r.all_changes and not r.new_brawlers
    assert "layout may have changed" in r.note


def test_no_layout_warning_when_section_simply_absent():
    # A genuinely balance-free update (no Balance Changes / NEW Brawlers block) is NOT a warning.
    html = _mk_html("Release Notes X", "2026-08-04", [
        ("Bug Fixes", _rt(_p("fixed things"))),
        ("Maintenance - Aug 5", _rt(_p("a hotfix")))])
    r = P.parse_release_notes(html, JUNE_URL)
    assert not r.layout_warning
    assert r.maintenance_sections == ["Maintenance - Aug 5"]


def test_safe_url_drops_query_and_encodes_path():
    # The discovery guard checks host but not path/query, and urljoin doesn't encode — so a
    # crafted URL could smuggle a forged marker / Markdown breakout into the link target.
    q = "https://supercell.com/en/games/brawlstars/blog/release-notes/release-notes-june-2026?x=<!-- bs-patch:deadbeef -->"
    safe = P._safe_url(q)
    assert "?" not in safe and "<!--" not in safe and "deadbeef" not in safe
    p = "https://supercell.com/en/games/brawlstars/blog/release-notes/x <!-- bs-patch:deadbeef -->)"
    safe2 = P._safe_url(p)
    assert "<!--" not in safe2 and ")" not in safe2 and " " not in safe2
    assert P._safe_url("not-an-absolute-url") == ""


def test_render_url_injection_is_neutralized():
    r = P.PatchReport(
        slug="s", title="T", publish_date="2026-08-04", fingerprint="realfp",
        url="https://supercell.com/en/games/brawlstars/blog/release-notes/x?q=<!-- bs-patch:deadbeef -->",
        buffs=[P.BalanceChange("Bo", 1, "buff", ["x"])])
    _t, body = P.render_issue(r)
    assert body.count("<!--") == 1                    # only the real trailing marker
    assert "bs-patch:deadbeef" not in body


def test_render_layout_warning_title():
    r = P.PatchReport(slug="s", url="https://supercell.com/x", title="Release Notes X",
                      publish_date="2026-08-04", fingerprint="f", layout_warning=True)
    assert P.render_issue(r)[0].startswith("⚠️ Release-notes layout")


def test_fetch_refuses_offsite_before_requesting():
    # _host_ok gate fires before any network call, so these raise without touching the network.
    assert P._host_ok("https://supercell.com/x") and P._host_ok("https://www.supercell.com/x")
    assert not P._host_ok("https://evil.com/x")
    assert not P._host_ok("http://169.254.169.254/latest/meta-data/")
    for bad in ("http://169.254.169.254/latest/meta-data/", "https://evil.com/release-notes/x"):
        try:
            P._fetch(bad)
            raise AssertionError(f"expected refusal for {bad}")
        except ValueError:
            pass


# --- buffies / hypercharges (content the catalog cannot see) --------------------

def test_find_brawler_scans_text_longest_first():
    idx = P._name_index()
    # "Bowling Bolt" is the hypercharge's name; the brawler is Bolt.
    assert P._find_brawler("BOWLING BOLT:\nBolt destroys all walls", idx)[1] is not None
    # Longest-match + punctuation-safe boundaries.
    assert P._find_brawler("Larry & Lawrie get a buff", idx)[0] == "Larry & Lawrie"
    assert P._find_brawler("8-Bit teleports", idx)[0] == "8-Bit"
    assert P._find_brawler("Bo places mines", idx)[0] == "Bo"   # not shadowed by "Bolt"
    assert P._find_brawler("no brawler named here", idx) == ("", None)


def test_buffies_heading_layout_is_itemized():
    html = _mk_html("Release Notes X", "2026-08-04", [
        ("NEW Buffies", _rt(
            _p("More Brawlers are joining the Buffies family!", "italic"),
            _h3("Rico"), _ul("Gadget 1: Multiball Launcher (Rework)"),
            _h3("\n8-Bit"), _ul("Gadget 1: Cheat Cartridge"),
        ))])
    r = P.parse_release_notes(html, JUNE_URL)
    got = [(c.name, c.kind) for c in r.content_changes]
    assert got == [("Rico", "buffie"), ("8-Bit", "buffie")]   # leading \n stripped
    assert all(c.brawler_id is not None for c in r.content_changes)
    assert r.content_changes[0].details == ["Gadget 1: Multiball Launcher (Rework)"]


def test_hypercharge_headingless_layout_uses_name_scan():
    html = _mk_html("Release Notes X", "2026-08-04", [
        ("NEW Hypercharges", _rt(_ul(
            "STARR NOVA: THE HYPERCHARGE \nStarr Nova becomes invulnerable while dashing",
            "BOWLING BOLT:\nBolt destroys all walls while his Hyper Super is active.",
        )))])
    r = P.parse_release_notes(html, JUNE_URL)
    kinds = {c.kind for c in r.content_changes}
    assert kinds == {"hypercharge"}
    ids = [c.brawler_id for c in r.content_changes]
    assert len(ids) == 2 and all(i is not None for i in ids)   # Bolt not dropped


def test_content_changes_alone_count_as_reportable():
    # A buffie-only update must still alert: it has no Balance Changes / NEW Brawlers section.
    html = _mk_html("Release Notes X", "2026-08-04", [
        ("NEW Buffies", _rt(_h3("Rico"), _ul("Gadget rework")))])
    r = P.parse_release_notes(html, JUNE_URL)
    assert r.content_changes and not r.all_changes and not r.new_brawlers
    assert not r.layout_warning
    _title, body = P.render_issue(r)
    assert "Rico" in body and "buffie" in body


def test_prose_headings_do_not_become_content_changes():
    # A list under a PROSE heading must not fall through to the name scan — otherwise the prose
    # ("...unlike Shelly...") mints a bogus entry for whichever brawler it happens to mention.
    html = _mk_html("Release Notes X", "2026-08-04", [
        ("NEW Buffies", _rt(_h3("How Buffies Work"), _ul("Buffies work unlike Shelly's gadget"),
                            _h3("Rico"), _ul("Gadget rework")))])
    r = P.parse_release_notes(html, JUNE_URL)
    assert [c.name for c in r.content_changes] == ["Rico"]


def test_content_changes_dedupe_per_brawler():
    html = _mk_html("Release Notes X", "2026-08-04", [
        ("NEW Buffies", _rt(_h3("Rico"), _ul("first"), _h3("Rico"), _ul("second")))])
    r = P.parse_release_notes(html, JUNE_URL)
    assert len(r.content_changes) == 1
    assert r.content_changes[0].details == ["first", "second"]


def test_new_brawler_info_carries_rarity_and_class():
    info = {b.name: b for b in _june().new_brawler_info}
    assert (info["Nori"].rarity, info["Nori"].cls) == ("Legendary", "Assassin")
    assert (info["Wendy"].rarity, info["Wendy"].cls) == ("Mythic", "Support")


def test_canon_class_rejects_junk():
    assert P._canon_class("Assassin") == "Assassin"
    assert P._canon_class("damage dealer") == "Damage Dealer"
    assert P._canon_class("Legendary") == ""     # a rarity is not a class
    assert P._canon_class("") == ""


# --- discovery + link guard -----------------------------------------------------

def test_discovery_picks_newest_release_notes():
    found = P.find_latest_release_notes(fetch=lambda _url: _fixture("blog_index.html"))
    assert found is not None
    url, publish_date, title = found
    assert url.endswith("/release-notes/release-notes-june-2026")
    assert title == "Release Notes June 2026"


def test_offsite_link_guard():
    ok = "/en/games/brawlstars/blog/release-notes/x"
    assert P._is_release_notes_link(ok)
    assert P._is_release_notes_link("https://supercell.com" + ok)
    for bad in ("https://evil.com/blog/release-notes/x",
                "https://supercell.com.evil.com/blog/release-notes/x",
                "https://evilsupercell.com/blog/release-notes/x",
                "//evil.com/blog/release-notes/x",
                "/blog/esports/not-release-notes"):
        assert not P._is_release_notes_link(bad), bad


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
