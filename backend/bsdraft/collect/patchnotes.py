"""Scrape the official Brawl Stars release notes for balance changes — the *leading* signal.

The drift detector (:mod:`bsdraft.engine.drift`) infers "the meta moved" from our own match
data, which necessarily *lags* a patch by days: it needs enough ranked games in the recent
window to clear the z-test. This module reads Supercell's official release-notes blog directly,
so a balance change — and especially a **new brawler**, which is invisible to the model until
``refresh_reference.py`` runs — is caught the moment it's announced, before the data confirms it.

How the blog is structured (learned by inspection — there's no patch/version API):

  * Release notes are **living monthly documents**: one page per major update, titled e.g.
    "Release Notes June 2026", *updated in place* as hotfixes land (each adds a
    ``Maintenance - <date>`` section and bumps ``publishDate``). A genuinely new major update
    starts a NEW page. Slugs are unguessable — the title month lags the date and months are
    skipped (April's notes carry a mid-May date; there is no separate May page) — so the latest
    page must be *discovered* via the blog index, never constructed.
  * Every page embeds its full content as JSON in a ``<script id="__NEXT_DATA__">`` blob
    (server-rendered — a plain GET suffices, no JS, no key, no IP lock). ``pageProps`` carries
    ``title``, ``publishDate`` and a ``bodyCollection`` of ``TextBlock`` sections keyed by a
    semantic ``title`` ("Balance Changes", "NEW Brawlers", "Maintenance - <date>", …). Each
    ``text`` is Contentful Rich Text (``{"json": {"nodeType": "document", "content": [...]}}``).
  * Inside "Balance Changes", a brawler is a **bold+underline paragraph** — the one reliable
    anchor across months. Direction is encoded two ways we both handle: ``heading-3`` "Buffs" /
    "Nerfs" / "Changes" dividers (Jun/Apr), or a per-name suffix ("MORTIS - Minor Nerf", Feb).
  * "NEW Brawlers" lists each as a ``heading-3`` "Name - Rarity - Class".

Because a page updates in place, dedup can't key on the slug alone: :attr:`PatchReport.fingerprint`
hashes the balance-relevant *content*, so a new maintenance section or new balance entry on the
same page is a fresh event, while a re-run over unchanged notes is not.

Pure ``httpx`` + stdlib (no torch/pandas) — safe to import on any tier.

    PYTHONPATH=backend python -m bsdraft.collect.patchnotes            # human summary, latest notes
    PYTHONPATH=backend python -m bsdraft.collect.patchnotes --json     # machine-readable (CI)
    PYTHONPATH=backend python -m bsdraft.collect.patchnotes --url <release-notes URL>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlsplit

import httpx

from bsdraft.constants import BRAWLER_CLASSES
from bsdraft.data import reference as R

BASE = "https://supercell.com"
BLOG_INDEX_URL = f"{BASE}/en/games/brawlstars/blog/"
# Only ever follow links into this section, on this host — never an off-page linkUrl.
_RELEASE_NOTES_SEG = "/blog/release-notes/"
_ALLOWED_HOST = "supercell.com"
_UA = "bsdraft-patchnotes/1.0 (+https://github.com/; Brawl Stars draft tool)"

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)


# --------------------------------------------------------------------------- data

@dataclass
class BalanceChange:
    name: str                     # brawler name exactly as printed on the page
    brawler_id: Optional[int]     # resolved against the reference catalog, or None if unmatched
    direction: str                # "buff" | "nerf" | "change" | "adjusted"
    details: List[str] = field(default_factory=list)  # the bullet lines under the name


@dataclass
class ContentChange:
    """A per-brawler change to something the *catalog cannot see*: buffies, hypercharges, gears.
    Verified: no catalog endpoint exposes these (``/v1/{gadgets,starpowers,gears}`` are 404 and
    brawler records carry no such field), so the release notes are the only automated source.
    They still move a brawler's real strength, which is why they belong in the alert."""
    kind: str                     # "buffie" | "hypercharge" | "gear"
    name: str                     # brawler name as printed
    brawler_id: Optional[int]
    details: List[str] = field(default_factory=list)


@dataclass
class NewBrawler:
    """A "NEW Brawlers" entry, printed as "Name - Rarity - Class" (e.g. "Nori - Legendary -
    Assassin"). The class matters well beyond cosmetics: the keyless catalog API tags brand-new
    brawlers as class "Unknown", so the release notes are the only automated source for the
    class that ``data/class_overrides.json`` needs — and that class drives the composition /
    game-plan reasoning (:mod:`bsdraft.engine.composition`)."""
    name: str
    rarity: str = ""
    cls: str = ""


@dataclass
class PatchReport:
    slug: str
    url: str
    title: str
    publish_date: str
    fingerprint: str              # hash of balance-relevant content (dedup key; see module doc)
    new_brawlers: List[str] = field(default_factory=list)          # names (back-compat view)
    new_brawler_info: List[NewBrawler] = field(default_factory=list)  # + rarity/class when printed
    buffs: List[BalanceChange] = field(default_factory=list)
    nerfs: List[BalanceChange] = field(default_factory=list)
    other_changes: List[BalanceChange] = field(default_factory=list)
    content_changes: List[ContentChange] = field(default_factory=list)
    maintenance_sections: List[str] = field(default_factory=list)
    note: str = ""
    # True when a recognized section ("Balance Changes"/"NEW Brawlers") was *present* on the page
    # but parsed to nothing — a likely layout change, distinct from a genuinely balance-free
    # update (where the section is simply absent). The alerting job escalates on this.
    layout_warning: bool = False

    @property
    def all_changes(self) -> List[BalanceChange]:
        return self.buffs + self.nerfs + self.other_changes

    @property
    def brawlers_touched(self) -> List[str]:
        """Distinct brawler names appearing in any balance bucket, in first-seen order."""
        seen: Dict[str, None] = {}
        for c in self.all_changes:
            seen.setdefault(c.name, None)
        return list(seen)

    def summary(self) -> str:
        lines = [
            f"{self.title}  ({self.publish_date})",
            f"  {self.url}",
            f"  fingerprint {self.fingerprint}",
        ]
        if self.new_brawlers:
            lines.append(f"  NEW brawler(s): {', '.join(self.new_brawlers)}")
        for label, bucket in (("buffs", self.buffs), ("nerfs", self.nerfs),
                              ("other", self.other_changes)):
            if bucket:
                lines.append(f"  {label} ({len(bucket)}): "
                             + ", ".join(c.name for c in bucket))
        if self.content_changes:
            lines.append("  content: " + ", ".join(
                f"{c.name} ({c.kind})" for c in self.content_changes))
        if self.maintenance_sections:
            lines.append(f"  maintenance sections: {', '.join(self.maintenance_sections)}")
        if not (self.new_brawlers or self.all_changes or self.content_changes):
            lines.append("  (no balance changes or new brawlers parsed)")
        if self.layout_warning:
            lines.append("  ⚠ LAYOUT WARNING — a recognized section parsed empty (see note)")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- fetch

def _host_ok(url: str) -> bool:
    """True iff ``url``'s host is supercell.com (or a subdomain). Strips any port/credentials."""
    host = (urlsplit(url).hostname or "").lower()
    return host == _ALLOWED_HOST or host.endswith("." + _ALLOWED_HOST)


def _fetch(url: str, timeout: float = 30.0, max_redirects: int = 5) -> str:
    """GET ``url`` as text, following the site's trailing-slash redirects **manually** so the
    on-host guard is re-checked *before every hop* — ``follow_redirects=True`` would let a 3xx
    (an open redirect, or an injected off-site link) walk the request off supercell.com to an
    internal address, defeating the guard :func:`_is_release_notes_link` advertises. Only ever
    fetches supercell.com."""
    with httpx.Client(follow_redirects=False, timeout=timeout,
                      headers={"User-Agent": _UA}) as client:
        for _ in range(max_redirects + 1):
            if not _host_ok(url):
                raise ValueError(f"refusing to fetch off-site URL: {url!r}")
            resp = client.get(url)
            if resp.is_redirect:
                url = urljoin(url, resp.headers.get("location", ""))
                continue
            resp.raise_for_status()
            return resp.text
    raise ValueError(f"too many redirects starting from {url!r}")


def _next_data(html: str) -> dict:
    m = _NEXT_DATA_RE.search(html)
    if not m:
        raise ValueError("no __NEXT_DATA__ blob — page layout changed or not a blog page?")
    return json.loads(m.group(1))


def _is_release_notes_link(link: str) -> bool:
    """A same-site path (or absolute supercell.com URL) inside the release-notes section.
    Guards discovery against following an off-site ``linkUrl`` injected into the JSON — any
    host component (including a protocol-relative ``//evil.com/…``) must be supercell.com."""
    if not isinstance(link, str) or not link:
        return False
    parts = urlsplit(link)
    host = (parts.hostname or "").lower()  # None for a bare path; strips any port/creds
    if host and host != _ALLOWED_HOST and not host.endswith("." + _ALLOWED_HOST):
        return False
    return _RELEASE_NOTES_SEG in parts.path


def _slug_of(url: str) -> str:
    return urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- rich text

def _text_of(node: dict) -> str:
    """Concatenate all descendant text values of a rich-text node (plain text)."""
    if not isinstance(node, dict):
        return ""
    if node.get("nodeType") == "text":
        return node.get("value", "")
    return "".join(_text_of(c) for c in node.get("content", []) or [])


def _marks_of(node: dict) -> set:
    """Marks on a paragraph's first **non-blank** text run (bold/underline/italic).

    Contentful frequently splits a paragraph so the first run is a bare "\\n" or " " carrying no
    marks, with the styled name in the next run. Reading the literally-first run therefore missed
    31 of 52 brawler entries on the real June 2026 notes (Mandy, Gale, Larry & Lawrie, Starr
    Nova, …) — they were silently absent from the alert."""
    for c in node.get("content", []) or []:
        if isinstance(c, dict) and c.get("nodeType") == "text" and (c.get("value") or "").strip():
            return {m.get("type") for m in c.get("marks", []) or []}
    return set()


def _list_items(node: dict) -> List[str]:
    """Plain-text of each top-level ``list-item`` in an unordered/ordered list."""
    out = []
    for li in node.get("content", []) or []:
        if isinstance(li, dict) and li.get("nodeType") == "list-item":
            txt = _text_of(li).strip()
            if txt:
                out.append(txt)
    return out


_DIRECTION_KEYS = (
    ("nerf", "nerf"),
    ("buff", "buff"),
    ("rework", "change"),
    ("change", "change"),
    ("adjust", "change"),
    ("remov", "change"),
)


def _direction(text: str) -> Optional[str]:
    """Map a heading ("Nerfs") or name suffix ("Minor Nerf", "Change") to a bucket, or None."""
    t = (text or "").lower()
    for key, bucket in _DIRECTION_KEYS:
        if key in t:
            return bucket
    return None


def _split_name_suffix(text: str) -> Tuple[str, Optional[str]]:
    """"MORTIS - Minor Nerf" -> ("MORTIS", "nerf"); "Piper" -> ("Piper", None). Splits only on
    " - " (space-hyphen-space), so hyphenated names like "Jae-Yong" stay intact."""
    parts = re.split(r"\s+-\s+", text.strip(), maxsplit=1)
    name = parts[0].strip()
    suffix_dir = _direction(parts[1]) if len(parts) > 1 else None
    return name, suffix_dir


# --------------------------------------------------------------------------- name resolution

# Names the release notes spell differently from the catalog. Kept explicit (rather than fuzzy
# matching) so a mismatch is an auditable decision: "Grey" in the June 2026 notes is the brawler
# the catalog calls "Gray" — confirmed by the change text ("Gadget - Grand Piano", Gray's gadget).
_NAME_ALIASES = {"grey": "gray"}


def _norm_name(name: str) -> str:
    """Canonical key for matching printed names to the catalog, tolerant of case and the
    punctuation/spacing that varies month to month: "JAE YONG"/"Jae-Yong" -> "JAEYONG",
    "LARRY & LAWRIE" -> "LARRYLAWRIE", "8-BIT" -> "8BIT", "Mr. P" -> "MRP"."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return _NAME_ALIASES.get(key, key)


def _name_index() -> Dict[str, R.Brawler]:
    return {_norm_name(b.name): b for b in R.load_brawlers()}


_NAME_RE: Optional["re.Pattern"] = None


def _name_pattern() -> "re.Pattern":
    """Alternation over every catalog brawler name, longest first so "Larry & Lawrie" wins over
    "Larry" and "Bolt" isn't shadowed by "Bo". Bounded by non-alphanumerics rather than \\b so
    names like "8-Bit", "R-T" and "Mr. P" match cleanly."""
    global _NAME_RE
    if _NAME_RE is None:
        names = sorted((b.name for b in R.load_brawlers()), key=len, reverse=True)
        _NAME_RE = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(n) for n in names)
                              + r")(?![A-Za-z0-9])", re.I)
    return _NAME_RE


def _find_brawler(text: str, index: Dict[str, R.Brawler]) -> Tuple[str, Optional[int]]:
    """First catalog brawler name mentioned anywhere in ``text``. Needed because a heading-less
    entry may lead with the *feature's* name rather than the brawler's — e.g. the hypercharge
    item "BOWLING BOLT:\\nBolt destroys all walls…" belongs to Bolt, not to a "Bowling Bolt"."""
    m = _name_pattern().search(text or "")
    if not m:
        return "", None
    return m.group(1), _resolve_id(m.group(1), index)


def _resolve_id(name: str, index: Dict[str, R.Brawler]) -> Optional[int]:
    b = index.get(_norm_name(name))
    return b.id if b else None


# --------------------------------------------------------------------------- parse

def _parse_balance_block(rich: dict, index: Dict[str, R.Brawler]) -> List[BalanceChange]:
    """Walk the top-level rich-text sequence of the "Balance Changes" block. A recognized
    ``heading-3`` sets the running direction (Jun/Apr scheme); a bold+underline paragraph is a
    brawler whose name may itself carry a direction suffix (Feb scheme, which wins); the
    unordered-list(s) that follow are that brawler's change bullets."""
    doc = rich.get("json", rich) if isinstance(rich, dict) else {}
    nodes = doc.get("content", []) or []
    changes: List[BalanceChange] = []
    section_dir: Optional[str] = None
    current: Optional[BalanceChange] = None
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nt = n.get("nodeType")
        if nt in ("heading-2", "heading-3", "heading-4"):
            d = _direction(_text_of(n))
            if d is not None:
                section_dir = d
            current = None
            continue
        if nt == "paragraph" and {"bold", "underline"} <= _marks_of(n):
            raw = _text_of(n).strip()
            if not raw or len(raw) > 80:  # a bold+underline sentence isn't a brawler name
                current = None
                continue
            name, suffix_dir = _split_name_suffix(raw)
            direction = suffix_dir or section_dir or "adjusted"
            current = BalanceChange(name=name, brawler_id=_resolve_id(name, index),
                                    direction=direction)
            changes.append(current)
            continue
        if nt in ("unordered-list", "ordered-list") and current is not None:
            current.details.extend(_list_items(n))
            continue
        # any other block ends the current brawler's bullet run
        if nt == "paragraph" and _text_of(n).strip():
            current = None
    return changes


def _parse_new_brawlers(rich: dict) -> List[NewBrawler]:
    """Entries from a "NEW Brawlers" block — each a ``heading-3`` "Name - Rarity - Class".
    Rarity/class are best-effort: only a class in the official taxonomy is kept, so a stray
    heading can't inject a bogus class into ``class_overrides.json`` downstream."""
    doc = rich.get("json", rich) if isinstance(rich, dict) else {}
    out: List[NewBrawler] = []
    for n in doc.get("content", []) or []:
        if not (isinstance(n, dict) and n.get("nodeType") in ("heading-2", "heading-3", "heading-4")):
            continue
        parts = [p.strip() for p in re.split(r"\s+-\s+", _text_of(n).strip()) if p.strip()]
        if not parts:
            continue
        rarity = parts[1] if len(parts) > 1 else ""
        cls = next((p for p in parts[1:] if _canon_class(p)), "")
        out.append(NewBrawler(name=parts[0], rarity="" if rarity == cls else rarity,
                              cls=_canon_class(cls)))
    return out


def _parse_content_block(rich: dict, kind: str, index: Dict[str, R.Brawler]) -> List[ContentChange]:
    """Extract per-brawler entries from a "NEW Buffies" / "NEW Hypercharges" / gears block.

    Two shapes appear, so both are handled:
      * a ``heading-3`` naming the brawler ("Rico"), followed by the list(s) describing it —
        the Buffies layout;
      * a bare list whose items lead with the brawler ("STARR NOVA: THE HYPERCHARGE …") — the
        Hypercharges layout.
    Only entries whose name resolves to a real brawler are kept, so prose headings don't
    masquerade as content changes."""
    doc = rich.get("json", rich) if isinstance(rich, dict) else {}
    nodes = [n for n in (doc.get("content", []) or []) if isinstance(n, dict)]
    # Which layout is this? If the section uses headings at all, a list belongs to whatever
    # heading precedes it — so a list under a *prose* heading ("How Buffies Work") must not be
    # name-scanned, or it mints a bogus entry for whichever brawler the prose happens to mention.
    heading_layout = any(n.get("nodeType") in ("heading-2", "heading-3", "heading-4")
                         for n in nodes)
    out: List[ContentChange] = []
    by_id: Dict[int, ContentChange] = {}
    current: Optional[ContentChange] = None

    def _record(name: str, bid: int, details: List[str]) -> ContentChange:
        """One entry per brawler per section; repeat mentions extend the first."""
        existing = by_id.get(bid)
        if existing is None:
            existing = ContentChange(kind=kind, name=name, brawler_id=bid)
            by_id[bid] = existing
            out.append(existing)
        existing.details.extend(details)
        return existing

    for n in nodes:
        nt = n.get("nodeType")
        if nt in ("heading-2", "heading-3", "heading-4"):
            name = _split_name_suffix(_text_of(n))[0]
            bid = _resolve_id(name, index) if name else None
            current = _record(name, bid, []) if bid is not None else None
        elif nt in ("unordered-list", "ordered-list"):
            items = _list_items(n)
            if current is not None:
                current.details.extend(items)
            elif not heading_layout:
                for item in items:  # headingless layout: "NAME: rest of the line"
                    lead, _, rest = item.partition(":")
                    name = lead.strip()
                    bid = _resolve_id(name, index) if name else None
                    if bid is None:
                        # The lead was the feature's name ("BOWLING BOLT:"), not the brawler's.
                        # Scan the description first — it names the owner ("Bolt destroys …") —
                        # before falling back to the whole item.
                        name, bid = _find_brawler(rest, index)
                        if bid is None:
                            name, bid = _find_brawler(item, index)
                    if bid is not None:
                        _record(name, bid, [re.sub(r"\s+", " ", item).strip()])
    return out


def _content_kind(section_key: str) -> Optional[str]:
    """Map a canonical section title to the content kind it describes, or None."""
    for needle, kind in (("buffie", "buffie"), ("hypercharge", "hypercharge"), ("gear", "gear")):
        if needle in section_key:
            return kind
    return None


def _canon_class(text: str) -> str:
    """Map printed text to a member of the official 7-class taxonomy, else "" — the whitelist
    that keeps a parsed heading from writing a junk class into the overrides file."""
    t = re.sub(r"[^a-z]", "", (text or "").lower())
    return next((c for c in BRAWLER_CLASSES if re.sub(r"[^a-z]", "", c.lower()) == t), "")


def _block_text(block: dict) -> str:
    t = block.get("text")
    if isinstance(t, dict):
        return _text_of(t.get("json", t))
    return t if isinstance(t, str) else ""


def _canon(title: str) -> str:
    """Canonical section-title key: lowercased, emoji/punctuation dropped, whitespace collapsed.
    So a decorated "⚖️ Balance Changes" or "Maintenance - March 19 " still routes correctly —
    Supercell demonstrably emoji-decorates section titles (see the February fixture)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", title.lower())).strip()


def parse_release_notes(html: str, url: str) -> PatchReport:
    """Parse one release-notes page's HTML into a :class:`PatchReport`."""
    pp = _next_data(html).get("props", {}).get("pageProps", {})
    title = (pp.get("title") or "").strip()
    publish_date = (pp.get("publishDate") or "").strip()
    body = pp.get("bodyCollection") or []
    index = _name_index()

    buffs: List[BalanceChange] = []
    nerfs: List[BalanceChange] = []
    other: List[BalanceChange] = []
    new_brawlers: List[NewBrawler] = []
    content: List[ContentChange] = []
    maintenance: List[str] = []
    fingerprint_parts: List[str] = []
    saw_balance = saw_new_brawlers = False

    for block in body:
        if not isinstance(block, dict):
            continue
        sec = (block.get("title") or "").strip()
        key = _canon(sec)
        # Substring rather than equality: a retitled "Balance Changes & Adjustments" or
        # "NEW Brawlers and Skins" must still route, or the whole patch reports empty.
        if "balance change" in key:
            saw_balance = True
            for c in _parse_balance_block(block.get("text") or {}, index):
                (buffs if c.direction == "buff"
                 else nerfs if c.direction == "nerf"
                 else other).append(c)
            fingerprint_parts.append(sec + "\n" + _block_text(block))
        elif "new brawler" in key:
            saw_new_brawlers = True
            new_brawlers.extend(_parse_new_brawlers(block.get("text") or {}))
            fingerprint_parts.append(sec + "\n" + _block_text(block))
        elif key.startswith("maintenance"):
            maintenance.append(sec)
            fingerprint_parts.append(sec + "\n" + _block_text(block))
        elif _content_kind(key):
            content.extend(_parse_content_block(block.get("text") or {},
                                                _content_kind(key), index))
            fingerprint_parts.append(sec + "\n" + _block_text(block))

    # Distinguish a *broken parse* (a recognized section is present but yielded nothing — a
    # likely layout change) from a genuinely balance-free update (the section is simply absent).
    # Only the former is a scraper-health problem worth escalating.
    layout_warning = ((saw_balance and not (buffs or nerfs or other))
                      or (saw_new_brawlers and not new_brawlers))
    note = ""
    if layout_warning:
        note = ("a 'Balance Changes'/'NEW Brawlers' section is present but parsed to nothing — "
                "the page's rich-text layout may have changed; check it manually and update "
                "backend/bsdraft/collect/patchnotes.py")

    slug = _slug_of(url)
    return PatchReport(
        slug=slug, url=url, title=title, publish_date=publish_date,
        fingerprint=_fingerprint(slug, fingerprint_parts),
        new_brawlers=[b.name for b in new_brawlers], new_brawler_info=new_brawlers,
        buffs=buffs, nerfs=nerfs, other_changes=other, content_changes=content,
        maintenance_sections=maintenance, note=note, layout_warning=layout_warning,
    )


def _fingerprint(slug: str, parts: List[str]) -> str:
    """Stable short hash over the slug + balance-relevant block text. Whitespace-normalized so
    cosmetic reflows don't churn it, but any real content change (new maintenance section, new
    balance entry, new brawler) does. New page (new slug) or in-place update both -> new hash."""
    norm = "\n".join(re.sub(r"\s+", " ", p).strip() for p in parts)
    h = hashlib.sha256((slug + "\n" + norm).encode("utf-8")).hexdigest()
    return h[:16]


# --------------------------------------------------------------------------- discovery

def find_latest_release_notes(fetch: Callable[[str], str] = _fetch,
                              max_pages: int = 3) -> Optional[Tuple[str, str, str]]:
    """Find the current living release-notes page via the blog index. Returns
    ``(url, publish_date, title)`` for the newest ``/release-notes/`` article, or None. Scans
    page 1 (which normally carries it), walking a few more pages only as a fallback."""
    best: Optional[Tuple[str, str, str]] = None
    for page in range(1, max_pages + 1):
        url = BLOG_INDEX_URL if page == 1 else f"{BLOG_INDEX_URL}page/{page}/"
        try:
            pp = _next_data(fetch(url)).get("props", {}).get("pageProps", {})
        except (httpx.HTTPError, ValueError):
            break
        for a in pp.get("articles", []) or []:
            link = a.get("linkUrl") or ""
            if not _is_release_notes_link(link):
                continue
            pd = (a.get("publishDate") or "").strip()
            full = urljoin(BASE, link)
            if best is None or pd > best[1]:
                best = (full, pd, (a.get("title") or "").strip())
        if best is not None:
            break  # page 1 (newest-first) had it — no need to paginate
    return best


def fetch_latest(fetch: Callable[[str], str] = _fetch) -> Optional[PatchReport]:
    """Discover and parse the latest release-notes page. None if none is found."""
    found = find_latest_release_notes(fetch)
    if not found:
        return None
    url, _pd, _title = found
    return parse_release_notes(fetch(url), url)


# --------------------------------------------------------------------------- issue rendering

# Marker embedded in the issue body so the alerting job can dedup on content: a re-run over
# unchanged notes finds its own fingerprint and stays quiet; a living-doc edit (new maintenance
# section / new balance entry) mints a new fingerprint and files a fresh alert.
FINGERPRINT_MARKER = "bs-patch"

# The new-brawler rollout, mirrored from the meta-alert job — a brawler is invisible to the
# model (encodes to embedding index 0) until the reference catalog is refreshed and retrained.
_ROLLOUT = (
    "PYTHONPATH=backend python backend/scripts/refresh_reference.py\n"
    "PYTHONPATH=backend python backend/scripts/train.py\n"
    "PYTHONPATH=backend python backend/scripts/export_model.py\n"
    "PYTHONPATH=backend python -m bsdraft.collect.publish --model\n"
    "git add data/reference/ && git commit && git push   # Render deploy picks up the catalog"
)


def _cell(text: str) -> str:
    """Make scraped text safe inside a one-line Markdown table cell: no pipes/newlines to break
    the table, no comment delimiters that could spoof the fingerprint marker."""
    s = re.sub(r"\s+", " ", text).replace("|", "\\|").replace("<!--", "").replace("-->", "")
    return s.strip()


def _safe_url(url: str) -> str:
    """A Markdown-safe link target: scheme+host+percent-encoded path only, query/fragment
    dropped. The discovery guard validates a link's *host* but not its path/query, and ``urljoin``
    does no encoding — so without this a crafted path/query (spaces, ``<!--``, ``)``) would reach
    the issue body verbatim and could break out of the link or forge the fingerprint marker."""
    p = urlsplit(url)
    if not p.scheme or not p.hostname:
        return ""  # not a usable absolute URL — omit the link target rather than emit junk
    netloc = p.hostname + (f":{p.port}" if p.port else "")  # rebuilt without any userinfo
    return f"{p.scheme}://{netloc}{quote(p.path, safe='/-._~')}"


def _change_table(rows: List[BalanceChange]) -> str:
    out = ["| Brawler | Changes |", "|---|---|"]
    for c in rows:
        details = "; ".join(_cell(d) for d in c.details) or "—"
        out.append(f"| {_cell(c.name)} | {details} |")
    return "\n".join(out)


def render_issue(report: PatchReport) -> Tuple[str, str]:
    """Render ``(title, body_markdown)`` for a GitHub issue from a report. GitHub-agnostic
    Markdown; the caller owns labels and dedup. All scraped text flows through :func:`_cell`."""
    date = report.publish_date[:10] or "?"
    if report.new_brawlers:
        title = f"New brawler(s) + balance patch: {', '.join(report.new_brawlers)} ({date})"
    elif report.all_changes:
        title = f"Balance patch: {report.title} ({date})"
    elif report.layout_warning:
        title = f"⚠️ Release-notes layout may have changed — scraper parsed no balance items ({date})"
    else:
        title = f"Release notes updated: {report.title} ({date})"
    title = title[:200]

    lines = [
        f"The official release notes **[{_cell(report.title)}]({_safe_url(report.url)})** changed.",
        "",
        "This is the **leading** signal, straight from the patch notes; the data-driven drift "
        "detector (`/api/meta`) confirms the on-ladder impact once enough ranked games "
        "accumulate. See `backend/bsdraft/collect/patchnotes.py`.",
        "",
    ]
    if report.new_brawlers:
        lines += [
            f"## 🆕 New brawler(s): {', '.join(_cell(n) for n in report.new_brawlers)}",
            "",
            "Invisible to the model (they encode to embedding index 0) until the reference is "
            "refreshed and retrained:",
            "",
            "```bash",
            _ROLLOUT,
            "```",
            "",
        ]
    for heading, bucket in (("⬆️ Buffs", report.buffs), ("⬇️ Nerfs", report.nerfs),
                            ("🔧 Other changes", report.other_changes)):
        if bucket:
            lines += [f"## {heading} ({len(bucket)})", "", _change_table(bucket), ""]
    if report.content_changes:
        kinds = sorted({c.kind for c in report.content_changes})
        lines += [f"## ✨ New {'/'.join(k + 's' for k in kinds)} "
                  f"({len(report.content_changes)})", "",
                  "Not visible to any catalog endpoint — the release notes are the only source, "
                  "and these move real brawler strength:", "",
                  "| Brawler | Kind | Detail |", "|---|---|---|"]
        for c in report.content_changes:
            detail = "; ".join(_cell(d) for d in c.details) or "—"
            lines.append(f"| {_cell(c.name)} | {c.kind} | {detail} |")
        lines.append("")
    if report.maintenance_sections:
        lines += ["## 🛠 Maintenance sections on this page", "",
                  *(f"- {_cell(s)}" for s in report.maintenance_sections), "",
                  "_These in-place hotfix sections can carry balance tweaks in a looser format "
                  "that isn't itemized in the tables above — open the notes to confirm._", ""]
    if report.note:
        lines += [f"> ⚠️ {_cell(report.note)}", ""]
    lines += [
        f"_Fingerprint `{report.fingerprint}` — a new alert fires only when the notes' balance "
        "content changes (a new page, or a hotfix section added in place). Close to acknowledge._",
        "",
        f"<!-- {FINGERPRINT_MARKER}:{report.fingerprint} -->",
    ]
    return title, "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def _to_dict(report: PatchReport) -> dict:
    d = asdict(report)
    d["brawlers_touched"] = report.brawlers_touched
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape the latest Brawl Stars release notes for balance changes.")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument("--url", default=None,
                    help="parse this specific release-notes URL instead of discovering the latest")
    ap.add_argument("--issue-body", metavar="PATH", default=None,
                    help="CI mode: write the rendered GitHub-issue Markdown to PATH and print a "
                         "one-line JSON header "
                         "{fingerprint,title,has_new_brawlers,has_content,layout_warning}")
    args = ap.parse_args()
    try:
        report = parse_release_notes(_fetch(args.url), args.url) if args.url else fetch_latest()
    except (httpx.HTTPError, ValueError) as e:
        raise SystemExit(f"patchnotes scrape failed: {e}")
    if report is None:
        raise SystemExit("no release-notes page found on the blog index")

    if args.issue_body:
        title, body = render_issue(report)
        with open(args.issue_body, "w", encoding="utf-8") as fh:
            fh.write(body)
        print(json.dumps({
            "fingerprint": report.fingerprint,
            "title": title,
            "has_new_brawlers": bool(report.new_brawlers),
            "has_content": bool(report.new_brawlers or report.all_changes
                                or report.content_changes),
            "layout_warning": report.layout_warning,
        }, ensure_ascii=False))
    elif args.json:
        print(json.dumps(_to_dict(report), ensure_ascii=False))
    else:
        print(report.summary())


if __name__ == "__main__":
    main()
