# Changelog

Notable, user-visible changes to [brawldraft.com](https://brawldraft.com). The site deploys
continuously from `main`, so entries are **dated, not versioned** — newest first. Routine
retrains, doc edits, and internal refactors are left out unless they changed what users see.

## 2026-08-18

- **Bolt no longer shows Brock's gadgets.** The upstream catalog API serves "Rocket Laces" and
  "Rocket Fuel" under Bolt as well as Brock, and the committed snapshot had carried the duplicate
  since day one — polluting Bolt's loadout advice (the kit-description effect classifier read
  Brock's rockets as Bolt's). The snapshot is fixed, and the catalog fetch now strips any
  accessory served under two brawlers (its description names the real owner) or refuses the
  payload when it can't tell — so the next auto-refresh can't reintroduce it. A test now pins
  accessory-id uniqueness across the committed catalog.
- **Removed the buffie signal everywhere.** The roster API reports which buffies you *own* but
  never how many *exist* per brawler, so the "MISSING BUFFIE" tag misfired on every brawler with
  no buffies released (R-T among them — verified against maxed top-100 rosters). Dropped from the
  pick-card gap tags, the mastery build score (renormalized 3:2:2 over star power / gadget /
  gears), and the purchase advisor.
- **Ranked power floor now actually enforced on the live site.** The live roster service predated
  the gate and omitted `power`, so under-floor brawlers (e.g. a Power-9 Sandy at Legendary) still
  appeared in personalized picks. Redeployed; the picker now marks them "needs Power 11 in
  <bracket>" and they're excluded from your pick's recommendations.
- **Boosted-rotation season flips are now hands-off.** The `valid_until` fail-safe was only
  checked at process start, but the deployed API is kept warm for days — it could serve an expired
  FREE rotation long after a season flipped. The rotation logic now runs on every request against
  an explicit **UTC** clock (all serving hosts agree; hand-staged dates are targetable), and an
  upcoming rotation can be staged with an `active_from` date or exact instant: Season 1
  (Berry / Tara / Meg) serves through 2026-08-18 UTC, and Season 2 (Trunk / Willow / Kaze) takes
  over automatically at 2026-08-19 10:00 UTC — the overnight window between them deliberately
  serves no FREE set, since a wrong badge is worse than a missing one. Non-string dates fail safe
  instead of erroring, a boosted-watch rewrite carries staged dates forward for seasons whose
  names still match, and the committed file is schema-checked in tests.
- Loadout hover: both gear slots are starred on your own seat, not just the single best gear.

## 2026-08-17

- **Purchase advisor** shipped at `/purchases`: enter your tag and get your highest-value next
  purchases — power-11 climbs, gadgets, star powers, gears, hypercharges, new-brawler unlocks —
  ranked by meta strength × purchase impact, with costs as context.
- **Ranked power-floor gate**: drafts never recommend a brawler you can't select in your bracket
  (below Power 9 through Diamond, below Power 11 from Mythic up).

## 2026-08-12

- **Partial-draft-native model**: masked training bakes mid-draft evaluation directly into the
  win-probability net; the deep-search toggle is retired (the lookahead is now implicit).

## 2026-08-11

- **Seat-scoped personalization**: mark which of your team's three picks is *you* — only that
  seat's suggestions filter to your roster and history; teammates draft from the full meta.
- **Season's free "boosted" brawlers**: the three maxed brawlers Ranked hands everyone are
  recommendable even when unowned, with FREE badges (scraped from the official release notes).
- **Loadout hover**: hover any drafted brawler for gadget / star-power / gear advice, filtered to
  what you own on your own seat.
- **Data-driven loadout picks**: gadget/star-power suggestions backed by measured win rates from
  single-item-owner inference, not just heuristics.
- Fixes: ban slots are 3–6 (cursor no longer snaps back to skipped bans); tag-less visitors can no
  longer see the operator's roster; the crawler survives network outages.

## 2026-08-10

- **Console redesign**: the draft board became the tactical-telemetry console (THE CALL, pick
  orders, signal meters, confidence).
- Wendy + Nori added to the reference; the model vocabulary is pinned into the artifact so map
  refreshes can't silently shift embeddings.
- CI watchers added: balance-notes scrape, catalog diff, pipeline staleness.
- Scoring fusion rebalanced toward the trained model per the 995k-match ablation rerun.

## 2026-08-07 → 08-08

- Content pages (draft guide, how-it-works, FAQ), social share card, logo + brand assets, and
  env-gated (dark) AdSense wiring with a privacy page.

## 2026-07-16

- **Balance-change watch**: meta-report artifact, automatic retrain when drift trips, and a daily
  alert pipeline.
- `/api/rank` resolves tiers from a published rank-index artifact; the player tag is remembered
  across visits.

## 2026-06-17 → 06-22 — initial public launch

- Live at **brawldraft.com**: Python win-probability model + draft engine behind FastAPI, Next.js
  board, $0/mo deploy (Render + Cloudflare Pages) self-updating via home crawler → GitHub Release
  artifacts.
- Per-visitor roster personalization through a Cloudflare Tunnel (the Supercell key is IP-locked
  to the home machine); live Ranked-tier resolution; rank-bracket-conditioned stats; blind-pick
  mode for Diamond and below; live model hot-swap; meta-shift banner.
