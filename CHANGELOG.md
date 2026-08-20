# Changelog

Notable, user-visible changes to [brawldraft.com](https://brawldraft.com). The site deploys
continuously from `main`, so entries are **dated, not versioned** — newest first. Routine
retrains, doc edits, and internal refactors are left out unless they changed what users see.

## 2026-08-20

- **The map picker now offers only the maps Ranked actually rotates.** It was listing every map
  still in the game's files for the five ranked modes — 113 of them, including pairs you can't
  queue, like "Heist: Pit Stop". The list is now the 27 maps that appear in collected ranked
  games, which is the rotation: four per mode (Gem Grab shows seven while the season boundary
  straddles two rotations). Maps we've never seen played were ones the model had nothing to say
  about anyway.
- **Your Ranked tier no longer reports last season's rank as fact.** On a season reset the live
  profile lookup is the only source that knows you've been reset — our match data is a crawl
  snapshot with no season stamp. When that lookup came back "no tier yet this season", the badge
  quietly fell through to the pre-reset snapshot and showed the tier you'd just lost. It now
  says you haven't placed. And when the live lookup can't run at all, the badge still shows the
  snapshot but marks it with a `?` and says so on hover, instead of stating it flatly.

## 2026-08-19

- **Upgrade planner now ranks by value per coin, prerequisites included.** The first cut ranked
  purchases by raw win-rate value with cost as a side note — so its top suggestion could be a star
  power on a brawler still at Power 2 with no items, which you couldn't even field. Every
  recommendation is now the full *package* from where your account actually stands (the power
  climb to your bracket's floor, a first gadget + star power on an unbuilt brawler, the Starr Road
  unlock), priced as a whole and ranked by how much ranked win rate it buys per coin. Ranked's
  power floor is treated as the hard gate it is: below Power 9 (through Diamond) or Power 11
  (Mythic and up) a brawler gets one "make it ranked-ready" card and no item cards. Your live
  Ranked tier sets the floor (pin P9/P11 yourself if the lookup fails — unknown assumes Power 11,
  the safer guess); the bracket's own stats drive the meta read; roster depth discounts a 30th
  option versus a 1st; this season's free brawlers are discounted; unlocks only appear for the
  Starr Road tier you can buy from, with credit prices corrected (Epic 925 / Mythic 1,900 /
  Legendary 3,800 / Ultra Legendary 5,500). Cards show the package steps with per-step cost, a
  value-per-coin meter, and kind filters (Power / Gadgets / Star Powers / Gears / Hypercharges /
  Unlocks).

## 2026-08-18

- **Recommendations now favor the specific map over the game mode.** The mode-archetype nudge
  (e.g. Controllers in Gem Grab, Tanks in Brawl Ball) was quietly about as influential as real map
  win-rates and applied the same on every map of a mode — so the same brawlers surfaced regardless
  of the actual map. It's now scaled down by how much real data exists for that brawler on that
  map: on well-played maps the pick follows the genuine win-rate, while freshly-rotated maps still
  lean on the archetype guidance where there's no data yet. Brawlers no longer flat-top every map
  of their mode.
- **Faster analysis.** The recommend step is ~3.7× faster — the win-probability model now scores
  every candidate in one batched pass instead of one at a time. Identical picks, just quicker.
- **Keyboard-first drafting: Tab and arrow keys.** Press **Tab** to jump straight to your first
  pick, skipping any unused ban slots — handy when only a few bans are used. Browse the brawler
  grid with the **arrow keys**: press ↓ from the search box to drop into the grid, arrow between
  brawlers, and hit Enter to place the highlighted one. Tab still steps through form fields (like
  the tag box) normally and never traps focus, and the same keys work identically on Windows, macOS,
  and Linux.
- **Removed the confusing gear "level" from the loadout popover.** Owned gears used to show a bare
  "Lv3." Brawl Stars removed gear upgrade levels back in 2022 — gears are now a flat purchase at full
  power — so that number (always 3) meant nothing. Owned gears now just show their name.
- **Loadout advice now adjusts to the enemy comp.** The hover popover's gadget / star-power / gear
  picks were frozen at pick time; now the drafted enemy team feeds a bounded overlay: class-count
  reads (dive-heavy, 2 Tanks, poke-heavy) plus a CC-heavy read that fires when *every* enemy
  carries real crowd control in their kit (keyword scan corrected by a full-roster audit — Frank's
  pull, Sandy's sleep and friends were being missed; Carl/Janet's self-dashes no longer count).
  Adjusted items show signed chips ("+ vs dive"), the popover header names the reads it saw, and a
  pick that wins *only because* of the comp is badged ★ PICK · COMP. Gears join in via curated
  counter offsets (Shield vs dive/CC, Damage vs tanks, Health/Speed vs poke). Capped at ±0.15 fit
  so the comp nudges rather than overrules the mode read, measured win rates stay authoritative
  where they exist, and advice refreshes as picks land (Mythic+ drafts only — blind-pick brackets
  can't see the enemy team).
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
