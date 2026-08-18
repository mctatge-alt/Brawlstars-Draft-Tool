# Purchase advisor — "what to upgrade next"

A personalized page (`/purchases`) that ranks a player's highest-value next purchases from their
live roster: power-11 climbs, gadgets, star powers, gears, hypercharges, and new-brawler unlocks.
It is the inverse of the [loadout advisor](item-winrate.md): that tells you which *owned* item to
equip; this tells you which *unowned* item is most worth acquiring. (Buffies are deliberately not
advised — see the buffie note under *Data reality*.)

Read this before touching `engine/purchases.py`, the `/api/purchases` endpoint, the
`economy.json` reference, or `frontend/components/PurchaseAdvisor.tsx`.

## Data reality — what the API can and can't tell us

The Supercell player endpoint exposes **ownership**, never **currency balances**. So we can compute
what a player is *missing* and what it *costs*, but never what they can *afford*. Confirmed against
a live roster (105-brawler account), every field the advisor needs is present and parsed by
`engine/mastery.py::parse_roster`:

- `power` (1–11) per brawler → exact power-deficit and its coin/power-point cost.
- owned gadget / star-power ids → diff against the catalog (`R.load_brawlers()`) to find the missing one.
- owned gears (id+name+level) → diff against the six universal gears (`reference/gears.json`).
- `has_hypercharge` (bool) → hypercharge slot state.
- roster membership → a brawler *absent* from the roster is a Credit-unlock candidate.

**Blind spots:** all currency balances; the equipped loadout (ownership only); and the *catalogs*
of what gears / hypercharges / buffies exist per brawler (none are catalog-backed). Hypercharge
availability is handled by a curated policy (below).

**Buffies are not advised.** The roster carries a `buffies: {gadget, starPower, hyperCharge}` object,
but its `True` flags only say which buffies you *own* — never how many *exist* for the brawler. A
brawler with no buffie released (e.g. R-T) is all-`False`, indistinguishable from one whose buffies
you just haven't unlocked (verified against maxed top-100 rosters). The earlier model read the fixed
3-key object as three fillable slots, so it flagged "Buy a Buffie" on every buffie-less brawler. With
no reliable slot total, buffies are left out of the advisor (and of `engine/mastery.py` scoring)
entirely. Reviving them would need a curated `buffie_availability` policy like the hypercharge one.

## Scoring (locked product decisions)

`value = meta_strength(brawler) × impact_prior(kind)`, nudged by the measured item win-rate delta
where the [itemstats](item-winrate.md) table has a significant cell (a tie-breaker between the two
items of a type — its estimand is item-vs-other-owners, not own-vs-not-own, so it never dominates).

- **Rank by win-rate value**, not value-per-coin — cost is context only (balances are unknowable).
- **Pure meta strength** — ordering ignores how much the player mains a brawler; ownership only
  decides what's *buyable*.
- **Account-wide across the ranked map pool** — `meta_strength` is the games-weighted average of
  the brawler's win rate over the *current* ranked maps (`stats.brawler_rate` per map), falling back
  to the global rate on thin data.
- **New-brawler unlocks included** — unowned brawlers scored by meta strength × a new-brawler prior.

Power levels have **no intrinsic modeled value** (Ranked normalizes everyone to Power 11), so a
power upgrade never stands alone — its cost is *folded into* the gated item it unlocks, with a
"requires Power N" note (this is how the "upgrade to Power 11" case surfaces, on the hypercharge
rec). Each rec carries a `confidence` tag: `measured` (win-rate cell) / `heuristic` (prior) /
`eligibility_only` (hypercharge — no value model yet).

## The economy table — `data/reference/economy.json`

Hand-maintained researched constants (costs, power gates, impact priors, hypercharge availability),
loaded via `reference.load_economy()` (stdlib, fail-safe to `{}`). **None of this is in any API and
it drifts on balance updates** — treat every value as approximate. Because the advisor ranks by
value and shows cost only as context, a stale number misinforms a cost label, not the ranking. A
patchnotes-fed refresh is a possible follow-up.

Hypercharge availability uses a `mode`/`list` policy: `all_except` treats every Power-11 brawler as
eligible unless listed (matches the ~all-brawlers-have-one reality, and only fires at Power 11);
flip to `only` + an explicit list for conservative, under-covering behavior. Missing section ⇒ no
hypercharge recs (fail-safe).

## Wiring

- **Endpoint** `POST /api/purchases` ([api/main.py](../backend/bsdraft/api/main.py)) mirrors
  `/api/recommend`: the client fetches the roster from the keyed tunnel (`ROSTER_BASE`) and POSTs it
  to the public host (`API_BASE`), which holds the stats + itemstats and can't fetch a roster itself
  (IP-locked out of Supercell). `OwnedBrawler` carries `power`/`has_hypercharge` for the advisor.
- **Engine** `engine/purchases.py` → `DraftEngine.recommend_purchases`; reuses the loaded `stats`
  and `IS.get_itemstats()`. No new artifact; serve path stays torch/pandas-free.
- **Frontend** `app/purchases/page.tsx` + `components/PurchaseAdvisor.tsx`, tag reused from
  `localStorage["bsdraft.tag"]`, `getPurchases()` in `lib/api.ts`.

## Constraints

- **Live-only, no fallback.** Full ownership comes only from the keyed roster tunnel; the public
  host and the (stale, power-less) `profiles.jsonl` can't substitute. The page is dead during the
  recurring 403 IP-rotation outages or when the home machine is offline — it degrades to a clear
  "roster service is down" state. Same single-point-of-failure as board personalization.
- **Local testing:** the roster path needs the IP-locked key + CORS, so verify end-to-end on the
  deployed site (or, on the home machine, run the backend locally with the key and point the
  frontend at it — see [dev-commands.md](dev-commands.md)).
