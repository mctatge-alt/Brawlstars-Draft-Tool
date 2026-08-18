// Typed client for the draft API (FastAPI backend).

export type Brawler = { id: number; name: string; cls: string; rarity: string; image_url: string };
export type GameMap = { id: number; name: string; mode: string; image_url: string; games: number };
export type Reference = { brawlers: Brawler[]; maps: GameMap[]; modes: string[]; brackets: string[]; boosted: number[] };

export type PickRec = {
  brawler_id: number; name: string; cls: string; score: number; map_winrate: number;
  synergy: number | null; counter: number | null; role_fit: number;
  win_prob: number | null; confidence: number;
  mastery: number | null; personal_winrate: number | null; personal_games: number | null;
  owned: boolean; gaps: string[];
  breakdown: Record<string, number>;
};
export type BanRec = {
  brawler_id: number; name: string; cls: string; threat: number;
  map_winrate: number; use_rate: number; confidence: number;
  // Projected swing in your win probability from banning this brawler, given everything already
  // banned and who picks first — the sort key. Null when the backend has no model to project
  // with, where the list falls back to raw threat order.
  ban_value: number | null;
  replacement: string | null;   // who the enemy builds around instead
  self_deny: boolean;           // the draft projects this brawler onto *your* side
};
export type Warning = { text: string; severity: string };
export type RoleTip = { name: string; cls: string; role: string };
export type ThreatTip = { name: string; cls: string; tip: string };
export type GamePlan = {
  objective: string; win_condition: string; archetype: string; playstyle: string;
  roles: RoleTip[]; threats: ThreatTip[]; tips: string[]; avoid: string[]; compensate: string[];
};
export type RecommendResponse = {
  phase: string; picks: PickRec[]; bans: BanRec[];
  composition: Record<string, number>; warnings: Warning[];
  game_plan: GamePlan | null; next_to_act: string | null;
};

export type OwnedGear = { id: number; name: string; level: number };
export type OwnedBrawler = {
  id: number; mastery: number; gaps: string[];
  // Specific items the player owns on this brawler — populated by /api/roster (empty on the
  // recommend path). Used to restrict loadout suggestions on the user's own pick to what they have.
  owned_star_powers: number[]; owned_gadgets: number[]; owned_gears: OwnedGear[];
  // Progression state for the purchase advisor (populated by /api/roster; absent on the recommend
  // path). Optional so older backends / the recommend payload still type-check.
  power?: number; has_hypercharge?: boolean;
};
export type RosterResponse = {
  loaded: boolean; tag: string; name: string; owned: OwnedBrawler[]; error?: string | null;
};

export type LoadoutItem = {
  id: number | null; name: string; kind: "gadget" | "star_power" | "gear";
  image_url: string; effect: string; description: string;
  fit: number; recommended: boolean; why: string; source: string;
};
export type LoadoutResponse = {
  brawler_id: number; brawler_name: string; cls: string; mode: string;
  gadgets: LoadoutItem[]; star_powers: LoadoutItem[]; gears: LoadoutItem[]; note: string;
};

export type PurchaseKind =
  "power_upgrade" | "gadget" | "star_power" | "gear" | "hypercharge" | "new_brawler";
export type PurchaseRec = {
  brawler_id: number; brawler_name: string; kind: PurchaseKind;
  value_score: number; meta_winrate: number;
  confidence: "measured" | "heuristic" | "eligibility_only";
  cost: Record<string, number>;              // e.g. { coins: 2000, power_points: 890 }
  rationale: string;
  item_id: number | null; item_name: string | null; target_power: number | null;
  item_delta: number | null; gate: string | null;
};
export type PurchasesResponse = {
  tag: string; name: string; scope: string; recommendations: PurchaseRec[];
};

export type RankInfo = {
  found: boolean; tag: string; tier: number | null; tier_label: string | null;
  bracket: string | null; source: string | null; error?: string | null;
};

export type Health = {
  status: string; model: boolean; matches: number; roster: boolean;
  refresh_seconds: number; last_check: number | null; last_change: number | null;
};

export type MetaShift = {
  brawler_id: number; name: string; kind: string;
  wr_before: number; wr_after: number; use_before: number; use_after: number; z: number;
};
export type Meta = {
  shifted: boolean; n_recent: number; n_prior: number;
  new_brawlers: string[]; shifts: MetaShift[]; note: string;
};

export type TopPick = {
  brawler_id: number; name: string; cls: string; score: number; map_winrate: number;
};
export type TopPicksBody = {
  map_id: number; mode: string; our_team: number[]; their_team: number[]; bans: number[];
  rank_bracket?: string | null; top: number;
};
export type TopPicksResponse = {
  map_id: number; mode: string; rank_bracket: string | null; picks: TopPick[];
};

export type RecommendBody = {
  map_id: number; mode: string; our_team: number[]; their_team: number[]; bans: number[];
  we_pick_first: boolean; solo_queue: boolean; rank_bracket?: string | null; phase: "pick" | "ban";
  personalize: boolean; personal_tag?: string | null; top: number;
  // The player's owned brawlers + mastery + loadout gaps, sent so the public backend (which can't
  // fetch the roster itself — IP-locked out of Supercell) can personalize the suggestions.
  roster?: OwnedBrawler[] | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
// Roster needs a live Supercell call, which only works from an IP whitelisted with the key.
// Point this at a whitelisted host (e.g. a Cloudflare Tunnel to the home machine) to enable
// per-visitor personalization on the public site; defaults to the main API otherwise.
const ROSTER_BASE = process.env.NEXT_PUBLIC_ROSTER_BASE || API_BASE;

export async function getReference(): Promise<Reference> {
  const res = await fetch(`${API_BASE}/api/reference`);
  if (!res.ok) throw new Error(`reference: ${res.status}`);
  return res.json();
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`health: ${res.status}`);
  return res.json();
}

export async function getMeta(): Promise<Meta> {
  const res = await fetch(`${API_BASE}/api/meta`);
  if (!res.ok) throw new Error(`meta: ${res.status}`);
  return res.json();
}

export async function getRank(tag: string): Promise<RankInfo> {
  // Through ROSTER_BASE (the keyed tunnel), not API_BASE: a live battle-log lookup gives the
  // player's *current* tier, whereas the keyless API can only return the crawl snapshot, which
  // goes stale across a Ranked season reset. Falls back to API_BASE when no tunnel is set.
  const res = await fetch(`${ROSTER_BASE}/api/rank?tag=${encodeURIComponent(tag)}`);
  if (!res.ok) throw new Error(`rank: ${res.status}`);
  return res.json();
}

export async function getRoster(tag?: string | null): Promise<RosterResponse> {
  const qs = tag ? `?tag=${encodeURIComponent(tag)}` : "";
  const res = await fetch(`${ROSTER_BASE}/api/roster${qs}`);
  if (!res.ok) throw new Error(`roster: ${res.status}`);
  return res.json();
}

export async function getTopPicks(body: TopPicksBody): Promise<TopPicksResponse> {
  const res = await fetch(`${API_BASE}/api/top_picks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`top_picks: ${res.status}`);
  return res.json();
}

export async function recommend(body: RecommendBody): Promise<RecommendResponse> {
  const res = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`recommend: ${res.status}`);
  return res.json();
}

export async function getPurchases(
  roster: OwnedBrawler[], tag?: string | null, name?: string | null, top = 24,
): Promise<PurchasesResponse> {
  // Scored on API_BASE (holds the stats + item win-rate table); the roster was fetched from
  // ROSTER_BASE (the keyed tunnel) and is POSTed here, mirroring the recommend path — the public
  // backend can't fetch a roster itself (IP-locked out of Supercell).
  const res = await fetch(`${API_BASE}/api/purchases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ roster, tag: tag || null, name: name || null, top }),
  });
  if (!res.ok) throw new Error(`purchases: ${res.status}`);
  return res.json();
}

export async function getLoadout(brawlerId: number, mode: string, mapId?: number | null): Promise<LoadoutResponse> {
  const qs = new URLSearchParams({ brawler: String(brawlerId), mode });
  if (mapId != null) qs.set("map_id", String(mapId));
  const res = await fetch(`${API_BASE}/api/loadout?${qs.toString()}`);
  if (!res.ok) throw new Error(`loadout: ${res.status}`);
  return res.json();
}
