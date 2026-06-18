// Typed client for the draft API (FastAPI backend).

export type Brawler = { id: number; name: string; cls: string; rarity: string; image_url: string };
export type GameMap = { id: number; name: string; mode: string; image_url: string; games: number };
export type Reference = { brawlers: Brawler[]; maps: GameMap[]; modes: string[]; brackets: string[] };

export type PickRec = {
  brawler_id: number; name: string; cls: string; score: number; map_winrate: number;
  synergy: number | null; counter: number | null; role_fit: number;
  win_prob: number | null; confidence: number; projected_winprob: number | null;
  mastery: number | null; personal_winrate: number | null; personal_games: number | null;
  owned: boolean; gaps: string[];
  breakdown: Record<string, number>;
};
export type BanRec = {
  brawler_id: number; name: string; cls: string; threat: number;
  map_winrate: number; use_rate: number; confidence: number;
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

export type OwnedBrawler = { id: number; mastery: number; gaps: string[] };
export type RosterResponse = {
  loaded: boolean; tag: string; name: string; owned: OwnedBrawler[]; error?: string | null;
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
export type TopPicksResponse = {
  map_id: number; mode: string; rank_bracket: string | null; picks: TopPick[];
};

export type RecommendBody = {
  map_id: number; mode: string; our_team: number[]; their_team: number[]; bans: number[];
  we_pick_first: boolean; solo_queue: boolean; rank_bracket?: string | null; phase: "pick" | "ban";
  use_search: boolean; personalize: boolean; personal_tag?: string | null; top: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

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
  const res = await fetch(`${API_BASE}/api/rank?tag=${encodeURIComponent(tag)}`);
  if (!res.ok) throw new Error(`rank: ${res.status}`);
  return res.json();
}

export async function getRoster(): Promise<RosterResponse> {
  const res = await fetch(`${API_BASE}/api/roster`);
  if (!res.ok) throw new Error(`roster: ${res.status}`);
  return res.json();
}

export async function getTopPicks(
  mapId: number, mode: string, bracket?: string | null
): Promise<TopPicksResponse> {
  const qs = new URLSearchParams({ map_id: String(mapId), mode });
  if (bracket) qs.set("rank_bracket", bracket);
  const res = await fetch(`${API_BASE}/api/top_picks?${qs.toString()}`);
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
