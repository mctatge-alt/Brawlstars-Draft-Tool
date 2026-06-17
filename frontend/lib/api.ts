// Typed client for the draft API (FastAPI backend).

export type Brawler = { id: number; name: string; cls: string; rarity: string; image_url: string };
export type GameMap = { id: number; name: string; mode: string; image_url: string; games: number };
export type Reference = { brawlers: Brawler[]; maps: GameMap[]; modes: string[] };

export type PickRec = {
  brawler_id: number; name: string; cls: string; score: number; map_winrate: number;
  synergy: number | null; counter: number | null; role_fit: number;
  win_prob: number | null; confidence: number; projected_winprob: number | null;
  mastery: number | null; owned: boolean; gaps: string[];
  breakdown: Record<string, number>;
};
export type BanRec = {
  brawler_id: number; name: string; cls: string; threat: number;
  map_winrate: number; use_rate: number; confidence: number;
};
export type Warning = { text: string; severity: string };
export type RecommendResponse = {
  phase: string; picks: PickRec[]; bans: BanRec[];
  composition: Record<string, number>; warnings: Warning[]; next_to_act: string | null;
};

export type OwnedBrawler = { id: number; mastery: number; gaps: string[] };
export type RosterResponse = {
  loaded: boolean; tag: string; name: string; owned: OwnedBrawler[]; error?: string | null;
};

export type RecommendBody = {
  map_id: number; mode: string; our_team: number[]; their_team: number[]; bans: number[];
  we_pick_first: boolean; solo_queue: boolean; phase: "pick" | "ban";
  use_search: boolean; personalize: boolean; top: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function getReference(): Promise<Reference> {
  const res = await fetch(`${API_BASE}/api/reference`);
  if (!res.ok) throw new Error(`reference: ${res.status}`);
  return res.json();
}

export async function getRoster(): Promise<RosterResponse> {
  const res = await fetch(`${API_BASE}/api/roster`);
  if (!res.ok) throw new Error(`roster: ${res.status}`);
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
