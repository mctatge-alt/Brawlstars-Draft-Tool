"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Brawler, PickRec, BanRec, Reference, RecommendResponse, Warning, RosterResponse, GamePlan, Health, Meta, RankInfo,
  getReference, getRoster, recommend, getHealth, getMeta, getRank,
} from "@/lib/api";

const CLASS_COLOR: Record<string, string> = {
  Tank: "#e0566f", Assassin: "#b15be0", Controller: "#3b82f6", Marksman: "#3ec46d",
  Support: "#e8c34a", "Damage Dealer": "#e8843a", Artillery: "#39c3c0", Unclassified: "#6b7280",
};
const SEV_COLOR: Record<string, string> = { critical: "#e0566f", warn: "#e8c34a", info: "#5aa0ff" };

type Zone = "ban" | "our" | "their";
type Slot = { zone: Zone; index: number };
type Step =
  | { kind: "ban"; slot: Slot; index: number }
  | { kind: "pick"; side: "us" | "them"; slot: Slot; n: number }
  | { kind: "done" };

const BAN_N = 6, TEAM_N = 3;
const PICK_ORDER = [0, 1, 1, 0, 0, 1]; // 1-2-2-1 snake; 0 = first-pick team
const pct = (x: number) => `${Math.round(x * 100)}%`;

function pickSlotSequence(wePickFirst: boolean): { side: "us" | "them"; zone: Zone; index: number }[] {
  const seq: { side: "us" | "them"; zone: Zone; index: number }[] = [];
  const counts = { us: 0, them: 0 };
  for (const team of PICK_ORDER) {
    const side: "us" | "them" = (team === 0) === wePickFirst ? "us" : "them";
    const zone: Zone = side === "us" ? "our" : "their";
    seq.push({ side, zone, index: counts[side] });
    counts[side]++;
  }
  return seq;
}

function Bar({ label, value }: { label: string; value: number }) {
  const w = Math.max(2, Math.min(100, (value - 0.35) / 0.30 * 100));
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-14 text-right text-[var(--muted)] capitalize">{label}</span>
      <div className="flex-1 h-1.5 rounded bg-[#0c1119] overflow-hidden">
        <div className="h-full rounded" style={{ width: `${w}%`, background: value >= 0.5 ? "#3ec46d" : "#e0566f" }} />
      </div>
      <span className="w-9 tabular-nums text-[var(--muted)]">{value.toFixed(2)}</span>
    </div>
  );
}

function Avatar({ b, size = 56, dim, ring }: { b?: Brawler; size?: number; dim?: boolean; ring?: string }) {
  const border = ring || (b ? CLASS_COLOR[b.cls] || "#26303f" : "#26303f");
  if (!b) return <div style={{ width: size, height: size, borderColor: border }} className="rounded-lg bg-[#0c1119] border" />;
  return (
    <img src={b.image_url} alt={b.name} title={b.name} width={size} height={size}
      className="rounded-lg object-cover border"
      style={{ width: size, height: size, opacity: dim ? 0.3 : 1, borderColor: border }} />
  );
}

function RankWidget({ tag, setTag, rankInfo, loading, onCheck, bracket, onBracket, brackets }: {
  tag: string; setTag: (s: string) => void; rankInfo: RankInfo | null; loading: boolean;
  onCheck: () => void; bracket: string | null; onBracket: (b: string | null) => void; brackets: string[];
}) {
  const opts = bracket && !brackets.includes(bracket) ? [...brackets, bracket] : brackets;
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--panel)] px-4 py-3 mb-4 flex flex-wrap items-center gap-x-3 gap-y-2">
      <span className="text-sm font-semibold">🏅 Your rank</span>
      <div className="flex items-center gap-1.5">
        <input value={tag} onChange={(e) => setTag(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onCheck()} placeholder="#PLAYERTAG"
          className="bg-[var(--panel2)] border border-[var(--border)] rounded-md px-2.5 py-1.5 text-sm w-36 uppercase" />
        <button onClick={onCheck} disabled={loading || !tag.trim()}
          className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--panel2)] disabled:opacity-50">
          {loading ? "…" : "Check"}
        </button>
      </div>
      {rankInfo?.found && rankInfo.tier_label && (
        <span className="text-sm px-2.5 py-1 rounded-full font-semibold"
          style={{ background: "#e8c34a22", color: "#e8c34a" }}
          title={rankInfo.source === "live" ? "from a live lookup" : "from our match data"}>
          {rankInfo.tier_label}
        </span>
      )}
      {rankInfo && !rankInfo.found && (
        <span className="text-xs text-[var(--muted)]">{rankInfo.error || "not found"} — pick your bracket →</span>
      )}
      <label className="ml-auto text-xs text-[var(--muted)] flex items-center gap-1.5">
        Recommendations for
        <select value={bracket ?? ""} onChange={(e) => onBracket(e.target.value || null)}
          className="bg-[var(--panel2)] border border-[var(--border)] rounded-md px-2 py-1 text-sm">
          <option value="">All ranks</option>
          {opts.map((b) => <option key={b} value={b}>{b}</option>)}
        </select>
      </label>
    </div>
  );
}

function MetaBanner({ meta }: { meta: Meta }) {
  const buffs = meta.shifts.filter((s) => s.kind === "buff").slice(0, 3).map((s) => s.name);
  const nerfs = meta.shifts.filter((s) => s.kind === "nerf").slice(0, 3).map((s) => s.name);
  const parts: string[] = [];
  if (meta.new_brawlers.length) parts.push(`new: ${meta.new_brawlers.join(", ")}`);
  if (buffs.length) parts.push(`▲ ${buffs.join(", ")}`);
  if (nerfs.length) parts.push(`▼ ${nerfs.join(", ")}`);
  return (
    <div className="rounded-lg px-4 py-2.5 mb-4 flex items-center gap-3"
      style={{ background: "#e8c34a18", border: "1px solid #e8c34a55" }}>
      <span className="text-lg">📈</span>
      <div>
        <div className="font-semibold text-sm" style={{ color: "#e8c34a" }}>
          Meta shift detected — stats are catching up to the live meta
        </div>
        <div className="text-xs text-[var(--muted)]">{parts.join(" · ") || "recent balance change"}</div>
      </div>
    </div>
  );
}

function StatusBanner({ step }: { step: Step }) {
  const cfg =
    step.kind === "ban"
      ? { color: "#e0566f", icon: "🚫", text: `Ban phase — ban ${step.index} of ${BAN_N}`, sub: "Tap a brawler to ban it" }
      : step.kind === "done"
      ? { color: "#3ec46d", icon: "✓", text: "Draft complete", sub: "Your game plan is below" }
      : step.side === "us"
      ? { color: "#3b82f6", icon: "🔵", text: "Your pick", sub: "Tap a recommendation, or any brawler" }
      : { color: "#e0566f", icon: "🔴", text: "Enemy's pick", sub: "Tap the brawler they chose" };
  return (
    <div className="rounded-lg px-4 py-2.5 mb-4 flex items-center gap-3"
      style={{ background: cfg.color + "18", border: `1px solid ${cfg.color}55` }}>
      <span className="text-lg">{cfg.icon}</span>
      <div>
        <div className="font-semibold text-sm" style={{ color: cfg.color }}>{cfg.text}</div>
        <div className="text-xs text-[var(--muted)]">{cfg.sub}</div>
      </div>
    </div>
  );
}

export default function DraftBoard() {
  const [ref, setRef] = useState<Reference | null>(null);
  const [roster, setRoster] = useState<RosterResponse | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [tag, setTag] = useState("");
  const [rankInfo, setRankInfo] = useState<RankInfo | null>(null);
  const [bracket, setBracket] = useState<string | null>(null);
  const [rankLoading, setRankLoading] = useState(false);
  const [mapId, setMapId] = useState<number | null>(null);
  const [bans, setBans] = useState<(number | null)[]>(Array(BAN_N).fill(null));
  const [our, setOur] = useState<(number | null)[]>(Array(TEAM_N).fill(null));
  const [their, setTheir] = useState<(number | null)[]>(Array(TEAM_N).fill(null));
  const [wePickFirst, setWePickFirst] = useState(true);
  const [withBans, setWithBans] = useState(true);
  const [solo, setSolo] = useState(true);
  const [recs, setRecs] = useState<RecommendResponse | null>(null);
  const [warnings, setWarnings] = useState<Warning[]>([]);
  const [query, setQuery] = useState("");
  const [useSearch, setUseSearch] = useState(false);
  const [personalize, setPersonalize] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getReference().then((r) => {
      setRef(r);
      const best = [...r.maps].filter((m) => m.games > 0).sort((a, b) => b.games - a.games)[0] || r.maps[0];
      if (best) setMapId(best.id);
    }).catch((e) => setErr(String(e)));
    getRoster().then(setRoster).catch(() => {});
    getHealth().then(setHealth).catch(() => {});
    getMeta().then(setMeta).catch(() => {});
    const savedBracket = localStorage.getItem("bsdraft.bracket");
    if (savedBracket) setBracket(savedBracket);
    const savedTag = localStorage.getItem("bsdraft.tag");
    if (savedTag) {
      setTag(savedTag);
      getRank(savedTag).then((info) => {
        setRankInfo(info);
        if (info.found && info.bracket) setBracket(info.bracket);
      }).catch(() => {});
    }
  }, []);

  const byId = useMemo(() => {
    const m = new Map<number, Brawler>();
    ref?.brawlers.forEach((b) => m.set(b.id, b));
    return m;
  }, [ref]);
  const map = useMemo(() => ref?.maps.find((m) => m.id === mapId) || null, [ref, mapId]);
  const mode = map?.mode || "";
  const used = useMemo(
    () => new Set([...bans, ...our, ...their].filter((x): x is number => x != null)),
    [bans, our, their]
  );
  const ownedSet = useMemo(() => new Set((roster?.owned || []).map((o) => o.id)), [roster]);
  const personalizeReady = !!roster?.loaded;

  // --- guided draft sequence: ban×6 → 1-2-2-1 snake; current step = first empty in order ---
  const pickSeq = useMemo(() => pickSlotSequence(wePickFirst), [wePickFirst]);
  const step: Step = useMemo(() => {
    if (withBans) {
      const bi = bans.findIndex((x) => x == null);
      if (bi >= 0) return { kind: "ban", slot: { zone: "ban", index: bi }, index: bi + 1 };
    }
    for (let i = 0; i < pickSeq.length; i++) {
      const s = pickSeq[i];
      const arr = s.side === "us" ? our : their;
      if (arr[s.index] == null) return { kind: "pick", side: s.side, slot: { zone: s.zone, index: s.index }, n: BAN_N + i + 1 };
    }
    return { kind: "done" };
  }, [withBans, bans, our, their, pickSeq]);
  const phase: "ban" | "pick" = step.kind === "ban" ? "ban" : "pick";

  useEffect(() => {
    if (!mapId || !mode) return;
    const body = {
      map_id: mapId, mode,
      our_team: our.filter((x): x is number => x != null),
      their_team: their.filter((x): x is number => x != null),
      bans: bans.filter((x): x is number => x != null),
      we_pick_first: wePickFirst, solo_queue: solo, phase,
      use_search: useSearch, personalize: personalize && personalizeReady,
      rank_bracket: bracket, top: 12,
    };
    setLoading(true);
    const t = setTimeout(() => {
      recommend(body)
        .then((r) => { setRecs(r); setWarnings(r.warnings || []); })
        .catch((e) => setErr(String(e)))
        .finally(() => setLoading(false));
    }, 120);
    return () => clearTimeout(t);
  }, [mapId, mode, our, their, bans, wePickFirst, solo, phase, useSearch, personalize, personalizeReady, bracket]);

  const setZone = (zone: Zone, idx: number, val: number | null) => {
    const apply = (arr: (number | null)[]) => arr.map((x, i) => (i === idx ? val : x));
    if (zone === "ban") setBans(apply);
    else if (zone === "our") setOur(apply);
    else setTheir(apply);
  };

  // tap a brawler → fill the current step's slot and auto-advance
  const place = (bid: number) => {
    if (used.has(bid) || step.kind === "done") return;
    setZone(step.slot.zone, step.slot.index, bid);
  };

  const reset = () => {
    setBans(Array(BAN_N).fill(null));
    setOur(Array(TEAM_N).fill(null));
    setTheir(Array(TEAM_N).fill(null));
  };

  const selectBracket = (b: string | null) => {
    setBracket(b);
    if (b) localStorage.setItem("bsdraft.bracket", b);
    else localStorage.removeItem("bsdraft.bracket");
  };

  const checkRank = async () => {
    const t = tag.trim();
    if (!t) return;
    setRankLoading(true);
    try {
      const info = await getRank(t);
      setRankInfo(info);
      if (info.found) {
        setTag(info.tag);
        localStorage.setItem("bsdraft.tag", info.tag);
        if (info.bracket) selectBracket(info.bracket);
      }
    } catch {
      setRankInfo({ found: false, tag: t, tier: null, tier_label: null, bracket: null, source: null, error: "lookup failed" });
    } finally {
      setRankLoading(false);
    }
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (ref?.brawlers || [])
      .filter((b) => !q || b.name.toLowerCase().includes(q))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [ref, query]);

  const rotation = useMemo(
    () => (ref?.maps || []).filter((m) => m.games > 0).sort((a, b) =>
      a.mode === b.mode ? b.games - a.games : a.mode.localeCompare(b.mode)),
    [ref]
  );

  const isCurrent = (zone: Zone, index: number) =>
    step.kind !== "done" && step.slot.zone === zone && step.slot.index === index;

  const SlotBox = ({ zone, index, accent }: { zone: Zone; index: number; accent: string }) => {
    const arr = zone === "ban" ? bans : zone === "our" ? our : their;
    const bid = arr[index];
    const b = bid != null ? byId.get(bid) : undefined;
    const current = isCurrent(zone, index);
    return (
      <button
        onClick={() => bid != null && setZone(zone, index, null)}
        className="relative rounded-lg transition"
        style={{ boxShadow: current ? `0 0 0 2px ${accent}, 0 0 12px ${accent}88` : "none" }}
        title={b ? `${b.name} — tap to remove` : current ? "next slot — tap a brawler" : ""}
      >
        <Avatar b={b} size={zone === "ban" ? 44 : 60} dim={zone === "ban"} ring={current ? accent : undefined} />
        {bid == null && (
          <span className="absolute inset-0 grid place-items-center text-lg" style={{ color: current ? accent : "var(--muted)" }}>+</span>
        )}
      </button>
    );
  };

  const recTitle =
    step.kind === "ban" ? "Ban suggestions"
    : step.kind === "done" ? "Draft complete"
    : step.side === "us" ? "Your pick — suggestions"
    : "Strong picks here";

  return (
    <div className="min-h-screen p-4 md:p-6 max-w-6xl mx-auto">
      <header className="flex flex-wrap items-center gap-3 mb-5">
        <h1 className="text-xl font-bold tracking-tight mr-2">
          ⚔️ Brawl Draft <span className="text-[var(--muted)] font-normal text-sm">ranked assistant</span>
        </h1>
        <select
          value={mapId ?? ""} onChange={(e) => setMapId(Number(e.target.value))}
          className="bg-[var(--panel2)] border border-[var(--border)] rounded-md px-3 py-1.5 text-sm">
          {rotation.map((m) => (
            <option key={m.id} value={m.id}>{m.mode} — {m.name}</option>
          ))}
        </select>
        <button onClick={() => setWePickFirst((v) => !v)}
          className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--panel2)]">
          {wePickFirst ? "You pick 1st" : "Enemy picks 1st"}
        </button>
        <button onClick={() => setWithBans((v) => !v)}
          className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--panel2)]"
          title="Toggle the ban phase">
          {withBans ? "Bans: on" : "Bans: off"}
        </button>
        <button onClick={() => setSolo((v) => !v)}
          className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--panel2)]">
          {solo ? "Solo queue" : "Premade"}
        </button>
        <button onClick={() => setUseSearch((v) => !v)}
          className="text-sm px-3 py-1.5 rounded-md border bg-[var(--panel2)]"
          style={{ borderColor: useSearch ? "#3ec46d" : "var(--border)", color: useSearch ? "#3ec46d" : "var(--muted)" }}
          title="Seat-aware minimax lookahead over the remaining snake">
          🔮 Deep search {useSearch ? "on" : "off"}
        </button>
        <button onClick={() => personalizeReady && setPersonalize((v) => !v)}
          disabled={!personalizeReady}
          className="text-sm px-3 py-1.5 rounded-md border bg-[var(--panel2)] disabled:opacity-40"
          style={{ borderColor: personalize ? "#e8c34a" : "var(--border)", color: personalize ? "#e8c34a" : "var(--muted)" }}
          title={personalizeReady ? `Personalize to ${roster?.name}'s roster & mastery` : "no roster loaded"}>
          👤 {personalize && roster?.name ? roster.name : "Personalize"}{personalizeReady ? (personalize ? " ✓" : "") : " —"}
        </button>
        <span className="ml-auto text-xs text-[var(--muted)] flex items-center gap-1"
          title="Ranked matches in the live dataset — auto-refreshes every few minutes">
          {health != null && <>📊 <span className="tabular-nums">{health.matches.toLocaleString()}</span> matches analyzed</>}
        </span>
        <button onClick={reset} className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] text-[var(--muted)]">
          Reset
        </button>
      </header>

      <RankWidget tag={tag} setTag={setTag} rankInfo={rankInfo} loading={rankLoading}
        onCheck={checkRank} bracket={bracket} onBracket={selectBracket} brackets={ref?.brackets || []} />

      {err && <div className="mb-4 text-sm text-[#e0566f]">API error: {err}. Is the backend running on :8000?</div>}

      {meta?.shifted && <MetaBanner meta={meta} />}
      <StatusBanner step={step} />

      <div className="grid lg:grid-cols-[1fr_360px] gap-5">
        {/* LEFT: board + picker */}
        <div>
          <div className="rounded-xl border border-[var(--border)] bg-[var(--panel)] p-4 mb-4">
            {map && (
              <div className="flex items-center gap-3 mb-4">
                <img src={map.image_url} alt={map.name} className="w-16 h-16 rounded-lg object-cover border border-[var(--border)]" />
                <div>
                  <div className="font-semibold">{map.name}</div>
                  <div className="text-sm text-[var(--muted)]">{map.mode} · {map.games.toLocaleString()} games</div>
                </div>
              </div>
            )}
            {withBans && (
              <>
                <div className="text-xs uppercase tracking-wide text-[var(--muted)] mb-1.5">Bans</div>
                <div className="flex gap-2 mb-4">
                  {bans.map((_, i) => <SlotBox key={i} zone="ban" index={i} accent="#e0566f" />)}
                </div>
              </>
            )}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs uppercase tracking-wide mb-1.5" style={{ color: "#5aa0ff" }}>Your team</div>
                <div className="flex gap-2">{our.map((_, i) => <SlotBox key={i} zone="our" index={i} accent="#3b82f6" />)}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide mb-1.5" style={{ color: "#ff7a7a" }}>Enemy team</div>
                <div className="flex gap-2">{their.map((_, i) => <SlotBox key={i} zone="their" index={i} accent="#e0566f" />)}</div>
              </div>
            </div>
            {recs?.composition && Object.keys(recs.composition).length > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-4 pt-3 border-t border-[var(--border)]">
                <span className="text-xs text-[var(--muted)] mr-1">Your comp:</span>
                {Object.entries(recs.composition).map(([cls, n]) => (
                  <span key={cls} className="text-[11px] px-2 py-0.5 rounded-full"
                    style={{ background: (CLASS_COLOR[cls] || "#333") + "22", color: CLASS_COLOR[cls] || "#aaa" }}>
                    {cls}{n > 1 ? ` ×${n}` : ""}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-xl border border-[var(--border)] bg-[var(--panel)] p-4">
            <div className="flex items-center gap-2 mb-3">
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search brawlers…"
                className="flex-1 bg-[var(--panel2)] border border-[var(--border)] rounded-md px-3 py-2 text-sm" />
              {personalize && personalizeReady && <span className="text-xs text-[#e8c34a] whitespace-nowrap">owned only</span>}
            </div>
            <div className="grid grid-cols-6 sm:grid-cols-8 gap-2 max-h-[340px] overflow-y-auto pr-1">
              {filtered.map((b) => {
                const isUsed = used.has(b.id);
                const unowned = personalize && personalizeReady && !ownedSet.has(b.id);
                return (
                  <button key={b.id} onClick={() => place(b.id)} disabled={isUsed || unowned || step.kind === "done"}
                    className="flex flex-col items-center gap-1 group"
                    title={unowned ? `${b.name} — not owned` : `${b.name} (${b.cls})`}>
                    <Avatar b={b} size={48} dim={isUsed || unowned} />
                    <span className="text-[10px] truncate w-full text-center"
                      style={{ color: unowned ? "#566173" : "var(--muted)" }}>{b.name}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* RIGHT: warnings + recommendations */}
        <div className="rounded-xl border border-[var(--border)] bg-[var(--panel)] p-4 h-fit lg:sticky lg:top-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">
              {recTitle}
              {loading && <span className="text-xs text-[var(--muted)] font-normal ml-2">analyzing…</span>}
            </h2>
          </div>

          {warnings.length > 0 && (
            <div className="mb-3 space-y-1 rounded-lg bg-[var(--panel2)] p-2.5">
              {warnings.map((w, i) => (
                <div key={i} className="flex items-start gap-2 text-xs">
                  <span style={{ color: SEV_COLOR[w.severity] || "#888" }}>●</span>
                  <span className="text-[var(--muted)]">{w.text}</span>
                </div>
              ))}
            </div>
          )}

          <div className="space-y-2">
            {step.kind === "done" && <div className="text-sm text-[var(--muted)]">✓ Draft set — see your game plan below.</div>}
            {step.kind !== "done" && phase === "ban" &&
              (recs?.bans || []).map((r, i) => <BanCard key={r.brawler_id} r={r} i={i} b={byId.get(r.brawler_id)} onClick={() => place(r.brawler_id)} />)}
            {step.kind !== "done" && phase === "pick" &&
              (recs?.picks || []).map((r, i) => <PickCard key={r.brawler_id} r={r} i={i} b={byId.get(r.brawler_id)} onClick={() => place(r.brawler_id)} />)}
            {!recs && <div className="text-sm text-[var(--muted)]">Loading…</div>}
          </div>
        </div>
      </div>
      {recs?.game_plan && our.some((x) => x != null) && <GamePlanPanel gp={recs.game_plan} />}
      <footer className="text-center text-xs text-[var(--muted)] mt-6">
        Recommendations fuse a trained win-prob model with empirical map stats · not affiliated with Supercell
      </footer>
    </div>
  );
}

function ScoreBadge({ value, label }: { value: number; label: string }) {
  return (
    <div className="text-right">
      <div className="text-lg font-bold tabular-nums" style={{ color: value >= 0.5 ? "#3ec46d" : "#e8c34a" }}>{pct(value)}</div>
      <div className="text-[10px] text-[var(--muted)] -mt-1">{label}</div>
    </div>
  );
}

function PickCard({ r, i, b, onClick }: { r: PickRec; i: number; b?: Brawler; onClick: () => void }) {
  const lookahead = r.projected_winprob != null;
  return (
    <button onClick={onClick} className="w-full text-left rounded-lg border border-[var(--border)] bg-[var(--panel2)] p-2.5 hover:border-[#3b82f6] transition">
      <div className="flex items-center gap-2.5">
        <span className="text-[var(--muted)] text-sm w-4">{i + 1}</span>
        <Avatar b={b} size={40} />
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-sm truncate">{r.name}</div>
          <div className="text-[11px]" style={{ color: CLASS_COLOR[r.cls] || "#aaa" }}>{r.cls}</div>
        </div>
        <ScoreBadge value={lookahead ? (r.projected_winprob as number) : r.score} label={lookahead ? "🔮 lookahead" : "score"} />
      </div>
      <div className="mt-2 space-y-1">
        <Bar label="map" value={r.map_winrate} />
        {r.synergy != null && <Bar label="synergy" value={r.synergy} />}
        {r.counter != null && <Bar label="counter" value={r.counter} />}
        <Bar label="role" value={r.role_fit} />
        {r.win_prob != null && <Bar label="model" value={r.win_prob} />}
        {r.mastery != null && <Bar label="mastery" value={r.mastery} />}
      </div>
      <div className="mt-1.5 flex items-center gap-1.5 flex-wrap text-[10px] text-[var(--muted)]">
        <span>confidence {pct(r.confidence)}</span>
        {(r.gaps || []).map((g) => (
          <span key={g} className="px-1.5 py-0.5 rounded" style={{ background: "#e8843a22", color: "#e8a24a" }}>{g}</span>
        ))}
      </div>
    </button>
  );
}

function BanCard({ r, i, b, onClick }: { r: BanRec; i: number; b?: Brawler; onClick: () => void }) {
  return (
    <button onClick={onClick} className="w-full text-left rounded-lg border border-[var(--border)] bg-[var(--panel2)] p-2.5 hover:border-[#e0566f] transition">
      <div className="flex items-center gap-2.5">
        <span className="text-[var(--muted)] text-sm w-4">{i + 1}</span>
        <Avatar b={b} size={40} />
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-sm truncate">{r.name}</div>
          <div className="text-[11px]" style={{ color: CLASS_COLOR[r.cls] || "#aaa" }}>{r.cls}</div>
        </div>
        <ScoreBadge value={r.threat} label="threat" />
      </div>
      <div className="mt-2 space-y-1">
        <Bar label="win rate" value={r.map_winrate} />
        <div className="flex items-center gap-2 text-[11px]">
          <span className="w-14 text-right text-[var(--muted)]">use rate</span>
          <div className="flex-1 h-1.5 rounded bg-[#0c1119] overflow-hidden">
            <div className="h-full rounded bg-[#e8843a]" style={{ width: pct(Math.min(1, r.use_rate * 2)) }} />
          </div>
          <span className="w-9 tabular-nums text-[var(--muted)]">{pct(r.use_rate)}</span>
        </div>
      </div>
    </button>
  );
}

function GamePlanPanel({ gp }: { gp: GamePlan }) {
  const Section = ({ label, color, mark, items }: { label: string; color: string; mark: string; items: string[] }) =>
    items.length === 0 ? null : (
      <div>
        <div className="text-xs uppercase tracking-wide mb-1.5" style={{ color }}>{label}</div>
        <ul className="space-y-1">
          {items.map((t, i) => <li key={i} className="text-xs text-[var(--muted)]">{mark} {t}</li>)}
        </ul>
      </div>
    );
  return (
    <div className="mt-5 rounded-xl border border-[var(--border)] bg-[var(--panel)] p-5">
      <div className="flex items-center gap-2 mb-3">
        <h2 className="font-semibold">📋 Game plan</h2>
        <span className="text-[11px] px-2 py-0.5 rounded-full bg-[var(--panel2)] text-[var(--muted)]">{gp.archetype}</span>
      </div>
      <div className="rounded-lg p-3 mb-4" style={{ background: "#3b82f615", border: "1px solid #3b82f640" }}>
        <div className="text-[11px] uppercase tracking-wide mb-0.5" style={{ color: "#5aa0ff" }}>Win condition</div>
        <div className="text-sm">{gp.win_condition}</div>
        {gp.objective && <div className="text-xs text-[var(--muted)] mt-1">Objective: {gp.objective}</div>}
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <div className="text-xs uppercase tracking-wide text-[var(--muted)] mb-1.5">Your roles</div>
          <div className="space-y-1.5">
            {gp.roles.map((r) => (
              <div key={r.name} className="text-xs">
                <span className="font-semibold" style={{ color: CLASS_COLOR[r.cls] || "#aaa" }}>{r.name}</span>
                <span className="text-[var(--muted)]"> — {r.role}</span>
              </div>
            ))}
          </div>
          <div className="text-xs text-[var(--muted)] mt-2 italic">{gp.playstyle}</div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-[var(--muted)] mb-1.5">Vs their threats</div>
          <div className="space-y-1.5">
            {gp.threats.length === 0 && <div className="text-xs text-[var(--muted)]">No enemy picks on the board yet.</div>}
            {gp.threats.map((t) => (
              <div key={t.name} className="text-xs">
                <span className="font-semibold" style={{ color: CLASS_COLOR[t.cls] || "#aaa" }}>{t.name}</span>
                <span className="text-[var(--muted)]"> — {t.tip}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="grid md:grid-cols-3 gap-4 mt-4 pt-3 border-t border-[var(--border)]">
        <Section label="Do" color="#3ec46d" mark="✓" items={gp.tips} />
        <Section label="Avoid" color="#e0566f" mark="✕" items={gp.avoid} />
        <Section label="Compensate" color="#e8c34a" mark="⚠" items={gp.compensate} />
      </div>
    </div>
  );
}
