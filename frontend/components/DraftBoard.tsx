"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Brawler, PickRec, BanRec, Reference, RecommendResponse, Warning, RosterResponse, GamePlan,
  getReference, getRoster, recommend,
} from "@/lib/api";

const CLASS_COLOR: Record<string, string> = {
  Tank: "#e0566f", Assassin: "#b15be0", Controller: "#3b82f6", Marksman: "#3ec46d",
  Support: "#e8c34a", "Damage Dealer": "#e8843a", Artillery: "#39c3c0", Unclassified: "#6b7280",
};
const SEV_COLOR: Record<string, string> = { critical: "#e0566f", warn: "#e8c34a", info: "#5aa0ff" };

type Zone = "ban" | "our" | "their";
type Slot = { zone: Zone; index: number };

const BAN_N = 6, TEAM_N = 3;
const pct = (x: number) => `${Math.round(x * 100)}%`;

function Bar({ label, value }: { label: string; value: number }) {
  const w = Math.max(2, Math.min(100, (value - 0.35) / 0.30 * 100));
  const good = value >= 0.5;
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-14 text-right text-[var(--muted)] capitalize">{label}</span>
      <div className="flex-1 h-1.5 rounded bg-[#0c1119] overflow-hidden">
        <div className="h-full rounded" style={{ width: `${w}%`, background: good ? "#3ec46d" : "#e0566f" }} />
      </div>
      <span className="w-9 tabular-nums text-[var(--muted)]">{value.toFixed(2)}</span>
    </div>
  );
}

function Avatar({ b, size = 56, dim }: { b?: Brawler; size?: number; dim?: boolean }) {
  if (!b) return <div style={{ width: size, height: size }} className="rounded-lg bg-[#0c1119] border border-[var(--border)]" />;
  return (
    <img src={b.image_url} alt={b.name} title={b.name} width={size} height={size}
      className="rounded-lg object-cover border border-[var(--border)]"
      style={{ width: size, height: size, opacity: dim ? 0.3 : 1, borderColor: CLASS_COLOR[b.cls] || "#26303f" }} />
  );
}

export default function DraftBoard() {
  const [ref, setRef] = useState<Reference | null>(null);
  const [roster, setRoster] = useState<RosterResponse | null>(null);
  const [mapId, setMapId] = useState<number | null>(null);
  const [bans, setBans] = useState<(number | null)[]>(Array(BAN_N).fill(null));
  const [our, setOur] = useState<(number | null)[]>(Array(TEAM_N).fill(null));
  const [their, setTheir] = useState<(number | null)[]>(Array(TEAM_N).fill(null));
  const [active, setActive] = useState<Slot | null>(null);
  const [wePickFirst, setWePickFirst] = useState(true);
  const [solo, setSolo] = useState(true);
  const [phase, setPhase] = useState<"ban" | "pick">("pick");
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

  useEffect(() => {
    if (!mapId || !mode) return;
    const body = {
      map_id: mapId, mode,
      our_team: our.filter((x): x is number => x != null),
      their_team: their.filter((x): x is number => x != null),
      bans: bans.filter((x): x is number => x != null),
      we_pick_first: wePickFirst, solo_queue: solo, phase,
      use_search: useSearch, personalize: personalize && personalizeReady, top: 12,
    };
    setLoading(true);
    const t = setTimeout(() => {
      recommend(body)
        .then((r) => { setRecs(r); setWarnings(r.warnings || []); })
        .catch((e) => setErr(String(e)))
        .finally(() => setLoading(false));
    }, 120);
    return () => clearTimeout(t);
  }, [mapId, mode, our, their, bans, wePickFirst, solo, phase, useSearch, personalize, personalizeReady]);

  const setZone = (zone: Zone, idx: number, val: number | null) => {
    const apply = (arr: (number | null)[]) => arr.map((x, i) => (i === idx ? val : x));
    if (zone === "ban") setBans(apply);
    else if (zone === "our") setOur(apply);
    else setTheir(apply);
  };

  const firstEmpty = (arr: (number | null)[]) => arr.findIndex((x) => x == null);
  const defaultTarget = useCallback((): Slot | null => {
    if (phase === "ban") { const i = firstEmpty(bans); if (i >= 0) return { zone: "ban", index: i }; }
    const o = firstEmpty(our); if (o >= 0) return { zone: "our", index: o };
    const t = firstEmpty(their); if (t >= 0) return { zone: "their", index: t };
    const b = firstEmpty(bans); if (b >= 0) return { zone: "ban", index: b };
    return null;
  }, [phase, bans, our, their]);

  const place = (bid: number) => {
    if (used.has(bid)) return;
    const slot = active ?? defaultTarget();
    if (!slot) return;
    setZone(slot.zone, slot.index, bid);
    setActive(null);
  };

  const reset = () => {
    setBans(Array(BAN_N).fill(null)); setOur(Array(TEAM_N).fill(null));
    setTheir(Array(TEAM_N).fill(null)); setActive(null); setPhase("pick");
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

  const SlotBox = ({ zone, index, accent }: { zone: Zone; index: number; accent: string }) => {
    const arr = zone === "ban" ? bans : zone === "our" ? our : their;
    const bid = arr[index];
    const b = bid != null ? byId.get(bid) : undefined;
    const isActive = active?.zone === zone && active?.index === index;
    return (
      <button
        onClick={() => (bid != null ? setZone(zone, index, null) : setActive(isActive ? null : { zone, index }))}
        className="relative rounded-lg transition"
        style={{ outline: isActive ? `2px solid ${accent}` : "none", outlineOffset: 2 }}
        title={b ? `${b.name} — click to remove` : "click, then choose a brawler"}
      >
        <Avatar b={b} size={zone === "ban" ? 44 : 60} dim={zone === "ban"} />
        {bid == null && <span className="absolute inset-0 grid place-items-center text-[var(--muted)] text-lg">+</span>}
      </button>
    );
  };

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
        <div className="flex rounded-md overflow-hidden border border-[var(--border)] text-sm">
          {(["ban", "pick"] as const).map((p) => (
            <button key={p} onClick={() => setPhase(p)}
              className="px-3 py-1.5 capitalize"
              style={{ background: phase === p ? "var(--accent)" : "transparent", color: phase === p ? "#fff" : "var(--muted)" }}>
              {p}
            </button>
          ))}
        </div>
        <button onClick={() => setWePickFirst((v) => !v)}
          className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--panel2)]">
          {wePickFirst ? "We pick 1st" : "Enemy picks 1st"}
        </button>
        <button onClick={() => setSolo((v) => !v)}
          className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--panel2)]">
          {solo ? "Solo queue" : "Premade"}
        </button>
        <button onClick={() => setUseSearch((v) => !v)}
          className="text-sm px-3 py-1.5 rounded-md border bg-[var(--panel2)]"
          style={{ borderColor: useSearch ? "#3ec46d" : "var(--border)", color: useSearch ? "#3ec46d" : "var(--muted)" }}
          title="Seat-aware minimax lookahead over the remaining 1-2-2-1 snake">
          🔮 Deep search {useSearch ? "on" : "off"}
        </button>
        <button onClick={() => personalizeReady && setPersonalize((v) => !v)}
          disabled={!personalizeReady}
          className="text-sm px-3 py-1.5 rounded-md border bg-[var(--panel2)] disabled:opacity-40"
          style={{ borderColor: personalize ? "#e8c34a" : "var(--border)", color: personalize ? "#e8c34a" : "var(--muted)" }}
          title={personalizeReady ? `Personalize to ${roster?.name}'s roster & mastery` : "no roster loaded"}>
          👤 {personalize && roster?.name ? roster.name : "Personalize"}{personalizeReady ? (personalize ? " ✓" : "") : " —"}
        </button>
        <button onClick={reset} className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] text-[var(--muted)] ml-auto">
          Reset
        </button>
      </header>

      {err && <div className="mb-4 text-sm text-[#e0566f]">API error: {err}. Is the backend running on :8000?</div>}

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
            <div className="text-xs uppercase tracking-wide text-[var(--muted)] mb-1.5">Bans</div>
            <div className="flex gap-2 mb-4">
              {bans.map((_, i) => <SlotBox key={i} zone="ban" index={i} accent="#e0566f" />)}
            </div>
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
              {personalize && personalizeReady && (
                <span className="text-xs text-[#e8c34a] whitespace-nowrap">owned only</span>
              )}
            </div>
            <div className="grid grid-cols-6 sm:grid-cols-8 gap-2 max-h-[340px] overflow-y-auto pr-1">
              {filtered.map((b) => {
                const isUsed = used.has(b.id);
                const unowned = personalize && personalizeReady && !ownedSet.has(b.id);
                return (
                  <button key={b.id} onClick={() => place(b.id)} disabled={isUsed || unowned}
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
              {phase === "ban" ? "Ban suggestions" : "Pick suggestions"}
              {loading && <span className="text-xs text-[var(--muted)] font-normal ml-2">analyzing…</span>}
            </h2>
            {recs?.next_to_act && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-[var(--panel2)] text-[var(--muted)]">
                turn: {recs.next_to_act === "us" ? "you" : "enemy"}
              </span>
            )}
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
            {phase === "ban"
              ? (recs?.bans || []).map((r, i) => <BanCard key={r.brawler_id} r={r} i={i} b={byId.get(r.brawler_id)} onClick={() => place(r.brawler_id)} />)
              : (recs?.picks || []).map((r, i) => <PickCard key={r.brawler_id} r={r} i={i} b={byId.get(r.brawler_id)} onClick={() => place(r.brawler_id)} />)}
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
