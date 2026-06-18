"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Brawler, PickRec, BanRec, Reference, RecommendResponse, Warning, RosterResponse, GamePlan, Health, Meta, RankInfo, TopPick,
  getReference, getRoster, recommend, getHealth, getMeta, getRank, getTopPicks,
} from "@/lib/api";

const CLASS_COLOR: Record<string, string> = {
  Tank: "#e0566f", Assassin: "#b15be0", Controller: "#3b82f6", Marksman: "#3ec46d",
  Support: "#e8c34a", "Damage Dealer": "#e8843a", Artillery: "#39c3c0", Unclassified: "#6b7280",
};
const SEV_COLOR: Record<string, string> = { critical: "#e0566f", warn: "#e8c34a", info: "#5aa0ff" };
// Ranked tier accent colors, low → high (matched to the in-game rank emblems:
// Diamond cyan, Mythic purple, Legendary red, Masters brownish-red, Pro green)
const BRACKET_COLOR: Record<string, string> = {
  Bronze: "#c8814b", Silver: "#a7b6c8", Gold: "#e8c34a", Diamond: "#45d4e8",
  Mythic: "#b15be0", Legendary: "#e0566f", Masters: "#b9533a", Pro: "#34c759",
};
const bracketColor = (b?: string | null) => (b && BRACKET_COLOR[b]) || "#e8c34a";

const TIER_SUB: Record<string, number> = { I: 1, II: 2, III: 3 };
// "Legendary II" -> { name: "Legendary", sub: 2 };  "Pro" -> { name: "Pro", sub: 0 }
function splitTier(label: string): { name: string; sub: number } {
  const parts = label.trim().split(/\s+/);
  const last = parts[parts.length - 1];
  return TIER_SUB[last] ? { name: parts.slice(0, -1).join(" "), sub: TIER_SUB[last] } : { name: label, sub: 0 };
}

// stacked upward chevrons (military-rank insignia) marking the sub-tier 1–3
function TierChevrons({ n }: { n: number }) {
  return (
    <span className="inline-flex flex-col items-center" style={{ gap: 1 }} aria-hidden="true">
      {Array.from({ length: n }).map((_, i) => (
        <svg key={i} width="11" height="5" viewBox="0 0 11 5">
          <polyline points="1.5,4 5.5,1.4 9.5,4" fill="none" stroke="currentColor"
            strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ))}
    </span>
  );
}

// color-morphing "first pick" button — blue = you pick first, red = enemy first
function FirstPickToggle({ wePickFirst, onToggle }: { wePickFirst: boolean; onToggle: () => void }) {
  const c = wePickFirst
    ? { bg: "#2e8bf0", glow: "rgba(46,139,240,0.6)", side: "You" }
    : { bg: "#ef4d4d", glow: "rgba(239,77,77,0.6)", side: "Enemy" };
  return (
    <button onClick={onToggle}
      className="firstpick-btn shine ml-auto shrink-0 relative overflow-hidden rounded-lg pl-3 pr-2 py-2 inline-flex items-center gap-2 text-white"
      style={{ backgroundColor: c.bg, boxShadow: `0 8px 20px -8px ${c.glow}, inset 0 1px 0 rgba(255,255,255,0.28)` }}
      title="Who picks first — click to switch"
      aria-label={`First pick: ${c.side}. Click to switch.`}>
      <span className="text-[11px] font-extrabold tracking-[0.16em] uppercase whitespace-nowrap">First pick</span>
      <span className="text-[11px] font-bold leading-none w-14 text-center py-1 rounded-md bg-white/20">{c.side}</span>
    </button>
  );
}

type Zone = "ban" | "our" | "their";
type Slot = { zone: Zone; index: number };
type Step =
  | { kind: "ban"; slot: Slot; index: number }
  | { kind: "pick"; side: "us" | "them"; slot: Slot; n: number }
  | { kind: "done" };

const BAN_N = 6, TEAM_N = 3;
const ROSTER_POLL_MS = 5 * 60 * 1000; // re-check the player's roster/inventory every 5 min
const PICK_ORDER = [0, 1, 1, 0, 0, 1]; // 1-2-2-1 snake; 0 = first-pick team
const pct = (x: number) => `${Math.round(x * 100)}%`;
const cssVars = (vars: Record<string, string>) => vars as React.CSSProperties;

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

// eased number that animates from 0 (on mount) or its previous value toward `target`
function useCountUp(target: number, duration = 650) {
  const [val, setVal] = useState(0);
  const fromRef = useRef(0);
  const rafRef = useRef(0);
  useEffect(() => {
    const from = fromRef.current;
    let startTs = 0;
    cancelAnimationFrame(rafRef.current);
    const tick = (now: number) => {
      if (!startTs) startTs = now;
      const t = Math.min(1, (now - startTs) / duration);
      const e = 1 - Math.pow(1 - t, 3); // easeOutCubic
      setVal(from + (target - from) * e);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
      else fromRef.current = target;
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, duration]);
  return val;
}

function CountUp({ value, className }: { value: number; className?: string }) {
  const v = useCountUp(value);
  return <span className={className}>{Math.round(v).toLocaleString()}</span>;
}

function Bar({ label, value }: { label: string; value: number }) {
  const w = Math.max(2, Math.min(100, (value - 0.35) / 0.30 * 100));
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-14 text-right text-[var(--muted)] capitalize">{label}</span>
      <div className="flex-1 h-1.5 rounded bg-[#0c1119] overflow-hidden">
        <div className="h-full rounded bar-fill" style={{ width: `${w}%`, background: value >= 0.5 ? "#3ec46d" : "#e0566f" }} />
      </div>
      <span className="w-9 tabular-nums text-[var(--muted)]">{value.toFixed(2)}</span>
    </div>
  );
}

function Avatar({ b, size = 56, dim, ring, className }: { b?: Brawler; size?: number; dim?: boolean; ring?: string; className?: string }) {
  const border = ring || (b ? CLASS_COLOR[b.cls] || "#26303f" : "#26303f");
  if (!b) return <div style={{ width: size, height: size, borderColor: border }} className={`rounded-lg bg-[#0c1119] border ${className || ""}`} />;
  return (
    <img src={b.image_url} alt={b.name} title={b.name} width={size} height={size}
      className={`rounded-lg object-cover border ${className || ""}`}
      style={{ width: size, height: size, opacity: dim ? 0.3 : 1, borderColor: border }} />
  );
}

function RankWidget({ tag, setTag, rankInfo, loading, onCheck }: {
  tag: string; setTag: (s: string) => void; rankInfo: RankInfo | null; loading: boolean; onCheck: () => void;
}) {
  return (
    <div className="rounded-xl glass backdrop-blur-xl backdrop-saturate-150 px-4 py-3 mb-4 flex flex-wrap items-center gap-x-3 gap-y-2 anim-fade-up" style={{ animationDelay: "60ms" }}>
      <div className="flex items-center gap-1.5">
        <input value={tag} onChange={(e) => setTag(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onCheck()} placeholder="#PLAYERTAG"
          className="bg-[var(--panel2)] border border-[var(--border)] rounded-md px-2.5 py-1.5 text-sm w-36 uppercase outline-none focus:border-[var(--accent)] ctl" />
        <button onClick={onCheck} disabled={loading || !tag.trim()}
          className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--panel2)] disabled:opacity-50 ctl">
          {loading ? "…" : "Enter ↵"}
        </button>
      </div>
      {rankInfo?.found && rankInfo.tier_label && (() => {
        const c = bracketColor(rankInfo.bracket);
        const { name, sub } = splitTier(rankInfo.tier_label!);
        return (
          <span className="ml-auto text-sm px-2.5 py-1 rounded-full font-semibold shine anim-pop inline-flex items-center gap-1.5"
            style={{ background: c + "22", color: c, boxShadow: `0 0 18px -6px ${c}` }}
            aria-label={rankInfo.tier_label!}
            title={(rankInfo.tier_label! + (rankInfo.source === "live" ? " — from a live lookup" : " — from our match data"))}>
            {name}
            {sub > 0 && <TierChevrons n={sub} />}
          </span>
        );
      })()}
      {rankInfo && !rankInfo.found && (
        <span className="ml-auto text-xs text-[var(--muted)]">{rankInfo.error || "tag not found"}</span>
      )}
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
    <div className="relative overflow-hidden rounded-lg px-4 py-2.5 mb-4 flex items-center gap-3 anim-fade-up"
      style={{ background: "#e8c34a18", border: "1px solid #e8c34a55" }}>
      <span className="absolute left-0 top-0 bottom-0 w-[3px]" style={{ background: "#e8c34a" }} />
      <span className="text-lg floaty">📈</span>
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
    <div className="relative overflow-hidden rounded-lg px-4 py-2.5 mb-4 flex items-center gap-3 anim-fade-up"
      style={{ background: cfg.color + "18", border: `1px solid ${cfg.color}55` }}>
      <span className="absolute left-0 top-0 bottom-0 w-[3px] slot-current" style={cssVars({ "--ring": cfg.color, background: cfg.color })} />
      <span className="text-lg floaty">{cfg.icon}</span>
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
  const [rankLoading, setRankLoading] = useState(false);
  const [mapId, setMapId] = useState<number | null>(null);
  const [bans, setBans] = useState<(number | null)[]>(Array(BAN_N).fill(null));
  const [our, setOur] = useState<(number | null)[]>(Array(TEAM_N).fill(null));
  const [their, setTheir] = useState<(number | null)[]>(Array(TEAM_N).fill(null));
  const [wePickFirst, setWePickFirst] = useState(true);
  const [activeOverride, setActiveOverride] = useState<Slot | null>(null);
  const [solo, setSolo] = useState(true);
  const [recs, setRecs] = useState<RecommendResponse | null>(null);
  const [topPicks, setTopPicks] = useState<TopPick[]>([]);
  const [railOk, setRailOk] = useState(true);  // hide the rail if the endpoint isn't available
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
    getHealth().then(setHealth).catch(() => {});
    getMeta().then(setMeta).catch(() => {});
    const savedTag = localStorage.getItem("bsdraft.tag");
    if (savedTag) {
      setTag(savedTag);
      getRank(savedTag).then(setRankInfo).catch(() => {});
    }
  }, []);

  // The roster (owned brawlers + loadout/mastery) only changes when the player unlocks or
  // upgrades something, so a one-shot fetch can go stale in a long session. Re-poll it on an
  // interval and when the tab regains focus; the backend caches it briefly, so this stays
  // cheap on the live API.
  useEffect(() => {
    let cancelled = false;
    let last = 0;
    const pull = () => {
      last = Date.now();
      getRoster().then((r) => { if (!cancelled) setRoster(r); }).catch(() => {});
    };
    pull();
    const id = setInterval(pull, ROSTER_POLL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible" && Date.now() - last > 60_000) pull();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => { cancelled = true; clearInterval(id); document.removeEventListener("visibilitychange", onVisible); };
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
  // recommendations are tuned to the player's own rank bracket (null = all ranks)
  const bracket = rankInfo?.found ? rankInfo.bracket : null;
  // once the player's tag is confirmed, fold in their own win rates (resolved from our data,
  // so it works without an API key); null leaves picks at the population meta.
  const personalTag = rankInfo?.found ? rankInfo.tag : null;

  // --- draft order: ban×6 → 1-2-2-1 snake (by first pick). The active slot is the
  //     slot the user clicked if it's still empty, else the next empty slot in order. ---
  const pickSeq = useMemo(() => pickSlotSequence(wePickFirst), [wePickFirst]);
  const order = useMemo<Slot[]>(() => [
    ...Array.from({ length: BAN_N }, (_, i): Slot => ({ zone: "ban", index: i })),
    ...pickSeq.map((s): Slot => ({ zone: s.zone, index: s.index })),
  ], [pickSeq]);
  const active = useMemo<Slot | null>(() => {
    const empty = (s: Slot) => (s.zone === "ban" ? bans : s.zone === "our" ? our : their)[s.index] == null;
    if (activeOverride && empty(activeOverride)) return activeOverride;
    return order.find(empty) ?? null;
  }, [activeOverride, order, bans, our, their]);
  const step: Step = useMemo(() => {
    if (!active) return { kind: "done" };
    if (active.zone === "ban") return { kind: "ban", slot: active, index: active.index + 1 };
    return { kind: "pick", side: active.zone === "our" ? "us" : "them", slot: active, n: 0 };
  }, [active]);
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
      personal_tag: personalTag,
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
  }, [mapId, mode, our, their, bans, wePickFirst, solo, phase, useSearch, personalize, personalizeReady, bracket, personalTag]);

  // "Full-loadout" rail: strongest brawlers on this map in a vacuum (empty board, no roster).
  // Independent of the live draft and personalization, so it only refetches on map/bracket.
  useEffect(() => {
    if (!mapId || !mode) return;
    let cancelled = false;
    getTopPicks(mapId, mode, bracket)
      .then((r) => { if (!cancelled) { setTopPicks(r.picks); setRailOk(true); } })
      .catch(() => { if (!cancelled) setRailOk(false); });
    return () => { cancelled = true; };
  }, [mapId, mode, bracket]);

  const setZone = (zone: Zone, idx: number, val: number | null) => {
    const apply = (arr: (number | null)[]) => arr.map((x, i) => (i === idx ? val : x));
    if (zone === "ban") setBans(apply);
    else if (zone === "our") setOur(apply);
    else setTheir(apply);
  };

  // pick a brawler → fill the active slot, then fall back to next-in-order
  const place = (bid: number) => {
    if (!active || used.has(bid)) return;
    setZone(active.zone, active.index, bid);
    setActiveOverride(null);
  };

  const reset = () => {
    setBans(Array(BAN_N).fill(null));
    setOur(Array(TEAM_N).fill(null));
    setTheir(Array(TEAM_N).fill(null));
    setActiveOverride(null);
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
        onClick={() => {
          if (bid != null) setZone(zone, index, null);
          setActiveOverride({ zone, index });
        }}
        className={`relative shrink-0 rounded-lg transition ${current ? "slot-current" : ""}`}
        style={current ? cssVars({ "--ring": accent }) : undefined}
        title={b ? `${b.name} — click to replace` : "click to pick this slot"}
      >
        <span key={bid ?? "empty"} className="block anim-pop">
          <Avatar b={b} size={zone === "ban" ? 44 : 60} dim={zone === "ban"} ring={current ? accent : undefined} />
        </span>
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
      <header className="flex flex-wrap items-center gap-3 mb-5 anim-fade-up">
        <h1 className="text-xl font-bold tracking-tight mr-2">
          <span className="inline-block floaty mr-1">⚔️</span>
          <span className="brand-gradient">Brawl Draft</span>{" "}
          <span className="text-[var(--muted)] font-normal text-sm">ranked assistant</span>
        </h1>
        <select
          value={mapId ?? ""} onChange={(e) => setMapId(Number(e.target.value))}
          className="bg-[var(--panel2)] border border-[var(--border)] rounded-md px-3 py-1.5 text-sm ctl">
          {rotation.map((m) => (
            <option key={m.id} value={m.id}>{m.mode} — {m.name}</option>
          ))}
        </select>
        <button onClick={() => setSolo((v) => !v)}
          className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--panel2)] ctl">
          {solo ? "Solo queue" : "Premade"}
        </button>
        <button onClick={() => setUseSearch((v) => !v)}
          className="text-sm px-3 py-1.5 rounded-md border bg-[var(--panel2)] ctl"
          style={{ borderColor: useSearch ? "#3ec46d" : "var(--border)", color: useSearch ? "#3ec46d" : "var(--muted)", boxShadow: useSearch ? "0 0 16px -5px #3ec46d, inset 0 0 0 1px #3ec46d33" : undefined }}
          title="Seat-aware minimax lookahead over the remaining snake">
          🔮 Deep search {useSearch ? "on" : "off"}
        </button>
        <button onClick={() => personalizeReady && setPersonalize((v) => !v)}
          disabled={!personalizeReady}
          className="text-sm px-3 py-1.5 rounded-md border bg-[var(--panel2)] disabled:opacity-40 ctl"
          style={{ borderColor: personalize ? "#e8c34a" : "var(--border)", color: personalize ? "#e8c34a" : "var(--muted)", boxShadow: personalize ? "0 0 16px -5px #e8c34a, inset 0 0 0 1px #e8c34a33" : undefined }}
          title={personalizeReady ? `Personalize to ${roster?.name}'s roster & mastery` : "no roster loaded"}>
          👤 {personalize && roster?.name ? roster.name : "Personalize"}{personalizeReady ? (personalize ? " ✓" : "") : " —"}
        </button>
        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs text-[var(--muted)] flex items-center gap-1"
            title="Ranked matches in the live dataset — auto-refreshes every few minutes">
            {health != null && <>📊 <CountUp value={health.matches} className="tabular-nums text-[var(--text)] font-semibold" /> matches analyzed</>}
          </span>
          <button onClick={reset} className="text-sm px-3 py-1.5 rounded-md border border-[var(--border)] text-[var(--muted)] ctl">
            Reset
          </button>
        </div>
      </header>

      <RankWidget tag={tag} setTag={setTag} rankInfo={rankInfo} loading={rankLoading} onCheck={checkRank} />

      {err && <div className="mb-4 text-sm text-[#e0566f]">API error: {err}. Is the backend running on :8000?</div>}

      {meta?.shifted && <MetaBanner meta={meta} />}
      <StatusBanner key={step.kind === "pick" ? `pick-${step.side}-${step.slot.index}` : step.kind} step={step} />

      <div className={`grid gap-5 ${railOk ? "lg:grid-cols-[1fr_360px_auto]" : "lg:grid-cols-[1fr_360px]"}`}>
        {/* LEFT: board + picker */}
        <div>
          <div className="rounded-xl glass backdrop-blur-xl backdrop-saturate-150 p-4 mb-4 anim-fade-up" style={{ animationDelay: "120ms" }}>
            <div className="flex items-center gap-3 mb-4">
              {map && (
                <>
                  <img src={map.image_url} alt={map.name} className="w-16 h-16 rounded-lg object-cover border border-[var(--border)]" />
                  <div>
                    <div className="font-semibold">{map.name}</div>
                    <div className="text-sm text-[var(--muted)]">{map.mode} · {map.games.toLocaleString()} games</div>
                  </div>
                </>
              )}
              <FirstPickToggle wePickFirst={wePickFirst} onToggle={() => setWePickFirst((v) => !v)} />
            </div>
            <div className="text-xs uppercase tracking-wide text-[var(--muted)] mb-1.5">Bans</div>
            <div className="flex flex-wrap gap-2 mb-4">
              {bans.map((_, i) => <SlotBox key={i} zone="ban" index={i} accent="#e0566f" />)}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
                  <span key={cls} className="text-[11px] px-2 py-0.5 rounded-full anim-pop"
                    style={{ background: (CLASS_COLOR[cls] || "#333") + "22", color: CLASS_COLOR[cls] || "#aaa" }}>
                    {cls}{n > 1 ? ` ×${n}` : ""}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-xl glass backdrop-blur-xl backdrop-saturate-150 p-4 anim-fade-up" style={{ animationDelay: "180ms" }}>
            <div className="flex items-center gap-2 mb-3">
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search brawlers…"
                className="flex-1 bg-[var(--panel2)] border border-[var(--border)] rounded-md px-3 py-2 text-sm outline-none focus:border-[var(--accent)] ctl" />
              {personalize && personalizeReady && <span className="text-xs text-[#e8c34a] whitespace-nowrap">owned only</span>}
            </div>
            <div className="grid grid-cols-6 sm:grid-cols-8 gap-2 max-h-[340px] overflow-y-auto pr-1">
              {filtered.map((b) => {
                const isUsed = used.has(b.id);
                const unowned = personalize && personalizeReady && !ownedSet.has(b.id);
                return (
                  <button key={b.id} onClick={() => place(b.id)} disabled={isUsed || unowned || step.kind === "done"}
                    className="flex flex-col items-center gap-1 group disabled:cursor-not-allowed"
                    title={unowned ? `${b.name} — not owned` : `${b.name} (${b.cls})`}>
                    <span className="tile-img" style={cssVars({ "--c": CLASS_COLOR[b.cls] || "#26303f" })}>
                      <Avatar b={b} size={48} dim={isUsed || unowned} />
                    </span>
                    <span className="text-[10px] truncate w-full text-center"
                      style={{ color: unowned ? "#566173" : "var(--muted)" }}>{b.name}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* RIGHT: warnings + recommendations */}
        <div className="rounded-xl glass backdrop-blur-xl backdrop-saturate-150 p-4 h-fit lg:sticky lg:top-4 anim-fade-up" style={{ animationDelay: "240ms" }}>
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">
              {recTitle}
              {loading && <span className="text-xs text-[var(--muted)] font-normal ml-2 animate-pulse">analyzing…</span>}
            </h2>
          </div>

          {warnings.length > 0 && (
            <div className="mb-3 space-y-1 rounded-lg bg-[var(--panel2)] p-2.5 anim-fade-in">
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
            {!recs && [0, 1, 2, 3].map((i) => <SkeletonCard key={i} />)}
          </div>
        </div>

        {/* RAIL: full-loadout meta top 10 (vacuum, icons only) — hidden if the endpoint is unavailable */}
        {railOk && (
          <TopPicksRail picks={topPicks} byId={byId} used={used}
            onPick={place} disabled={step.kind === "done"} />
        )}
      </div>
      {recs?.game_plan && our.some((x) => x != null) && <GamePlanPanel gp={recs.game_plan} />}
      <footer className="text-center text-xs text-[var(--muted)] mt-6">
        Recommendations fuse a trained win-prob model with empirical map stats · not affiliated with Supercell
      </footer>
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--panel2)] p-2.5">
      <div className="flex items-center gap-2.5">
        <div className="w-10 h-10 rounded-lg skeleton" />
        <div className="flex-1 space-y-2">
          <div className="h-3 w-24 rounded skeleton" />
          <div className="h-2 w-14 rounded skeleton" />
        </div>
        <div className="h-5 w-10 rounded skeleton" />
      </div>
      <div className="mt-3 space-y-1.5">
        <div className="h-1.5 w-full rounded skeleton" />
        <div className="h-1.5 w-3/4 rounded skeleton" />
      </div>
    </div>
  );
}

function ScoreBadge({ value, label }: { value: number; label: string }) {
  const shown = useCountUp(value);
  const color = value >= 0.5 ? "#3ec46d" : "#e8c34a";
  return (
    <div className="text-right">
      <div className="text-lg font-bold tabular-nums" style={{ color, textShadow: `0 0 18px ${color}55` }}>{pct(shown)}</div>
      <div className="text-[10px] text-[var(--muted)] -mt-1">{label}</div>
    </div>
  );
}

function PickCard({ r, i, b, onClick }: { r: PickRec; i: number; b?: Brawler; onClick: () => void }) {
  const lookahead = r.projected_winprob != null;
  return (
    <button onClick={onClick}
      className="w-full text-left rounded-lg border border-[var(--border)] bg-[var(--panel2)] p-2.5 hover:border-[#3b82f6] card-rec anim-fade-up"
      style={cssVars({ "--glow": "rgba(59,130,246,0.45)", animationDelay: `${i * 45}ms` })}>
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
        {r.personal_winrate != null && <Bar label="you" value={r.personal_winrate} />}
      </div>
      <div className="mt-1.5 flex items-center gap-1.5 flex-wrap text-[10px] text-[var(--muted)]">
        <span>confidence {pct(r.confidence)}</span>
        {r.personal_games != null && (
          <span className="px-1.5 py-0.5 rounded" style={{ background: "#5aa0ff22", color: "#7fb4ff" }}
            title="your recent ranked games with this brawler (recency-weighted)">
            you · {Math.round(r.personal_games)}g
          </span>
        )}
        {(r.gaps || []).map((g) => (
          <span key={g} className="px-1.5 py-0.5 rounded" style={{ background: "#e8843a22", color: "#e8a24a" }}>{g}</span>
        ))}
      </div>
    </button>
  );
}

function BanCard({ r, i, b, onClick }: { r: BanRec; i: number; b?: Brawler; onClick: () => void }) {
  return (
    <button onClick={onClick}
      className="w-full text-left rounded-lg border border-[var(--border)] bg-[var(--panel2)] p-2.5 hover:border-[#e0566f] card-rec anim-fade-up"
      style={cssVars({ "--glow": "rgba(224,86,111,0.45)", animationDelay: `${i * 45}ms` })}>
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
            <div className="h-full rounded bar-fill bg-[#e8843a]" style={{ width: pct(Math.min(1, r.use_rate * 2)) }} />
          </div>
          <span className="w-9 tabular-nums text-[var(--muted)]">{pct(r.use_rate)}</span>
        </div>
      </div>
    </button>
  );
}

// Skinny, icons-only rail of the map's strongest brawlers in a vacuum — judged at a full
// loadout (all gadgets, gears & star powers) with no roster filter. Stable across the draft;
// a constant "who's generally strong here" reference beside the live, personalized picks.
function TopPicksRail({ picks, byId, used, onPick, disabled }: {
  picks: TopPick[]; byId: Map<number, Brawler>; used: Set<number>;
  onPick: (id: number) => void; disabled: boolean;
}) {
  const blurb =
    "The strongest brawlers on this map if you owned every brawler at a full loadout — " +
    "all gadgets, gears & star powers. A general meta tier list, independent of the live " +
    "draft and your roster.";
  return (
    <div className="rounded-xl glass backdrop-blur-xl backdrop-saturate-150 p-2 h-fit lg:sticky lg:top-4 anim-fade-up"
      style={{ animationDelay: "300ms" }}>
      <div className="text-center mb-2 px-0.5 cursor-help" title={blurb}>
        <div className="text-base leading-none floaty">👑</div>
        <div className="text-[9px] uppercase tracking-wide text-[var(--muted)] leading-tight mt-0.5">
          Top 10<br />full loadout
        </div>
      </div>
      <div className="flex flex-row flex-wrap justify-center gap-1.5 lg:flex-col lg:items-center">
        {picks.length === 0
          ? [0, 1, 2, 3, 4].map((i) => <div key={i} className="w-12 h-12 rounded-lg skeleton" />)
          : picks.map((p, i) => {
              const b = byId.get(p.brawler_id);
              const isUsed = used.has(p.brawler_id);
              return (
                <button key={p.brawler_id} onClick={() => onPick(p.brawler_id)}
                  disabled={isUsed || disabled}
                  className="group anim-pop disabled:cursor-not-allowed"
                  style={cssVars({ animationDelay: `${i * 40}ms` })}
                  title={`#${i + 1}  ${p.name}\n${pct(p.score)} pick score · ${pct(p.map_winrate)} map win rate\nassumes a full loadout (all gadgets, gears & star powers)`}>
                  <span className="tile-img" style={cssVars({ "--c": CLASS_COLOR[p.cls] || "#26303f" })}>
                    <Avatar b={b} size={48} dim={isUsed} />
                  </span>
                </button>
              );
            })}
      </div>
    </div>
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
    <div className="mt-5 rounded-xl glass backdrop-blur-xl backdrop-saturate-150 p-5 anim-fade-up">
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
