"use client";

// "What to upgrade next" — enters a player tag, fetches their live roster (owned items + power
// levels via the keyed roster tunnel), and ranks the highest-value purchases they haven't made:
// power-11 climbs, gadgets, star powers, gears, hypercharges, and new-brawler unlocks.
// The inverse of the board's loadout popover — it surfaces the best UNOWNED item rather than
// locking it. Scored by meta strength × purchase impact on the backend (see engine/purchases.py);
// cost is shown as context only (the API can't see your currency balances).
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getReference, getRoster, getPurchases, type Brawler, type PurchaseRec, type PurchaseKind,
} from "@/lib/api";
import Logo from "@/components/Logo";

const NAV = [
  { href: "/", label: "Draft board" },
  { href: "/purchases", label: "Upgrades" },
  { href: "/guide", label: "Draft guide" },
  { href: "/how-it-works", label: "How it works" },
  { href: "/faq", label: "FAQ" },
];

const KIND: Record<PurchaseKind, { label: string; color: string; glyph: string }> = {
  new_brawler:   { label: "UNLOCK",       color: "var(--blue)",   glyph: "◈" },
  hypercharge:   { label: "HYPERCHARGE",  color: "var(--gold)",   glyph: "⚡" },
  star_power:    { label: "STAR POWER",   color: "var(--accent)", glyph: "★" },
  gadget:        { label: "GADGET",       color: "var(--green)",  glyph: "⚙" },
  gear:          { label: "GEAR",         color: "var(--muted)",  glyph: "⛭" },
  power_upgrade: { label: "POWER",        color: "var(--accent)", glyph: "▲" },
};

const COST_LABEL: Record<string, string> = {
  coins: "Coins", power_points: "Power Points", credits: "Credits",
};

const CONF: Record<PurchaseRec["confidence"], { label: string; color: string; title: string }> = {
  measured:         { label: "MEASURED", color: "var(--green)",
                      title: "Backed by a measured item win-rate from match data" },
  heuristic:        { label: "ESTIMATE", color: "var(--muted)",
                      title: "Ranked by the brawler's meta strength × a purchase-impact prior" },
  eligibility_only: { label: "ELIGIBLE", color: "var(--gold)",
                      title: "A high-value slot you're eligible to fill — no measured value model yet" },
};

// Roster fetch fails with raw operator text; translate to something a visitor can act on.
function rosterFailReason(error: string): string {
  const e = error.toLowerCase();
  if (e.includes("404") || e.includes("not found")) return "no player with that tag — check it and load again";
  if (e.includes("429")) return "roster service is busy — try again in a moment";
  if (e.includes("403") || e.includes("auth/ip") || e.includes("no api token") || e.includes("no player tag"))
    return "the roster service is down right now — try again later";
  return "couldn't reach the roster service — try again later";
}

const pct = (v: number) => `${Math.round(v * 100)}%`;
const num = (n: number) => n.toLocaleString();

function DocNav({ current }: { current: string }) {
  return (
    <nav className="panel flex flex-wrap items-center gap-x-1 gap-y-1 px-3 py-2 mb-6">
      <a href="/" className="flex items-center gap-2 mr-3" aria-label="Brawl Draft home">
        <Logo size={22} />
        <span className="brand-gradient text-[14px]">BRAWL DRAFT</span>
      </a>
      <div className="flex flex-wrap gap-x-1 ml-auto">
        {NAV.map((n) => {
          const on = n.href === current;
          return (
            <a key={n.href} href={n.href} aria-current={on ? "page" : undefined}
              className="mono text-[11px] tracking-[0.06em] uppercase px-2.5 py-1.5 border ctl"
              style={on
                ? { color: "var(--text)", borderColor: "var(--accent)", boxShadow: "inset 0 0 0 1px color-mix(in srgb, var(--accent) 40%, transparent)" }
                : { color: "var(--muted)", borderColor: "transparent" }}>
              {n.label}
            </a>
          );
        })}
      </div>
    </nav>
  );
}

function ValueMeter({ v }: { v: number }) {
  // Scores land roughly in 0.1..0.6; scale so the strongest recs nearly fill the bar.
  const w = Math.max(6, Math.min(100, (v / 0.6) * 100));
  return <div className="meter w-full"><i style={{ width: `${w}%`, background: "var(--gold)" }} /></div>;
}

function CostChips({ cost, gate }: { cost: Record<string, number>; gate: string | null }) {
  const entries = Object.entries(cost).filter(([, n]) => n > 0);
  if (!entries.length && !gate) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5 mt-2">
      {entries.map(([k, n]) => (
        <span key={k} className="mono text-[10px] px-1.5 py-0.5 border border-[var(--line)] text-[var(--muted)] tabular-nums">
          {num(n)} {COST_LABEL[k] || k}
        </span>
      ))}
      {gate && (
        <span className="mono text-[10px] px-1.5 py-0.5 border tabular-nums"
          style={{ borderColor: "color-mix(in srgb, var(--gold) 45%, transparent)", color: "var(--gold)" }}
          title="This purchase unlocks at a higher power level — the climb is included in the cost above.">
          ▲ {gate}
        </span>
      )}
    </div>
  );
}

function RecCard({ r, rank, b }: { r: PurchaseRec; rank: number; b?: Brawler }) {
  const kind = KIND[r.kind];
  const conf = CONF[r.confidence];
  return (
    <div className="card-rec panel flex gap-3 p-3 anim-rise" style={{ "--glow": kind.color } as React.CSSProperties}>
      <div className="mono text-[13px] tabular-nums text-[var(--dim)] w-5 shrink-0 pt-1 text-right">{rank}</div>
      {b
        ? <img src={b.image_url} alt={b.name} title={b.name} width={48} height={48}
            className="shrink-0 object-cover self-start"
            style={{ width: 48, height: 48, border: `1px solid ${kind.color}` }} />
        : <div className="shrink-0 self-start" style={{ width: 48, height: 48, background: "var(--panel2)", border: "1px solid var(--line)" }} />}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 flex-wrap leading-tight">
          <span className="mono text-[8px] px-1.5 py-0.5 font-bold tracking-[0.08em] shrink-0"
            style={{ background: kind.color, color: "#0a0a0c" }}>{kind.glyph} {kind.label}</span>
          <span className="text-[14px] font-bold text-[var(--text)] truncate">{r.brawler_name}</span>
          {r.item_name && <span className="mono text-[11px] text-[var(--muted)] truncate">{r.item_name}</span>}
          <span className="mono text-[9px] px-1 py-0.5 ml-auto shrink-0" style={{ color: conf.color }}
            title={conf.title}>{conf.label}</span>
        </div>
        <p className="text-[12px] text-[var(--muted)] mt-1.5 leading-snug">{r.rationale}</p>
        <div className="flex items-center gap-2 mt-2">
          <span className="mono text-[9px] tracking-[0.12em] text-[var(--dim)] shrink-0">VALUE</span>
          <ValueMeter v={r.value_score} />
          <span className="mono text-[10px] tabular-nums shrink-0" style={{ color: "var(--gold)" }}
            title="The brawler's smoothed win rate across the ranked map pool">
            WR {pct(r.meta_winrate)}
          </span>
        </div>
        <CostChips cost={r.cost} gate={r.gate} />
      </div>
    </div>
  );
}

function TagBar({ tag, setTag, onLoad, onClear, loading }: {
  tag: string; setTag: (s: string) => void; onLoad: () => void; onClear: () => void; loading: boolean;
}) {
  return (
    <div className="panel px-3 py-2.5 mb-4 flex flex-wrap items-center gap-x-3 gap-y-2">
      <span className="label">◇ Player</span>
      <form className="flex items-center gap-1.5" onSubmit={(e) => { e.preventDefault(); onLoad(); }}>
        <div className="relative flex items-center">
          <input value={tag} onChange={(e) => setTag(e.target.value.toUpperCase())}
            id="bs-player-tag" name="bs-player-tag" autoComplete="on"
            autoCapitalize="characters" spellCheck={false} enterKeyHint="search"
            placeholder="#GZ95SFSKJ3"
            className="mono bg-[var(--panel2)] border border-[var(--line)] pl-2.5 pr-7 py-1.5 text-[13px] w-44 outline-none focus:border-[var(--accent)] ctl" />
          {tag && (
            <button type="button" onClick={onClear} aria-label="Forget saved tag" title="Forget saved tag"
              className="absolute right-1.5 grid place-items-center w-5 h-5 leading-none text-[var(--muted)] hover:text-[var(--red)] ctl">✕</button>
          )}
        </div>
        <button type="submit" disabled={loading || !tag.trim()} className="seg px-3 py-1.5 disabled:opacity-40">
          {loading ? "…" : "LOAD ↵"}
        </button>
      </form>
      <span className="mono text-[10px] text-[var(--dim)] ml-auto hidden sm:inline">
        find your tag in-game under your profile
      </span>
    </div>
  );
}

function SkeletonList() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="panel flex gap-3 p-3">
          <div className="skeleton w-12 h-12 shrink-0" />
          <div className="flex-1 space-y-2 pt-1">
            <div className="skeleton h-3 w-1/3" />
            <div className="skeleton h-2.5 w-2/3" />
            <div className="skeleton h-2 w-1/2" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function PurchaseAdvisor() {
  const [byId, setById] = useState<Map<number, Brawler>>(new Map());
  const [tag, setTag] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<"idle" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [ownedCount, setOwnedCount] = useState(0);
  const [recs, setRecs] = useState<PurchaseRec[]>([]);
  const reqId = useRef(0);

  const load = useCallback(async (rawTag: string) => {
    const t = rawTag.trim();
    if (!t) return;
    const mine = ++reqId.current;
    setLoading(true); setError(null);
    try {
      const roster = await getRoster(t);
      if (mine !== reqId.current) return;
      if (!roster.loaded) {
        setStatus("error"); setError(roster.error || "roster unavailable"); setRecs([]);
        return;
      }
      localStorage.setItem("bsdraft.tag", roster.tag || t);
      const res = await getPurchases(roster.owned, roster.tag || t, roster.name, 24);
      if (mine !== reqId.current) return;
      setName(roster.name); setOwnedCount(roster.owned.length);
      setRecs(res.recommendations); setStatus("ready");
    } catch (e) {
      if (mine !== reqId.current) return;
      setStatus("error"); setError(String(e)); setRecs([]);
    } finally {
      if (mine === reqId.current) setLoading(false);
    }
  }, []);

  // Load the reference (brawler art/names) and auto-run for a previously saved tag.
  useEffect(() => {
    let live = true;
    getReference().then((ref) => {
      if (!live) return;
      setById(new Map(ref.brawlers.map((b) => [b.id, b])));
    }).catch(() => {});
    const saved = typeof window !== "undefined" ? localStorage.getItem("bsdraft.tag") : null;
    if (saved) { setTag(saved); load(saved); }
    return () => { live = false; };
  }, [load]);

  const clear = useCallback(() => {
    reqId.current++;
    setTag(""); setRecs([]); setStatus("idle"); setError(null); setName("");
    localStorage.removeItem("bsdraft.tag");
  }, []);

  const body = useMemo(() => {
    if (loading && !recs.length) return <SkeletonList />;
    if (status === "error")
      return (
        <div className="panel p-5 text-center" style={{ borderColor: "color-mix(in srgb, var(--red) 40%, transparent)" }}>
          <div className="mono text-[13px]" style={{ color: "var(--red)" }}>⚠ {rosterFailReason(error || "")}</div>
          <div className="mono text-[10px] text-[var(--dim)] mt-2" title={error || undefined}>
            Your roster loads live from Supercell, so this needs the roster service to be online.
          </div>
        </div>
      );
    if (status === "ready" && !recs.length)
      return (
        <div className="panel p-6 text-center">
          <div className="text-[15px] font-bold" style={{ color: "var(--green)" }}>◆ Fully maxed</div>
          <p className="text-[12px] text-[var(--muted)] mt-2">
            Nothing left to buy on {name || "this account"} that we'd rank — every meta brawler you own is built out.
          </p>
        </div>
      );
    if (status === "ready")
      return (
        <>
          <div className="mono text-[10px] text-[var(--dim)] mb-3">
            ▸ analyzed {ownedCount} owned brawler{ownedCount === 1 ? "" : "s"}
            {name && <> on <span className="text-[var(--muted)]">{name}</span></>} · ranked by value across the ranked map pool
          </div>
          <div className="space-y-2">
            {recs.map((r, i) => (
              <RecCard key={`${r.brawler_id}-${r.kind}-${r.item_id ?? i}`} r={r} rank={i + 1} b={byId.get(r.brawler_id)} />
            ))}
          </div>
        </>
      );
    // idle
    return (
      <div className="panel p-6 text-center">
        <div className="text-[15px] font-bold text-[var(--text)]">Enter your player tag to begin</div>
        <p className="text-[12px] text-[var(--muted)] mt-2 max-w-md mx-auto leading-relaxed">
          We read your account's owned brawlers, power levels, and loadouts, then rank the purchases
          that most improve your ranked win rate — from a first gadget to a game-swinging hypercharge.
        </p>
      </div>
    );
  }, [loading, status, error, recs, name, ownedCount, byId]);

  return (
    <div className="p-3 md:p-5 max-w-3xl mx-auto w-full">
      <DocNav current="/purchases" />

      <header className="mb-6">
        <div className="label mb-3" style={{ color: "var(--accent)" }}>▸ UPGRADE PLANNER</div>
        <h1 className="display text-[clamp(1.8rem,5vw,2.8rem)] mb-3">What to upgrade next</h1>
        <p className="text-[14px] leading-relaxed text-[var(--muted)] max-w-2xl">
          Your highest-value next purchases, ranked by how much they lift your ranked win rate.
          Costs are shown for context — we read what you own, not what you can afford.
        </p>
      </header>

      <TagBar tag={tag} setTag={setTag} onLoad={() => load(tag)} onClear={clear} loading={loading} />
      {body}
    </div>
  );
}
