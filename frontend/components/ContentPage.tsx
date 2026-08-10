import Logo from "@/components/Logo";

export type Section = { heading: string; body: string; bullets?: string[] };
export type Content = { title: string; intro: string; sections: Section[] };

// Site-wide nav for the written pages. The board lives at "/", so it leads.
const NAV = [
  { href: "/", label: "Draft board" },
  { href: "/guide", label: "Draft guide" },
  { href: "/how-it-works", label: "How it works" },
  { href: "/faq", label: "FAQ" },
];

// Minimal inline formatter: the drafted copy only ever uses **bold**, so a full markdown
// dependency would be dead weight in a static export. Splitting on the delimiter keeps the
// odd indices as the emphasized runs.
function inline(text: string) {
  return text.split(/\*\*(.+?)\*\*/g).map((part, i) =>
    i % 2 === 1 ? <strong key={i} className="font-semibold text-[var(--text)]">{part}</strong> : <span key={i}>{part}</span>
  );
}

function Prose({ text }: { text: string }) {
  return (
    <>
      {text.split(/\n\n+/).map((para, i) => (
        <p key={i} className="mb-3 last:mb-0">{inline(para)}</p>
      ))}
    </>
  );
}

// Tactical top bar shared by the written pages, so the docs read as the same console as the
// board: mono nav, hairline rules, sharp corners. Long-form prose stays in the readable sans.
function DocNav({ current }: { current: string }) {
  return (
    <nav className="panel flex flex-wrap items-center gap-x-1 gap-y-1 px-3 py-2 mb-8">
      <a href="/" className="flex items-center gap-2 mr-3" aria-label="Brawl Draft home">
        <Logo size={22} />
        <span className="brand-gradient text-[14px]">BRAWL DRAFT</span>
      </a>
      <span className="label hidden sm:inline mr-2">// DOCS</span>
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

export default function ContentPage({ content, current }: { content: Content; current: string }) {
  return (
    <div className="min-h-screen p-3 md:p-5 max-w-3xl mx-auto">
      <DocNav current={current} />

      <header className="mb-9">
        <div className="label mb-3" style={{ color: "var(--accent)" }}>▸ FIELD MANUAL</div>
        <h1 className="display text-[clamp(1.9rem,5vw,3rem)] mb-4">{content.title}</h1>
        <div className="h-px w-full bg-[var(--line)] mb-5" />
        <div className="text-[15px] leading-relaxed text-[var(--muted)]">
          <Prose text={content.intro} />
        </div>
      </header>

      <div className="space-y-9">
        {content.sections.map((s, i) => (
          <section key={s.heading}>
            <div className="flex items-baseline gap-2.5 mb-3">
              <span className="mono text-[11px] tabular-nums text-[var(--dim)] shrink-0 pt-0.5">{String(i + 1).padStart(2, "0")}</span>
              <h2 className="text-lg font-bold tracking-tight text-[var(--text)]">{s.heading}</h2>
            </div>
            <div className="text-[15px] leading-relaxed text-[var(--muted)] pl-[calc(11px+0.625rem)]">
              <Prose text={s.body} />
              {s.bullets && s.bullets.length > 0 && (
                <ul className="mt-3 space-y-2">
                  {s.bullets.map((b, j) => (
                    <li key={j} className="pl-4 relative">
                      <span className="mono absolute left-0 top-0" style={{ color: "var(--accent)" }}>▸</span>
                      {inline(b)}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        ))}
      </div>

      <footer className="mt-12 pt-5 border-t border-[var(--line)] text-xs text-[var(--muted)]">
        <div className="flex flex-wrap gap-x-1 gap-y-1.5 mb-3">
          {NAV.filter((n) => n.href !== current).map((n) => (
            <a key={n.href} href={n.href} className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">{n.label}</a>
          ))}
          <a href="/privacy" className="mono text-[10px] uppercase tracking-[0.08em] px-2 py-1 border border-[var(--line)] hover:border-[var(--line-strong)] hover:text-[var(--text)] ctl">Privacy</a>
        </div>
        <p className="mono text-[10px] leading-relaxed text-[var(--dim)]">
          This content is not affiliated with, endorsed, sponsored, or specifically approved by Supercell and Supercell is not
          responsible for it (
          <a href="https://supercell.com/en/fan-content-policy/" className="underline hover:text-[var(--text)]"
            target="_blank" rel="noopener noreferrer">Fan Content Policy</a>
          ).
        </p>
      </footer>
    </div>
  );
}
