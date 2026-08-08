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

export default function ContentPage({ content, current }: { content: Content; current: string }) {
  return (
    <div className="min-h-screen p-4 md:p-6 max-w-3xl mx-auto">
      <header className="mb-8">
        <a href="/" className="inline-flex items-center gap-2 mb-5" aria-label="Brawl Draft home">
          <Logo size={30} />
          <span className="font-bold tracking-tight">
            <span className="brand-gradient">Brawl Draft</span>
          </span>
        </a>
        <nav className="flex flex-wrap gap-x-4 gap-y-1.5 text-sm mb-7">
          {NAV.map((n) => (
            <a key={n.href} href={n.href}
              className={n.href === current
                ? "text-[var(--text)] font-semibold"
                : "text-[var(--muted)] hover:text-[var(--text)] transition-colors"}
              aria-current={n.href === current ? "page" : undefined}>
              {n.label}
            </a>
          ))}
        </nav>
        <h1 className="text-3xl font-bold tracking-tight mb-4">{content.title}</h1>
        <div className="text-[15px] leading-relaxed text-[var(--muted)]">
          <Prose text={content.intro} />
        </div>
      </header>

      <div className="space-y-8">
        {content.sections.map((s) => (
          <section key={s.heading}>
            <h2 className="text-lg font-semibold mb-2.5 text-[var(--text)]">{s.heading}</h2>
            <div className="text-[15px] leading-relaxed text-[var(--muted)]">
              <Prose text={s.body} />
              {s.bullets && s.bullets.length > 0 && (
                <ul className="mt-3 space-y-2">
                  {s.bullets.map((b, i) => (
                    <li key={i} className="pl-4 relative">
                      <span className="absolute left-0 top-[0.6em] w-1.5 h-1.5 rounded-full bg-[var(--border-soft)]" />
                      {inline(b)}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        ))}
      </div>

      <footer className="mt-12 pt-5 border-t border-[var(--border)] text-xs text-[var(--muted)]">
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 mb-3">
          {NAV.filter((n) => n.href !== current).map((n) => (
            <a key={n.href} href={n.href} className="hover:text-[var(--text)] transition-colors">{n.label}</a>
          ))}
          <a href="/privacy" className="hover:text-[var(--text)] transition-colors">Privacy</a>
        </div>
        <p>
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
