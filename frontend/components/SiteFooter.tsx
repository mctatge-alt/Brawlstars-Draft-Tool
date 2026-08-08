// Server-rendered on purpose. DraftBoard is a client component that returns a BootScreen until
// the API answers, so anything inside it is absent from the exported HTML — which means crawlers
// (and the AdSense reviewer) would see the homepage as an empty shell with no links to the
// written pages. Keeping the footer out here guarantees the nav and the Supercell notice are in
// the static markup, and visible even if the backend is cold or down.
export default function SiteFooter({ blurb }: { blurb?: string }) {
  return (
    <footer className="max-w-6xl mx-auto px-4 md:px-6 pb-6 mt-6 pt-4 border-t border-[var(--border)] text-center text-xs text-[var(--muted)]">
      <nav className="flex flex-wrap justify-center gap-x-4 gap-y-1.5 mb-3 text-[13px]">
        <a href="/guide" className="hover:text-[var(--text)] transition-colors">Draft guide</a>
        <a href="/how-it-works" className="hover:text-[var(--text)] transition-colors">How it works</a>
        <a href="/faq" className="hover:text-[var(--text)] transition-colors">FAQ</a>
        <a href="/privacy" className="hover:text-[var(--text)] transition-colors">Privacy</a>
      </nav>
      {blurb && <>{blurb} · </>}
      This content is not affiliated with, endorsed, sponsored, or specifically approved by Supercell and Supercell is not
      responsible for it (
      <a href="https://supercell.com/en/fan-content-policy/" className="underline hover:text-[var(--text)]"
        target="_blank" rel="noopener noreferrer">Fan Content Policy</a>
      )
    </footer>
  );
}
