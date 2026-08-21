import katex from "katex";

// KaTeX runs at *build* time, inside a server component — the exported HTML already contains
// the typeset math, so the page ships no math JS and never flashes raw TeX. `throwOnError`
// stays on deliberately: a malformed equation should fail `next build`, not render in red on
// the live site. Output keeps the default HTML+MathML so screen readers and copy-paste work.
function render(tex: string, displayMode: boolean) {
  return katex.renderToString(tex, { displayMode, throwOnError: true, strict: "warn" });
}

/** Inline math, sized to sit in a line of prose. */
export function Tex({ children }: { children: string }) {
  return <span dangerouslySetInnerHTML={{ __html: render(children, false) }} />;
}

/** A centred display equation. Scrolls horizontally on narrow screens rather than overflowing. */
export function TexBlock({ children }: { children: string }) {
  return <div dangerouslySetInnerHTML={{ __html: render(children, true) }} />;
}

/**
 * A titled equation panel: the gold-ruled box the page uses to set out each piece of the math,
 * with an optional "where ..." legend underneath.
 */
export function Equation({ title, tex, children }: { title?: string; tex: string | string[]; children?: React.ReactNode }) {
  const lines = Array.isArray(tex) ? tex : [tex];
  return (
    <div className="eqbox">
      {title && <div className="eq-title">{title}</div>}
      {lines.map((t, i) => <TexBlock key={i}>{t}</TexBlock>)}
      {children && <div className="eq-where">{children}</div>}
    </div>
  );
}
