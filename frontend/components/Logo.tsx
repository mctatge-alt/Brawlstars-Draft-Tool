// The Brawl Draft mark: a "B" built from two D-shaped pick slots — blue team over red
// team — with the gold playhead between them pointing at the next pick. Kept inline (vs
// <img src="/logo.svg">) so the header never flashes an empty box while the SVG loads.
// Mirrors public/logo.svg and app/icon.svg — change all three together.
export default function Logo({ size = 26, className }: { size?: number; className?: string }) {
  return (
    <svg viewBox="0 0 512 512" width={size} height={size} className={className} aria-hidden="true">
      <defs>
        <linearGradient id="bd-logo-bg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#121b2b" />
          <stop offset="1" stopColor="#0b1220" />
        </linearGradient>
      </defs>
      <rect width="512" height="512" rx="112" fill="url(#bd-logo-bg)" />
      <path fill="#3b82f6" fillRule="evenodd" d="M160 112H298A54 54 0 0 1 298 220H160ZM204 140H240A26 26 0 0 1 240 192H204Z" />
      <path fill="#e8c34a" d="M160 240H312L352 256 312 272H160Z" />
      <path fill="#e0566f" fillRule="evenodd" d="M160 292H298A54 54 0 0 1 298 400H160ZM204 320H240A26 26 0 0 1 240 372H204Z" />
    </svg>
  );
}
