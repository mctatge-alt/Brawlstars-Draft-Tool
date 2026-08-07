"use client";

import { useEffect, useRef } from "react";

// Both come from the Cloudflare Pages build env (inlined at build time, like NEXT_PUBLIC_API_BASE).
// Until BOTH are set the component renders nothing, so the site ships ad-free by default and
// flips on with a rebuild — no code change.
const CLIENT = process.env.NEXT_PUBLIC_ADSENSE_CLIENT; // e.g. ca-pub-1234567890123456
// `satisfies` (not a Record annotation) keeps the keys literal, so `name: keyof typeof SLOTS`
// actually rejects unknown slot names at compile time instead of widening to string.
const SLOTS = {
  footer: process.env.NEXT_PUBLIC_ADSENSE_SLOT_FOOTER, // numeric ad-unit id from the AdSense console
} satisfies Record<string, string | undefined>;

declare global {
  interface Window { adsbygoogle?: unknown[] }
}

// One responsive display unit. Placement rules this component assumes (AdSense placement
// policy, support.google.com/adsense/answer/1346295): never inside the active draft flow,
// well clear of tappable game-like UI (the mt-24 ≈ 96px gap — policy suggests ~150px from a
// game window's edge, and the footer adds more), and labeled with one of the two permitted
// strings ("Advertisements" / "Sponsored Links"). The min-height reserves space up front so
// a late fill doesn't shift the board (CLS).
export default function AdSlot({ name }: { name: keyof typeof SLOTS }) {
  const slot = SLOTS[name];
  const pushed = useRef(false);

  useEffect(() => {
    if (!CLIENT || !slot || pushed.current) return;
    try {
      (window.adsbygoogle = window.adsbygoogle || []).push({});
      pushed.current = true;
    } catch {
      // adsbygoogle not loaded (blocked, or script still fetching its queue) — fine, the
      // reserved box just stays empty.
    }
  }, [slot]);

  if (!CLIENT || !slot) return null;
  return (
    <div className="mt-24" style={{ minHeight: 100 }}>
      <div className="text-[9px] uppercase tracking-wide text-[var(--muted)] mb-1 text-center">Advertisements</div>
      <ins
        className="adsbygoogle"
        style={{ display: "block", minHeight: 90 }}
        data-ad-client={CLIENT}
        data-ad-slot={slot}
        data-ad-format="auto"
        data-full-width-responsive="true"
      />
    </div>
  );
}
