# AdSense Go-Live — enabling the env-gated ad slots

The frontend ships with AdSense wired but dark. This is the checklist for turning ads on
without breaking ad serving. Read before touching `frontend/components/AdSlot.tsx`,
`frontend/app/layout.tsx`'s ad loader, or `frontend/public/ads.txt`.

## How the gating works

`components/AdSlot.tsx` + the script loader in `app/layout.tsx` render **nothing** until
both env vars are set in the Cloudflare Pages build environment:

- `NEXT_PUBLIC_ADSENSE_CLIENT` — the `ca-pub-…` publisher id.
- `NEXT_PUBLIC_ADSENSE_SLOT_FOOTER` — the footer slot id.

Both are inlined at build time (static export), so enabling ads requires a Pages rebuild,
not just an env change.

## Go-live checklist

1. Complete AdSense console setup (site added, review passed).
2. Set the two env vars above in the Cloudflare Pages build env.
3. **In the SAME commit**, add `frontend/public/ads.txt` with the real
   `google.com, pub-…, DIRECT, …` line. A served `ads.txt` **without** the publisher line
   halts ad serving ("Unauthorized").

## Brand assets

AdSense uploads (logo rasters, 5:1 light-theme PNG) live under `frontend/public/brand/` —
see the README there.
