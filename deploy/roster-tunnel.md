# Per-visitor roster via Cloudflare Tunnel

Goal: let the live site (`brawlstars-draft-tool.pages.dev`) personalize to **any visitor's**
roster (owned brawlers + power levels), by routing just `/api/roster` to an API instance on
the home machine — the one whose IP is whitelisted with the Supercell key.

Everything else (recommend, top_picks, reference, meta) stays on Render. The frontend is
already wired: it calls `${NEXT_PUBLIC_ROSTER_BASE}/api/roster?tag=…`, falling back to the
main API when that env var is unset.

```
visitor ─▶ pages.dev (Cloudflare) ─▶ Render API        (meta / recommend / top_picks)
                       └──────────▶ roster.brawldraft.com ─▶ Cloudflare Tunnel ─▶ home API ─▶ Supercell
                                                                            (whitelisted IP)
```

## Prerequisites (yours)
- A **domain on your Cloudflare account** (Cloudflare Registrar ≈ $8–10/yr, or move an existing one in). Needed for a stable tunnel hostname.
- Home IP already whitelisted with the Supercell key — true today (the crawler uses it).

## Steps

**1. Run the home API** (serves the live roster)
```
sed "s#/ABSOLUTE/PATH/TO/Brawlstars-Draft-Tool#$PWD#g" deploy/com.bsdraft.api.plist > ~/Library/LaunchAgents/com.bsdraft.api.plist
launchctl load ~/Library/LaunchAgents/com.bsdraft.api.plist
curl 'http://127.0.0.1:8000/api/roster?tag=YOURTAG'      # -> {"loaded":true,...}
```

**2. Install + authenticate cloudflared** (browser opens — pick your domain)
```
brew install cloudflared
cloudflared tunnel login
```

**3. Create the tunnel** (note the UUID it prints)
```
cloudflared tunnel create bsdraft-roster
```

**4. Fill `deploy/cloudflared.yml`** — set `<TUNNEL-UUID>`, `mitchelltatge` (your macOS user), and
`roster.brawldraft.com`.

**5. Route the hostname to the tunnel**
```
cloudflared tunnel route dns bsdraft-roster roster.brawldraft.com
```

**6. Run the tunnel** (fill paths + `which cloudflared` into the plist first)
```
cp deploy/com.bsdraft.tunnel.plist ~/Library/LaunchAgents/com.bsdraft.tunnel.plist
launchctl load ~/Library/LaunchAgents/com.bsdraft.tunnel.plist
curl 'https://roster.brawldraft.com/api/roster?tag=YOURTAG'    # -> {"loaded":true,...}
```

**7. Point the live frontend at it** — Cloudflare Pages → project → Settings → Environment
variables: `NEXT_PUBLIC_ROSTER_BASE = https://roster.brawldraft.com`, then redeploy.

Done — entering a tag on the live site now loads that player's roster and filters to what
they own.

## Hardening (already wired)
- Home API binds `127.0.0.1` only; the tunnel exposes **only** `^/api/roster` (see `cloudflared.yml`).
- `CORS_ORIGINS` in the API plist locks responses to the Pages origin. Add your custom domain (comma-separated) if you put the site on one.

## Caveats
- The home machine must stay on (it already runs the crawler).
- If your ISP rotates your IP, re-whitelist the new one in the Supercell developer portal.
- Cost: just the domain (~$8/yr). No monthly fees.
