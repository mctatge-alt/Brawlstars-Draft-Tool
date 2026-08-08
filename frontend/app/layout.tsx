import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  // metadataBase turns the opengraph-image.png file convention into the absolute URL that
  // scrapers require — without it Next emits a relative path and warns at build time.
  metadataBase: new URL("https://brawldraft.com"),
  title: "Brawl Draft — Ranked Draft Assistant",
  description: "AI-powered Brawl Stars ranked draft tool: bans, picks, and win-probability.",
  openGraph: {
    type: "website",
    siteName: "Brawl Draft",
    title: "Brawl Draft — Ranked Draft Assistant",
    description: "AI-powered Brawl Stars ranked draft tool: bans, picks, and win-probability.",
    url: "/",
  },
  // Next reuses opengraph-image.png for twitter:image too, so the card needs no duplicate PNG —
  // only the card type, which is what makes X render it full-bleed instead of as a thumbnail.
  twitter: { card: "summary_large_image" },
};

// Set in the Cloudflare Pages build env to turn ads on (see components/AdSlot.tsx).
const ADSENSE_CLIENT = process.env.NEXT_PUBLIC_ADSENSE_CLIENT;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        {children}
        {/* Plain <script async> (not next/script): React 19 hoists async scripts into <head>
            during prerender, so the tag is present in the exported static HTML — which is what
            AdSense's "code snippet" site verification crawls for. next/script injects at
            hydration and fails that check. */}
        {ADSENSE_CLIENT && (
          <script
            async
            src={`https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${ADSENSE_CLIENT}`}
            crossOrigin="anonymous"
          />
        )}
      </body>
    </html>
  );
}
