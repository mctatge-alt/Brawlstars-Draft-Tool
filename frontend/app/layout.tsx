import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Brawl Draft — Ranked Draft Assistant",
  description: "AI-powered Brawl Stars ranked draft tool: bans, picks, and win-probability.",
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
