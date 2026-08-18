import type { Metadata } from "next";
import PurchaseAdvisor from "@/components/PurchaseAdvisor";
import SiteFooter from "@/components/SiteFooter";

export const metadata: Metadata = {
  title: "What to Upgrade Next — Brawl Stars Purchase Advisor | Brawl Draft",
  description:
    "Enter your Brawl Stars tag and see your highest-value next purchases — gadgets, star powers, gears, hypercharges, power-11 upgrades, and new meta brawlers — ranked by ranked win-rate impact.",
  alternates: { canonical: "/purchases" },
  openGraph: {
    type: "website",
    siteName: "Brawl Draft",
    title: "What to Upgrade Next — Brawl Stars Purchase Advisor",
    description:
      "Your highest-value next Brawl Stars purchases, ranked by how much they lift your ranked win rate.",
    url: "/purchases",
  },
  twitter: { card: "summary_large_image" },
};

// min-h-screen + footer live here (server-rendered) rather than inside the client component, so the
// nav and Supercell notice are in the exported static HTML even before the roster loads — mirrors
// app/page.tsx. See components/SiteFooter.tsx for why the footer stays out of client components.
export default function PurchasesPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-1">
        <PurchaseAdvisor />
      </div>
      <SiteFooter blurb="Upgrade picks fuse measured item win-rates with brawler meta strength" />
    </div>
  );
}
