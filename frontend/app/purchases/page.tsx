import type { Metadata } from "next";
import PurchaseAdvisor from "@/components/PurchaseAdvisor";
import SiteFooter from "@/components/SiteFooter";

export const metadata: Metadata = {
  title: "What to Upgrade Next — Brawl Stars Purchase Advisor | Brawl Draft",
  description:
    "Enter your Brawl Stars tag and see your most efficient next purchases — power climbs to your Ranked floor, gadgets, star powers, gears, hypercharges, and new meta brawlers — ranked by ranked win rate per coin, prerequisites included.",
  alternates: { canonical: "/purchases" },
  openGraph: {
    type: "website",
    siteName: "Brawl Draft",
    title: "What to Upgrade Next — Brawl Stars Purchase Advisor",
    description:
      "Your most efficient next Brawl Stars purchases, ranked by how much ranked win rate they buy per coin — with every power climb and prerequisite priced in.",
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
      <SiteFooter blurb="Upgrade picks are ranked by ranked win rate per coin, prerequisites and the Ranked power floor included" />
    </div>
  );
}
