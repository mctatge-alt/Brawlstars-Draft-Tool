import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Brawl Draft — Ranked Draft Assistant",
  description: "AI-powered Brawl Stars ranked draft tool: bans, picks, and win-probability.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
