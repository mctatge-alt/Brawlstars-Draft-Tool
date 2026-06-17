import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",               // static site -> ./out (Cloudflare Pages, no Node runtime needed)
  images: { unoptimized: true },  // required for static export if next/image is ever used
};

export default nextConfig;
