import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    unoptimized: true
  },
  turbopack: {
    root: /*turbopackIgnore: true*/ process.cwd()
  }
};

export default nextConfig;
