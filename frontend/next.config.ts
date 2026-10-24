import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    // Keep old bookmarks working without the production /index route collision.
    return [{ source: "/index", destination: "/indexing", permanent: true }];
  },
};

export default nextConfig;
