import type { NextConfig } from "next";

const engineUrl =
  process.env.ENGINE_URL || process.env.NEXT_PUBLIC_ENGINE_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/engine/:path*",
        destination: `${engineUrl}/engine/:path*`,
      },
    ];
  },
};

export default nextConfig;
