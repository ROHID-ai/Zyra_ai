import type { NextConfig } from "next";

const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
const lanHost = process.env.NEXT_PUBLIC_LAN_HOST ?? "192.168.0.10";

const nextConfig: NextConfig = {
  // Allow phone/tablet on same Wi‑Fi to load dev assets (Next.js 16+)
  allowedDevOrigins: [
    lanHost,
    `http://${lanHost}:3000`,
    `https://${lanHost}:3000`,
  ],
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
